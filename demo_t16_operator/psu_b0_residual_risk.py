"""Observable residual-risk gate for the PSU spectral preconditioner.

The gate never sees a reference field at deployment. It uses the initial
whitened residual, exact adjoint gradient, declared view/noise metadata, and
the candidate-versus-Sobolev directions. A split-conformal lower bound on the
candidate's expected field-error gain determines whether the learned
preconditioner or the fixed Sobolev fallback is used for the complete solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .psu_b0_spectral_preconditioner import (
    IterativeReconstruction,
    _relative_objective,
    _weighted_measurement_terms,
)


RESIDUAL_RISK_SCHEMA = "psu-b0-observable-conformal-residual-risk-1.0"
RISK_FEATURE_NAMES = (
    "active_fraction",
    "white_rms_active_mean",
    "white_rms_active_std",
    "white_rms_active_max",
    "white_component_log_ratio_abs",
    "white_component_correlation_abs",
    "gradient_log_rms",
    "gradient_spectral_centroid",
    "gradient_high_frequency_fraction",
    "gradient_axis_anisotropy",
    "direction_cosine",
    "direction_log_norm_ratio",
    "direction_relative_correction",
    "candidate_log_gain_span",
    "candidate_log_gain_center",
    "controller_coefficient_rms",
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=1).clamp_min(1)
    selected = torch.where(mask, values, torch.zeros_like(values))
    return torch.sum(selected, dim=1) / count


def _masked_std(
    values: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    count = mask.sum(dim=1).clamp_min(1)
    centered = torch.where(mask, values - mean[:, None], torch.zeros_like(values))
    return torch.sqrt(torch.sum(centered.square(), dim=1) / count)


def _gradient_spectral_features(
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spectrum = torch.fft.rfftn(gradient[:, 0], dim=(-3, -2, -1))
    power = spectrum.abs().square()
    nz, ny, nx = gradient.shape[-3:]
    fz = torch.fft.fftfreq(
        nz,
        device=gradient.device,
        dtype=gradient.dtype,
    ) / 0.5
    fy = torch.fft.fftfreq(
        ny,
        device=gradient.device,
        dtype=gradient.dtype,
    ) / 0.5
    fx = torch.fft.rfftfreq(
        nx,
        device=gradient.device,
        dtype=gradient.dtype,
    ) / 0.5
    zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square() + zz.square())
    total = power.sum(dim=(1, 2, 3)).clamp_min(1e-20)
    centroid = torch.sum(power * radius[None], dim=(1, 2, 3)) / total
    high = torch.sum(
        power * (radius >= 0.72).to(power)[None],
        dim=(1, 2, 3),
    ) / total
    axis = torch.stack(
        (
            torch.sum(power * xx.square()[None], dim=(1, 2, 3)),
            torch.sum(power * yy.square()[None], dim=(1, 2, 3)),
            torch.sum(power * zz.square()[None], dim=(1, 2, 3)),
        ),
        dim=1,
    )
    axis_mean = axis.mean(dim=1).clamp_min(1e-20)
    anisotropy = (axis.max(dim=1).values - axis.min(dim=1).values) / axis_mean
    return centroid, high, anisotropy


def observable_risk_features(
    gradient: torch.Tensor,
    *,
    residual_uv: torch.Tensor,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    candidate_direction: torch.Tensor,
    fallback_direction: torch.Tensor,
    candidate_diagnostics: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build truth-free first-stage risk features."""

    if gradient.ndim != 5 or gradient.shape[1] != 1:
        raise ValueError("gradient must have shape [batch,1,z,y,x]")
    batch, ray_count, components = residual_uv.shape
    view_count = view_mask.shape[1]
    if components != 2 or ray_count != view_count * int(rays_per_view):
        raise ValueError("residual rays do not match the view layout")
    if sigma_by_view.shape != view_mask.shape or len(view_mask) != batch:
        raise ValueError("view metadata must align with the batch")
    active = view_mask > 0.5
    active_count = active.sum(dim=1).clamp_min(1)
    active_fraction = active_count.to(gradient) / float(view_count)

    white = residual_uv.reshape(batch, view_count, int(rays_per_view), 2)
    white = white / sigma_by_view[:, :, None, None]
    per_view_rms = torch.sqrt(
        torch.mean(white.square(), dim=(2, 3)).clamp_min(1e-20)
    )
    white_mean = _masked_mean(per_view_rms, active)
    white_std = _masked_std(per_view_rms, active, white_mean)
    white_max = torch.max(
        torch.where(active, per_view_rms, torch.full_like(per_view_rms, -torch.inf)),
        dim=1,
    ).values

    active_rays = active[:, :, None, None].to(white)
    denominator = (active_count * int(rays_per_view)).to(white).clamp_min(1)
    component_energy = torch.sum(white.square() * active_rays, dim=(1, 2))
    component_rms = torch.sqrt(
        component_energy / denominator[:, None]
    ).clamp_min(1e-12)
    component_log_ratio = torch.abs(
        torch.log(component_rms[:, 0] / component_rms[:, 1])
    )
    component_cross = torch.sum(
        white[..., 0] * white[..., 1] * active_rays[..., 0],
        dim=(1, 2),
    ) / denominator
    component_correlation = torch.abs(
        component_cross
        / (component_rms[:, 0] * component_rms[:, 1]).clamp_min(1e-12)
    )

    gradient_rms = torch.sqrt(
        torch.mean(gradient.square(), dim=(1, 2, 3, 4)).clamp_min(1e-20)
    )
    spectral_centroid, high_fraction, axis_anisotropy = (
        _gradient_spectral_features(gradient)
    )
    candidate_flat = candidate_direction.flatten(1)
    fallback_flat = fallback_direction.flatten(1)
    candidate_norm = torch.linalg.vector_norm(
        candidate_flat,
        dim=1,
    ).clamp_min(1e-20)
    fallback_norm = torch.linalg.vector_norm(
        fallback_flat,
        dim=1,
    ).clamp_min(1e-20)
    cosine = torch.sum(candidate_flat * fallback_flat, dim=1) / (
        candidate_norm * fallback_norm
    )
    correction = torch.linalg.vector_norm(
        candidate_flat - fallback_flat,
        dim=1,
    ) / fallback_norm
    gain_minimum = candidate_diagnostics["gain_minimum"].clamp_min(1e-20)
    gain_maximum = candidate_diagnostics["gain_maximum"].clamp_min(1e-20)
    gain_span = torch.log(gain_maximum / gain_minimum)
    gain_center = 0.5 * torch.log(gain_maximum * gain_minimum)
    coefficients = candidate_diagnostics.get("controller_coefficients")
    if coefficients is None:
        coefficient_rms = torch.zeros_like(active_fraction)
    else:
        coefficient_rms = torch.sqrt(torch.mean(coefficients.square(), dim=1))

    features = torch.stack(
        (
            active_fraction,
            white_mean,
            white_std,
            white_max,
            component_log_ratio,
            component_correlation,
            torch.log(gradient_rms),
            spectral_centroid,
            high_fraction,
            axis_anisotropy,
            cosine,
            torch.log(candidate_norm / fallback_norm),
            correction,
            gain_span,
            gain_center,
            coefficient_rms,
        ),
        dim=1,
    )
    if features.shape[1] != len(RISK_FEATURE_NAMES):
        raise RuntimeError("risk feature schema and implementation disagree")
    if torch.any(~torch.isfinite(features)):
        raise ValueError("risk features contain non-finite values")
    return features


@dataclass(frozen=True)
class RidgeRiskFit:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    ridge_lambda: float
    validation_rmse: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        standardized = (values - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.intercept

    def distance(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        standardized = (values - self.feature_mean) / self.feature_scale
        return np.sqrt(np.mean(np.square(standardized), axis=1))


def fit_ridge_risk_model(
    train_features: np.ndarray,
    train_gain_percent: np.ndarray,
    validation_features: np.ndarray,
    validation_gain_percent: np.ndarray,
    *,
    ridge_grid: tuple[float, ...],
) -> RidgeRiskFit:
    """Fit and validation-select a deterministic standardized ridge model."""

    train_x = np.asarray(train_features, dtype=np.float64)
    train_y = np.asarray(train_gain_percent, dtype=np.float64).reshape(-1)
    validation_x = np.asarray(validation_features, dtype=np.float64)
    validation_y = np.asarray(validation_gain_percent, dtype=np.float64).reshape(-1)
    if train_x.ndim != 2 or validation_x.ndim != 2:
        raise ValueError("features must be matrices")
    if train_x.shape[1] != validation_x.shape[1] or train_x.shape[1] != len(
        RISK_FEATURE_NAMES
    ):
        raise ValueError("feature widths do not match the risk schema")
    if len(train_x) != len(train_y) or len(validation_x) != len(validation_y):
        raise ValueError("features and gains must align")
    if not ridge_grid or any(float(value) <= 0.0 for value in ridge_grid):
        raise ValueError("ridge_grid must contain positive values")
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized_train = (train_x - mean) / scale
    standardized_validation = (validation_x - mean) / scale
    target_mean = float(np.mean(train_y))
    centered_target = train_y - target_mean
    gram = standardized_train.T @ standardized_train
    rhs = standardized_train.T @ centered_target
    identity = np.eye(train_x.shape[1], dtype=np.float64)
    candidates = []
    for value in ridge_grid:
        ridge_lambda = float(value)
        coefficients = np.linalg.solve(gram + ridge_lambda * identity, rhs)
        prediction = standardized_validation @ coefficients + target_mean
        rmse = float(np.sqrt(np.mean((prediction - validation_y) ** 2)))
        candidates.append((rmse, ridge_lambda, coefficients))
    rmse, selected, coefficients = min(
        candidates,
        key=lambda row: (row[0], row[1]),
    )
    return RidgeRiskFit(
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        intercept=target_mean,
        ridge_lambda=selected,
        validation_rmse=rmse,
    )


def one_sided_conformal_quantile(
    predicted_gain_percent: np.ndarray,
    actual_gain_percent: np.ndarray,
    *,
    alpha: float,
) -> float:
    """Return the finite-sample one-sided overprediction quantile."""

    predicted = np.asarray(predicted_gain_percent, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_gain_percent, dtype=np.float64).reshape(-1)
    if len(predicted) != len(actual) or len(predicted) < 2:
        raise ValueError("conformal arrays must be aligned and nontrivial")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    nonconformity = predicted - actual
    rank = int(np.ceil((len(nonconformity) + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), len(nonconformity))
    return float(np.sort(nonconformity)[rank - 1])


class CalibratedResidualRiskDirection(nn.Module):
    """Use one first-stage truth-free decision for the complete reconstruction."""

    def __init__(
        self,
        *,
        candidate: nn.Module,
        fallback: nn.Module,
        stages: int,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        coefficients: np.ndarray,
        intercept: float,
        overprediction_quantile: float,
        distance_threshold: float,
        minimum_lower_gain_percent: float,
        minimum_active_views: int,
        maximum_active_views: int,
    ) -> None:
        super().__init__()
        if int(stages) < 1:
            raise ValueError("stages must be positive")
        arrays = [
            np.asarray(feature_mean, dtype=np.float32),
            np.asarray(feature_scale, dtype=np.float32),
            np.asarray(coefficients, dtype=np.float32),
        ]
        if any(value.shape != (len(RISK_FEATURE_NAMES),) for value in arrays):
            raise ValueError("risk model arrays must match the feature schema")
        if np.any(arrays[1] <= 0.0):
            raise ValueError("feature scales must be positive")
        self.candidate = candidate
        self.fallback = fallback
        self.stages = int(stages)
        self.minimum_active_views = int(minimum_active_views)
        self.maximum_active_views = int(maximum_active_views)
        self.intercept = float(intercept)
        self.overprediction_quantile = float(overprediction_quantile)
        self.distance_threshold = float(distance_threshold)
        self.minimum_lower_gain_percent = float(minimum_lower_gain_percent)
        self.register_buffer("feature_mean", torch.from_numpy(arrays[0]))
        self.register_buffer("feature_scale", torch.from_numpy(arrays[1]))
        self.register_buffer("coefficients", torch.from_numpy(arrays[2]))
        self._cached_trust: torch.Tensor | None = None
        self._cached_prediction: torch.Tensor | None = None
        self._cached_lower_bound: torch.Tensor | None = None
        self._cached_distance: torch.Tensor | None = None

    def forward(
        self,
        gradient: torch.Tensor,
        *,
        residual_uv: torch.Tensor,
        sigma_by_view: torch.Tensor,
        view_mask: torch.Tensor,
        rays_per_view: int,
        stage_fraction: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        candidate, candidate_diagnostics = self.candidate(
            gradient,
            residual_uv=residual_uv,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
            stage_fraction=stage_fraction,
        )
        fallback, fallback_diagnostics = self.fallback(
            gradient,
            residual_uv=residual_uv,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
            stage_fraction=stage_fraction,
        )
        first_stage = abs(float(stage_fraction) - 1.0 / self.stages) <= 1e-7
        if first_stage:
            features = observable_risk_features(
                gradient,
                residual_uv=residual_uv,
                sigma_by_view=sigma_by_view,
                view_mask=view_mask,
                rays_per_view=rays_per_view,
                candidate_direction=candidate,
                fallback_direction=fallback,
                candidate_diagnostics=candidate_diagnostics,
            )
            standardized = (
                features - self.feature_mean.to(features)
            ) / self.feature_scale.to(features)
            prediction = (
                standardized @ self.coefficients.to(features) + self.intercept
            )
            lower_bound = prediction - self.overprediction_quantile
            distance = torch.sqrt(torch.mean(standardized.square(), dim=1))
            active_count = torch.sum(view_mask > 0.5, dim=1)
            trust = (
                (active_count >= self.minimum_active_views)
                & (active_count <= self.maximum_active_views)
                & (lower_bound >= self.minimum_lower_gain_percent)
                & (distance <= self.distance_threshold)
            )
            self._cached_trust = trust
            self._cached_prediction = prediction
            self._cached_lower_bound = lower_bound
            self._cached_distance = distance
        if (
            self._cached_trust is None
            or self._cached_prediction is None
            or self._cached_lower_bound is None
            or self._cached_distance is None
            or len(self._cached_trust) != len(gradient)
        ):
            raise RuntimeError("risk direction must begin at the first stage")
        trust = self._cached_trust
        selector = trust[:, None, None, None, None]
        direction = torch.where(selector, candidate, fallback)
        diagnostics = {
            key: value
            for key, value in candidate_diagnostics.items()
            if key not in {"gain_minimum", "gain_maximum", "gain_geometric_mean"}
        }
        diagnostics.update(
            {
                "gain_minimum": torch.where(
                    trust,
                    candidate_diagnostics["gain_minimum"],
                    fallback_diagnostics["gain_minimum"],
                ),
                "gain_maximum": torch.where(
                    trust,
                    candidate_diagnostics["gain_maximum"],
                    fallback_diagnostics["gain_maximum"],
                ),
                "gain_geometric_mean": torch.where(
                    trust,
                    candidate_diagnostics["gain_geometric_mean"],
                    fallback_diagnostics["gain_geometric_mean"],
                ),
                "residual_risk_trust": trust.to(gradient),
                "predicted_gain_percent": self._cached_prediction,
                "lower_gain_bound_percent": self._cached_lower_bound,
                "feature_distance": self._cached_distance,
            }
        )
        return direction, diagnostics


def exact_line_search_reconstruction_with_risk_gate(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    stages: int,
    direction: CalibratedResidualRiskDirection,
    denominator_floor: float = 1e-20,
) -> IterativeReconstruction:
    """Run the gated solve with the same logical calls as the ungated solver."""

    count = int(stages)
    active, sigma = _weighted_measurement_terms(
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )
    support = operator.support[None, None].to(observation_uv)
    current = torch.zeros(
        (len(observation_uv), 1, *operator.grid_shape),
        dtype=observation_uv.dtype,
        device=observation_uv.device,
    )
    residual = active * observation_uv
    initial_objective = torch.sum(
        (residual / sigma).square(),
        dim=(1, 2),
    ).clamp_min(float(denominator_floor))
    history: list[dict[str, torch.Tensor]] = []
    for stage in range(count):
        weighted_residual = residual / sigma.square()
        gradient = operator.adjoint(weighted_residual)
        proposed, diagnostics = direction(
            gradient,
            residual_uv=residual,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
            stage_fraction=(stage + 1) / count,
        )
        search = proposed * support
        projected = active * operator(search)
        numerator = torch.sum(weighted_residual * projected, dim=(1, 2))
        denominator = torch.sum(
            (projected / sigma).square(),
            dim=(1, 2),
        ).clamp_min(float(denominator_floor))
        alpha = torch.clamp_min(numerator / denominator, 0.0)
        objective_before = _relative_objective(
            residual,
            sigma,
            initial_objective,
        )
        current = current + alpha[:, None, None, None, None] * search
        residual = residual - alpha[:, None, None] * projected
        objective_after = _relative_objective(
            residual,
            sigma,
            initial_objective,
        )
        history.append(
            {
                "stage": torch.full_like(alpha, stage + 1, dtype=torch.int64),
                "alpha": alpha,
                "directional_derivative": numerator,
                "relative_objective_before": objective_before,
                "relative_objective_after": objective_after,
                **diagnostics,
            }
        )
    return IterativeReconstruction(
        volume=current,
        residual_uv=residual,
        history=history,
        forward_calls=count,
        adjoint_calls=count,
    )
