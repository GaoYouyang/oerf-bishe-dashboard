"""Canonical observable multi-veto gate for the next PSU B0 risk candidate.

This module is post-open method-development code. It is intentionally separate
from the frozen OCRRG implementation so that the original audit remains
reproducible. No threshold or result in this file is confirmatory evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .psu_b0_residual_risk import (
    RISK_FEATURE_NAMES,
    observable_risk_features,
)


MULTIVETO_FEATURE_SCHEMA = "psu-b0-canonical-support-projected-risk-2.0"
SPECTRAL_STRESS_WEIGHTS = {
    "direction_relative_correction": 1.0,
    "candidate_log_gain_span": 1.0,
    "gradient_spectral_centroid": -1.0,
    "gradient_high_frequency_fraction": -1.0,
}
CORRELATED_CAMERA_STRESS_WEIGHTS = {
    "white_component_correlation_abs": 1.0,
    "gradient_log_rms": -1.0,
    "white_rms_active_mean": -1.0,
    "gradient_axis_anisotropy": 1.0,
}


def canonical_observable_risk_features(
    gradient: torch.Tensor,
    *,
    residual_uv: torch.Tensor,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    candidate_direction: torch.Tensor,
    fallback_direction: torch.Tensor,
    candidate_diagnostics: dict[str, torch.Tensor],
    support: torch.Tensor,
) -> torch.Tensor:
    """Apply the solver support before every direction-derived risk feature."""

    if support.ndim == 3:
        projected_support = support[None, None]
    elif support.ndim == 5 and support.shape[0] == support.shape[1] == 1:
        projected_support = support
    else:
        raise ValueError("support must have shape [z,y,x] or [1,1,z,y,x]")
    if tuple(projected_support.shape[-3:]) != tuple(gradient.shape[-3:]):
        raise ValueError("support and gradient grids do not match")
    projected_support = projected_support.to(gradient)
    return observable_risk_features(
        gradient,
        residual_uv=residual_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
        candidate_direction=candidate_direction * projected_support,
        fallback_direction=fallback_direction * projected_support,
        candidate_diagnostics=candidate_diagnostics,
    )


def _stress_indices_and_weights(
    definition: dict[str, float],
) -> tuple[list[int], list[float]]:
    return (
        [RISK_FEATURE_NAMES.index(name) for name in definition],
        [float(value) for value in definition.values()],
    )


def observable_stress_scores(
    standardized_features: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """Return post-open spectral and correlated-camera stress scores."""

    spectral_indices, spectral_weights = _stress_indices_and_weights(
        SPECTRAL_STRESS_WEIGHTS
    )
    camera_indices, camera_weights = _stress_indices_and_weights(
        CORRELATED_CAMERA_STRESS_WEIGHTS
    )
    if isinstance(standardized_features, torch.Tensor):
        values = standardized_features
        spectral = torch.mean(
            values[:, spectral_indices]
            * torch.as_tensor(
                spectral_weights,
                dtype=values.dtype,
                device=values.device,
            )[None],
            dim=1,
        )
        camera = torch.mean(
            values[:, camera_indices]
            * torch.as_tensor(
                camera_weights,
                dtype=values.dtype,
                device=values.device,
            )[None],
            dim=1,
        )
        return spectral, camera
    values = np.asarray(standardized_features, dtype=np.float64)
    spectral = np.mean(
        values[:, spectral_indices]
        * np.asarray(spectral_weights, dtype=np.float64)[None],
        axis=1,
    )
    camera = np.mean(
        values[:, camera_indices]
        * np.asarray(camera_weights, dtype=np.float64)[None],
        axis=1,
    )
    return spectral, camera


@dataclass(frozen=True)
class BalancedViewMaskPlan:
    masks: torch.Tensor
    requested_count_by_active_views: dict[int, int]
    unique_pattern_count_by_active_views: dict[int, int]
    pattern_reuse_count_by_active_views: dict[int, int]


def balanced_view_masks(
    *,
    count: int,
    view_count: int,
    active_view_counts: Sequence[int],
    seed: int,
) -> BalancedViewMaskPlan:
    """Build a deterministic balanced plan, allowing unavoidable pattern reuse.

    Nine active views have only one possible mask. Requiring every field to use
    a globally unique mask therefore makes balanced 6/7/8/9-view development
    mathematically impossible. This planner balances field counts and reports
    the reuse explicitly instead.
    """

    total = int(count)
    views = int(view_count)
    active = tuple(int(value) for value in active_view_counts)
    if total < 1 or views < 1 or not active:
        raise ValueError("mask-plan inputs must be nonempty and positive")
    if len(set(active)) != len(active):
        raise ValueError("active_view_counts must be unique")
    if any(value < 1 or value > views for value in active):
        raise ValueError("active-view count lies outside the camera set")
    base, remainder = divmod(total, len(active))
    requested = {
        value: base + int(index < remainder)
        for index, value in enumerate(active)
    }
    generator = np.random.default_rng(int(seed))
    rows = []
    unique_counts = {}
    reuse_counts = {}
    for active_count in active:
        patterns = list(combinations(range(views), active_count))
        order = generator.permutation(len(patterns))
        patterns = [patterns[int(index)] for index in order]
        needed = requested[active_count]
        selected = [patterns[index % len(patterns)] for index in range(needed)]
        unique_counts[active_count] = len(set(selected))
        reuse_counts[active_count] = needed - unique_counts[active_count]
        for pattern in selected:
            mask = torch.zeros(views, dtype=torch.float32)
            mask[list(pattern)] = 1.0
            rows.append((active_count, mask))
    order = generator.permutation(len(rows))
    masks = torch.stack([rows[int(index)][1] for index in order], dim=0)
    return BalancedViewMaskPlan(
        masks=masks,
        requested_count_by_active_views=requested,
        unique_pattern_count_by_active_views=unique_counts,
        pattern_reuse_count_by_active_views=reuse_counts,
    )


class ObservableMultiVetoDirection(nn.Module):
    """Select one learned or Sobolev direction using a canonical v2 contract."""

    def __init__(
        self,
        *,
        candidate: nn.Module,
        fallback: nn.Module,
        support: torch.Tensor,
        stages: int,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        coefficients: np.ndarray,
        intercept: float,
        overprediction_quantile: float,
        distance_threshold: float,
        minimum_lower_gain_percent: float,
        spectral_stress_threshold: float,
        camera_stress_threshold: float,
        six_view_extra_margin_percent: float,
        minimum_active_views: int,
        maximum_active_views: int,
    ) -> None:
        super().__init__()
        width = len(RISK_FEATURE_NAMES)
        arrays = (
            np.asarray(feature_mean, dtype=np.float32),
            np.asarray(feature_scale, dtype=np.float32),
            np.asarray(coefficients, dtype=np.float32),
        )
        if any(value.shape != (width,) for value in arrays):
            raise ValueError("risk arrays must match the canonical feature width")
        if np.any(arrays[1] <= 0.0):
            raise ValueError("feature scales must be positive")
        if int(stages) < 1:
            raise ValueError("stages must be positive")
        if support.ndim != 3:
            raise ValueError("support must have shape [z,y,x]")
        self.candidate = candidate
        self.fallback = fallback
        self.stages = int(stages)
        self.intercept = float(intercept)
        self.overprediction_quantile = float(overprediction_quantile)
        self.distance_threshold = float(distance_threshold)
        self.minimum_lower_gain_percent = float(minimum_lower_gain_percent)
        self.spectral_stress_threshold = float(spectral_stress_threshold)
        self.camera_stress_threshold = float(camera_stress_threshold)
        self.six_view_extra_margin_percent = float(
            six_view_extra_margin_percent
        )
        self.minimum_active_views = int(minimum_active_views)
        self.maximum_active_views = int(maximum_active_views)
        self.register_buffer("support", support.detach().to(torch.float32))
        self.register_buffer("feature_mean", torch.from_numpy(arrays[0]))
        self.register_buffer("feature_scale", torch.from_numpy(arrays[1]))
        self.register_buffer("coefficients", torch.from_numpy(arrays[2]))
        self._cached: dict[str, torch.Tensor] | None = None

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
            features = canonical_observable_risk_features(
                gradient,
                residual_uv=residual_uv,
                sigma_by_view=sigma_by_view,
                view_mask=view_mask,
                rays_per_view=rays_per_view,
                candidate_direction=candidate,
                fallback_direction=fallback,
                candidate_diagnostics=candidate_diagnostics,
                support=self.support,
            )
            standardized = (
                features - self.feature_mean.to(features)
            ) / self.feature_scale.to(features)
            prediction = (
                standardized @ self.coefficients.to(features) + self.intercept
            )
            lower = prediction - self.overprediction_quantile
            distance = torch.sqrt(torch.mean(standardized.square(), dim=1))
            spectral_stress, camera_stress = observable_stress_scores(standardized)
            active_count = torch.sum(view_mask > 0.5, dim=1)
            required_margin = torch.full_like(
                lower,
                self.minimum_lower_gain_percent,
            )
            required_margin = required_margin + (
                active_count == 6
            ).to(lower) * self.six_view_extra_margin_percent
            trust = (
                (active_count >= self.minimum_active_views)
                & (active_count <= self.maximum_active_views)
                & (lower >= required_margin)
                & (distance <= self.distance_threshold)
                & (spectral_stress <= self.spectral_stress_threshold)
                & (camera_stress <= self.camera_stress_threshold)
            )
            self._cached = {
                "trust": trust,
                "prediction": prediction,
                "lower": lower,
                "distance": distance,
                "spectral_stress": spectral_stress,
                "camera_stress": camera_stress,
                "required_margin": required_margin,
            }
        if self._cached is None or len(self._cached["trust"]) != len(gradient):
            raise RuntimeError("multi-veto direction must begin at the first stage")
        trust = self._cached["trust"]
        direction = torch.where(
            trust[:, None, None, None, None],
            candidate,
            fallback,
        )
        diagnostics: dict[str, torch.Tensor] = {
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
                "multiveto_trust": trust.to(gradient),
                "predicted_gain_percent": self._cached["prediction"],
                "lower_gain_bound_percent": self._cached["lower"],
                "feature_distance": self._cached["distance"],
                "spectral_stress": self._cached["spectral_stress"],
                "camera_stress": self._cached["camera_stress"],
                "required_gain_margin_percent": self._cached[
                    "required_margin"
                ],
            }
        )
        return direction, diagnostics
