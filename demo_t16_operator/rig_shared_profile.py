"""Explicit small-grid profile calibration utilities for the v5b pilot.

The routines in this module deliberately use a support-restricted linear ridge
problem.  They are a mechanism check for rig-shared calibration and local
identifiability, not a scalable BOST reconstruction method and not a neural
operator.  The explicit solve lets the matrix-free implementation planned for
larger grids be checked against a transparent Schur-complement reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RidgeProfileFit:
    """One field fit at one candidate optical parameter."""

    support_values: np.ndarray
    field: np.ndarray
    whitened_sse: float
    regularization: float
    measurement_count: int

    @property
    def objective(self) -> float:
        return 0.5 * (self.whitened_sse + self.regularization)

    @property
    def reduced_objective(self) -> float:
        return self.objective / float(self.measurement_count)


@dataclass(frozen=True)
class ProfileCandidate:
    """Block-level profile score and nuisance fits for one radius."""

    index: int
    radius: float
    data_score: float
    metadata_penalty: float
    total_score: float
    fits: tuple[RidgeProfileFit, ...]


@dataclass(frozen=True)
class ProfileSelection:
    """Result of sharing one optical radius across a block of fields."""

    selected_index: int
    candidates: tuple[ProfileCandidate, ...]

    @property
    def selected(self) -> ProfileCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True)
class ProfileFisherResult:
    """Local scalar-parameter information after profiling the field nuisance."""

    raw_parameter_energy: float
    profile_information: float
    retained_fraction: float
    approximate_standard_error: float


def _validate_operator(operator: np.ndarray, observation: np.ndarray) -> None:
    if operator.ndim != 4:
        raise ValueError("operator must have shape [depth,view,detector,voxel]")
    if observation.shape != operator.shape[:3]:
        raise ValueError("observation and operator measurement shapes disagree")
    if not np.all(np.isfinite(operator)) or not np.all(np.isfinite(observation)):
        raise ValueError("operator and observation must be finite")


def _expanded_sigma(noise_std: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    sigma = np.asarray(noise_std, dtype=np.float64)
    if sigma.ndim == 1 and sigma.shape == (shape[1],):
        sigma = sigma[None, :, None]
    try:
        expanded = np.broadcast_to(sigma, shape)
    except ValueError as error:
        raise ValueError("noise_std cannot be broadcast to the observation") from error
    if np.any(~np.isfinite(expanded)) or np.any(expanded <= 0.0):
        raise ValueError("noise_std must be finite and strictly positive")
    return expanded


def _view_mask(views: Sequence[int] | np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(views)
    if values.dtype == bool:
        if values.shape != (count,):
            raise ValueError("boolean view mask has the wrong length")
        mask = values.copy()
    else:
        mask = np.zeros(count, dtype=bool)
        indices = np.asarray(tuple(int(value) for value in values), dtype=int)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("view indices are empty or outside the rig")
        mask[indices] = True
    if not np.any(mask):
        raise ValueError("at least one view is required")
    return mask


def _whitened_system(
    operator: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(operator, dtype=np.float64)
    values = np.asarray(observation, dtype=np.float64)
    _validate_operator(matrix, values)
    support_mask = np.asarray(support, dtype=bool).reshape(-1)
    if support_mask.shape != (matrix.shape[-1],) or not np.any(support_mask):
        raise ValueError("support and operator voxel dimensions disagree")
    active_views = _view_mask(views, matrix.shape[1])
    active = np.broadcast_to(active_views[None, :, None], values.shape)
    sigma = _expanded_sigma(np.asarray(noise_std), values.shape)
    rows = matrix.reshape(-1, matrix.shape[-1])[active.reshape(-1)]
    rows = rows[:, support_mask]
    data = values.reshape(-1)[active.reshape(-1)]
    scales = sigma.reshape(-1)[active.reshape(-1)]
    return rows / scales[:, None], data / scales, support_mask


def whitened_support_system(
    operator: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expose the support-restricted whitened system for transparent audits."""

    return _whitened_system(operator, observation, noise_std, views, support)


def fit_support_ridge(
    operator: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    ridge_lambda: float,
) -> RidgeProfileFit:
    """Fit one field with an explicit support-restricted linear ridge solve."""

    regularization = float(ridge_lambda)
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("ridge_lambda must be finite and strictly positive")
    matrix, data, support_mask = _whitened_system(
        operator, observation, noise_std, views, support
    )
    hessian = matrix.T @ matrix
    hessian.flat[:: hessian.shape[0] + 1] += regularization
    support_values = np.linalg.solve(hessian, matrix.T @ data)
    residual = data - matrix @ support_values
    full = np.zeros(len(support_mask), dtype=np.float64)
    full[support_mask] = support_values
    return RidgeProfileFit(
        support_values=support_values,
        field=full.reshape(np.asarray(support).shape),
        whitened_sse=float(residual @ residual),
        regularization=float(regularization * (support_values @ support_values)),
        measurement_count=int(len(data)),
    )


def whitened_normal_mean_diagonal(
    operator: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
) -> float:
    """Return ``trace(A_w^T A_w) / p`` for regularization-scale audits."""

    matrix, _, _ = _whitened_system(
        operator, observation, noise_std, views, support
    )
    return float(np.sum(np.square(matrix)) / matrix.shape[1])


def profile_shared_radius(
    operator_bank: np.ndarray,
    radii: Sequence[float],
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    ridge_lambda: float,
    *,
    metadata_radius: float | None = None,
    metadata_sigma: float | None = None,
    metadata_weight: float = 0.0,
) -> ProfileSelection:
    """Profile field nuisances while sharing one radius across all fields."""

    bank = np.asarray(operator_bank, dtype=np.float64)
    radius_values = np.asarray(tuple(radii), dtype=np.float64)
    if bank.ndim != 5 or bank.shape[0] != len(radius_values):
        raise ValueError("operator_bank and radii disagree")
    if len(observations) == 0 or len(observations) != len(noise_std):
        raise ValueError("observations and noise_std must have equal nonzero length")
    weight = float(metadata_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("metadata_weight must be finite and non-negative")
    use_metadata = weight > 0.0
    if use_metadata:
        if metadata_radius is None or metadata_sigma is None:
            raise ValueError("metadata radius and sigma are required when weighted")
        if not np.isfinite(metadata_sigma) or float(metadata_sigma) <= 0.0:
            raise ValueError("metadata_sigma must be finite and strictly positive")

    candidates: list[ProfileCandidate] = []
    for index, radius in enumerate(radius_values):
        fits = tuple(
            fit_support_ridge(
                bank[index], observation, sigma, views, support, ridge_lambda
            )
            for observation, sigma in zip(observations, noise_std, strict=True)
        )
        data_score = float(np.sum([fit.objective for fit in fits]))
        metadata_penalty = 0.0
        if use_metadata:
            metadata_penalty = float(
                0.5
                * weight
                * ((float(radius) - float(metadata_radius)) / float(metadata_sigma)) ** 2
            )
        candidates.append(
            ProfileCandidate(
                index=index,
                radius=float(radius),
                data_score=data_score,
                metadata_penalty=metadata_penalty,
                total_score=data_score + metadata_penalty,
                fits=fits,
            )
        )
    selected_index = int(np.argmin([value.total_score for value in candidates]))
    return ProfileSelection(selected_index, tuple(candidates))


def apply_metadata_prior(
    selection: ProfileSelection,
    metadata_radius: float,
    metadata_sigma: float,
    metadata_weight: float,
) -> ProfileSelection:
    """Re-score already fitted candidates without repeating nuisance solves."""

    sigma = float(metadata_sigma)
    weight = float(metadata_weight)
    if not np.isfinite(metadata_radius):
        raise ValueError("metadata_radius must be finite")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("metadata_sigma must be finite and strictly positive")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("metadata_weight must be finite and non-negative")
    candidates = tuple(
        ProfileCandidate(
            index=value.index,
            radius=value.radius,
            data_score=value.data_score,
            metadata_penalty=float(
                0.5 * weight * ((value.radius - float(metadata_radius)) / sigma) ** 2
            ),
            total_score=float(
                value.data_score
                + 0.5 * weight * ((value.radius - float(metadata_radius)) / sigma) ** 2
            ),
            fits=value.fits,
        )
        for value in selection.candidates
    )
    return ProfileSelection(
        int(np.argmin([value.total_score for value in candidates])), candidates
    )


def operator_radius_derivative(
    operator_bank: np.ndarray, radii: Sequence[float], index: int
) -> np.ndarray:
    """Finite-difference derivative of the candidate operator bank."""

    bank = np.asarray(operator_bank, dtype=np.float64)
    values = np.asarray(tuple(radii), dtype=np.float64)
    selected = int(index)
    if bank.ndim != 5 or bank.shape[0] != len(values) or len(values) < 2:
        raise ValueError("at least two radius operators are required")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("radii must be strictly increasing")
    if not 0 <= selected < len(values):
        raise ValueError("derivative index is outside the operator bank")
    if selected == 0:
        lower, upper = 0, 1
    elif selected == len(values) - 1:
        lower, upper = len(values) - 2, len(values) - 1
    else:
        lower, upper = selected - 1, selected + 1
    return (bank[upper] - bank[lower]) / float(values[upper] - values[lower])


def profile_fisher_scalar(
    operator: np.ndarray,
    operator_derivative: np.ndarray,
    fit: RidgeProfileFit,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    ridge_lambda: float,
) -> ProfileFisherResult:
    """Compute the explicit scalar Schur-complement information reference."""

    matrix, _, support_mask = _whitened_system(
        operator, observation, noise_std, views, support
    )
    derivative, _, derivative_support = _whitened_system(
        operator_derivative,
        np.zeros_like(observation, dtype=np.float64),
        noise_std,
        views,
        support,
    )
    if not np.array_equal(support_mask, derivative_support):
        raise RuntimeError("support changed while constructing the derivative")
    if fit.support_values.shape != (matrix.shape[1],):
        raise ValueError("fit and operator support dimensions disagree")
    parameter_direction = derivative @ fit.support_values
    raw_energy = float(parameter_direction @ parameter_direction)
    hessian = matrix.T @ matrix
    hessian.flat[:: hessian.shape[0] + 1] += float(ridge_lambda)
    nuisance_explanation = matrix @ np.linalg.solve(
        hessian, matrix.T @ parameter_direction
    )
    information = float(parameter_direction @ (parameter_direction - nuisance_explanation))
    information = max(0.0, information)
    fraction = information / max(raw_energy, 1e-15)
    standard_error = float("inf") if information <= 0.0 else float(1.0 / np.sqrt(information))
    return ProfileFisherResult(
        raw_parameter_energy=raw_energy,
        profile_information=information,
        retained_fraction=float(np.clip(fraction, 0.0, 1.0)),
        approximate_standard_error=standard_error,
    )


def whitened_view_rms(
    operator: np.ndarray,
    field: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
) -> float:
    """Evaluate a fitted field on a disjoint set of cameras."""

    matrix = np.asarray(operator, dtype=np.float64)
    values = np.asarray(observation, dtype=np.float64)
    _validate_operator(matrix, values)
    active_views = _view_mask(views, matrix.shape[1])
    active = np.broadcast_to(active_views[None, :, None], values.shape)
    sigma = _expanded_sigma(np.asarray(noise_std), values.shape)
    prediction = matrix.reshape(-1, matrix.shape[-1]) @ np.asarray(field).reshape(-1)
    residual = (values.reshape(-1) - prediction) / sigma.reshape(-1)
    return float(np.sqrt(np.mean(residual[active.reshape(-1)] ** 2)))
