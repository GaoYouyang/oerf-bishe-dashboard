"""Nested cross-view selection for the v5c finite-aperture pilot.

The selector uses only declared inner cameras.  Each inner camera is held out
once while the remaining cameras fit the field nuisance.  Radius and a
dimensionless ridge ratio are selected by held-out data error; outer cameras
remain available for a later deployment gate and the audit camera is never
read here.

This is an explicit small-matrix reference, not a scalable neural operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from .rig_shared_profile import (
        RidgeProfileFit,
        fit_support_ridge,
        whitened_normal_mean_diagonal,
        whitened_view_rms,
    )
except ImportError:
    from rig_shared_profile import (
        RidgeProfileFit,
        fit_support_ridge,
        whitened_normal_mean_diagonal,
        whitened_view_rms,
    )


@dataclass(frozen=True)
class CrossViewCandidate:
    """One radius-ridge candidate scored on inner-camera validation folds."""

    radius_index: int
    radius: float
    kappa: float
    mean_validation_mse: float
    fold_validation_mse: tuple[float, ...]
    median_effective_lambda: float


@dataclass(frozen=True)
class CrossViewSelection:
    """Nested inner-camera selection and fold-deletion stability audit."""

    selected_candidate_index: int
    candidates: tuple[CrossViewCandidate, ...]
    validation_views: tuple[int, ...]
    fold_score_deletion_candidate_indices: tuple[int, ...]
    fold_score_deletion_radius_stability_fraction: float
    fold_score_deletion_kappa_stability_fraction: float
    relative_score_margin: float
    relative_radius_margin: float

    @property
    def selected(self) -> CrossViewCandidate:
        return self.candidates[self.selected_candidate_index]


@dataclass(frozen=True)
class ScaledRidgeRefit:
    """All-inner-view refits at one selected radius and dimensionless ridge."""

    fits: tuple[RidgeProfileFit, ...]
    effective_lambdas: tuple[float, ...]


def _validated_views(views: Sequence[int] | np.ndarray, count: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in views)
    if len(values) < 2:
        raise ValueError("nested cross-view selection requires at least two views")
    if len(set(values)) != len(values):
        raise ValueError("nested cross-view views must be unique")
    if min(values) < 0 or max(values) >= count:
        raise ValueError("nested cross-view view is outside the operator")
    return values


def _validated_kappas(kappas: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in kappas)
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("kappas must be finite and strictly positive")
    if len(set(values)) != len(values):
        raise ValueError("kappas must be unique")
    return tuple(sorted(values))


def _slice_views(
    operator: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Physically isolate selected cameras before lower-level validation."""

    matrix = np.asarray(operator, dtype=np.float64)
    values = np.asarray(observation, dtype=np.float64)
    if matrix.ndim != 4 or values.shape != matrix.shape[:3]:
        raise ValueError("operator and observation measurement shapes disagree")
    selected = tuple(int(value) for value in views)
    if not selected:
        raise ValueError("at least one view is required")
    if len(set(selected)) != len(selected):
        raise ValueError("views must be unique")
    if min(selected) < 0 or max(selected) >= matrix.shape[1]:
        raise ValueError("view is outside the operator")
    sliced_matrix = matrix[:, selected, :, :]
    sliced_values = values[:, selected, :]
    sigma = np.asarray(noise_std, dtype=np.float64)
    if sigma.ndim == 1 and sigma.shape == (matrix.shape[1],):
        sliced_sigma = sigma[np.asarray(selected, dtype=int)]
    else:
        try:
            expanded = np.broadcast_to(sigma, values.shape)
        except ValueError as error:
            raise ValueError("noise_std cannot be broadcast to the observation") from error
        sliced_sigma = expanded[:, selected, :]
    local_views = tuple(range(len(selected)))
    return sliced_matrix, sliced_values, sliced_sigma, local_views


def _validated_support(support: np.ndarray, voxel_count: int) -> np.ndarray:
    values = np.asarray(support)
    if values.dtype != bool:
        raise ValueError("support must be an explicit boolean mask")
    if values.size != voxel_count or not np.any(values):
        raise ValueError("support and operator voxel dimensions disagree")
    return values


def scaled_ridge_fit(
    operator: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappa: float,
) -> tuple[RidgeProfileFit, float]:
    """Fit ridge with ``lambda = kappa * mean(diag(A_w.T A_w))``."""

    ratio = float(kappa)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("kappa must be finite and strictly positive")
    matrix, values, sigma, local_views = _slice_views(
        operator, observation, noise_std, views
    )
    support_mask = _validated_support(support, matrix.shape[-1])
    normal_scale = whitened_normal_mean_diagonal(
        matrix, values, sigma, local_views, support_mask
    )
    if not np.isfinite(normal_scale) or normal_scale <= 0.0:
        raise ValueError("whitened normal scale must be finite and positive")
    effective_lambda = float(ratio * normal_scale)
    return (
        fit_support_ridge(
            matrix,
            values,
            sigma,
            local_views,
            support_mask,
            effective_lambda,
        ),
        effective_lambda,
    )


def select_radius_kappa_crossview(
    operator_bank: np.ndarray,
    radii: Sequence[float],
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    inner_views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappas: Sequence[float],
) -> CrossViewSelection:
    """Select a shared radius and dimensionless ridge using inner views only."""

    bank = np.asarray(operator_bank, dtype=np.float64)
    radius_values = np.asarray(tuple(radii), dtype=np.float64)
    if bank.ndim != 5 or bank.shape[0] != len(radius_values) or len(radius_values) == 0:
        raise ValueError("operator_bank and radii disagree")
    if np.any(~np.isfinite(radius_values)) or np.any(np.diff(radius_values) <= 0.0):
        raise ValueError("radii must be finite and strictly increasing")
    if len(observations) == 0 or len(observations) != len(noise_std):
        raise ValueError("observations and noise_std must have equal nonzero length")
    views = _validated_views(inner_views, bank.shape[2])
    ratios = _validated_kappas(kappas)
    support_mask = _validated_support(support, bank.shape[-1])

    candidates: list[CrossViewCandidate] = []
    for radius_index, radius in enumerate(radius_values):
        operator = bank[radius_index]
        for kappa in ratios:
            fold_scores: list[float] = []
            effective_lambdas: list[float] = []
            for validation_view in views:
                fit_views = tuple(view for view in views if view != validation_view)
                sample_scores: list[float] = []
                for observation, sigma in zip(observations, noise_std, strict=True):
                    fit, effective_lambda = scaled_ridge_fit(
                        operator,
                        observation,
                        sigma,
                        fit_views,
                        support_mask,
                        kappa,
                    )
                    validation_operator, validation_observation, validation_sigma, _ = (
                        _slice_views(
                            operator,
                            observation,
                            sigma,
                            [validation_view],
                        )
                    )
                    validation_rms = whitened_view_rms(
                        validation_operator,
                        fit.field,
                        validation_observation,
                        validation_sigma,
                        [0],
                    )
                    sample_scores.append(float(validation_rms**2))
                    effective_lambdas.append(effective_lambda)
                fold_scores.append(float(np.mean(sample_scores)))
            candidates.append(
                CrossViewCandidate(
                    radius_index=radius_index,
                    radius=float(radius),
                    kappa=kappa,
                    mean_validation_mse=float(np.mean(fold_scores)),
                    fold_validation_mse=tuple(fold_scores),
                    median_effective_lambda=float(np.median(effective_lambdas)),
                )
            )

    scores = np.asarray(
        [candidate.mean_validation_mse for candidate in candidates], dtype=float
    )
    selected = int(np.argmin(scores))
    ordered = np.sort(scores)
    relative_margin = (
        float((ordered[1] - ordered[0]) / max(abs(ordered[0]), 1e-12))
        if len(ordered) > 1
        else float("inf")
    )
    radius_best_scores = np.asarray(
        [
            min(
                candidate.mean_validation_mse
                for candidate in candidates
                if candidate.radius_index == radius_index
            )
            for radius_index in range(len(radius_values))
        ],
        dtype=float,
    )
    ordered_radius_scores = np.sort(radius_best_scores)
    relative_radius_margin = (
        float(
            (ordered_radius_scores[1] - ordered_radius_scores[0])
            / max(abs(ordered_radius_scores[0]), 1e-12)
        )
        if len(ordered_radius_scores) > 1
        else float("inf")
    )

    jackknife: list[int] = []
    for omitted_fold in range(len(views)):
        reduced_scores = np.asarray(
            [
                np.mean(
                    [
                        score
                        for fold, score in enumerate(candidate.fold_validation_mse)
                        if fold != omitted_fold
                    ]
                )
                for candidate in candidates
            ],
            dtype=float,
        )
        jackknife.append(int(np.argmin(reduced_scores)))

    selected_candidate = candidates[selected]
    radius_stability = float(
        np.mean(
            [
                candidates[index].radius_index == selected_candidate.radius_index
                for index in jackknife
            ]
        )
    )
    kappa_stability = float(
        np.mean(
            [
                candidates[index].kappa == selected_candidate.kappa
                for index in jackknife
            ]
        )
    )
    return CrossViewSelection(
        selected_candidate_index=selected,
        candidates=tuple(candidates),
        validation_views=views,
        fold_score_deletion_candidate_indices=tuple(jackknife),
        fold_score_deletion_radius_stability_fraction=radius_stability,
        fold_score_deletion_kappa_stability_fraction=kappa_stability,
        relative_score_margin=relative_margin,
        relative_radius_margin=relative_radius_margin,
    )


def refit_scaled_selection(
    selection: CrossViewSelection,
    operator_bank: np.ndarray,
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    fit_views: Sequence[int] | np.ndarray,
    support: np.ndarray,
) -> ScaledRidgeRefit:
    """Refit every field on all declared inner cameras after selection."""

    bank = np.asarray(operator_bank, dtype=np.float64)
    chosen = selection.selected
    if bank.ndim != 5 or not 0 <= chosen.radius_index < bank.shape[0]:
        raise ValueError("selected radius is outside the operator bank")
    if len(observations) == 0 or len(observations) != len(noise_std):
        raise ValueError("observations and noise_std must have equal nonzero length")
    fits: list[RidgeProfileFit] = []
    effective_lambdas: list[float] = []
    for observation, sigma in zip(observations, noise_std, strict=True):
        fit, effective_lambda = scaled_ridge_fit(
            bank[chosen.radius_index],
            observation,
            sigma,
            fit_views,
            support,
            chosen.kappa,
        )
        fits.append(fit)
        effective_lambdas.append(effective_lambda)
    return ScaledRidgeRefit(tuple(fits), tuple(effective_lambdas))


def whitened_per_view_rms(
    operator: np.ndarray,
    field: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
) -> tuple[float, ...]:
    """Return one independently inspectable whitened RMS per camera."""

    values = tuple(int(view) for view in views)
    if not values:
        raise ValueError("at least one evaluation view is required")
    if len(set(values)) != len(values):
        raise ValueError("evaluation views must be unique")
    scores: list[float] = []
    for view in values:
        matrix, camera_observation, sigma, _ = _slice_views(
            operator, observation, noise_std, [view]
        )
        scores.append(
            whitened_view_rms(matrix, field, camera_observation, sigma, [0])
        )
    return tuple(scores)
