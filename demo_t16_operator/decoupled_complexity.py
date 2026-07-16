"""Complexity-first aperture selection for the v5d development study.

Each outer inner-camera fold compares aperture radii on a camera that is not
used to choose the ridge ratio.  GCV, UPRE, Morozov, equal effective degrees of
freedom, or a second camera-CV loop selects complexity on the remaining views.
This is an explicit small-matrix reference for diagnosis, not confirmatory
evidence and not a scalable neural reconstruction method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

try:
    from .nested_crossview import (
        ScaledRidgeRefit,
        _validated_kappas,
        _validated_support,
        _validated_views,
        scaled_ridge_fit,
    )
    from .rig_shared_profile import (
        RidgeProfileFit,
        whitened_support_system,
        whitened_view_rms,
    )
except ImportError:
    from nested_crossview import (
        ScaledRidgeRefit,
        _validated_kappas,
        _validated_support,
        _validated_views,
        scaled_ridge_fit,
    )
    from rig_shared_profile import (
        RidgeProfileFit,
        whitened_support_system,
        whitened_view_rms,
    )


ComplexityMethod = Literal[
    "gcv",
    "upre",
    "morozov",
    "df_corrected_morozov",
    "equal_df",
    "nested_cv",
]
METHODS: tuple[ComplexityMethod, ...] = (
    "gcv",
    "upre",
    "morozov",
    "df_corrected_morozov",
    "equal_df",
    "nested_cv",
)


@dataclass(frozen=True)
class RidgeDiagnostics:
    """Exact small-matrix diagnostics for one ridge fit."""

    fit: RidgeProfileFit
    effective_lambda: float
    effective_degrees_of_freedom: float
    squared_hat_trace: float
    effective_degrees_of_freedom_fraction: float
    whitened_discrepancy: float
    residual_noise_degrees_of_freedom: float
    degrees_of_freedom_corrected_discrepancy: float
    generalized_cross_validation: float
    unbiased_predictive_risk: float


@dataclass(frozen=True)
class ComplexityChoice:
    """One complexity decision made without the outer validation camera."""

    method: ComplexityMethod
    kappa_index: int
    kappa: float
    selection_score: float
    mean_effective_degrees_of_freedom_fraction: float
    mean_whitened_discrepancy: float
    mean_degrees_of_freedom_corrected_discrepancy: float


@dataclass(frozen=True)
class DecoupledSurfacePoint:
    """One precomputed point on a radius-fold-kappa development surface."""

    kappa: float
    mean_generalized_cross_validation: float
    mean_unbiased_predictive_risk: float
    mean_effective_degrees_of_freedom_fraction: float
    mean_whitened_discrepancy: float
    mean_degrees_of_freedom_corrected_discrepancy: float
    nested_cross_validation_mse: float
    outer_validation_mse: float


@dataclass(frozen=True)
class DecoupledSurface:
    """Reusable complexity paths shared by all v5d selection rules."""

    radii: tuple[float, ...]
    kappas: tuple[float, ...]
    validation_views: tuple[int, ...]
    paths: tuple[tuple[tuple[DecoupledSurfacePoint, ...], ...], ...]
    nested_cross_validation_computed: bool


@dataclass(frozen=True)
class DecoupledRadiusCandidate:
    """One radius scored after fold-local complexity selection."""

    radius_index: int
    radius: float
    mean_validation_mse: float
    fold_validation_mse: tuple[float, ...]
    fold_selected_kappas: tuple[float, ...]
    fold_complexity_scores: tuple[float, ...]
    fold_effective_degrees_of_freedom_fractions: tuple[float, ...]
    fold_whitened_discrepancies: tuple[float, ...]
    fold_degrees_of_freedom_corrected_discrepancies: tuple[float, ...]


@dataclass(frozen=True)
class DecoupledSelection:
    """Radius selection with complexity chosen inside every camera fold."""

    method: ComplexityMethod
    selected_candidate_index: int
    candidates: tuple[DecoupledRadiusCandidate, ...]
    validation_views: tuple[int, ...]
    fold_score_deletion_candidate_indices: tuple[int, ...]
    fold_score_deletion_radius_stability_fraction: float
    relative_radius_margin: float
    discrepancy_target: float
    effective_degrees_of_freedom_target: float

    @property
    def selected(self) -> DecoupledRadiusCandidate:
        return self.candidates[self.selected_candidate_index]


@dataclass(frozen=True)
class DecoupledRefit:
    """All-view refits after a complexity-first radius decision."""

    choice: ComplexityChoice
    refit: ScaledRidgeRefit


def _validated_method(method: str) -> ComplexityMethod:
    if method not in METHODS:
        raise ValueError(f"unknown complexity method: {method}")
    return method  # type: ignore[return-value]


def ridge_diagnostics(
    operator: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappa: float,
) -> RidgeDiagnostics:
    """Fit ridge and compute exact hat-matrix diagnostics."""

    fit, effective_lambda = scaled_ridge_fit(
        operator,
        observation,
        noise_std,
        views,
        support,
        kappa,
    )
    matrix, _, _ = whitened_support_system(
        operator,
        observation,
        noise_std,
        views,
        support,
    )
    eigenvalues = np.linalg.eigvalsh(matrix.T @ matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    effective_df = float(
        np.sum(eigenvalues / (eigenvalues + float(effective_lambda)))
    )
    shrinkage_eigenvalues = eigenvalues / (
        eigenvalues + float(effective_lambda)
    )
    squared_hat_trace = float(np.sum(np.square(shrinkage_eigenvalues)))
    measurement_count, parameter_count = matrix.shape
    capacity = float(min(measurement_count, parameter_count))
    df_fraction = effective_df / max(capacity, 1.0)
    discrepancy = float(fit.whitened_sse / measurement_count)
    residual_noise_df = float(
        measurement_count - 2.0 * effective_df + squared_hat_trace
    )
    if residual_noise_df <= 1e-12:
        corrected_discrepancy = float("inf")
    else:
        corrected_discrepancy = float(fit.whitened_sse / residual_noise_df)
    denominator = max(1.0 - effective_df / measurement_count, 1e-12)
    gcv = float(discrepancy / denominator**2)
    upre = float(
        (fit.whitened_sse + 2.0 * effective_df - measurement_count)
        / measurement_count
    )
    return RidgeDiagnostics(
        fit=fit,
        effective_lambda=effective_lambda,
        effective_degrees_of_freedom=effective_df,
        squared_hat_trace=squared_hat_trace,
        effective_degrees_of_freedom_fraction=df_fraction,
        whitened_discrepancy=discrepancy,
        residual_noise_degrees_of_freedom=residual_noise_df,
        degrees_of_freedom_corrected_discrepancy=corrected_discrepancy,
        generalized_cross_validation=gcv,
        unbiased_predictive_risk=upre,
    )


def _nested_cv_score(
    operator: np.ndarray,
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    views: tuple[int, ...],
    support: np.ndarray,
    kappa: float,
) -> float:
    if len(views) < 2:
        raise ValueError("nested_cv complexity selection requires at least two views")
    fold_scores: list[float] = []
    for validation_view in views:
        fit_views = tuple(view for view in views if view != validation_view)
        sample_scores: list[float] = []
        for observation, sigma in zip(observations, noise_std, strict=True):
            fit, _ = scaled_ridge_fit(
                operator,
                observation,
                sigma,
                fit_views,
                support,
                kappa,
            )
            validation_rms = whitened_view_rms(
                operator,
                fit.field,
                observation,
                sigma,
                [validation_view],
            )
            sample_scores.append(float(validation_rms**2))
        fold_scores.append(float(np.mean(sample_scores)))
    return float(np.mean(fold_scores))


def build_decoupled_surface(
    operator_bank: np.ndarray,
    radii: Sequence[float],
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    inner_views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappas: Sequence[float],
    *,
    include_nested_cross_validation: bool = True,
) -> DecoupledSurface:
    """Precompute every quantity shared by the v5d selection rules."""

    bank = np.asarray(operator_bank, dtype=np.float64)
    radius_values = np.asarray(tuple(radii), dtype=np.float64)
    if bank.ndim != 5 or bank.shape[0] != len(radius_values) or not len(radius_values):
        raise ValueError("operator_bank and radii disagree")
    if np.any(~np.isfinite(radius_values)) or np.any(np.diff(radius_values) <= 0.0):
        raise ValueError("radii must be finite and strictly increasing")
    if len(observations) == 0 or len(observations) != len(noise_std):
        raise ValueError("observations and noise_std must have equal nonzero length")
    views = _validated_views(inner_views, bank.shape[2])
    if len(views) < 3:
        raise ValueError("decoupled radius selection requires at least three views")
    support_mask = _validated_support(support, bank.shape[-1])
    ratios = _validated_kappas(kappas)

    radius_paths: list[tuple[tuple[DecoupledSurfacePoint, ...], ...]] = []
    for radius_index in range(len(radius_values)):
        operator = bank[radius_index]
        fold_paths: list[tuple[DecoupledSurfacePoint, ...]] = []
        for validation_view in views:
            fit_views = tuple(view for view in views if view != validation_view)
            points: list[DecoupledSurfacePoint] = []
            for kappa in ratios:
                diagnostics = tuple(
                    ridge_diagnostics(
                        operator,
                        observation,
                        sigma,
                        fit_views,
                        support_mask,
                        kappa,
                    )
                    for observation, sigma in zip(
                        observations, noise_std, strict=True
                    )
                )
                outer_scores = tuple(
                    float(
                        whitened_view_rms(
                            operator,
                            item.fit.field,
                            observation,
                            sigma,
                            [validation_view],
                        )
                        ** 2
                    )
                    for item, observation, sigma in zip(
                        diagnostics, observations, noise_std, strict=True
                    )
                )
                nested_score = float("nan")
                if include_nested_cross_validation:
                    nested_score = _nested_cv_score(
                        operator,
                        observations,
                        noise_std,
                        fit_views,
                        support_mask,
                        kappa,
                    )
                points.append(
                    DecoupledSurfacePoint(
                        kappa=kappa,
                        mean_generalized_cross_validation=float(
                            np.mean(
                                [
                                    item.generalized_cross_validation
                                    for item in diagnostics
                                ]
                            )
                        ),
                        mean_unbiased_predictive_risk=float(
                            np.mean(
                                [item.unbiased_predictive_risk for item in diagnostics]
                            )
                        ),
                        mean_effective_degrees_of_freedom_fraction=float(
                            np.mean(
                                [
                                    item.effective_degrees_of_freedom_fraction
                                    for item in diagnostics
                                ]
                            )
                        ),
                        mean_whitened_discrepancy=float(
                            np.mean(
                                [item.whitened_discrepancy for item in diagnostics]
                            )
                        ),
                        mean_degrees_of_freedom_corrected_discrepancy=float(
                            np.mean(
                                [
                                    item.degrees_of_freedom_corrected_discrepancy
                                    for item in diagnostics
                                ]
                            )
                        ),
                        nested_cross_validation_mse=nested_score,
                        outer_validation_mse=float(np.mean(outer_scores)),
                    )
                )
            fold_paths.append(tuple(points))
        radius_paths.append(tuple(fold_paths))
    return DecoupledSurface(
        radii=tuple(float(value) for value in radius_values),
        kappas=ratios,
        validation_views=views,
        paths=tuple(radius_paths),
        nested_cross_validation_computed=bool(include_nested_cross_validation),
    )


def select_radius_from_surface(
    method: str,
    surface: DecoupledSurface,
    *,
    discrepancy_target: float = 1.0,
    effective_degrees_of_freedom_target: float = 0.5,
) -> DecoupledSelection:
    """Select one radius from a shared precomputed development surface."""

    selected_method = _validated_method(method)
    discrepancy_level = float(discrepancy_target)
    df_target = float(effective_degrees_of_freedom_target)
    if not np.isfinite(discrepancy_level) or discrepancy_level <= 0.0:
        raise ValueError("discrepancy_target must be finite and positive")
    if not np.isfinite(df_target) or not 0.0 < df_target < 1.0:
        raise ValueError(
            "effective_degrees_of_freedom_target must lie strictly between zero and one"
        )
    if selected_method == "nested_cv" and not surface.nested_cross_validation_computed:
        raise ValueError("surface does not contain nested cross-validation scores")
    if len(surface.paths) != len(surface.radii) or not surface.paths:
        raise ValueError("surface radius paths are incomplete")

    candidates: list[DecoupledRadiusCandidate] = []
    for radius_index, (radius, fold_paths) in enumerate(
        zip(surface.radii, surface.paths, strict=True)
    ):
        if len(fold_paths) != len(surface.validation_views):
            raise ValueError("surface fold paths are incomplete")
        fold_scores: list[float] = []
        fold_choices: list[ComplexityChoice] = []
        for points in fold_paths:
            if tuple(point.kappa for point in points) != surface.kappas:
                raise ValueError("surface kappa path is inconsistent")
            method_scores: list[float] = []
            for point in points:
                if selected_method == "gcv":
                    score = point.mean_generalized_cross_validation
                elif selected_method == "upre":
                    score = point.mean_unbiased_predictive_risk
                elif selected_method == "morozov":
                    score = abs(point.mean_whitened_discrepancy - discrepancy_level)
                elif selected_method == "df_corrected_morozov":
                    score = abs(
                        point.mean_degrees_of_freedom_corrected_discrepancy
                        - discrepancy_level
                    )
                elif selected_method == "equal_df":
                    score = abs(
                        point.mean_effective_degrees_of_freedom_fraction - df_target
                    )
                else:
                    score = point.nested_cross_validation_mse
                method_scores.append(float(score))
            selected_kappa_index = int(
                np.argmin(np.asarray(method_scores, dtype=float))
            )
            selected_point = points[selected_kappa_index]
            fold_scores.append(selected_point.outer_validation_mse)
            fold_choices.append(
                ComplexityChoice(
                    method=selected_method,
                    kappa_index=selected_kappa_index,
                    kappa=selected_point.kappa,
                    selection_score=method_scores[selected_kappa_index],
                    mean_effective_degrees_of_freedom_fraction=(
                        selected_point.mean_effective_degrees_of_freedom_fraction
                    ),
                    mean_whitened_discrepancy=(
                        selected_point.mean_whitened_discrepancy
                    ),
                    mean_degrees_of_freedom_corrected_discrepancy=(
                        selected_point.mean_degrees_of_freedom_corrected_discrepancy
                    ),
                )
            )
        candidates.append(
            DecoupledRadiusCandidate(
                radius_index=radius_index,
                radius=radius,
                mean_validation_mse=float(np.mean(fold_scores)),
                fold_validation_mse=tuple(fold_scores),
                fold_selected_kappas=tuple(choice.kappa for choice in fold_choices),
                fold_complexity_scores=tuple(
                    choice.selection_score for choice in fold_choices
                ),
                fold_effective_degrees_of_freedom_fractions=tuple(
                    choice.mean_effective_degrees_of_freedom_fraction
                    for choice in fold_choices
                ),
                fold_whitened_discrepancies=tuple(
                    choice.mean_whitened_discrepancy for choice in fold_choices
                ),
                fold_degrees_of_freedom_corrected_discrepancies=tuple(
                    choice.mean_degrees_of_freedom_corrected_discrepancy
                    for choice in fold_choices
                ),
            )
        )

    radius_scores = np.asarray(
        [candidate.mean_validation_mse for candidate in candidates], dtype=float
    )
    selected_index = int(np.argmin(radius_scores))
    ordered = np.sort(radius_scores)
    relative_margin = (
        float((ordered[1] - ordered[0]) / max(abs(ordered[0]), 1e-12))
        if len(ordered) > 1
        else float("inf")
    )
    jackknife: list[int] = []
    for omitted_fold in range(len(surface.validation_views)):
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
    radius_stability = float(
        np.mean([index == selected_index for index in jackknife])
    )
    return DecoupledSelection(
        method=selected_method,
        selected_candidate_index=selected_index,
        candidates=tuple(candidates),
        validation_views=surface.validation_views,
        fold_score_deletion_candidate_indices=tuple(jackknife),
        fold_score_deletion_radius_stability_fraction=radius_stability,
        relative_radius_margin=relative_margin,
        discrepancy_target=discrepancy_level,
        effective_degrees_of_freedom_target=df_target,
    )


def choose_complexity(
    method: str,
    operator: np.ndarray,
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    fit_views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappas: Sequence[float],
    *,
    discrepancy_target: float = 1.0,
    effective_degrees_of_freedom_target: float = 0.5,
) -> ComplexityChoice:
    """Choose kappa on fit views without reading an outer validation camera."""

    selected_method = _validated_method(method)
    if len(observations) == 0 or len(observations) != len(noise_std):
        raise ValueError("observations and noise_std must have equal nonzero length")
    matrix = np.asarray(operator, dtype=np.float64)
    if matrix.ndim != 4:
        raise ValueError("operator must have shape [depth,view,detector,voxel]")
    views = _validated_views(fit_views, matrix.shape[1])
    support_mask = _validated_support(support, matrix.shape[-1])
    ratios = _validated_kappas(kappas)
    discrepancy_level = float(discrepancy_target)
    df_target = float(effective_degrees_of_freedom_target)
    if not np.isfinite(discrepancy_level) or discrepancy_level <= 0.0:
        raise ValueError("discrepancy_target must be finite and positive")
    if not np.isfinite(df_target) or not 0.0 < df_target < 1.0:
        raise ValueError(
            "effective_degrees_of_freedom_target must lie strictly between zero and one"
        )

    scores: list[float] = []
    summaries: list[tuple[float, float, float]] = []
    for kappa in ratios:
        diagnostics = tuple(
            ridge_diagnostics(
                matrix,
                observation,
                sigma,
                views,
                support_mask,
                kappa,
            )
            for observation, sigma in zip(observations, noise_std, strict=True)
        )
        mean_df = float(
            np.mean(
                [
                    item.effective_degrees_of_freedom_fraction
                    for item in diagnostics
                ]
            )
        )
        mean_discrepancy = float(
            np.mean([item.whitened_discrepancy for item in diagnostics])
        )
        mean_corrected_discrepancy = float(
            np.mean(
                [
                    item.degrees_of_freedom_corrected_discrepancy
                    for item in diagnostics
                ]
            )
        )
        summaries.append((mean_df, mean_discrepancy, mean_corrected_discrepancy))
        if selected_method == "gcv":
            score = float(
                np.mean([item.generalized_cross_validation for item in diagnostics])
            )
        elif selected_method == "upre":
            score = float(
                np.mean([item.unbiased_predictive_risk for item in diagnostics])
            )
        elif selected_method == "morozov":
            score = abs(mean_discrepancy - discrepancy_level)
        elif selected_method == "df_corrected_morozov":
            score = abs(mean_corrected_discrepancy - discrepancy_level)
        elif selected_method == "equal_df":
            score = abs(mean_df - df_target)
        else:
            score = _nested_cv_score(
                matrix,
                observations,
                noise_std,
                views,
                support_mask,
                kappa,
            )
        scores.append(float(score))

    selected_index = int(np.argmin(np.asarray(scores, dtype=float)))
    selected_df, selected_discrepancy, selected_corrected_discrepancy = summaries[
        selected_index
    ]
    return ComplexityChoice(
        method=selected_method,
        kappa_index=selected_index,
        kappa=ratios[selected_index],
        selection_score=scores[selected_index],
        mean_effective_degrees_of_freedom_fraction=selected_df,
        mean_whitened_discrepancy=selected_discrepancy,
        mean_degrees_of_freedom_corrected_discrepancy=(
            selected_corrected_discrepancy
        ),
    )


def select_radius_decoupled(
    method: str,
    operator_bank: np.ndarray,
    radii: Sequence[float],
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    inner_views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappas: Sequence[float],
    *,
    discrepancy_target: float = 1.0,
    effective_degrees_of_freedom_target: float = 0.5,
) -> DecoupledSelection:
    """Compare radii outside fold-local complexity selection."""

    selected_method = _validated_method(method)
    surface = build_decoupled_surface(
        operator_bank,
        radii,
        observations,
        noise_std,
        inner_views,
        support,
        kappas,
        include_nested_cross_validation=selected_method == "nested_cv",
    )
    return select_radius_from_surface(
        selected_method,
        surface,
        discrepancy_target=discrepancy_target,
        effective_degrees_of_freedom_target=(
            effective_degrees_of_freedom_target
        ),
    )


def refit_decoupled_selection(
    selection: DecoupledSelection,
    operator_bank: np.ndarray,
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    fit_views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappas: Sequence[float],
) -> DecoupledRefit:
    """Choose final complexity on all fit views, then refit every field."""

    bank = np.asarray(operator_bank, dtype=np.float64)
    chosen = selection.selected
    if bank.ndim != 5 or not 0 <= chosen.radius_index < bank.shape[0]:
        raise ValueError("selected radius is outside the operator bank")
    choice = choose_complexity(
        selection.method,
        bank[chosen.radius_index],
        observations,
        noise_std,
        fit_views,
        support,
        kappas,
        discrepancy_target=selection.discrepancy_target,
        effective_degrees_of_freedom_target=(
            selection.effective_degrees_of_freedom_target
        ),
    )
    fits: list[RidgeProfileFit] = []
    effective_lambdas: list[float] = []
    for observation, sigma in zip(observations, noise_std, strict=True):
        fit, effective_lambda = scaled_ridge_fit(
            bank[chosen.radius_index],
            observation,
            sigma,
            fit_views,
            support,
            choice.kappa,
        )
        fits.append(fit)
        effective_lambdas.append(effective_lambda)
    return DecoupledRefit(
        choice=choice,
        refit=ScaledRidgeRefit(tuple(fits), tuple(effective_lambdas)),
    )
