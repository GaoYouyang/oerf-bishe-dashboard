"""Covariance-aware counterpart of the v5d decoupled complexity surface."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from .covariance_whitening import (
        covariance_scaled_ridge_fit,
        covariance_whitened_support_system,
        covariance_whitened_view_rms,
    )
    from .decoupled_complexity import (
        DecoupledSurface,
        DecoupledSurfacePoint,
        RidgeDiagnostics,
    )
    from .nested_crossview import _validated_kappas, _validated_support, _validated_views
except ImportError:
    from covariance_whitening import (
        covariance_scaled_ridge_fit,
        covariance_whitened_support_system,
        covariance_whitened_view_rms,
    )
    from decoupled_complexity import (
        DecoupledSurface,
        DecoupledSurfacePoint,
        RidgeDiagnostics,
    )
    from nested_crossview import _validated_kappas, _validated_support, _validated_views


def covariance_ridge_diagnostics(
    operator: np.ndarray,
    observation: np.ndarray,
    covariance: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappa: float,
) -> RidgeDiagnostics:
    fit, effective_lambda = covariance_scaled_ridge_fit(
        operator, observation, covariance, views, support, kappa
    )
    matrix, _, _ = covariance_whitened_support_system(
        operator, observation, covariance, views, support
    )
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix.T @ matrix), 0.0, None)
    shrinkage = eigenvalues / (eigenvalues + float(effective_lambda))
    effective_df = float(np.sum(shrinkage))
    squared_hat_trace = float(np.sum(np.square(shrinkage)))
    measurement_count, parameter_count = matrix.shape
    residual_noise_df = float(
        measurement_count - 2.0 * effective_df + squared_hat_trace
    )
    corrected = (
        float(fit.whitened_sse / residual_noise_df)
        if residual_noise_df > 1e-12
        else float("inf")
    )
    discrepancy = float(fit.whitened_sse / measurement_count)
    denominator = max(1.0 - effective_df / measurement_count, 1e-12)
    return RidgeDiagnostics(
        fit=fit,
        effective_lambda=effective_lambda,
        effective_degrees_of_freedom=effective_df,
        squared_hat_trace=squared_hat_trace,
        effective_degrees_of_freedom_fraction=(
            effective_df / max(float(min(measurement_count, parameter_count)), 1.0)
        ),
        whitened_discrepancy=discrepancy,
        residual_noise_degrees_of_freedom=residual_noise_df,
        degrees_of_freedom_corrected_discrepancy=corrected,
        generalized_cross_validation=float(discrepancy / denominator**2),
        unbiased_predictive_risk=float(
            (fit.whitened_sse + 2.0 * effective_df - measurement_count)
            / measurement_count
        ),
    )


def _nested_cv_score(
    operator: np.ndarray,
    observations: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    views: tuple[int, ...],
    support: np.ndarray,
    kappa: float,
) -> float:
    if len(views) < 2:
        raise ValueError("nested covariance CV needs at least two fit views")
    fold_scores: list[float] = []
    for validation_view in views:
        fit_views = tuple(view for view in views if view != validation_view)
        sample_scores: list[float] = []
        for observation, covariance in zip(
            observations, covariances, strict=True
        ):
            fit, _ = covariance_scaled_ridge_fit(
                operator,
                observation,
                covariance,
                fit_views,
                support,
                kappa,
            )
            sample_scores.append(
                covariance_whitened_view_rms(
                    operator,
                    fit.field,
                    observation,
                    covariance,
                    [validation_view],
                )
                ** 2
            )
        fold_scores.append(float(np.mean(sample_scores)))
    return float(np.mean(fold_scores))


def build_covariance_decoupled_surface(
    operator_bank: np.ndarray,
    radii: Sequence[float],
    observations: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    inner_views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappas: Sequence[float],
    *,
    include_nested_cross_validation: bool = False,
) -> DecoupledSurface:
    """Build a radius/fold/kappa surface after full covariance whitening."""

    bank = np.asarray(operator_bank, dtype=np.float64)
    radius_values = np.asarray(tuple(radii), dtype=np.float64)
    if bank.ndim != 5 or bank.shape[0] != len(radius_values) or not len(radius_values):
        raise ValueError("operator_bank and radii disagree")
    if np.any(~np.isfinite(radius_values)) or np.any(np.diff(radius_values) <= 0.0):
        raise ValueError("radii must be finite and strictly increasing")
    if len(observations) == 0 or len(observations) != len(covariances):
        raise ValueError("observations and covariances must have equal nonzero length")
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
                    covariance_ridge_diagnostics(
                        operator,
                        observation,
                        covariance,
                        fit_views,
                        support_mask,
                        kappa,
                    )
                    for observation, covariance in zip(
                        observations, covariances, strict=True
                    )
                )
                outer_scores = tuple(
                    covariance_whitened_view_rms(
                        operator,
                        item.fit.field,
                        observation,
                        covariance,
                        [validation_view],
                    )
                    ** 2
                    for item, observation, covariance in zip(
                        diagnostics, observations, covariances, strict=True
                    )
                )
                nested_score = float("nan")
                if include_nested_cross_validation:
                    nested_score = _nested_cv_score(
                        operator,
                        observations,
                        covariances,
                        fit_views,
                        support_mask,
                        kappa,
                    )
                points.append(
                    DecoupledSurfacePoint(
                        kappa=kappa,
                        mean_generalized_cross_validation=float(
                            np.mean(
                                [item.generalized_cross_validation for item in diagnostics]
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
