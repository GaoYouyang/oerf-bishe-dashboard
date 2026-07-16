"""Separate optical-parameter calibration from final field regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from .decoupled_complexity import ComplexityChoice, choose_complexity
    from .nested_crossview import ScaledRidgeRefit, scaled_ridge_fit
except ImportError:
    from decoupled_complexity import ComplexityChoice, choose_complexity
    from nested_crossview import ScaledRidgeRefit, scaled_ridge_fit


@dataclass(frozen=True)
class RadiusMethodRefit:
    """One fixed-radius refit with an independently chosen field complexity."""

    radius_index: int
    method: str
    choice: ComplexityChoice
    refit: ScaledRidgeRefit


def refit_fixed_radius_with_method(
    radius_index: int,
    method: str,
    operator_bank: np.ndarray,
    observations: Sequence[np.ndarray],
    noise_std: Sequence[np.ndarray],
    fit_views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappas: Sequence[float],
) -> RadiusMethodRefit:
    """Choose field complexity after the optical radius has been frozen."""

    bank = np.asarray(operator_bank, dtype=np.float64)
    index = int(radius_index)
    if bank.ndim != 5 or not 0 <= index < bank.shape[0]:
        raise ValueError("radius_index is outside the operator bank")
    if len(observations) == 0 or len(observations) != len(noise_std):
        raise ValueError("observations and noise_std must have equal nonzero length")
    choice = choose_complexity(
        method,
        bank[index],
        observations,
        noise_std,
        fit_views,
        support,
        kappas,
    )
    fits = []
    effective_lambdas = []
    for observation, sigma in zip(observations, noise_std, strict=True):
        fit, effective_lambda = scaled_ridge_fit(
            bank[index],
            observation,
            sigma,
            fit_views,
            support,
            choice.kappa,
        )
        fits.append(fit)
        effective_lambdas.append(effective_lambda)
    return RadiusMethodRefit(
        radius_index=index,
        method=method,
        choice=choice,
        refit=ScaledRidgeRefit(tuple(fits), tuple(effective_lambdas)),
    )


def error_reduction_percent(candidate_error: float, baseline_error: float) -> float:
    """Positive means the candidate reduced an error metric."""

    candidate = float(candidate_error)
    baseline = float(baseline_error)
    if not np.isfinite(candidate) or not np.isfinite(baseline):
        raise ValueError("errors must be finite")
    return float(100.0 * (baseline - candidate) / max(abs(baseline), 1e-12))


def outer_only_route(
    radius_changed: bool,
    outer_error_reductions_percent: Sequence[float],
    *,
    minimum_per_view_reduction_percent: float,
) -> bool:
    """Route with optical change and outer cameras only."""

    reductions = np.asarray(tuple(outer_error_reductions_percent), dtype=float)
    threshold = float(minimum_per_view_reduction_percent)
    if reductions.size == 0 or np.any(~np.isfinite(reductions)):
        raise ValueError("outer reductions must be finite and nonempty")
    if not np.isfinite(threshold):
        raise ValueError("outer threshold must be finite")
    return bool(radius_changed and np.all(reductions >= threshold))
