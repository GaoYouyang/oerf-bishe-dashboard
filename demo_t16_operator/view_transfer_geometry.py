"""Geometry diagnostics and residual-field transfer for held-out BOST views.

The functions in this module are deliberately small-matrix references.  They
use source-camera observations and target-camera operators, but never target
observations.  This makes the audit firewall explicit while testing whether
view geometry can support a later transfer rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from .dual_regularization import error_reduction_percent
    from .rig_shared_profile import fit_support_ridge, whitened_support_system
except ImportError:
    from dual_regularization import error_reduction_percent
    from rig_shared_profile import fit_support_ridge, whitened_support_system


@dataclass(frozen=True)
class GroupLeverage:
    """Predictive leverage of one camera group against an inner-view fit."""

    total: float
    mean_per_measurement: float
    maximum_diagonal: float


@dataclass(frozen=True)
class ResidualTransferPrediction:
    """Target-view residual prediction produced without target observations."""

    residual_field: np.ndarray
    source_fit_whitened_rms: float
    predicted_baseline_rms: tuple[float, ...]
    predicted_candidate_rms: tuple[float, ...]
    predicted_error_reductions_percent: tuple[float, ...]


def _validated_2d(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def row_space_basis(matrix: np.ndarray, *, relative_tolerance: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis for a matrix row space."""

    values = _validated_2d(matrix, "matrix")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    _, singular_values, right = np.linalg.svd(values, full_matrices=False)
    if singular_values[0] <= 0.0:
        raise ValueError("matrix has zero numerical rank")
    keep = singular_values > tolerance * singular_values[0]
    if not np.any(keep):
        raise ValueError("matrix has zero numerical rank")
    return right[keep].T


def projection_similarity(
    first: np.ndarray, second: np.ndarray, *, relative_tolerance: float = 1e-10
) -> float:
    """Normalized overlap of two row-space projection matrices."""

    first_basis = row_space_basis(first, relative_tolerance=relative_tolerance)
    second_basis = row_space_basis(second, relative_tolerance=relative_tolerance)
    overlap = float(np.sum(np.square(first_basis.T @ second_basis)))
    scale = float(np.sqrt(first_basis.shape[1] * second_basis.shape[1]))
    return float(np.clip(overlap / scale, 0.0, 1.0))


def gram_cosine(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine similarity between two normal/Fisher information matrices."""

    first_values = _validated_2d(first, "first")
    second_values = _validated_2d(second, "second")
    if first_values.shape[1] != second_values.shape[1]:
        raise ValueError("matrices must act on the same parameter space")
    first_gram = first_values.T @ first_values
    second_gram = second_values.T @ second_values
    denominator = float(
        np.linalg.norm(first_gram, ord="fro")
        * np.linalg.norm(second_gram, ord="fro")
    )
    if denominator <= 0.0:
        raise ValueError("normal matrix has zero energy")
    return float(np.clip(np.sum(first_gram * second_gram) / denominator, -1.0, 1.0))


def operator_change_cosine(
    first_candidate: np.ndarray,
    first_baseline: np.ndarray,
    second_candidate: np.ndarray,
    second_baseline: np.ndarray,
) -> float:
    """Cosine of the optical-operator change seen by two camera groups."""

    first = _validated_2d(first_candidate, "first_candidate") - _validated_2d(
        first_baseline, "first_baseline"
    )
    second = _validated_2d(second_candidate, "second_candidate") - _validated_2d(
        second_baseline, "second_baseline"
    )
    if first.shape != second.shape:
        raise ValueError("operator changes must share a measurement coordinate system")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        raise ValueError("operator change has zero energy")
    return float(np.clip(np.sum(first * second) / denominator, -1.0, 1.0))


def group_predictive_leverage(
    inner_matrix: np.ndarray, target_matrix: np.ndarray, ridge_lambda: float
) -> GroupLeverage:
    """Compute target group leverage against a ridge-stabilized inner Hessian."""

    inner = _validated_2d(inner_matrix, "inner_matrix")
    target = _validated_2d(target_matrix, "target_matrix")
    if inner.shape[1] != target.shape[1]:
        raise ValueError("inner and target matrices must share parameter columns")
    regularization = float(ridge_lambda)
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("ridge_lambda must be finite and positive")
    hessian = inner.T @ inner
    hessian.flat[:: hessian.shape[0] + 1] += regularization
    solved = np.linalg.solve(hessian, target.T)
    diagonal = np.sum(target * solved.T, axis=1)
    diagonal = np.maximum(diagonal, 0.0)
    return GroupLeverage(
        total=float(np.sum(diagonal)),
        mean_per_measurement=float(np.mean(diagonal)),
        maximum_diagonal=float(np.max(diagonal)),
    )


def similarity_weighted_gain(
    similarities: Sequence[float], gains: Sequence[float]
) -> float:
    """Average source-camera gains using nonnegative, parameter-free weights."""

    weights = np.asarray(tuple(similarities), dtype=np.float64)
    values = np.asarray(tuple(gains), dtype=np.float64)
    if weights.size == 0 or weights.shape != values.shape:
        raise ValueError("similarities and gains must have equal nonzero length")
    if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(values)):
        raise ValueError("similarities and gains must be finite")
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        return float(np.mean(values))
    return float(np.sum(weights * values) / np.sum(weights))


def camera_support_matrix(
    operator: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    camera_index: int,
    support: np.ndarray,
) -> np.ndarray:
    """Expose one camera's support-restricted, diagonally whitened operator."""

    matrix, _, _ = whitened_support_system(
        operator, observation, noise_std, (int(camera_index),), support
    )
    return matrix


def residual_field_transfer(
    baseline_operator: np.ndarray,
    candidate_operator: np.ndarray,
    baseline_field: np.ndarray,
    candidate_field: np.ndarray,
    observation: np.ndarray,
    noise_std: np.ndarray,
    source_views: Sequence[int],
    target_views: Sequence[int],
    support: np.ndarray,
    ridge_lambda: float,
) -> ResidualTransferPrediction:
    """Predict target residual changes from a source-only residual-field fit."""

    baseline = np.asarray(baseline_operator, dtype=np.float64)
    candidate = np.asarray(candidate_operator, dtype=np.float64)
    values = np.asarray(observation, dtype=np.float64)
    if baseline.ndim != 4 or candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline operators must share a 4-D shape")
    if values.shape != baseline.shape[:3]:
        raise ValueError("observation and operator measurement shapes disagree")
    source = tuple(int(value) for value in source_views)
    target = tuple(int(value) for value in target_views)
    if not source or not target or set(source) & set(target):
        raise ValueError("source and target views must be nonempty and disjoint")
    if min(source + target) < 0 or max(source + target) >= baseline.shape[1]:
        raise ValueError("source or target view is outside the operator")

    baseline_prediction = np.einsum(
        "dvnp,p->dvn", baseline, np.asarray(baseline_field).reshape(-1), optimize=True
    )
    candidate_prediction = np.einsum(
        "dvnp,p->dvn", candidate, np.asarray(candidate_field).reshape(-1), optimize=True
    )
    source_residual = np.zeros_like(values)
    source_residual[:, source, :] = (
        values[:, source, :] - baseline_prediction[:, source, :]
    )
    residual_fit = fit_support_ridge(
        baseline,
        source_residual,
        noise_std,
        source,
        support,
        ridge_lambda,
    )
    transferred_baseline = np.einsum(
        "dvnp,p->dvn", baseline, residual_fit.field.reshape(-1), optimize=True
    )
    transferred_candidate = transferred_baseline - (
        candidate_prediction - baseline_prediction
    )

    sigma = np.asarray(noise_std, dtype=np.float64)
    if sigma.ndim == 1 and sigma.shape == (baseline.shape[1],):
        sigma = sigma[None, :, None]
    try:
        expanded_sigma = np.broadcast_to(sigma, values.shape)
    except ValueError as error:
        raise ValueError("noise_std cannot be broadcast to the observation") from error
    if np.any(~np.isfinite(expanded_sigma)) or np.any(expanded_sigma <= 0.0):
        raise ValueError("noise_std must be finite and positive")

    source_mask = np.zeros(baseline.shape[1], dtype=bool)
    source_mask[list(source)] = True
    source_active = np.broadcast_to(source_mask[None, :, None], values.shape)
    fitted_source_residual = source_residual - np.einsum(
        "dvnp,p->dvn", baseline, residual_fit.field.reshape(-1), optimize=True
    )
    source_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    fitted_source_residual[source_active]
                    / expanded_sigma[source_active]
                )
            )
        )
    )

    baseline_rms: list[float] = []
    candidate_rms: list[float] = []
    gains: list[float] = []
    for view in target:
        baseline_value = float(
            np.sqrt(
                np.mean(
                    np.square(
                        transferred_baseline[:, view, :]
                        / expanded_sigma[:, view, :]
                    )
                )
            )
        )
        candidate_value = float(
            np.sqrt(
                np.mean(
                    np.square(
                        transferred_candidate[:, view, :]
                        / expanded_sigma[:, view, :]
                    )
                )
            )
        )
        baseline_rms.append(baseline_value)
        candidate_rms.append(candidate_value)
        gains.append(error_reduction_percent(candidate_value, baseline_value))
    return ResidualTransferPrediction(
        residual_field=residual_fit.field,
        source_fit_whitened_rms=source_rms,
        predicted_baseline_rms=tuple(baseline_rms),
        predicted_candidate_rms=tuple(candidate_rms),
        predicted_error_reductions_percent=tuple(gains),
    )
