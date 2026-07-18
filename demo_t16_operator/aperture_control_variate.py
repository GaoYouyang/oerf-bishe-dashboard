"""Unbiased control-variate estimators for finite-aperture operator means.

For one detector pixel, finite-aperture BOS requires an expectation over a
two-dimensional pupil coordinate.  The routines here operate on arbitrary
vector-valued integrands, including a complete linear operator.  The primary
estimator fits a low-order pupil polynomial on one IID half and corrects it on
the independent other half, then swaps the folds.  The residual correction is
never optional: without it the polynomial is only a biased surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CONTROL_VARIATE_SCHEMA = "cross-fitted-aperture-control-variate-1.0"


@dataclass(frozen=True)
class ControlVariateEstimate:
    estimate: np.ndarray
    plain_mean: np.ndarray
    fold_residual_rms: tuple[float, float]
    coefficient_frobenius_norms: tuple[float, float]
    sample_count: int
    basis_dimension: int
    basis: str
    ridge: float


def concentric_square_to_disk(square_points: np.ndarray) -> np.ndarray:
    """Map points from ``[0,1]^2`` to a uniform unit disk without a pole."""

    points = np.asarray(square_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("square_points must have shape [sample, 2]")
    if np.any(~np.isfinite(points)) or np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("square_points must be finite and lie in [0, 1]^2")
    sx = 2.0 * points[:, 0] - 1.0
    sy = 2.0 * points[:, 1] - 1.0
    output = np.zeros_like(points)
    center = (sx == 0.0) & (sy == 0.0)
    x_dominant = (~center) & (np.abs(sx) > np.abs(sy))
    y_dominant = (~center) & (~x_dominant)

    radius = np.zeros(len(points), dtype=np.float64)
    angle = np.zeros(len(points), dtype=np.float64)
    radius[x_dominant] = sx[x_dominant]
    angle[x_dominant] = (math.pi / 4.0) * (
        sy[x_dominant] / sx[x_dominant]
    )
    radius[y_dominant] = sy[y_dominant]
    angle[y_dominant] = (math.pi / 2.0) - (math.pi / 4.0) * (
        sx[y_dominant] / sy[y_dominant]
    )
    output[:, 0] = radius * np.cos(angle)
    output[:, 1] = radius * np.sin(angle)
    return output


def sample_uniform_disk_iid(count: int, *, seed: int) -> np.ndarray:
    if int(count) < 1:
        raise ValueError("count must be positive")
    square = np.random.default_rng(int(seed)).random((int(count), 2))
    return concentric_square_to_disk(square)


def sample_uniform_disk_antithetic(count: int, *, seed: int) -> np.ndarray:
    if int(count) < 2 or int(count) % 2:
        raise ValueError("antithetic count must be a positive even integer")
    half = sample_uniform_disk_iid(int(count) // 2, seed=int(seed))
    return np.stack((half, -half), axis=1).reshape(int(count), 2)


def centered_pupil_basis(points: np.ndarray, *, basis: str) -> np.ndarray:
    """Return a pupil basis whose nonconstant columns have known zero mean."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape [sample, 2]")
    if np.any(~np.isfinite(values)) or np.any(np.sum(values * values, axis=1) > 1.0 + 1e-12):
        raise ValueError("points must be finite and lie in the unit disk")
    x, y = values[:, 0], values[:, 1]
    if basis == "affine":
        return np.column_stack((np.ones(len(values)), x, y))
    if basis == "quadratic":
        return np.column_stack(
            (
                np.ones(len(values)),
                x,
                y,
                x * x - 0.25,
                x * y,
                y * y - 0.25,
            )
        )
    raise ValueError("basis must be 'affine' or 'quadratic'")


def _fit_operator_coefficients(
    design: np.ndarray,
    values: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    matrix = np.asarray(design, dtype=np.float64)
    targets = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or targets.shape[0] != matrix.shape[0]:
        raise ValueError("design and values must share their sample dimension")
    flat = targets.reshape(targets.shape[0], -1)
    gram = matrix.T @ matrix
    relative = float(ridge)
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    penalty = relative * max(float(np.trace(gram)) / len(gram), 1.0)
    regularizer = np.eye(matrix.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty * regularizer,
        matrix.T @ flat,
    )
    return coefficients.reshape((matrix.shape[1],) + targets.shape[1:])


def _apply_basis(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.tensordot(design, coefficients, axes=(1, 0))


def cross_fitted_control_variate(
    points: np.ndarray,
    values: np.ndarray,
    *,
    basis: str = "quadratic",
    ridge: float = 1e-8,
) -> ControlVariateEstimate:
    """Estimate a uniform-disk mean with two-fold independent correction.

    Samples must be IID.  Fold 0 is corrected by a polynomial trained only on
    fold 1 and vice versa.  Since all nonconstant basis functions have exactly
    zero uniform-disk mean, each corrected fold has the original expectation.
    Applying this function to an operator bank yields one matrix that can be
    reused unchanged for both forward and adjoint calls.
    """

    pupil = np.asarray(points, dtype=np.float64)
    targets = np.asarray(values, dtype=np.float64)
    if pupil.ndim != 2 or pupil.shape[1] != 2:
        raise ValueError("points must have shape [sample, 2]")
    if targets.shape[0] != pupil.shape[0]:
        raise ValueError("points and values must share their sample dimension")
    if len(pupil) < 8 or len(pupil) % 2:
        raise ValueError("cross fitting requires an even sample count of at least 8")
    design = centered_pupil_basis(pupil, basis=basis)
    half = len(pupil) // 2
    if half < design.shape[1]:
        raise ValueError("each fold must contain at least as many samples as basis terms")
    folds = (np.arange(0, half), np.arange(half, len(pupil)))
    estimates: list[np.ndarray] = []
    residual_rms: list[float] = []
    coefficient_norms: list[float] = []
    for held_out, training in ((folds[0], folds[1]), (folds[1], folds[0])):
        coefficients = _fit_operator_coefficients(
            design[training], targets[training], ridge=float(ridge)
        )
        prediction = _apply_basis(design[held_out], coefficients)
        residual = targets[held_out] - prediction
        estimates.append(coefficients[0] + np.mean(residual, axis=0))
        residual_rms.append(float(np.sqrt(np.mean(residual * residual))))
        coefficient_norms.append(float(np.linalg.norm(coefficients)))
    estimate = 0.5 * (estimates[0] + estimates[1])
    return ControlVariateEstimate(
        estimate=estimate,
        plain_mean=np.mean(targets, axis=0),
        fold_residual_rms=(residual_rms[0], residual_rms[1]),
        coefficient_frobenius_norms=(coefficient_norms[0], coefficient_norms[1]),
        sample_count=len(pupil),
        basis_dimension=design.shape[1],
        basis=basis,
        ridge=float(ridge),
    )


def disk_product_quadrature(
    radial_order: int,
    angular_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic Gauss-in-radius/trapezoidal-in-angle disk nodes."""

    if int(radial_order) < 2 or int(angular_order) < 4:
        raise ValueError("disk quadrature orders are too small")
    nodes, weights = np.polynomial.legendre.leggauss(int(radial_order))
    radial_squared = 0.5 * (nodes + 1.0)
    radial_weights = 0.5 * weights
    angles = 2.0 * math.pi * np.arange(int(angular_order)) / float(angular_order)
    points = []
    output_weights = []
    for radius_squared, radial_weight in zip(
        radial_squared, radial_weights, strict=True
    ):
        radius = math.sqrt(float(radius_squared))
        for angle in angles:
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
            output_weights.append(float(radial_weight) / float(angular_order))
    return np.asarray(points, dtype=np.float64), np.asarray(
        output_weights, dtype=np.float64
    )


def weighted_operator_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    targets = np.asarray(values, dtype=np.float64)
    values_weights = np.asarray(weights, dtype=np.float64)
    if values_weights.ndim != 1 or targets.shape[0] != len(values_weights):
        raise ValueError("weights must match the leading sample dimension")
    if np.any(~np.isfinite(values_weights)) or np.any(values_weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.isclose(np.sum(values_weights), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("weights must sum to one")
    return np.tensordot(values_weights, targets, axes=(0, 0))
