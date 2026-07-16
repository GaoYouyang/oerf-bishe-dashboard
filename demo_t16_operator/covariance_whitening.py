"""Exact small-grid whitening for the correlated synthetic BOST noise model.

This module is an oracle diagnostic: ``camera_noise_covariance`` uses the clean
synthetic observation to reproduce the signal-dependent covariance used by the
generator.  Real experiments must estimate covariance from independent
flow-off/background repeats and may not substitute an unknown clean field.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from .rig_shared_profile import RidgeProfileFit
except ImportError:
    from rig_shared_profile import RidgeProfileFit


def camera_noise_covariance(
    clean: np.ndarray,
    camera_std: np.ndarray,
    *,
    correlation_fraction: float,
    signal_fraction: float,
) -> np.ndarray:
    """Return the per-camera covariance implied by ``correlated_camera_noise``."""

    values = np.asarray(clean, dtype=np.float64)
    scales = np.asarray(camera_std, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("clean must have shape [detector_z,view,detector_x]")
    if scales.shape != (values.shape[1],):
        raise ValueError("camera_std must contain one scale per view")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(scales)):
        raise ValueError("clean and camera_std must be finite")
    if np.any(scales <= 0.0):
        raise ValueError("camera_std must be strictly positive")
    correlation = float(correlation_fraction)
    signal = float(signal_fraction)
    if (
        not np.isfinite(correlation)
        or not np.isfinite(signal)
        or correlation < 0.0
        or signal < 0.0
        or correlation**2 + signal**2 >= 1.0
    ):
        raise ValueError("noise fractions must be non-negative with squared sum < 1")

    depth, view_count, detector = values.shape
    pixel_count = depth * detector
    z_index = np.repeat(np.arange(depth), detector)
    x_index = np.tile(np.arange(detector), depth)
    same_row = z_index[:, None] == z_index[None, :]
    same_column = x_index[:, None] == x_index[None, :]
    correlated_covariance = 0.5 * (
        same_row.astype(np.float64) + same_column.astype(np.float64)
    )
    identity = np.eye(pixel_count, dtype=np.float64)
    iid_variance = 1.0 - correlation**2 - signal**2
    global_rms = float(np.sqrt(np.mean(values**2)) + 1e-8)

    covariance = np.empty(
        (view_count, pixel_count, pixel_count), dtype=np.float64
    )
    for view in range(view_count):
        signal_scale = np.abs(values[:, view, :]).reshape(-1) / global_rms
        normalized = iid_variance * identity
        normalized = normalized + correlation**2 * correlated_covariance
        normalized = normalized + signal**2 * np.diag(np.square(signal_scale))
        covariance[view] = float(scales[view] ** 2) * normalized
    return covariance


def _validated_views(views: Sequence[int] | np.ndarray, count: int) -> tuple[int, ...]:
    selected = tuple(int(value) for value in views)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("views must be nonempty and unique")
    if min(selected) < 0 or max(selected) >= count:
        raise ValueError("view is outside the operator")
    return selected


def _validated_covariance(
    covariance: np.ndarray, view_count: int, pixel_count: int
) -> np.ndarray:
    values = np.asarray(covariance, dtype=np.float64)
    if values.shape != (view_count, pixel_count, pixel_count):
        raise ValueError("covariance must have shape [view,pixel,pixel]")
    if np.any(~np.isfinite(values)):
        raise ValueError("covariance must be finite")
    if not np.allclose(values, np.swapaxes(values, 1, 2), rtol=1e-10, atol=1e-12):
        raise ValueError("covariance matrices must be symmetric")
    for view in range(view_count):
        try:
            np.linalg.cholesky(values[view])
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance matrices must be positive definite") from error
    return values


def covariance_whitened_support_system(
    operator: np.ndarray,
    observation: np.ndarray,
    covariance: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``L^-1 A`` and ``L^-1 y`` in camera-block order."""

    matrix = np.asarray(operator, dtype=np.float64)
    values = np.asarray(observation, dtype=np.float64)
    if matrix.ndim != 4 or values.shape != matrix.shape[:3]:
        raise ValueError("operator and observation measurement shapes disagree")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(values)):
        raise ValueError("operator and observation must be finite")
    support_mask = np.asarray(support)
    if support_mask.dtype != bool or support_mask.size != matrix.shape[-1]:
        raise ValueError("support must be a boolean mask matching the voxel count")
    support_mask = support_mask.reshape(-1)
    if not np.any(support_mask):
        raise ValueError("support must activate at least one voxel")
    selected = _validated_views(views, matrix.shape[1])
    pixel_count = matrix.shape[0] * matrix.shape[2]
    covariances = _validated_covariance(
        covariance, matrix.shape[1], pixel_count
    )

    whitened_rows: list[np.ndarray] = []
    whitened_data: list[np.ndarray] = []
    for view in selected:
        rows = matrix[:, view, :, :].reshape(pixel_count, matrix.shape[-1])
        rows = rows[:, support_mask]
        data = values[:, view, :].reshape(pixel_count)
        cholesky = np.linalg.cholesky(covariances[view])
        whitened_rows.append(np.linalg.solve(cholesky, rows))
        whitened_data.append(np.linalg.solve(cholesky, data))
    return (
        np.concatenate(whitened_rows, axis=0),
        np.concatenate(whitened_data, axis=0),
        support_mask,
    )


def covariance_normal_mean_diagonal(
    operator: np.ndarray,
    observation: np.ndarray,
    covariance: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
) -> float:
    matrix, _, _ = covariance_whitened_support_system(
        operator, observation, covariance, views, support
    )
    return float(np.sum(np.square(matrix)) / matrix.shape[1])


def fit_support_ridge_covariance(
    operator: np.ndarray,
    observation: np.ndarray,
    covariance: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    ridge_lambda: float,
) -> RidgeProfileFit:
    regularization = float(ridge_lambda)
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("ridge_lambda must be finite and strictly positive")
    matrix, data, support_mask = covariance_whitened_support_system(
        operator, observation, covariance, views, support
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


def covariance_scaled_ridge_fit(
    operator: np.ndarray,
    observation: np.ndarray,
    covariance: np.ndarray,
    views: Sequence[int] | np.ndarray,
    support: np.ndarray,
    kappa: float,
) -> tuple[RidgeProfileFit, float]:
    ratio = float(kappa)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("kappa must be finite and strictly positive")
    scale = covariance_normal_mean_diagonal(
        operator, observation, covariance, views, support
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("covariance-whitened normal scale must be positive")
    effective_lambda = float(ratio * scale)
    return (
        fit_support_ridge_covariance(
            operator,
            observation,
            covariance,
            views,
            support,
            effective_lambda,
        ),
        effective_lambda,
    )


def covariance_whitened_view_rms(
    operator: np.ndarray,
    field: np.ndarray,
    observation: np.ndarray,
    covariance: np.ndarray,
    views: Sequence[int] | np.ndarray,
) -> float:
    matrix = np.asarray(operator, dtype=np.float64)
    values = np.asarray(observation, dtype=np.float64)
    if matrix.ndim != 4 or values.shape != matrix.shape[:3]:
        raise ValueError("operator and observation measurement shapes disagree")
    selected = _validated_views(views, matrix.shape[1])
    pixel_count = matrix.shape[0] * matrix.shape[2]
    covariances = _validated_covariance(
        covariance, matrix.shape[1], pixel_count
    )
    flat_field = np.asarray(field, dtype=np.float64).reshape(-1)
    residuals: list[np.ndarray] = []
    for view in selected:
        rows = matrix[:, view, :, :].reshape(pixel_count, matrix.shape[-1])
        residual = values[:, view, :].reshape(pixel_count) - rows @ flat_field
        cholesky = np.linalg.cholesky(covariances[view])
        residuals.append(np.linalg.solve(cholesky, residual))
    merged = np.concatenate(residuals)
    return float(np.sqrt(np.mean(np.square(merged))))
