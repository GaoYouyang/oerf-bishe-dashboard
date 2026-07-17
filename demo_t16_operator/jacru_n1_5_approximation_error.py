"""Deployable predictors for the JACRU forward approximation error screen.

The predictors in this module consume only geometry, measured observations,
and a low-fidelity warm reconstruction projection.  High-fidelity mismatch is
used as a fit target by the development runner, never as an inference input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from .jacru_synthetic_fixture import JACRUGeometry


Tensor = torch.Tensor


@dataclass(frozen=True)
class StandardizedRidge:
    """A small deterministic ridge model with an unpenalized intercept."""

    feature_names: tuple[str, ...]
    feature_mean: Tensor
    feature_scale: Tensor
    coefficients: Tensor
    intercept: Tensor
    alpha: float

    def predict(self, features: Tensor) -> Tensor:
        matrix = _validated_matrix(features, columns=len(self.feature_names))
        standardized = (matrix - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.intercept


def _validated_matrix(values: Tensor, *, columns: int | None = None) -> Tensor:
    matrix = torch.as_tensor(values, dtype=torch.float64)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError("feature column count does not match the fitted model")
    if not bool(torch.all(torch.isfinite(matrix))):
        raise ValueError("features must be finite")
    return matrix


def fit_standardized_ridge(
    features: Tensor,
    targets: Tensor,
    *,
    feature_names: Sequence[str],
    alpha: float,
) -> StandardizedRidge:
    """Fit scalar ridge regression after training-only standardization."""

    matrix = _validated_matrix(features)
    target = torch.as_tensor(targets, dtype=torch.float64).reshape(-1)
    names = tuple(str(name) for name in feature_names)
    if matrix.shape[0] != target.numel():
        raise ValueError("features and targets must contain the same row count")
    if matrix.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("feature_names must uniquely name every feature column")
    if not bool(torch.all(torch.isfinite(target))):
        raise ValueError("targets must be finite")
    ridge = float(alpha)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("alpha must be finite and nonnegative")

    mean = torch.mean(matrix, dim=0)
    scale = torch.std(matrix, dim=0, unbiased=False).clamp_min(1e-12)
    standardized = (matrix - mean) / scale
    intercept = torch.mean(target)
    centered = target - intercept
    gram = standardized.T @ standardized
    regularizer = ridge * torch.eye(gram.shape[0], dtype=torch.float64)
    coefficients = torch.linalg.solve(gram + regularizer, standardized.T @ centered)
    return StandardizedRidge(
        feature_names=names,
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        intercept=intercept,
        alpha=ridge,
    )


def _detector_coordinates(geometry: JACRUGeometry) -> tuple[Tensor, Tensor]:
    rows, columns = (int(value) for value in geometry.detector_shape)
    rays_per_camera = rows * columns
    if geometry.ray_count != geometry.camera_count * rays_per_camera:
        raise ValueError("geometry ray ordering does not match detector_shape")
    detector_v, detector_u = torch.meshgrid(
        torch.linspace(-1.0, 1.0, rows, dtype=torch.float64),
        torch.linspace(-1.0, 1.0, columns, dtype=torch.float64),
        indexing="ij",
    )
    u = detector_u.reshape(-1).repeat(geometry.camera_count)
    v = detector_v.reshape(-1).repeat(geometry.camera_count)
    return u, v


def _detector_derivatives(values: Tensor, geometry: JACRUGeometry) -> tuple[Tensor, Tensor, Tensor]:
    rows, columns = (int(value) for value in geometry.detector_shape)
    image = torch.as_tensor(values, dtype=torch.float64).reshape(
        geometry.camera_count, rows, columns, 2
    )
    derivative_u = torch.zeros_like(image)
    derivative_v = torch.zeros_like(image)
    derivative_u[:, :, 1:-1] = 0.5 * (image[:, :, 2:] - image[:, :, :-2])
    derivative_u[:, :, 0] = image[:, :, 1] - image[:, :, 0]
    derivative_u[:, :, -1] = image[:, :, -1] - image[:, :, -2]
    derivative_v[:, 1:-1] = 0.5 * (image[:, 2:] - image[:, :-2])
    derivative_v[:, 0] = image[:, 1] - image[:, 0]
    derivative_v[:, -1] = image[:, -1] - image[:, -2]
    laplacian = torch.zeros_like(image)
    laplacian[:, :, 1:-1] += image[:, :, 2:] - 2.0 * image[:, :, 1:-1] + image[:, :, :-2]
    laplacian[:, 1:-1] += image[:, 2:] - 2.0 * image[:, 1:-1] + image[:, :-2]
    return derivative_u.reshape(-1, 2), derivative_v.reshape(-1, 2), laplacian.reshape(-1, 2)


def visible_feature_blocks(
    *,
    geometry: JACRUGeometry,
    observation_uv: Tensor,
    warm_projection_uv: Tensor,
) -> Mapping[str, tuple[tuple[str, ...], Tensor]]:
    """Return nested feature sets built only from deployment-visible values.

    Signal values are normalized by measured RMS, which is itself observable.
    Every output row corresponds to one ray/component pair.
    """

    observation = torch.as_tensor(observation_uv, dtype=torch.float64)
    warm = torch.as_tensor(warm_projection_uv, dtype=torch.float64)
    expected = (geometry.ray_count, 2)
    if observation.shape != expected or warm.shape != expected:
        raise ValueError(f"observation and warm projection must have shape {expected}")
    if not bool(torch.all(torch.isfinite(observation))) or not bool(torch.all(torch.isfinite(warm))):
        raise ValueError("observation and warm projection must be finite")

    signal_scale = torch.sqrt(torch.mean(observation.square())).clamp_min(1e-12)
    observation_n = observation / signal_scale
    warm_n = warm / signal_scale
    residual_n = observation_n - warm_n
    obs_du, obs_dv, obs_lap = _detector_derivatives(observation_n, geometry)
    warm_du, warm_dv, warm_lap = _detector_derivatives(warm_n, geometry)

    u, v = _detector_coordinates(geometry)
    camera = geometry.camera_index.to(torch.int64)
    component = torch.arange(2, dtype=torch.int64).repeat(geometry.ray_count)
    def repeat_two(value: Tensor) -> Tensor:
        return value[:, None].expand(-1, 2).reshape(-1)
    camera_rows = repeat_two(camera.to(torch.float64))
    u_rows = repeat_two(u)
    v_rows = repeat_two(v)
    component_v = component.to(torch.float64)

    azimuth = torch.as_tensor(geometry.camera_azimuth_degrees, dtype=torch.float64)
    elevation = torch.as_tensor(geometry.camera_elevation_degrees, dtype=torch.float64)
    azimuth_rad = torch.deg2rad(azimuth)[camera]
    elevation_rad = torch.deg2rad(elevation)[camera]
    line_length = geometry.line_length.to(torch.float64)
    geometry_columns = {
        "component_v": component_v,
        "camera_index": camera_rows,
        "detector_u": u_rows,
        "detector_v": v_rows,
        "detector_u2": u_rows.square(),
        "detector_v2": v_rows.square(),
        "detector_uv": u_rows * v_rows,
        "sin_azimuth": repeat_two(torch.sin(azimuth_rad)),
        "cos_azimuth": repeat_two(torch.cos(azimuth_rad)),
        "sin_elevation": repeat_two(torch.sin(elevation_rad)),
        "cos_elevation": repeat_two(torch.cos(elevation_rad)),
        "line_length": repeat_two(line_length),
    }
    signal_columns = {
        "observation": observation_n.reshape(-1),
        "warm_projection": warm_n.reshape(-1),
        "warm_residual": residual_n.reshape(-1),
        "abs_observation": torch.abs(observation_n).reshape(-1),
        "abs_warm_projection": torch.abs(warm_n).reshape(-1),
    }
    curvature_columns = {
        "observation_du": obs_du.reshape(-1),
        "observation_dv": obs_dv.reshape(-1),
        "observation_laplacian": obs_lap.reshape(-1),
        "warm_du": warm_du.reshape(-1),
        "warm_dv": warm_dv.reshape(-1),
        "warm_laplacian": warm_lap.reshape(-1),
    }

    def pack(columns: Mapping[str, Tensor]) -> tuple[tuple[str, ...], Tensor]:
        names = tuple(columns)
        matrix = torch.stack([columns[name] for name in names], dim=1)
        return names, matrix

    return {
        "geometry_only": pack(geometry_columns),
        "geometry_observation": pack(
            {**geometry_columns, "observation": signal_columns["observation"]}
        ),
        "geometry_signal": pack({**geometry_columns, **signal_columns}),
        "curvature_visible": pack(
            {**geometry_columns, **signal_columns, **curvature_columns}
        ),
    }


def pca_oracle_prediction(
    *,
    training_vectors: Tensor,
    target_vector: Tensor,
    rank: int,
) -> Tensor:
    """Project a target mismatch onto a train-only PCA basis.

    This function intentionally consumes the target mismatch.  It is an
    evaluator-only representational oracle and must never be called by a
    deployable reconstruction path.
    """

    training = _validated_matrix(training_vectors)
    target = torch.as_tensor(target_vector, dtype=torch.float64).reshape(-1)
    if training.shape[1] != target.numel():
        raise ValueError("target dimension must match training vectors")
    count = int(rank)
    maximum = min(training.shape[0] - 1, training.shape[1])
    if count < 0 or count > maximum:
        raise ValueError(f"rank must lie in [0, {maximum}]")
    mean = torch.mean(training, dim=0)
    if count == 0:
        return mean
    _, _, right = torch.linalg.svd(training - mean, full_matrices=False)
    basis = right[:count]
    centered = target - mean
    return mean + basis.T @ (basis @ centered)
