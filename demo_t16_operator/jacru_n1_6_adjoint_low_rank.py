"""Low-rank, adjoint-weighted approximation-error models for JACRU.

The deployable path predicts coefficients in a measurement-space basis using
only geometry, measured data, and a low-order warm reconstruction.  Exact
high/low mismatch is consumed only while fitting targets or evaluating oracle
headroom.  Keeping the predicted correction in measurement space guarantees
that a fresh low-order adjoint, rather than a learned independent adjoint,
determines its effect on the reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import torch

from .jacru_synthetic_fixture import JACRUGeometry


Tensor = torch.Tensor
AdjointMap = Callable[[Tensor], Tensor]


def _finite_matrix(values: Tensor, *, name: str) -> Tensor:
    matrix = torch.as_tensor(values, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not bool(torch.all(torch.isfinite(matrix))):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _finite_vector(values: Tensor, *, name: str) -> Tensor:
    vector = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if vector.numel() < 1 or not bool(torch.all(torch.isfinite(vector))):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return vector


@dataclass(frozen=True)
class MultiOutputStandardizedRidge:
    """Small multi-output ridge map with an unpenalized intercept."""

    feature_names: tuple[str, ...]
    feature_mean: Tensor
    feature_scale: Tensor
    weights: Tensor
    alpha: float

    def standardized(self, features: Tensor) -> Tensor:
        matrix = _finite_matrix(features, name="features")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature column count does not match the fitted model")
        return (matrix - self.feature_mean) / self.feature_scale

    def predict(self, features: Tensor) -> Tensor:
        standardized = self.standardized(features)
        design = torch.cat(
            (standardized, torch.ones((standardized.shape[0], 1), dtype=standardized.dtype)),
            dim=1,
        )
        return design @ self.weights


def fit_multioutput_ridge(
    features: Tensor,
    targets: Tensor,
    *,
    feature_names: Sequence[str],
    alpha: float,
) -> MultiOutputStandardizedRidge:
    """Fit a deterministic multi-output ridge model."""

    matrix = _finite_matrix(features, name="features")
    response = _finite_matrix(targets, name="targets")
    names = tuple(str(name) for name in feature_names)
    penalty = float(alpha)
    if matrix.shape[0] != response.shape[0]:
        raise ValueError("features and targets must have the same row count")
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must uniquely name every feature column")
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("alpha must be finite and nonnegative")

    mean = torch.mean(matrix, dim=0)
    scale = torch.std(matrix, dim=0, unbiased=False).clamp_min(1e-12)
    standardized = (matrix - mean) / scale
    design = torch.cat(
        (standardized, torch.ones((matrix.shape[0], 1), dtype=matrix.dtype)), dim=1
    )
    gram = design.T @ design
    regularizer = torch.eye(gram.shape[0], dtype=gram.dtype) * penalty
    regularizer[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + regularizer, design.T @ response)
    return MultiOutputStandardizedRidge(
        feature_names=names,
        feature_mean=mean,
        feature_scale=scale,
        weights=weights,
        alpha=penalty,
    )


@dataclass(frozen=True)
class MeasurementBasis:
    """Train-only centered measurement-space PCA basis."""

    mean: Tensor
    vectors: Tensor

    @property
    def rank(self) -> int:
        return int(self.vectors.shape[0])

    def synthesize(self, coefficients: Tensor) -> Tensor:
        values = _finite_vector(coefficients, name="coefficients")
        if values.numel() != self.rank:
            raise ValueError("coefficient count must match basis rank")
        return self.mean + values @ self.vectors


def fit_measurement_pca(training_vectors: Tensor, *, rank: int) -> MeasurementBasis:
    """Fit a centered basis without exposing fresh target coefficients."""

    matrix = _finite_matrix(training_vectors, name="training_vectors")
    count = int(rank)
    maximum = min(matrix.shape[0] - 1, matrix.shape[1])
    if count < 1 or count > maximum:
        raise ValueError(f"rank must lie in [1, {maximum}]")
    mean = torch.mean(matrix, dim=0)
    _, _, vh = torch.linalg.svd(matrix - mean, full_matrices=False)
    return MeasurementBasis(mean=mean, vectors=vh[:count])


def measurement_optimal_coefficients(basis: MeasurementBasis, target: Tensor) -> Tensor:
    """Return the ordinary measurement-L2 projection coefficients."""

    values = _finite_vector(target, name="target")
    if values.numel() != basis.mean.numel():
        raise ValueError("target size must match the measurement basis")
    return basis.vectors @ (values - basis.mean)


@dataclass(frozen=True)
class AdjointCoefficientTarget:
    coefficients: Tensor
    residual_ratio: float
    target_adjoint_norm: float
    evaluator_adjoint_calls: int


def adjoint_optimal_coefficients(
    basis: MeasurementBasis,
    target: Tensor,
    *,
    observation_shape: Sequence[int],
    adjoint: AdjointMap,
    l2: float,
) -> AdjointCoefficientTarget:
    """Fit basis coefficients in the norm induced by the current adjoint.

    The exact target is intentionally required here.  This function is for
    offline fit targets and evaluator-only oracle rows, never fresh inference.
    """

    values = _finite_vector(target, name="target")
    shape = tuple(int(value) for value in observation_shape)
    penalty = float(l2)
    if math.prod(shape) != values.numel() or values.numel() != basis.mean.numel():
        raise ValueError("observation_shape, target, and basis dimensions must agree")
    if not callable(adjoint):
        raise TypeError("adjoint must be callable")
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("l2 must be finite and nonnegative")

    centered = values - basis.mean
    target_adjoint = _finite_vector(
        adjoint(centered.reshape(shape)), name="target_adjoint"
    )
    columns = []
    for vector in basis.vectors:
        column = _finite_vector(adjoint(vector.reshape(shape)), name="basis_adjoint")
        if column.numel() != target_adjoint.numel():
            raise ValueError("adjoint output shape drifted across basis vectors")
        columns.append(column)
    design = torch.stack(columns, dim=1)
    gram = design.T @ design + penalty * torch.eye(basis.rank, dtype=design.dtype)
    coefficients = torch.linalg.solve(gram, design.T @ target_adjoint)
    residual = target_adjoint - design @ coefficients
    target_norm = float(torch.linalg.vector_norm(target_adjoint))
    residual_ratio = float(torch.linalg.vector_norm(residual)) / max(target_norm, 1e-30)
    return AdjointCoefficientTarget(
        coefficients=coefficients,
        residual_ratio=residual_ratio,
        target_adjoint_norm=target_norm,
        evaluator_adjoint_calls=basis.rank + 1,
    )


def _rms(values: Tensor) -> Tensor:
    return torch.sqrt(torch.mean(values.square())).clamp_min(1e-12)


def _safe_cosine(left: Tensor, right: Tensor) -> Tensor:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return torch.sum(left * right) / denominator.clamp_min(1e-12)


def _field_roughness(field: Tensor) -> tuple[Tensor, Tensor]:
    gradients = [
        torch.diff(field, dim=axis).reshape(-1) for axis in range(field.ndim)
    ]
    gradient_rms = _rms(torch.cat(gradients))
    laplacian = torch.zeros_like(field)
    for axis in range(field.ndim):
        middle = [slice(None)] * field.ndim
        lower = [slice(None)] * field.ndim
        upper = [slice(None)] * field.ndim
        middle[axis] = slice(1, -1)
        lower[axis] = slice(None, -2)
        upper[axis] = slice(2, None)
        laplacian[tuple(middle)] += (
            field[tuple(lower)] - 2.0 * field[tuple(middle)] + field[tuple(upper)]
        )
    return gradient_rms, _rms(laplacian)


def visible_case_feature_blocks(
    *,
    geometry: JACRUGeometry,
    observation_uv: Tensor,
    warm_projection_uv: Tensor,
    warm_field: Tensor,
) -> Mapping[str, tuple[tuple[str, ...], Tensor]]:
    """Build fixed-size case features from deployment-visible quantities."""

    observation = torch.as_tensor(observation_uv, dtype=torch.float64)
    warm = torch.as_tensor(warm_projection_uv, dtype=torch.float64)
    field = torch.as_tensor(warm_field, dtype=torch.float64)
    expected = (geometry.ray_count, 2)
    if observation.shape != expected or warm.shape != expected:
        raise ValueError(f"observation and warm projection must have shape {expected}")
    if field.ndim != 3:
        raise ValueError("warm_field must have shape [z,y,x]")
    if not all(bool(torch.all(torch.isfinite(value))) for value in (observation, warm, field)):
        raise ValueError("visible features require finite inputs")

    signal_scale = _rms(observation)
    observation_n = observation / signal_scale
    warm_n = warm / signal_scale
    residual_n = observation_n - warm_n
    gradient_rms, laplacian_rms = _field_roughness(field / _rms(field))

    columns: dict[str, Tensor] = {}

    def add_stats(prefix: str, values: Tensor) -> None:
        flat = values.reshape(-1)
        columns[f"{prefix}_mean"] = torch.mean(flat)
        columns[f"{prefix}_rms"] = _rms(flat)
        columns[f"{prefix}_abs_mean"] = torch.mean(torch.abs(flat))
        columns[f"{prefix}_max_abs"] = torch.max(torch.abs(flat))

    add_stats("observation", observation_n)
    add_stats("warm_projection", warm_n)
    add_stats("warm_residual", residual_n)
    columns["observation_warm_cosine"] = _safe_cosine(observation_n, warm_n)
    columns["residual_observation_cosine"] = _safe_cosine(residual_n, observation_n)
    field_n = field / _rms(field)
    columns["warm_field_mean"] = torch.mean(field_n)
    columns["warm_field_abs_mean"] = torch.mean(torch.abs(field_n))
    columns["warm_field_max_abs"] = torch.max(torch.abs(field_n))
    columns["warm_field_gradient_rms"] = gradient_rms
    columns["warm_field_laplacian_rms"] = laplacian_rms
    summary_names = tuple(columns)

    camera_index = geometry.camera_index.to(torch.int64)
    for camera in range(geometry.camera_count):
        mask = camera_index == camera
        for prefix, values in (
            ("observation", observation_n),
            ("warm_projection", warm_n),
            ("warm_residual", residual_n),
        ):
            selected = values[mask]
            columns[f"camera_{camera}_{prefix}_rms"] = _rms(selected)
            columns[f"camera_{camera}_{prefix}_mean_u"] = torch.mean(selected[:, 0])
            columns[f"camera_{camera}_{prefix}_mean_v"] = torch.mean(selected[:, 1])
    camera_names = tuple(columns)

    azimuth = torch.deg2rad(
        torch.as_tensor(geometry.camera_azimuth_degrees, dtype=torch.float64)
    )
    elevation = torch.deg2rad(
        torch.as_tensor(geometry.camera_elevation_degrees, dtype=torch.float64)
    )
    line_length = geometry.line_length.to(torch.float64)
    for camera in range(geometry.camera_count):
        mask = camera_index == camera
        columns[f"camera_{camera}_sin_azimuth"] = torch.sin(azimuth[camera])
        columns[f"camera_{camera}_cos_azimuth"] = torch.cos(azimuth[camera])
        columns[f"camera_{camera}_sin_elevation"] = torch.sin(elevation[camera])
        columns[f"camera_{camera}_cos_elevation"] = torch.cos(elevation[camera])
        columns[f"camera_{camera}_line_length_mean"] = torch.mean(line_length[mask])
        columns[f"camera_{camera}_line_length_std"] = torch.std(
            line_length[mask], unbiased=False
        )
    full_names = tuple(columns)

    def pack(names: tuple[str, ...]) -> tuple[tuple[str, ...], Tensor]:
        return names, torch.stack([columns[name] for name in names])

    return {
        "summary": pack(summary_names),
        "camera": pack(camera_names),
        "camera_geometry": pack(full_names),
    }


@dataclass(frozen=True)
class FailClosedPrediction:
    residual: Tensor
    raw_coefficients: Tensor
    clipped_coefficients: Tensor
    feature_max_abs_z: float
    residual_rms: float
    fallback: bool
    fallback_reason: str | None


def coefficient_abs_limits(
    coefficients: Tensor, *, quantile: float, multiplier: float
) -> Tensor:
    matrix = _finite_matrix(coefficients, name="coefficients")
    probability = float(quantile)
    factor = float(multiplier)
    if not (0.0 < probability <= 1.0) or not math.isfinite(factor) or factor < 1.0:
        raise ValueError("quantile and multiplier are outside their safe domains")
    return torch.quantile(torch.abs(matrix), probability, dim=0).clamp_min(1e-10) * factor


def standardized_feature_limit(
    model: MultiOutputStandardizedRidge,
    fit_features: Tensor,
    *,
    quantile: float,
    multiplier: float,
) -> float:
    standardized = model.standardized(fit_features)
    row_max = torch.max(torch.abs(standardized), dim=1).values
    probability = float(quantile)
    factor = float(multiplier)
    if not (0.0 < probability <= 1.0) or not math.isfinite(factor) or factor < 1.0:
        raise ValueError("quantile and multiplier are outside their safe domains")
    return float(torch.quantile(row_max, probability) * factor)


def fail_closed_predict(
    *,
    model: MultiOutputStandardizedRidge,
    features: Tensor,
    basis: MeasurementBasis,
    coefficient_limits: Tensor,
    feature_max_abs_z_limit: float,
    residual_rms_limit: float,
) -> FailClosedPrediction:
    """Predict a bounded residual correction or return the zero fallback."""

    matrix = _finite_matrix(features, name="features")
    if matrix.shape[0] != 1:
        raise ValueError("fail_closed_predict accepts exactly one case")
    limits = _finite_vector(coefficient_limits, name="coefficient_limits")
    if limits.numel() != basis.rank or model.weights.shape[1] != basis.rank:
        raise ValueError("model, basis, and coefficient limits must share one rank")
    feature_limit = float(feature_max_abs_z_limit)
    rms_limit = float(residual_rms_limit)
    if not math.isfinite(feature_limit) or feature_limit <= 0.0:
        raise ValueError("feature_max_abs_z_limit must be positive")
    if not math.isfinite(rms_limit) or rms_limit <= 0.0:
        raise ValueError("residual_rms_limit must be positive")

    standardized = model.standardized(matrix)
    raw = model.predict(matrix)[0]
    clipped = torch.clamp(raw, min=-limits, max=limits)
    residual = basis.synthesize(clipped)
    feature_max = float(torch.max(torch.abs(standardized)))
    correction_rms = float(_rms(residual))
    reason: str | None = None
    if feature_max > feature_limit:
        reason = "feature_envelope"
    elif correction_rms > rms_limit:
        reason = "correction_rms"
    fallback = reason is not None
    return FailClosedPrediction(
        residual=torch.zeros_like(residual) if fallback else residual,
        raw_coefficients=raw,
        clipped_coefficients=clipped,
        feature_max_abs_z=feature_max,
        residual_rms=correction_rms,
        fallback=fallback,
        fallback_reason=reason,
    )
