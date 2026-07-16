#!/usr/bin/env python3
"""Leakage-resistant building blocks for V6B few-query operator adaptation.

The high-fidelity operator is deliberately represented only by
``BudgetedForwardOracle.measure``.  The candidate operator uses a known nominal
forward/adjoint pair and ray-conditioned local voxel-kernel coefficients.  A
fresh rig may fit only a small vector of kernel-channel gates from K forward
queries; no truth adjoint or full truth operator rows are required.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Offset3D = tuple[int, int, int]


class QueryBudgetExceeded(RuntimeError):
    """Raised when a high-fidelity forward-query budget is exhausted."""


class BudgetedForwardOracle:
    """Expose only budgeted high-fidelity forward actions.

    This is an accidental-leakage boundary, not a security sandbox.  The
    evaluator should still live in a separate process for a sealed fresh run.
    """

    __slots__ = (
        "__measure_fn",
        "_budget",
        "_input_size",
        "_output_size",
        "_query_count",
    )

    def __init__(
        self,
        measure_fn: Callable[[FloatArray], ArrayLike],
        *,
        input_size: int,
        output_size: int,
        budget: int,
    ) -> None:
        if not callable(measure_fn):
            raise TypeError("measure_fn must be callable")
        if int(input_size) <= 0 or int(output_size) <= 0:
            raise ValueError("oracle dimensions must be positive")
        if int(budget) < 0:
            raise ValueError("query budget must be non-negative")
        self.__measure_fn = measure_fn
        self._input_size = int(input_size)
        self._output_size = int(output_size)
        self._budget = int(budget)
        self._query_count = 0

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def remaining_queries(self) -> int:
        return self._budget - self._query_count

    def measure(self, probe: ArrayLike) -> FloatArray:
        values = _vector(probe, self._input_size, "probe")
        if self._query_count >= self._budget:
            raise QueryBudgetExceeded(
                f"high-fidelity forward budget exhausted at {self._budget} queries"
            )
        # Attempts count against the budget so a failing renderer cannot be
        # retried silently until it reveals a convenient response.
        self._query_count += 1
        observation = _vector(
            self.__measure_fn(values.copy()), self._output_size, "observation"
        )
        return observation.copy()


def voxel_kernel_offsets(radius: int) -> tuple[Offset3D, ...]:
    """Return depth-major cubic voxel-kernel offsets."""

    value = int(radius)
    if value < 0:
        raise ValueError("voxel kernel radius must be non-negative")
    return tuple(
        (depth_offset, y_offset, x_offset)
        for depth_offset in range(-value, value + 1)
        for y_offset in range(-value, value + 1)
        for x_offset in range(-value, value + 1)
    )


def shift_volume(values: ArrayLike, offset: Offset3D) -> FloatArray:
    """Apply a zero-padded shift S; ``shift_volume(z, -offset)`` is S^T."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 3:
        raise ValueError("voxel volume must be three-dimensional")
    if len(offset) != 3:
        raise ValueError("voxel offset must have three components")
    output = np.zeros_like(source)
    output_slices: list[slice] = []
    source_slices: list[slice] = []
    for size, delta_raw in zip(source.shape, offset, strict=True):
        delta = int(delta_raw)
        output_start = max(0, -delta)
        output_stop = min(size, size - delta)
        if output_start >= output_stop:
            return output
        output_slices.append(slice(output_start, output_stop))
        source_slices.append(slice(output_start + delta, output_stop + delta))
    output[tuple(output_slices)] = source[tuple(source_slices)]
    return output


@dataclass(frozen=True)
class NominalLinearOperator:
    """A validated matrix-free nominal forward/adjoint pair."""

    forward_fn: Callable[[FloatArray], ArrayLike]
    adjoint_fn: Callable[[FloatArray], ArrayLike]
    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        if not callable(self.forward_fn) or not callable(self.adjoint_fn):
            raise TypeError("nominal forward and adjoint must be callable")
        if int(self.input_size) <= 0 or int(self.output_size) <= 0:
            raise ValueError("nominal operator dimensions must be positive")

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> "NominalLinearOperator":
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or 0 in values.shape:
            raise ValueError("nominal matrix must be non-empty and two-dimensional")
        frozen = values.copy()
        frozen.setflags(write=False)
        return cls(
            forward_fn=lambda x: frozen @ x,
            adjoint_fn=lambda z: frozen.T @ z,
            input_size=frozen.shape[1],
            output_size=frozen.shape[0],
        )

    def forward(self, field: ArrayLike) -> FloatArray:
        values = _vector(field, self.input_size, "field")
        return _vector(self.forward_fn(values), self.output_size, "nominal output")

    def adjoint(self, residual: ArrayLike) -> FloatArray:
        values = _vector(residual, self.output_size, "residual")
        return _vector(self.adjoint_fn(values), self.input_size, "nominal adjoint")


class RayKernelChannelOperator:
    """Nominal operator plus gated ray-conditioned local-kernel correction."""

    def __init__(
        self,
        nominal: NominalLinearOperator,
        *,
        input_shape: Sequence[int],
        ray_coefficients: ArrayLike,
        gates: ArrayLike,
        offsets: Sequence[Offset3D],
    ) -> None:
        shape = tuple(int(value) for value in input_shape)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("input_shape must contain three positive dimensions")
        if int(np.prod(shape)) != nominal.input_size:
            raise ValueError("input_shape does not match nominal input size")
        offset_values = tuple(tuple(int(value) for value in item) for item in offsets)
        if not offset_values or any(len(item) != 3 for item in offset_values):
            raise ValueError("offsets must contain three-component voxel offsets")
        coefficients = np.asarray(ray_coefficients, dtype=np.float64)
        gate_values = np.asarray(gates, dtype=np.float64)
        expected = (nominal.output_size, len(offset_values))
        if coefficients.shape != expected:
            raise ValueError(f"ray_coefficients must have shape {expected}")
        if gate_values.shape != (len(offset_values),):
            raise ValueError("gates must have one value per voxel-kernel channel")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(gate_values)):
            raise ValueError("operator coefficients and gates must be finite")
        self.nominal = nominal
        self.input_shape = shape
        self.offsets = offset_values
        self.ray_coefficients = coefficients.copy()
        self.gates = gate_values.copy()

    @property
    def input_size(self) -> int:
        return self.nominal.input_size

    @property
    def output_size(self) -> int:
        return self.nominal.output_size

    def forward(self, field: ArrayLike) -> FloatArray:
        values = _vector(field, self.input_size, "field")
        volume = values.reshape(self.input_shape)
        output = self.nominal.forward(values).copy()
        weights = self.ray_coefficients * self.gates[None, :]
        for channel, offset in enumerate(self.offsets):
            shifted = shift_volume(volume, offset).reshape(-1)
            output += weights[:, channel] * self.nominal.forward(shifted)
        return output

    def adjoint(self, residual: ArrayLike) -> FloatArray:
        values = _vector(residual, self.output_size, "residual")
        output = self.nominal.adjoint(values).reshape(self.input_shape).copy()
        weights = self.ray_coefficients * self.gates[None, :]
        for channel, offset in enumerate(self.offsets):
            backprojected = self.nominal.adjoint(weights[:, channel] * values)
            transpose_offset = tuple(-value for value in offset)
            output += shift_volume(
                backprojected.reshape(self.input_shape), transpose_offset
            )
        return output.reshape(-1)

    def materialize(self, *, max_elements: int = 2_000_000) -> FloatArray:
        """Materialize only for small conformance tests and toy experiments."""

        elements = self.input_size * self.output_size
        if elements > int(max_elements):
            raise MemoryError(
                f"refusing to materialize {elements} elements; matrix-free use required"
            )
        basis = np.eye(self.input_size, dtype=np.float64)
        return np.column_stack(
            [self.forward(basis[:, index]) for index in range(self.input_size)]
        )


class LowRankSecantCorrection:
    """Matrix-free regularized multisecant correction fitted from K probes."""

    def __init__(self, residual_basis: ArrayLike, analysis: ArrayLike) -> None:
        residuals = np.asarray(residual_basis, dtype=np.float64)
        analysis_values = np.asarray(analysis, dtype=np.float64)
        if residuals.ndim != 2 or analysis_values.ndim != 2:
            raise ValueError("secant residual basis and analysis must be matrices")
        if residuals.shape[1] == 0 or residuals.shape[1] != analysis_values.shape[0]:
            raise ValueError("secant factors must share a positive rank dimension")
        if not np.all(np.isfinite(residuals)) or not np.all(
            np.isfinite(analysis_values)
        ):
            raise ValueError("secant factors must be finite")
        self.residual_basis = residuals.copy()
        self.analysis = analysis_values.copy()

    @classmethod
    def fit(
        cls,
        probes: ArrayLike,
        residuals: ArrayLike,
        *,
        relative_ridge: float,
    ) -> "LowRankSecantCorrection":
        probe_values = np.asarray(probes, dtype=np.float64)
        residual_values = np.asarray(residuals, dtype=np.float64)
        if probe_values.ndim != 2 or 0 in probe_values.shape:
            raise ValueError("secant probes must be a non-empty matrix")
        if residual_values.ndim != 2 or residual_values.shape[1] != probe_values.shape[1]:
            raise ValueError("secant residuals must have one column per probe")
        if float(relative_ridge) < 0:
            raise ValueError("relative_ridge must be non-negative")
        if not np.all(np.isfinite(probe_values)) or not np.all(
            np.isfinite(residual_values)
        ):
            raise ValueError("secant calibration arrays must be finite")

        if float(relative_ridge) == 0.0:
            analysis = np.linalg.pinv(probe_values)
        else:
            gram = probe_values.T @ probe_values
            scale = max(float(np.trace(gram)) / gram.shape[0], np.finfo(float).eps)
            penalty = float(relative_ridge) * scale
            analysis = np.linalg.solve(
                gram + penalty * np.eye(gram.shape[0], dtype=np.float64),
                probe_values.T,
            )
        return cls(residual_values, analysis)

    @property
    def input_size(self) -> int:
        return self.analysis.shape[1]

    @property
    def output_size(self) -> int:
        return self.residual_basis.shape[0]

    @property
    def rank_budget(self) -> int:
        return self.residual_basis.shape[1]

    def forward(self, field: ArrayLike) -> FloatArray:
        values = _vector(field, self.input_size, "field")
        return self.residual_basis @ (self.analysis @ values)

    def adjoint(self, residual: ArrayLike) -> FloatArray:
        values = _vector(residual, self.output_size, "residual")
        return self.analysis.T @ (self.residual_basis.T @ values)

    def materialize(self, *, max_elements: int = 2_000_000) -> FloatArray:
        elements = self.input_size * self.output_size
        if elements > int(max_elements):
            raise MemoryError(
                f"refusing to materialize {elements} secant elements"
            )
        return self.residual_basis @ self.analysis

    def scaled(self, factor: float) -> "LowRankSecantCorrection":
        value = float(factor)
        if not np.isfinite(value) or value < 0:
            raise ValueError("secant scale must be finite and non-negative")
        return LowRankSecantCorrection(value * self.residual_basis, self.analysis)


class HybridGateSecantOperator:
    """Compose a ray-kernel base with a K-query residual secant update."""

    def __init__(
        self,
        base: RayKernelChannelOperator,
        correction: LowRankSecantCorrection,
    ) -> None:
        if base.input_size != correction.input_size:
            raise ValueError("hybrid input dimensions do not match")
        if base.output_size != correction.output_size:
            raise ValueError("hybrid output dimensions do not match")
        self.base = base
        self.correction = correction

    @property
    def input_size(self) -> int:
        return self.base.input_size

    @property
    def output_size(self) -> int:
        return self.base.output_size

    def forward(self, field: ArrayLike) -> FloatArray:
        return self.base.forward(field) + self.correction.forward(field)

    def adjoint(self, residual: ArrayLike) -> FloatArray:
        return self.base.adjoint(residual) + self.correction.adjoint(residual)

    def materialize(self, *, max_elements: int = 2_000_000) -> FloatArray:
        elements = self.input_size * self.output_size
        if elements > int(max_elements):
            raise MemoryError(
                f"refusing to materialize {elements} hybrid elements"
            )
        return self.base.materialize(max_elements=max_elements) + self.correction.materialize(
            max_elements=max_elements
        )


def fit_residual_secant(
    base: RayKernelChannelOperator,
    *,
    probes: ArrayLike,
    observations: ArrayLike,
    relative_ridge: float,
) -> LowRankSecantCorrection:
    """Fit a residual update without consuming additional truth queries."""

    probe_values = _probe_matrix(probes, base.input_size)
    observation_values = np.asarray(observations, dtype=np.float64)
    expected = (base.output_size, probe_values.shape[1])
    if observation_values.shape != expected:
        raise ValueError(f"observations must have shape {expected}")
    predicted = np.column_stack(
        [base.forward(probe_values[:, index]) for index in range(probe_values.shape[1])]
    )
    return LowRankSecantCorrection.fit(
        probe_values,
        observation_values - predicted,
        relative_ridge=relative_ridge,
    )


def collect_forward_observations(
    oracle: BudgetedForwardOracle, probes: ArrayLike
) -> FloatArray:
    """Consume exactly one high-fidelity query per probe column."""

    values = _probe_matrix(probes, oracle.input_size)
    return np.column_stack(
        [oracle.measure(values[:, index]) for index in range(values.shape[1])]
    )
def build_gate_design(
    nominal: NominalLinearOperator,
    *,
    input_shape: Sequence[int],
    ray_coefficients: ArrayLike,
    offsets: Sequence[Offset3D],
    probes: ArrayLike,
    observations: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Build the K-query linear system for channel gates without truth rows."""

    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 3 or int(np.prod(shape)) != nominal.input_size:
        raise ValueError("input_shape does not match nominal input size")
    offset_values = tuple(tuple(int(value) for value in item) for item in offsets)
    coefficients = np.asarray(ray_coefficients, dtype=np.float64)
    if coefficients.shape != (nominal.output_size, len(offset_values)):
        raise ValueError("ray_coefficients shape does not match nominal output/offsets")
    probe_values = _probe_matrix(probes, nominal.input_size)
    observation_values = np.asarray(observations, dtype=np.float64)
    expected_observations = (nominal.output_size, probe_values.shape[1])
    if observation_values.shape != expected_observations:
        raise ValueError(f"observations must have shape {expected_observations}")

    design_blocks: list[FloatArray] = []
    residual_blocks: list[FloatArray] = []
    for index in range(probe_values.shape[1]):
        probe = probe_values[:, index]
        volume = probe.reshape(shape)
        nominal_output = nominal.forward(probe)
        channels = []
        for channel, offset in enumerate(offset_values):
            shifted = shift_volume(volume, offset).reshape(-1)
            channels.append(coefficients[:, channel] * nominal.forward(shifted))
        design_blocks.append(np.column_stack(channels))
        residual_blocks.append(observation_values[:, index] - nominal_output)
    return np.vstack(design_blocks), np.concatenate(residual_blocks)


def fit_channel_gates(
    design: ArrayLike,
    target: ArrayLike,
    *,
    relative_ridge: float,
    prior: ArrayLike | None = None,
) -> FloatArray:
    """Fit channel gates with a trace-scaled ridge penalty around ``prior``."""

    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("design must be a non-empty matrix")
    if values.shape != (matrix.shape[0],):
        raise ValueError("target length must match design rows")
    if float(relative_ridge) < 0:
        raise ValueError("relative_ridge must be non-negative")
    center = (
        np.ones(matrix.shape[1], dtype=np.float64)
        if prior is None
        else _vector(prior, matrix.shape[1], "gate prior")
    )
    gram = matrix.T @ matrix
    scale = max(float(np.trace(gram)) / matrix.shape[1], np.finfo(float).eps)
    penalty = float(relative_ridge) * scale
    system = gram + penalty * np.eye(matrix.shape[1], dtype=np.float64)
    right_hand_side = matrix.T @ values + penalty * center
    try:
        return np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, right_hand_side, rcond=None)[0]


def ridge_effective_degrees_of_freedom(
    design: ArrayLike, *, relative_ridge: float
) -> float:
    """Return trace(H) for the same trace-scaled ridge used by gate fitting."""

    leverage = _ridge_leverage_values(design, relative_ridge=relative_ridge)
    return float(np.sum(leverage))


def ridge_residual_noise_degrees_of_freedom(
    design: ArrayLike, *, relative_ridge: float
) -> float:
    """Return ``tr((I-H)^T(I-H))`` for homoscedastic ridge noise.

    This is the exact residual-noise multiplier for a fixed linear ridge
    smoother.  It reduces to ``n-rank(G)`` for an unregularized orthogonal
    projection but, unlike ``n-tr(H)``, remains correct when ``H`` is not
    idempotent.
    """

    matrix = np.asarray(design, dtype=np.float64)
    leverage = _ridge_leverage_values(matrix, relative_ridge=relative_ridge)
    residual_degrees = (
        matrix.shape[0] - 2.0 * float(np.sum(leverage))
        + float(np.sum(leverage**2))
    )
    return float(np.clip(residual_degrees, 0.0, matrix.shape[0]))


def ridge_residual_noise_energy_diagonal(
    design: ArrayLike,
    noise_variances: ArrayLike,
    *,
    relative_ridge: float,
) -> float:
    """Return ``tr((I-H) Sigma (I-H)^T)`` for diagonal ``Sigma``.

    The gate design stacks one output block per probe.  This helper keeps the
    probe-dependent noise variances instead of replacing them by a global
    homoscedastic average.  Correlated experimental noise still requires
    whitening or an explicit covariance operator.
    """

    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("design must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("design must contain only finite values")
    variances = np.asarray(noise_variances, dtype=np.float64)
    if variances.shape != (matrix.shape[0],):
        raise ValueError("noise_variances must have one entry per design row")
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError("noise_variances must be finite and non-negative")
    ridge = float(relative_ridge)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("relative_ridge must be finite and non-negative")

    gram = matrix.T @ matrix
    scale = max(float(np.trace(gram)) / matrix.shape[1], np.finfo(float).eps)
    penalty = ridge * scale
    system = gram + penalty * np.eye(matrix.shape[1], dtype=np.float64)
    try:
        analysis = np.linalg.solve(system, matrix.T)
    except np.linalg.LinAlgError:
        analysis = np.linalg.lstsq(system, matrix.T, rcond=None)[0]
    leverage = np.einsum("ij,ji->i", matrix, analysis, optimize=True)
    hat_squared_diagonal = np.einsum(
        "ij,jk,ik->i", matrix, analysis @ analysis.T, matrix, optimize=True
    )
    residual_squared_diagonal = np.maximum(
        1.0 - 2.0 * leverage + hat_squared_diagonal, 0.0
    )
    return float(np.dot(variances, residual_squared_diagonal))


def _ridge_leverage_values(
    design: ArrayLike, *, relative_ridge: float
) -> FloatArray:
    """Return the nonzero eigenvalues of the ridge hat matrix."""

    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("design must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("design must contain only finite values")
    ridge = float(relative_ridge)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("relative_ridge must be finite and non-negative")
    gram = matrix.T @ matrix
    scale = max(float(np.trace(gram)) / matrix.shape[1], np.finfo(float).eps)
    penalty = ridge * scale
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    if penalty == 0.0:
        tolerance = (
            np.finfo(float).eps
            * max(matrix.shape)
            * max(float(np.max(eigenvalues)), 1.0)
        )
        return np.where(eigenvalues > tolerance, 1.0, 0.0)
    return eigenvalues / (eigenvalues + penalty)


def positive_part_residual_signal_fraction(
    residuals: ArrayLike,
    *,
    expected_noise_energy: float,
    residual_noise_degrees_of_freedom: float | None = None,
) -> float:
    """Estimate residual signal fraction after subtracting a noise floor.

    ``residual_noise_degrees_of_freedom`` should be
    ``tr((I-H)^T(I-H))`` for a fixed homoscedastic linear smoother.  Real BOS
    use must replace the expected energy with an estimate from independent
    flow-off repeats and validate heteroscedastic whitening.
    """

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("residuals must be a non-empty matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("residuals must be finite")
    noise_energy = float(expected_noise_energy)
    if not np.isfinite(noise_energy) or noise_energy < 0:
        raise ValueError("expected_noise_energy must be finite and non-negative")
    if residual_noise_degrees_of_freedom is None:
        residual_degrees = float(values.size)
    else:
        residual_degrees = float(residual_noise_degrees_of_freedom)
        if (
            not np.isfinite(residual_degrees)
            or residual_degrees < 0
            or residual_degrees > values.size
        ):
            raise ValueError(
                "residual noise degrees of freedom must lie in [0, residual size]"
            )
    observed_energy = float(np.vdot(values, values))
    if observed_energy <= np.finfo(float).tiny:
        return 0.0
    residual_noise_energy = noise_energy * residual_degrees / values.size
    return float(np.clip(1.0 - residual_noise_energy / observed_energy, 0.0, 1.0))


def sha256_arrays(*arrays: ArrayLike) -> str:
    """Hash array dtype, shape and contiguous bytes for pre-score freezing."""

    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _vector(values: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (int(size),):
        raise ValueError(f"{name} must have shape ({int(size)},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _probe_matrix(values: ArrayLike, input_size: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != int(input_size) or array.shape[1] == 0:
        raise ValueError(f"probes must have shape ({int(input_size)}, K) with K > 0")
    if not np.all(np.isfinite(array)):
        raise ValueError("probes must contain only finite values")
    return array
