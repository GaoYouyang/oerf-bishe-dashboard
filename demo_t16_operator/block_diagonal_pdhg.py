"""CPU-only infrastructure for signed, factor-majorized block PDHG.

This module is intentionally independent of the formal T16/MPS runner.  It
implements a small matrix-free contract for a block operator

    K = sum_f coefficient_f * K_f,

where each factor maps one primal block to one dual block.  The signed
coefficients are retained by the operator application, while the
block-majorizer uses ``abs(coefficient_f) * norm_majorizer_f``.  This is the
usual conservative construction for a block norm matrix: if every supplied
factor majorizer is a valid upper bound, the normalized block matrix gives a
checkable sufficient step condition.

``estimate_matrix_free_norm`` uses power iteration followed by a declared
margin.  It is a reproducible numerical estimate, not a mathematical proof
of an operator norm bound.  Callers that need a certified bound can pass one
through ``SignedFactor(norm_bound=...)`` and should retain that certification
outside this prototype.

The public update is one separable PDHG step for ``f(x) + g(Kx)``.  Proximal
callbacks are optional and default to the identity, making the module useful
for testing operator and preconditioner plumbing without claiming convergence
or algorithmic success.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


Array = np.ndarray
BlockTuple = tuple[Array, ...]
LinearMap = Callable[[Array], Array]
Proximal = Callable[[Array, float], Array]


def _shape(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive dimensions")
    return result


def _finite_scalar(value: float, *, name: str, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _array(value: Array, *, name: str, expected_shape: tuple[int, ...] | None = None) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if expected_shape is not None and result.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {result.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class MatrixFreeNormEstimate:
    """Power-iteration norm estimate and its explicitly declared margin.

    ``estimate`` is the largest observed Rayleigh quotient magnitude and
    ``majorizer`` is ``estimate * safety_factor``.  The latter is suitable as
    a conservative *declared* factor majorizer for this prototype.  Since
    power iteration can underestimate a norm, ``certified`` is always false.
    """

    estimate: float
    majorizer: float
    iterations: int
    safety_factor: float
    certified: bool = False


def estimate_matrix_free_norm(
    apply: LinearMap,
    adjoint: LinearMap,
    *,
    input_shape: Sequence[int],
    power_iterations: int = 24,
    safety_factor: float = 1.1,
    seed: int = 0,
) -> MatrixFreeNormEstimate:
    """Estimate ``||A||`` using only ``A`` and its matrix-free adjoint.

    The functions are called on CPU ``float64`` arrays.  At least two power
    iterations and a margin strictly larger than one are required.  The
    adjoint is checked by shape and finite-value validation, but no dense
    matrix is formed and no formal upper-bound certificate is inferred.
    """

    shape = _shape(input_shape, name="input_shape")
    iterations = int(power_iterations)
    if iterations < 2:
        raise ValueError("power_iterations must be at least two")
    margin = _finite_scalar(safety_factor, name="safety_factor", positive=True)
    if margin <= 1.0:
        raise ValueError("safety_factor must be greater than one")
    rng = np.random.default_rng(int(seed))
    current = rng.standard_normal(shape)
    current /= np.linalg.norm(current)
    rayleigh_values: list[float] = []
    output_shape: tuple[int, ...] | None = None

    for _ in range(iterations):
        projected = _array(apply(current), name="apply(current)")
        if output_shape is None:
            output_shape = projected.shape
        elif projected.shape != output_shape:
            raise ValueError("matrix-free apply changed output shape")
        pulled_back = _array(
            adjoint(projected),
            name="adjoint(apply(current))",
            expected_shape=shape,
        )
        rayleigh = float(np.dot(current.ravel(), pulled_back.ravel()))
        if rayleigh < -1e-10:
            raise ValueError("apply and adjoint do not form a positive normal map")
        rayleigh_values.append(math.sqrt(max(rayleigh, 0.0)))
        pulled_norm = float(np.linalg.norm(pulled_back))
        if pulled_norm == 0.0:
            return MatrixFreeNormEstimate(0.0, 0.0, iterations, margin)
        current = pulled_back / pulled_norm

    estimate = max(rayleigh_values)
    return MatrixFreeNormEstimate(
        estimate=estimate,
        majorizer=estimate * margin,
        iterations=iterations,
        safety_factor=margin,
    )


@dataclass(frozen=True)
class SignedFactor:
    """One signed matrix-free factor between a primal and dual block.

    ``coefficient`` may be positive or negative.  ``norm_bound`` is optional;
    when omitted, :func:`build_signed_factor_majorizer` estimates it from the
    supplied ``apply`` and ``adjoint`` functions.  ``input_shape`` and
    ``output_shape`` are part of the contract so a one-step update can catch
    accidental block mixing without materializing a global matrix.
    """

    primal_block: int
    dual_block: int
    coefficient: float
    apply: LinearMap
    adjoint: LinearMap
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    norm_bound: float | None = None
    name: str = "factor"

    def __post_init__(self) -> None:
        if int(self.primal_block) < 0 or int(self.dual_block) < 0:
            raise ValueError("block indices must be nonnegative")
        _finite_scalar(self.coefficient, name="coefficient")
        _shape(self.input_shape, name="input_shape")
        _shape(self.output_shape, name="output_shape")
        if not callable(self.apply) or not callable(self.adjoint):
            raise TypeError("apply and adjoint must be callable")
        if self.norm_bound is not None:
            _finite_scalar(self.norm_bound, name="norm_bound")
            if float(self.norm_bound) < 0.0:
                raise ValueError("norm_bound must be nonnegative")


@dataclass(frozen=True)
class PreconditionerParameters:
    """Reusable positive block steps and the checked majorizer diagnostics.

    There is one scalar primal step and one scalar dual step per block.  If
    each block has one coordinate this is a diagonal preconditioner; larger
    blocks use the same data as a block-diagonal preconditioner.
    """

    primal_steps: Array
    dual_steps: Array
    factor_majorizer: Array
    normalized_majorizer: Array
    majorizer_norm: float
    safety_factor: float
    factor_norm_majorizers: tuple[float, ...]
    uses_estimated_factor_bounds: bool
    mode: Literal["block-diagonal", "diagonal"]

    def validate(self) -> None:
        """Reject nonpositive steps or a failed normalized-majorizer check."""

        _array(self.primal_steps, name="primal_steps")
        _array(self.dual_steps, name="dual_steps")
        if np.any(self.primal_steps <= 0.0) or np.any(self.dual_steps <= 0.0):
            raise ValueError("preconditioner steps must be strictly positive")
        _array(self.factor_majorizer, name="factor_majorizer")
        if np.any(self.factor_majorizer < 0.0):
            raise ValueError("factor_majorizer must be nonnegative")
        _array(self.normalized_majorizer, name="normalized_majorizer")
        margin = _finite_scalar(
            self.safety_factor,
            name="safety_factor",
            positive=True,
        )
        if margin <= 1.0:
            raise ValueError("safety_factor must be greater than one")
        norm = _finite_scalar(self.majorizer_norm, name="majorizer_norm")
        if norm < 0.0 or margin * norm >= 1.0 - 1e-12:
            raise ValueError(
                "normalized factor majorizer fails the strict PDHG safety condition"
            )

    def scale_primal(self, blocks: Sequence[Array]) -> BlockTuple:
        """Apply the scalar primal block metric to a tuple of arrays."""

        if len(blocks) != len(self.primal_steps):
            raise ValueError("primal block count does not match parameters")
        return tuple(
            _array(block, name=f"primal block {index}") * self.primal_steps[index]
            for index, block in enumerate(blocks)
        )

    def scale_dual(self, blocks: Sequence[Array]) -> BlockTuple:
        """Apply the scalar dual block metric to a tuple of arrays."""

        if len(blocks) != len(self.dual_steps):
            raise ValueError("dual block count does not match parameters")
        return tuple(
            _array(block, name=f"dual block {index}") * self.dual_steps[index]
            for index, block in enumerate(blocks)
        )


def _factor_majorizers(
    factors: Sequence[SignedFactor],
    *,
    power_iterations: int,
    safety_factor: float,
    seed: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for index, factor in enumerate(factors):
        if factor.norm_bound is None:
            estimate = estimate_matrix_free_norm(
                factor.apply,
                factor.adjoint,
                input_shape=factor.input_shape,
                power_iterations=power_iterations,
                safety_factor=safety_factor,
                seed=seed + index,
            )
            values.append(estimate.majorizer)
        else:
            values.append(float(factor.norm_bound))
    return tuple(values)


def build_signed_factor_majorizer(
    factors: Sequence[SignedFactor],
    *,
    primal_block_count: int,
    dual_block_count: int,
    safety_factor: float = 1.1,
    power_iterations: int = 24,
    seed: int = 0,
    diagonal: bool = False,
    allow_uncertified_estimates: bool = False,
) -> PreconditionerParameters:
    """Build positive block steps from signed, matrix-free factor bounds.

    For ``C[i, j] = sum_f abs(c_f) * m_f`` over factors from primal block
    ``j`` to dual block ``i``, the returned steps are

    ``tau[j] = 1 / (safety_factor**2 * sum_i C[i,j])`` and
    ``sigma[i] = 1 / (safety_factor**2 * sum_j C[i,j])``.

    By default every factor must provide a declared ``norm_bound``.  Missing
    bounds can be estimated only with the explicit
    ``allow_uncertified_estimates=True`` diagnostic opt-in; the returned
    parameters then retain ``uses_estimated_factor_bounds=True`` and cannot
    enter :func:`pdhg_step` without a second explicit opt-in.

    The normalized nonnegative matrix
    ``diag(sqrt(sigma)) @ C @ diag(sqrt(tau))`` is checked explicitly.  Its
    norm, multiplied by ``safety_factor``, must be strictly below one.  The
    squared margin leaves room below the boundary instead of relying on an
    equality case.  Empty
    rows or columns are rejected because they would have an unbounded default
    step.  This construction verifies the declared majorizer contract; it
    does not establish convergence for a particular objective or data set.
    """

    primal_count = int(primal_block_count)
    dual_count = int(dual_block_count)
    if primal_count < 1 or dual_count < 1:
        raise ValueError("block counts must be positive")
    if not factors:
        raise ValueError("at least one signed factor is required")
    margin = _finite_scalar(safety_factor, name="safety_factor", positive=True)
    if margin <= 1.0:
        raise ValueError("safety_factor must be greater than one")
    factor_tuple = tuple(factors)
    for factor in factor_tuple:
        if factor.primal_block >= primal_count or factor.dual_block >= dual_count:
            raise ValueError("factor block index exceeds declared block count")
    uses_estimated_bounds = any(
        factor.norm_bound is None for factor in factor_tuple
    )
    if uses_estimated_bounds and not allow_uncertified_estimates:
        raise ValueError(
            "every factor needs a declared norm_bound; matrix-free estimates "
            "require allow_uncertified_estimates=True"
        )
    factor_bounds = _factor_majorizers(
        factor_tuple,
        power_iterations=int(power_iterations),
        safety_factor=margin,
        seed=int(seed),
    )
    majorizer = np.zeros((dual_count, primal_count), dtype=np.float64)
    for factor, bound in zip(factor_tuple, factor_bounds):
        majorizer[factor.dual_block, factor.primal_block] += (
            abs(float(factor.coefficient)) * bound
        )
    row_sums = majorizer.sum(axis=1)
    column_sums = majorizer.sum(axis=0)
    if np.any(row_sums <= 0.0) or np.any(column_sums <= 0.0):
        raise ValueError("every primal and dual block needs a positive majorizer")
    squared_margin = margin * margin
    primal_steps = 1.0 / (squared_margin * column_sums)
    dual_steps = 1.0 / (squared_margin * row_sums)
    normalized = np.sqrt(dual_steps)[:, None] * majorizer * np.sqrt(
        primal_steps
    )[None, :]
    majorizer_norm = float(np.linalg.norm(normalized, ord=2))
    parameters = PreconditionerParameters(
        primal_steps=primal_steps,
        dual_steps=dual_steps,
        factor_majorizer=majorizer,
        normalized_majorizer=normalized,
        majorizer_norm=majorizer_norm,
        safety_factor=margin,
        factor_norm_majorizers=factor_bounds,
        uses_estimated_factor_bounds=uses_estimated_bounds,
        mode="diagonal" if diagonal else "block-diagonal",
    )
    parameters.validate()
    return parameters


def _as_blocks(blocks: Sequence[Array], *, name: str) -> BlockTuple:
    if not isinstance(blocks, (tuple, list)) or not blocks:
        raise ValueError(f"{name} must be a non-empty tuple or list of arrays")
    return tuple(_array(block, name=f"{name}[{index}]") for index, block in enumerate(blocks))


def apply_block_operator(
    factors: Sequence[SignedFactor],
    primal_blocks: Sequence[Array],
) -> BlockTuple:
    """Apply the signed factor sum ``K`` without building a global matrix."""

    x = _as_blocks(primal_blocks, name="primal_blocks")
    if not factors:
        raise ValueError("at least one signed factor is required")
    output: list[Array | None] = []
    for factor in factors:
        if factor.primal_block >= len(x):
            raise ValueError("factor primal block is missing")
        source = _array(
            x[factor.primal_block],
            name=f"primal block {factor.primal_block}",
            expected_shape=factor.input_shape,
        )
        value = _array(
            factor.apply(source),
            name=f"{factor.name}.apply",
            expected_shape=factor.output_shape,
        )
        contribution = float(factor.coefficient) * value
        while len(output) <= factor.dual_block:
            output.append(None)
        if output[factor.dual_block] is None:
            output[factor.dual_block] = np.zeros_like(value)
        elif output[factor.dual_block].shape != value.shape:
            raise ValueError("factors targeting one dual block have different shapes")
        output[factor.dual_block] += contribution
    if any(value is None for value in output):
        raise ValueError("dual block indices must be contiguous from zero")
    return tuple(value for value in output if value is not None)


def apply_block_adjoint(
    factors: Sequence[SignedFactor],
    dual_blocks: Sequence[Array],
) -> BlockTuple:
    """Apply the exact signed factor transpose ``K.T`` matrix-free."""

    y = _as_blocks(dual_blocks, name="dual_blocks")
    if not factors:
        raise ValueError("at least one signed factor is required")
    output: list[Array | None] = []
    for factor in factors:
        if factor.dual_block >= len(y):
            raise ValueError("factor dual block is missing")
        source = _array(
            y[factor.dual_block],
            name=f"dual block {factor.dual_block}",
            expected_shape=factor.output_shape,
        )
        value = _array(
            factor.adjoint(source),
            name=f"{factor.name}.adjoint",
            expected_shape=factor.input_shape,
        )
        contribution = float(factor.coefficient) * value
        while len(output) <= factor.primal_block:
            output.append(None)
        if output[factor.primal_block] is None:
            output[factor.primal_block] = np.zeros_like(value)
        elif output[factor.primal_block].shape != value.shape:
            raise ValueError("factors targeting one primal block have different shapes")
        output[factor.primal_block] += contribution
    if any(value is None for value in output):
        raise ValueError("primal block indices must be contiguous from zero")
    return tuple(value for value in output if value is not None)


def _identity_prox(value: Array, _: float) -> Array:
    return value


@dataclass(frozen=True)
class PDHGStep:
    """One immutable primal-dual update and its extrapolated primal state."""

    primal: BlockTuple
    dual: BlockTuple
    extrapolated_primal: BlockTuple


def pdhg_step(
    primal_blocks: Sequence[Array],
    dual_blocks: Sequence[Array],
    factors: Sequence[SignedFactor],
    parameters: PreconditionerParameters,
    *,
    extrapolated_primal: Sequence[Array] | None = None,
    primal_prox: Proximal | None = None,
    dual_prox_conjugate: Proximal | None = None,
    theta: float = 1.0,
    allow_uncertified_parameters: bool = False,
) -> PDHGStep:
    """Perform one separable, preconditioned PDHG update on CPU arrays.

    The update is

    ``y+ = prox_(Sigma g*) (y + Sigma K x_bar)`` and
    ``x+ = prox_(Tau f) (x - Tau K.T y+)``,

    followed by ``x_bar+ = x+ + theta * (x+ - x)``.  The factor signs are
    included in both operator applications.  This is a one-step numerical
    primitive only: it does not run the formal runner, select frozen
    configurations, or claim convergence.
    """

    parameters.validate()
    if (
        parameters.uses_estimated_factor_bounds
        and not allow_uncertified_parameters
    ):
        raise ValueError(
            "estimated factor bounds are diagnostic-only; pass "
            "allow_uncertified_parameters=True to opt in explicitly"
        )
    x = _as_blocks(primal_blocks, name="primal_blocks")
    y = _as_blocks(dual_blocks, name="dual_blocks")
    if len(x) != len(parameters.primal_steps) or len(y) != len(parameters.dual_steps):
        raise ValueError("state block counts do not match preconditioner parameters")
    if extrapolated_primal is None:
        x_bar = x
    else:
        x_bar = _as_blocks(extrapolated_primal, name="extrapolated_primal")
        if len(x_bar) != len(x):
            raise ValueError("extrapolated_primal block count does not match")
    extrapolation = _finite_scalar(theta, name="theta")
    if not 0.0 <= extrapolation <= 1.0:
        raise ValueError("theta must lie in [0,1]")
    dual_prox = dual_prox_conjugate or _identity_prox
    primal_prox_fn = primal_prox or _identity_prox
    kx = apply_block_operator(factors, x_bar)
    if len(kx) != len(y):
        raise ValueError("operator dual blocks do not match state")
    next_dual = tuple(
        _array(
            dual_prox(
                y[index] + parameters.dual_steps[index] * kx[index],
                float(parameters.dual_steps[index]),
            ),
            name=f"dual update {index}",
            expected_shape=y[index].shape,
        )
        for index in range(len(y))
    )
    kty = apply_block_adjoint(factors, next_dual)
    if len(kty) != len(x):
        raise ValueError("operator primal blocks do not match state")
    next_primal = tuple(
        _array(
            primal_prox_fn(
                x[index] - parameters.primal_steps[index] * kty[index],
                float(parameters.primal_steps[index]),
            ),
            name=f"primal update {index}",
            expected_shape=x[index].shape,
        )
        for index in range(len(x))
    )
    next_extrapolated = tuple(
        next_primal[index]
        + extrapolation * (next_primal[index] - x[index])
        for index in range(len(x))
    )
    return PDHGStep(
        primal=next_primal,
        dual=next_dual,
        extrapolated_primal=next_extrapolated,
    )


__all__ = [
    "MatrixFreeNormEstimate",
    "PDHGStep",
    "PreconditionerParameters",
    "SignedFactor",
    "apply_block_adjoint",
    "apply_block_operator",
    "build_signed_factor_majorizer",
    "estimate_matrix_free_norm",
    "pdhg_step",
]
