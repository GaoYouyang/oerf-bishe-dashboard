"""CPU tests for the standalone signed factor-majorized PDHG prototype."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from demo_t16_operator.block_diagonal_pdhg import (
    SignedFactor,
    apply_block_adjoint,
    apply_block_operator,
    build_signed_factor_majorizer,
    estimate_matrix_free_norm,
    pdhg_step,
)


def _factor(
    matrix: np.ndarray,
    *,
    primal_block: int = 0,
    dual_block: int = 0,
    coefficient: float = 1.0,
    norm_bound: float | None = None,
) -> SignedFactor:
    matrix = np.asarray(matrix, dtype=np.float64)
    return SignedFactor(
        primal_block=primal_block,
        dual_block=dual_block,
        coefficient=coefficient,
        apply=lambda value: matrix @ value,
        adjoint=lambda value: matrix.T @ value,
        input_shape=(matrix.shape[1],),
        output_shape=(matrix.shape[0],),
        norm_bound=norm_bound,
        name=f"factor_{primal_block}_{dual_block}",
    )


def test_matrix_free_norm_is_cpu_only_and_has_declared_margin() -> None:
    matrix = np.array([[1.0, 2.0], [-2.0, 1.0]])
    result = estimate_matrix_free_norm(
        lambda value: matrix @ value,
        lambda value: matrix.T @ value,
        input_shape=(2,),
        power_iterations=8,
        safety_factor=1.2,
        seed=17,
    )

    assert result.estimate == pytest.approx(np.sqrt(5.0), rel=1e-12)
    assert result.majorizer == pytest.approx(1.2 * np.sqrt(5.0), rel=1e-12)
    assert result.certified is False


def test_signed_factor_majorizer_uses_absolute_coefficients_and_checks_safety() -> None:
    identity = np.eye(2)
    factors = (
        _factor(identity, coefficient=1.0, norm_bound=1.0),
        _factor(identity, coefficient=-2.0, norm_bound=1.0),
        _factor(
            identity,
            primal_block=1,
            coefficient=-0.5,
            norm_bound=1.0,
        ),
        _factor(
            identity,
            primal_block=0,
            dual_block=1,
            coefficient=1.5,
            norm_bound=1.0,
        ),
        _factor(
            identity,
            primal_block=1,
            dual_block=1,
            coefficient=0.75,
            norm_bound=1.0,
        ),
    )
    parameters = build_signed_factor_majorizer(
        factors,
        primal_block_count=2,
        dual_block_count=2,
        safety_factor=1.2,
        diagonal=True,
    )

    np.testing.assert_allclose(
        parameters.factor_majorizer,
        np.array([[3.0, 0.5], [1.5, 0.75]]),
    )
    assert parameters.mode == "diagonal"
    assert np.all(parameters.primal_steps > 0.0)
    assert np.all(parameters.dual_steps > 0.0)
    assert parameters.safety_factor * parameters.majorizer_norm < 1.0
    assert parameters.uses_estimated_factor_bounds is False


def test_declared_block_bounds_control_the_actual_dense_normalized_operator() -> None:
    data = np.array([[1.0, 2.0], [-2.0, 1.0]])
    gradient = np.array([[1.0, -1.0], [0.0, 2.0]])
    factors = (
        _factor(
            data,
            dual_block=0,
            norm_bound=float(np.linalg.norm(data, ord=2)),
        ),
        _factor(
            gradient,
            dual_block=1,
            coefficient=-1.0,
            norm_bound=float(np.linalg.norm(gradient, ord=2)),
        ),
    )
    parameters = build_signed_factor_majorizer(
        factors,
        primal_block_count=1,
        dual_block_count=2,
        safety_factor=1.2,
    )

    dense_operator = np.vstack([data, -gradient])
    dual_metric = np.diag(
        np.repeat(np.sqrt(parameters.dual_steps), repeats=2)
    )
    primal_metric = np.sqrt(parameters.primal_steps[0]) * np.eye(2)
    normalized_norm = np.linalg.norm(
        dual_metric @ dense_operator @ primal_metric,
        ord=2,
    )
    assert normalized_norm < 1.0


def test_empty_primal_or_dual_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="every primal and dual block"):
        build_signed_factor_majorizer(
            (_factor(np.eye(2), norm_bound=1.0),),
            primal_block_count=2,
            dual_block_count=1,
        )


def test_uncertified_estimates_require_two_explicit_opt_ins() -> None:
    factor = _factor(np.eye(2))
    with pytest.raises(ValueError, match="declared norm_bound"):
        build_signed_factor_majorizer(
            (factor,),
            primal_block_count=1,
            dual_block_count=1,
        )

    parameters = build_signed_factor_majorizer(
        (factor,),
        primal_block_count=1,
        dual_block_count=1,
        allow_uncertified_estimates=True,
    )
    assert parameters.uses_estimated_factor_bounds is True
    with pytest.raises(ValueError, match="diagnostic-only"):
        pdhg_step(
            (np.ones(2),),
            (np.ones(2),),
            (factor,),
            parameters,
        )
    result = pdhg_step(
        (np.ones(2),),
        (np.ones(2),),
        (factor,),
        parameters,
        allow_uncertified_parameters=True,
    )
    assert result.primal[0].shape == (2,)


def test_signed_matrix_free_operator_and_adjoint_keep_signs() -> None:
    factors = (
        _factor(np.array([[2.0, 0.0], [0.0, 1.0]]), coefficient=-1.0),
        _factor(np.eye(2), coefficient=0.5),
    )
    primal = (np.array([1.0, -2.0]),)
    dual = apply_block_operator(factors, primal)
    expected_dual = -np.array([2.0, -2.0]) + 0.5 * np.array([1.0, -2.0])
    np.testing.assert_allclose(dual[0], expected_dual)
    adjoint = apply_block_adjoint(factors, (np.array([3.0, -4.0]),))
    expected_adjoint = -np.array([6.0, -4.0]) + 0.5 * np.array([3.0, -4.0])
    np.testing.assert_allclose(adjoint[0], expected_adjoint)


def test_one_pdhg_step_matches_manual_signed_update() -> None:
    matrix = np.array([[1.0, 2.0], [-1.0, 1.0]])
    factors = (_factor(matrix, coefficient=-1.0, norm_bound=3.0),)
    parameters = build_signed_factor_majorizer(
        factors,
        primal_block_count=1,
        dual_block_count=1,
        safety_factor=1.5,
    )
    primal = (np.array([0.4, -0.2]),)
    dual = (np.array([0.3, -0.1]),)
    result = pdhg_step(primal, dual, factors, parameters, theta=1.0)

    tau = parameters.primal_steps[0]
    sigma = parameters.dual_steps[0]
    signed_matrix = -matrix
    expected_dual = dual[0] + sigma * signed_matrix @ primal[0]
    expected_primal = primal[0] - tau * signed_matrix.T @ expected_dual
    np.testing.assert_allclose(result.dual[0], expected_dual)
    np.testing.assert_allclose(result.primal[0], expected_primal)
    np.testing.assert_allclose(
        result.extrapolated_primal[0],
        expected_primal + (expected_primal - primal[0]),
    )


@pytest.mark.parametrize("bad_step", [0.0, -1.0, np.nan])
def test_nonpositive_or_nonfinite_steps_are_rejected(bad_step: float) -> None:
    factors = (_factor(np.eye(1), norm_bound=1.0),)
    parameters = build_signed_factor_majorizer(
        factors,
        primal_block_count=1,
        dual_block_count=1,
    )
    parameters = replace(parameters, primal_steps=np.array([bad_step]))
    with pytest.raises(ValueError, match="steps"):
        pdhg_step((np.ones(1),), (np.ones(1),), factors, parameters)
