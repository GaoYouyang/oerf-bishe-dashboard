"""Tiny dense/callable tests for the local signed-factor PDHG oracle."""

from __future__ import annotations

import numpy as np
import pytest

from .psu_b0_signed_factor_majorizer import (
    PDHGState,
    SignedFactorSystem,
    assert_exact_svd_safety,
    build_majorizer_setup,
    dominance_gap,
    exact_svd_squared_norm,
    ones_pass,
    pdhg_objective,
    run_pdhg,
)


def _system() -> SignedFactorSystem:
    # Four declared primal columns include one deliberately empty column.
    E = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        dtype=np.float64,
    )
    G = np.array(
        [[1.0, -0.5, 0.0, 0.2], [-0.3, 0.7, 1.0, -0.4], [0.2, 0.0, -0.8, 0.6]],
        dtype=np.float64,
    )
    P0 = np.array([[1.0, 0.0, 0.5], [0.2, 1.0, 0.0]], dtype=np.float64)
    W0 = np.array([[1.0, -0.5], [0.0, 0.0], [-0.4, 0.8]], dtype=np.float64)
    P1 = np.array([[0.5, 0.4, 0.1]], dtype=np.float64)
    W1 = np.zeros((2, 1), dtype=np.float64)  # whole covariance block is inactive
    D = np.array(
        [
            [1.0, -0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # partial zero row: site sigma remains shared
            [-0.3, 0.4, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # whole TV site is inactive
        ],
        dtype=np.float64,
    )
    return SignedFactorSystem(E, (P0, P1), G, (W0, W1), D)


def test_factor_signs_and_dense_dominance_are_explicit() -> None:
    system = _system()
    setup = build_majorizer_setup(system, eta=0.75)
    assert np.all(system.E >= 0.0)
    assert all(np.all(P >= 0.0) for P in system.P_blocks)
    assert any(np.any(G < 0.0) for G in (system.G, *system.W_blocks, system.D_plus))
    for A, M in zip(setup.A_blocks, setup.M_blocks):
        assert dominance_gap(A, M) >= -1e-12
    assert dominance_gap(setup.D, setup.N) >= -1e-12


def test_callable_ones_pass_matches_dense_and_calls_each_direction_once() -> None:
    calls = {"forward": 0, "transpose": 0}

    def forward(value: np.ndarray) -> np.ndarray:
        calls["forward"] += 1
        return np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float64) @ value

    def transpose(value: np.ndarray) -> np.ndarray:
        calls["transpose"] += 1
        return np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float64).T @ value

    rows, cols = ones_pass(forward, transpose, rows=2, cols=2)
    np.testing.assert_allclose(rows, [3.0, 3.0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(cols, [2.0, 4.0], atol=0.0, rtol=0.0)
    assert calls == {"forward": 1, "transpose": 1}

    dense = build_majorizer_setup(_system(), eta=0.75)
    callable_setup = build_majorizer_setup(_system(), eta=0.75, use_callable_ones_pass=True)
    for left, right in zip(dense.omega_data_rows, callable_setup.omega_data_rows):
        np.testing.assert_allclose(left, right, atol=1e-15, rtol=1e-15)
    for left, right in zip(dense.omega_data_cols, callable_setup.omega_data_cols):
        np.testing.assert_allclose(left, right, atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(dense.omega_tv_rows, callable_setup.omega_tv_rows, atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(dense.omega_tv_cols, callable_setup.omega_tv_cols, atol=1e-15, rtol=1e-15)


def test_zero_rows_sites_and_primal_columns_are_eliminated_without_infinity() -> None:
    setup = build_majorizer_setup(_system(), eta=0.75)
    np.testing.assert_array_equal(setup.active_primal, [0, 1, 2])
    np.testing.assert_array_equal(setup.active_data_rows[0], [True, False, True])
    np.testing.assert_array_equal(setup.active_data_rows[1], [False, False])
    np.testing.assert_array_equal(setup.active_tv_sites, [True, False])
    assert setup.active_tv_site_count == 1
    assert np.all(np.isfinite(setup.tau))
    assert all(value is None for value in setup.sigma_data[1:])
    assert setup.sigma_tv_sites[0] > 0.0 and setup.sigma_tv_sites[1] == 0.0
    assert setup.sigma_vector.size == 2 + 3  # two data rows plus one 3-vector site

    empty = SignedFactorSystem(
        np.zeros((2, 1)),
        (np.ones((1, 1)),),
        np.zeros((1, 2)),
        (np.ones((1, 1)),),
        np.zeros((3, 2)),
    )
    with pytest.raises(ValueError, match="no active primal"):
        build_majorizer_setup(empty)


def test_tiny_nonzero_coupling_is_retained_and_tolerance_is_rejected() -> None:
    tiny = 1e-12
    system = SignedFactorSystem(
        np.eye(2),
        (np.eye(2),),
        np.eye(2),
        (np.diag([1.0, tiny]),),
        np.zeros((3, 2)),
    )
    setup = build_majorizer_setup(system)
    np.testing.assert_array_equal(setup.active_data_rows[0], [True, True])
    np.testing.assert_array_equal(setup.active_primal, [0, 1])
    assert setup.omega_data_rows[0][1] == pytest.approx(tiny)
    with pytest.raises(ValueError, match="exactly zero"):
        build_majorizer_setup(system, zero_tolerance=1e-9)


def test_exact_svd_metric_is_safe_and_enlarged_steps_fail() -> None:
    setup = build_majorizer_setup(_system(), eta=0.75)
    observed = assert_exact_svd_safety(setup)
    assert observed <= setup.eta**2 + 1e-12
    assert exact_svd_squared_norm(setup.K_active, setup.sigma_vector, setup.tau) == pytest.approx(observed)
    with pytest.raises(ValueError, match="exact SVD safety"):
        assert_exact_svd_safety(setup, tau=setup.tau * 4.0)


def test_negative_P_is_rejected_before_setup() -> None:
    system = _system()
    bad_P = list(system.P_blocks)
    bad_P[0] = bad_P[0].copy()
    bad_P[0][0, 0] = -1.0
    with pytest.raises(ValueError, match=r"P_blocks\[0\].*nonnegative"):
        SignedFactorSystem(system.E, tuple(bad_P), system.G, system.W_blocks, system.D_plus)


def test_omitting_abs_W_cannot_pass_dominance() -> None:
    system = _system()
    block = 0
    wrong = system.W_blocks[block] @ system.P_blocks[block] @ np.abs(system.G) @ system.E
    with pytest.raises(ValueError, match="majorizer"):
        # The signed W contains cancellations and a negative wrong bound.
        from .psu_b0_signed_factor_majorizer import assert_dominance

        assert_dominance(system.signed_data(block, np.eye(system.full_primal_count)), wrong)


def test_one_and_six_step_dense_callable_recurrence_match() -> None:
    dense = build_majorizer_setup(_system(), eta=0.75)
    callable_setup = build_majorizer_setup(_system(), eta=0.75, use_callable_ones_pass=True)
    targets = (np.array([0.25, -0.4]), np.zeros(0, dtype=np.float64))
    dense_states = run_pdhg(dense, targets, iterations=6, regularization_weight=0.2, penalty="huber")
    callable_states = run_pdhg(callable_setup, targets, iterations=6, regularization_weight=0.2, penalty="huber")
    assert len(dense_states) == len(callable_states) == 6
    for left, right in zip(dense_states, callable_states):
        np.testing.assert_allclose(left.x, right.x, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(left.x_bar, right.x_bar, atol=1e-12, rtol=1e-12)
        for left_dual, right_dual in zip(left.data_dual, right.data_dual):
            np.testing.assert_allclose(left_dual, right_dual, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(left.tv_dual, right.tv_dual, atol=1e-12, rtol=1e-12)
    assert pdhg_objective(dense, dense_states[0], targets, regularization_weight=0.2) >= 0.0


def test_huber_objective_uses_both_piecewise_branches_and_differs_from_tv() -> None:
    setup = build_majorizer_setup(_system(), eta=0.75)
    weight = 0.2
    delta = 0.4
    for x, expected_branch in (
        (np.array([0.1, 1.0, 0.0]), "quadratic"),
        (np.array([1.0, 3.0, -2.0]), "linear"),
    ):
        state = PDHGState(
            x=x,
            x_bar=np.zeros(3),
            data_dual=(np.zeros(2), np.zeros(0)),
            tv_dual=np.zeros((1, 3)),
        )
        targets = (setup.data_forward(state.x)[0], np.zeros(0))
        magnitudes = np.linalg.norm(setup.tv_forward(state.x), axis=1)
        assert bool(np.all(magnitudes <= delta)) == (expected_branch == "quadratic")
        expected_huber = weight * np.where(
            magnitudes <= delta,
            0.5 * magnitudes**2 / delta,
            magnitudes - 0.5 * delta,
        ).sum()

        observed_huber = pdhg_objective(
            setup,
            state,
            targets,
            regularization_weight=weight,
            penalty="huber",
            huber_delta=delta,
        )
        observed_tv = pdhg_objective(
            setup, state, targets, regularization_weight=weight, penalty="tv"
        )

        assert observed_huber == pytest.approx(expected_huber)
        assert observed_huber != pytest.approx(observed_tv)


def test_huber_recurrence_has_declared_numeric_shrink_before_projection() -> None:
    setup = build_majorizer_setup(_system(), eta=0.75)
    targets = (np.zeros(2), np.zeros(0))
    initial = PDHGState(
        x=np.zeros(3),
        x_bar=np.array([0.1, 0.2, -0.1]),
        data_dual=(np.zeros(2), np.zeros(0)),
        tv_dual=np.zeros((1, 3)),
    )
    weight = 10.0  # keep this example inside the dual ball
    delta = 0.4
    state = run_pdhg(
        setup,
        targets,
        iterations=1,
        regularization_weight=weight,
        penalty="huber",
        huber_delta=delta,
        initial=initial,
    )[0]
    sigma = setup.sigma_tv_sites[setup.active_tv_sites, None]
    expected = sigma * setup.tv_forward(initial.x_bar)
    expected /= 1.0 + sigma * delta / weight
    np.testing.assert_allclose(state.tv_dual, expected, atol=1e-15, rtol=1e-15)


def test_site_major_tv_forward_and_adjoint_are_a_cross_interface_pair() -> None:
    setup = build_majorizer_setup(_system(), eta=0.75)
    x = np.array([0.3, -0.7, 1.1])
    q = np.array([[2.0, -3.0, 5.0]])
    full_x = np.zeros(setup.system.full_primal_count)
    full_x[setup.active_primal] = x
    expected = setup.system.signed_tv(full_x)[setup.active_tv_sites]

    np.testing.assert_allclose(setup.tv_forward(x), expected, atol=1e-15, rtol=1e-15)
    assert float(np.sum(setup.tv_forward(x) * q)) == pytest.approx(
        float(np.dot(x, setup.tv_adjoint(q)))
    )


def test_lambda_zero_data_only_is_finite_and_uses_empty_tv_dual() -> None:
    setup = build_majorizer_setup(_system(), eta=0.75)
    targets = (np.array([0.1, 0.2]), np.zeros(0, dtype=np.float64))
    states = run_pdhg(setup, targets, iterations=1, regularization_weight=0.0)
    state = states[0]
    assert state.tv_dual.shape == (1, 3)
    assert np.all(state.tv_dual == 0.0)
    assert np.all(np.isfinite(state.x))
