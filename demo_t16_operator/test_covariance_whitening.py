from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.covariance_decoupled_complexity import (
    build_covariance_decoupled_surface,
    covariance_ridge_diagnostics,
)
from demo_t16_operator.covariance_whitening import (
    camera_noise_covariance,
    covariance_scaled_ridge_fit,
    covariance_whitened_support_system,
    covariance_whitened_view_rms,
)
from demo_t16_operator.decoupled_complexity import select_radius_from_surface
from demo_t16_operator.independent_reaction_bost import correlated_camera_noise
from demo_t16_operator.nested_crossview import scaled_ridge_fit


def toy_operator(view_count: int = 3) -> np.ndarray:
    rows = np.zeros((1, view_count, 2, 2), dtype=float)
    for view in range(view_count):
        rows[0, view] = np.array(
            [[1.0 + 0.1 * view, 0.2], [0.1, 0.8 + 0.05 * view]]
        )
    return rows


def test_camera_covariance_matches_generator_monte_carlo() -> None:
    clean = np.array([[[0.4, -0.7]], [[1.2, 0.2]]], dtype=float)
    camera_std = np.array([0.3])
    expected = camera_noise_covariance(
        clean,
        camera_std,
        correlation_fraction=0.2,
        signal_fraction=0.15,
    )[0]
    rng = np.random.default_rng(19)
    draws = np.stack(
        [
            correlated_camera_noise(
                clean,
                camera_std,
                rng,
                correlation_fraction=0.2,
                signal_fraction=0.15,
            ).reshape(-1)
            for _ in range(12000)
        ]
    )
    empirical = np.cov(draws, rowvar=False, ddof=0)
    assert np.allclose(empirical, expected, rtol=0.10, atol=0.006)


def test_covariance_system_has_identity_residual_covariance() -> None:
    operator = toy_operator()
    observation = np.ones(operator.shape[:3])
    covariance = np.stack([np.array([[2.0, 0.4], [0.4, 1.0]])] * 3)
    matrix, _, _ = covariance_whitened_support_system(
        operator,
        observation,
        covariance,
        [0, 2],
        np.ones(2, dtype=bool),
    )
    assert matrix.shape == (4, 2)
    for view in (0, 2):
        cholesky = np.linalg.cholesky(covariance[view])
        whitened = np.linalg.solve(cholesky, covariance[view])
        whitened = np.linalg.solve(cholesky, whitened.T).T
        assert np.allclose(whitened, np.eye(2))


def test_covariance_fit_matches_diagonal_fit() -> None:
    operator = toy_operator()
    truth = np.array([0.6, -0.2])
    observation = np.einsum("dvnp,p->dvn", operator, truth)
    sigma = np.array([0.2, 0.3, 0.4])
    covariance = np.stack([np.eye(2) * value**2 for value in sigma])
    support = np.ones(2, dtype=bool)
    diagonal_fit, diagonal_lambda = scaled_ridge_fit(
        operator, observation, sigma, [0, 1], support, 0.1
    )
    full_fit, full_lambda = covariance_scaled_ridge_fit(
        operator, observation, covariance, [0, 1], support, 0.1
    )
    assert full_lambda == pytest.approx(diagonal_lambda)
    assert np.allclose(full_fit.field, diagonal_fit.field)
    assert full_fit.whitened_sse == pytest.approx(diagonal_fit.whitened_sse)


def test_covariance_diagnostics_expose_residual_df_correction() -> None:
    operator = toy_operator()
    truth = np.array([0.4, -0.1])
    observation = np.einsum("dvnp,p->dvn", operator, truth)
    covariance = np.stack([np.eye(2) * 0.04] * 3)
    diagnostic = covariance_ridge_diagnostics(
        operator,
        observation,
        covariance,
        [0, 1, 2],
        np.ones(2, dtype=bool),
        0.1,
    )
    assert diagnostic.residual_noise_degrees_of_freedom > 0.0
    assert diagnostic.squared_hat_trace <= diagnostic.effective_degrees_of_freedom
    assert diagnostic.degrees_of_freedom_corrected_discrepancy >= diagnostic.whitened_discrepancy


def test_corrected_discrepancy_is_unit_calibrated_for_noise_only() -> None:
    operator = toy_operator()
    covariance = np.stack(
        [np.array([[0.04, 0.012], [0.012, 0.09]])] * 3
    )
    cholesky = [np.linalg.cholesky(item) for item in covariance]
    rng = np.random.default_rng(311)
    raw: list[float] = []
    corrected: list[float] = []
    for _ in range(1200):
        observation = np.zeros(operator.shape[:3], dtype=float)
        for view in range(3):
            observation[:, view, :] = (
                cholesky[view] @ rng.normal(size=2)
            ).reshape(1, 2)
        diagnostic = covariance_ridge_diagnostics(
            operator,
            observation,
            covariance,
            [0, 1, 2],
            np.ones(2, dtype=bool),
            0.1,
        )
        raw.append(diagnostic.whitened_discrepancy)
        corrected.append(
            diagnostic.degrees_of_freedom_corrected_discrepancy
        )
    assert np.mean(raw) < 1.0
    assert np.mean(corrected) == pytest.approx(1.0, abs=0.06)


def test_covariance_surface_keeps_outer_camera_out_of_complexity_choice() -> None:
    bank = np.stack([toy_operator(), toy_operator() * 1.03])
    truth = np.array([0.5, -0.25])
    observation = np.einsum("dvnp,p->dvn", bank[0], truth)
    covariance = np.stack([np.eye(2) * 0.04] * 3)
    support = np.ones(2, dtype=bool)
    kwargs = dict(
        operator_bank=bank,
        radii=[0.0, 0.1],
        observations=[observation],
        covariances=[covariance],
        inner_views=[0, 1, 2],
        support=support,
        kappas=[0.01, 0.1, 1.0],
    )
    baseline = build_covariance_decoupled_surface(**kwargs)
    poisoned = observation.copy()
    poisoned[:, 0, :] += 1000.0
    changed = build_covariance_decoupled_surface(
        **{**kwargs, "observations": [poisoned]}
    )
    for radius_index in range(2):
        baseline_path = baseline.paths[radius_index][0]
        changed_path = changed.paths[radius_index][0]
        assert [point.mean_generalized_cross_validation for point in baseline_path] == pytest.approx(
            [point.mean_generalized_cross_validation for point in changed_path]
        )


def test_covariance_surface_supports_corrected_morozov_selection() -> None:
    bank = np.stack([toy_operator(), toy_operator() * 1.08])
    truth = np.array([0.5, -0.25])
    observation = np.einsum("dvnp,p->dvn", bank[0], truth)
    covariance = np.stack([np.eye(2) * 0.04] * 3)
    surface = build_covariance_decoupled_surface(
        bank,
        [0.0, 0.1],
        [observation],
        [covariance],
        [0, 1, 2],
        np.ones(2, dtype=bool),
        [0.01, 0.1, 1.0],
    )
    selection = select_radius_from_surface(
        "df_corrected_morozov", surface, discrepancy_target=1.0
    )
    assert selection.method == "df_corrected_morozov"
    assert np.isfinite(selection.selected.mean_validation_mse)


def test_covariance_view_rms_uses_requested_camera_only() -> None:
    operator = toy_operator()
    truth = np.array([0.5, -0.2])
    observation = np.einsum("dvnp,p->dvn", operator, truth)
    observation[:, 2, :] += 10.0
    covariance = np.stack([np.eye(2)] * 3)
    assert covariance_whitened_view_rms(
        operator, truth, observation, covariance, [0, 1]
    ) == pytest.approx(0.0)
    assert covariance_whitened_view_rms(
        operator, truth, observation, covariance, [2]
    ) > 9.0
