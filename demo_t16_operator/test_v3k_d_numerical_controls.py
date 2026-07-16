from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.adjoint_landweber import (
    forward_project,
    geometry_normalization,
    global_landweber_trajectory,
    landweber_trajectory,
    quadratic_steepest_descent_trajectory,
)
from demo_t16_operator.run_v3k_d_strong_numerical_controls import lookup_prediction


def problem(seed: int = 7) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    operator = rng.normal(size=(3, 4, 6))
    truth = np.abs(rng.normal(size=(2, 1, 2, 3)))
    masks = np.asarray([[1, 1, 0], [1, 1, 0]], dtype=np.float64)
    observation = forward_project(truth, operator)
    start = np.full_like(truth, 0.2)
    support = np.ones((1, 2, 3), dtype=np.float64)
    normalization = geometry_normalization(operator, masks)
    key = "110"
    spectral = float(normalization[key]["spectral_constant"])
    return operator, truth, masks, observation, start, support, normalization, spectral


def masked_objective(
    volume: np.ndarray,
    observation: np.ndarray,
    operator: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    residual = (forward_project(volume, operator) - observation) * masks[:, None, :, None]
    return 0.5 * np.sum(residual * residual, axis=(1, 2, 3))


def test_global_step_matches_geometry_normalized_step_for_one_geometry() -> None:
    operator, _, masks, observation, start, support, normalization, spectral = problem()
    checkpoints = [1, 2, 4]
    local = landweber_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        1.25,
        checkpoints,
        normalization,
        method="standard",
    )
    global_path = global_landweber_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        1.25,
        checkpoints,
        spectral,
    )
    for iteration in checkpoints:
        np.testing.assert_allclose(global_path[iteration], local[iteration], atol=1e-12)


def test_quadratic_step_matches_closed_form_before_projection() -> None:
    operator, _, masks, observation, start, support, _, spectral = problem(11)
    residual = (observation - forward_project(start, operator)) * masks[:, None, :, None]
    gradient = np.einsum(
        "vnp,bdvn->bdp", operator, residual, optimize=True
    ).reshape(start.shape)
    projected_gradient = forward_project(gradient, operator) * masks[:, None, :, None]
    expected = np.sum(gradient * gradient, axis=(1, 2, 3)) / np.sum(
        projected_gradient * projected_gradient, axis=(1, 2, 3)
    )
    _, diagnostics = quadratic_steepest_descent_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        [1],
        np.full(len(start), spectral),
    )
    np.testing.assert_allclose(diagnostics["step_size"][0], expected, rtol=1e-12)


def test_quadratic_step_audits_projected_objective_and_shapes() -> None:
    operator, _, masks, observation, start, support, _, spectral = problem(13)
    trajectory, diagnostics = quadratic_steepest_descent_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        [1, 2, 4],
        np.full(len(start), spectral),
    )
    assert set(trajectory) == {1, 2, 4}
    assert diagnostics["step_size"].shape == (4, len(start))
    assert diagnostics["normalized_step_fraction"].shape == (4, len(start))
    assert np.all(diagnostics["step_size"] >= 0.0)
    assert np.all(
        diagnostics["objective_after"]
        <= diagnostics["objective_before"] + 1e-10
    )
    np.testing.assert_allclose(
        diagnostics["objective_after"][-1],
        masked_objective(trajectory[4], observation, operator, masks),
        atol=1e-10,
    )


def test_quadratic_step_keeps_nonnegative_hard_support() -> None:
    operator, _, masks, observation, start, support, _, spectral = problem(17)
    support = support.copy()
    support[..., 0, 0] = 0.0
    trajectory, _ = quadratic_steepest_descent_trajectory(
        -start,
        observation,
        operator,
        masks,
        support,
        [3],
        np.full(len(start), spectral),
    )
    assert np.all(trajectory[3] >= 0.0)
    assert np.all(trajectory[3][..., 0, 0] == 0.0)


@pytest.mark.parametrize("beta", [0.0, 2.0, -0.1])
def test_global_step_rejects_unstable_fraction(beta: float) -> None:
    operator, _, masks, observation, start, support, _, spectral = problem()
    with pytest.raises(ValueError, match="must lie in"):
        global_landweber_trajectory(
            start, observation, operator, masks, support, beta, [1], spectral
        )


def test_quadratic_step_rejects_bad_spectral_shape() -> None:
    operator, _, masks, observation, start, support, _, _ = problem()
    with pytest.raises(ValueError, match="one positive value per sample"):
        quadratic_steepest_descent_trajectory(
            start, observation, operator, masks, support, [1], np.asarray([1.0])
        )


def test_lookup_skips_a_regime_absent_from_a_development_split() -> None:
    operator, _, masks, observation, start, support, normalization, _ = problem()
    table = {
        "low": {"iterations": 1, "step_fraction": 1.0},
        "high": {"iterations": 2, "step_fraction": 1.25},
    }
    prediction = lookup_prediction(
        start,
        observation,
        masks,
        support,
        operator,
        normalization,
        np.asarray(["high", "high"]),
        table,
    )
    expected = landweber_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        1.25,
        [2],
        normalization,
        method="standard",
    )[2]
    np.testing.assert_allclose(prediction, expected)
