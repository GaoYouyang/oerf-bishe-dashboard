from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.adjoint_landweber import (
    forward_project,
    geometry_normalization,
    landweber_trajectory,
    projected_bb_trajectory,
)


def problem(seed: int = 31) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    operator = rng.normal(size=(3, 4, 6))
    truth = np.abs(rng.normal(size=(2, 1, 2, 3)))
    masks = np.asarray([[1, 1, 0], [1, 1, 0]], dtype=np.float64)
    observation = forward_project(truth, operator)
    start = np.full_like(truth, 0.25)
    support = np.ones((1, 2, 3), dtype=np.float64)
    normalization = geometry_normalization(operator, masks)
    spectral = float(normalization["110"]["spectral_constant"])
    constants = np.full(len(start), spectral)
    return operator, truth, masks, observation, start, support, normalization, constants


def data_gradient(
    volume: np.ndarray,
    observation: np.ndarray,
    operator: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    residual = (forward_project(volume, operator) - observation) * masks[:, None, :, None]
    flat = np.einsum("vnp,bdvn->bdp", operator, residual, optimize=True)
    return flat.reshape(volume.shape)


def run(
    variant: str,
    checkpoints: list[int],
    *,
    seed: int = 31,
    initial: float = 1.25,
    lower: float = 0.05,
    upper: float = 1.95,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray], tuple[np.ndarray, ...]]:
    values = problem(seed)
    operator, _, masks, observation, start, support, _, spectral = values
    path, diagnostics = projected_bb_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        checkpoints,
        spectral,
        variant,
        initial,
        lower,
        upper,
    )
    return path, diagnostics, values


def test_first_step_matches_geometry_landweber() -> None:
    path, diagnostics, values = run("bb1", [1])
    operator, _, masks, observation, start, support, normalization, _ = values
    expected = landweber_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        1.25,
        [1],
        normalization,
    )[1]
    np.testing.assert_allclose(path[1], expected, atol=1e-12)
    np.testing.assert_array_equal(diagnostics["step_source_code"][0], 0)


@pytest.mark.parametrize("variant,formula_code", [("bb1", 1), ("bb2", 2)])
def test_second_step_matches_bb_formula(variant: str, formula_code: int) -> None:
    path, diagnostics, values = run(variant, [1, 2], upper=100.0)
    operator, _, masks, observation, start, _, _, _ = values
    first = path[1]
    gradient0 = data_gradient(start, observation, operator, masks)
    gradient1 = data_gradient(first, observation, operator, masks)
    step_difference = first - start
    gradient_difference = gradient1 - gradient0
    s_dot_s = np.sum(step_difference * step_difference, axis=(1, 2, 3))
    s_dot_y = np.sum(step_difference * gradient_difference, axis=(1, 2, 3))
    y_dot_y = np.sum(gradient_difference * gradient_difference, axis=(1, 2, 3))
    expected = s_dot_s / s_dot_y if variant == "bb1" else s_dot_y / y_dot_y
    np.testing.assert_allclose(diagnostics["raw_step_size"][1], expected, rtol=1e-12)
    np.testing.assert_array_equal(diagnostics["step_source_code"][1], formula_code)


def test_alternating_uses_bb1_then_bb2() -> None:
    _, diagnostics, _ = run("alternating", [4], upper=100.0)
    np.testing.assert_array_equal(diagnostics["step_source_code"][1:4, 0], [1, 2, 1])


def test_bad_curvature_falls_back_without_nan() -> None:
    operator, truth, masks, observation, _, support, _, spectral = problem(47)
    path, diagnostics = projected_bb_trajectory(
        truth,
        observation,
        operator,
        masks,
        support,
        [2],
        spectral,
        "bb1",
        1.0,
    )
    assert np.all(diagnostics["fallback_used"][1])
    np.testing.assert_array_equal(diagnostics["step_source_code"][1], 3)
    assert np.all(np.isfinite(diagnostics["step_size"]))
    np.testing.assert_allclose(path[2], truth, atol=1e-12)


def test_step_is_clipped_in_normalized_coordinates() -> None:
    _, diagnostics, _ = run("bb1", [3], upper=0.4, initial=1.25)
    assert np.all(diagnostics["normalized_step_fraction"] <= 0.4 + 1e-12)
    assert np.all(diagnostics["normalized_step_fraction"] >= 0.05 - 1e-12)
    assert np.all(diagnostics["clipped_high"][0])


def test_projection_preserves_nonnegative_hard_support() -> None:
    operator, _, masks, observation, start, support, _, spectral = problem(53)
    support = support.copy()
    support[..., 0, 0] = 0.0
    path, _ = projected_bb_trajectory(
        -start,
        observation,
        operator,
        masks,
        support,
        [4],
        spectral,
        "alternating",
        1.0,
    )
    assert np.all(path[4] >= 0.0)
    assert np.all(path[4][..., 0, 0] == 0.0)


@pytest.mark.parametrize("variant", ["BB1", "random", ""])
def test_unknown_variant_is_rejected(variant: str) -> None:
    operator, _, masks, observation, start, support, _, spectral = problem()
    with pytest.raises(ValueError, match="variant"):
        projected_bb_trajectory(
            start,
            observation,
            operator,
            masks,
            support,
            [1],
            spectral,
            variant,
            1.0,
        )


@pytest.mark.parametrize("lower,upper", [(0.0, 1.0), (2.0, 1.0), (1.0, np.nan)])
def test_invalid_step_bounds_are_rejected(lower: float, upper: float) -> None:
    operator, _, masks, observation, start, support, _, spectral = problem()
    with pytest.raises(ValueError, match="bounds"):
        projected_bb_trajectory(
            start,
            observation,
            operator,
            masks,
            support,
            [1],
            spectral,
            "bb1",
            1.0,
            lower,
            upper,
        )
