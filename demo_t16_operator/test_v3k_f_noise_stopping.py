from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.adjoint_landweber import (
    forward_project,
    geometry_normalization,
    projected_bb_trajectory,
)
from demo_t16_operator.noise_stopping import (
    active_observation_noise_scale,
    camera_ncp_ks,
    effective_operator_calls,
    first_crossing,
    gather_path,
    generator_noise_scale,
    grouped_validation_assignment,
    residual_statistics,
    white_noise_ncp_thresholds,
)


def test_active_observation_noise_scale_matches_closed_form() -> None:
    observation = np.asarray([[[[3.0, 4.0], [99.0, 99.0]]]])
    masks = np.asarray([[1.0, 0.0]])
    q = np.asarray([0.2])
    expected = 0.2 * np.sqrt((9.0 + 16.0) / 2.0) / np.sqrt(1.04)
    np.testing.assert_allclose(
        active_observation_noise_scale(observation, masks, q), [expected]
    )


def test_generator_noise_scale_uses_full_clean_rms() -> None:
    clean = np.asarray([[[[1.0, 3.0], [5.0, 7.0]]]])
    expected = 0.1 * np.sqrt(np.mean(clean * clean))
    np.testing.assert_allclose(generator_noise_scale(clean, np.asarray([0.1])), [expected])


def test_ncp_ignores_inactive_camera_and_detects_smooth_structure() -> None:
    rng = np.random.default_rng(71)
    white = rng.normal(size=(1, 8, 2, 16))
    smooth = np.broadcast_to(np.sin(np.linspace(0.0, np.pi, 16)), (1, 8, 1, 16))
    residual = white.copy()
    residual[:, :, 0:1] = smooth
    masks = np.asarray([[1.0, 0.0]])
    distance = camera_ncp_ks(residual, masks)
    assert distance.shape == (1, 2)
    assert np.isfinite(distance[0, 0])
    assert np.isnan(distance[0, 1])
    assert distance[0, 0] > 0.2


def test_residual_statistics_preserve_equal_camera_weights() -> None:
    residual = np.zeros((1, 1, 2, 3, 8), dtype=np.float64)
    residual[..., 0, :] = 1.0
    residual[..., 1, :] = 3.0
    residual[..., 2, :] = 100.0
    masks = np.asarray([[1.0, 1.0, 0.0]])
    stats = residual_statistics(residual, masks, np.asarray([2.0]))
    np.testing.assert_allclose(
        stats["discrepancy_pooled"], [[np.sqrt((0.5**2 + 1.5**2) / 2.0)]]
    )
    np.testing.assert_allclose(stats["discrepancy_camera_max"], [[1.5]])


def test_white_noise_thresholds_are_reproducible() -> None:
    left = white_noise_ncp_thresholds(4, 8, 3, samples=400, quantile=0.95, seed=19)
    right = white_noise_ncp_thresholds(4, 8, 3, samples=400, quantile=0.95, seed=19)
    assert left == right
    assert 0.0 < left["mean"] < left["maximum"] < 1.0


def test_first_crossing_and_effective_calls_charge_the_check_forward() -> None:
    condition = np.asarray(
        [
            [True, False, False],
            [True, False, False],
            [True, True, False],
            [True, True, False],
        ]
    )
    stop = first_crossing(condition, maximum_iteration=4)
    np.testing.assert_array_equal(stop, [0, 2, 4])
    a_calls, at_calls = effective_operator_calls(stop, maximum_iteration=4)
    np.testing.assert_array_equal(a_calls, [1, 3, 4])
    np.testing.assert_array_equal(at_calls, [0, 2, 4])


def test_gather_path_selects_one_iterate_per_sample() -> None:
    path = np.arange(4 * 3 * 1 * 1 * 1).reshape(4, 3, 1, 1, 1)
    gathered = gather_path(path, np.asarray([0, 2, 3]))
    np.testing.assert_array_equal(gathered[:, 0, 0, 0], [0, 7, 11])


def test_grouped_assignment_keeps_all_layouts_together() -> None:
    sources = np.repeat(np.arange(5), 4)
    seeds = np.repeat(np.arange(100, 105), 4)
    tune, manifest = grouped_validation_assignment(sources, seeds, 3, 77)
    assert len(manifest) == 5
    assert len(np.unique(sources[tune])) == 3
    for source in np.unique(sources):
        values = tune[sources == source]
        assert np.all(values == values[0])


@pytest.mark.parametrize("bad_count", [0, 5])
def test_grouped_assignment_rejects_empty_side(bad_count: int) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        grouped_validation_assignment(np.arange(5), np.arange(5), bad_count, 1)


def test_projected_bb_can_record_reused_residuals() -> None:
    rng = np.random.default_rng(91)
    operator = rng.normal(size=(2, 3, 4))
    truth = np.abs(rng.normal(size=(2, 1, 2, 2)))
    masks = np.ones((2, 2), dtype=np.float64)
    observation = forward_project(truth, operator)
    start = np.full_like(truth, 0.2)
    support = np.ones((1, 2, 2), dtype=np.float64)
    normalization = geometry_normalization(operator, masks)
    spectral = np.full(2, float(normalization["11"]["spectral_constant"]))
    path, diagnostics = projected_bb_trajectory(
        start,
        observation,
        operator,
        masks,
        support,
        [1, 2],
        spectral,
        "alternating",
        1.0,
        record_residual=True,
    )
    assert diagnostics["residual_before"].shape == (2, 2, 1, 2, 3)
    np.testing.assert_allclose(
        diagnostics["residual_before"][0], forward_project(start, operator) - observation
    )
    np.testing.assert_allclose(
        diagnostics["residual_before"][1], forward_project(path[1], operator) - observation
    )
