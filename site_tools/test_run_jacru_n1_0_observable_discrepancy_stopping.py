from __future__ import annotations

import copy

import pytest

from site_tools import run_jacru_n1_0_observable_discrepancy_stopping as n10


def _row(iteration: int, residual: float) -> dict[str, str]:
    return {
        "projection_iterations": str(iteration),
        "measured_reprojection_relative_l2": str(residual),
        "system_residual_fraction": str(residual / 2.0),
        "field_relative_l2": str(10.0 + iteration),
        "h1_seminorm_relative_error": str(20.0 + iteration),
        "clean_reprojection_relative_l2": str(30.0 + iteration),
    }


def test_first_crossing_selects_earliest_observable_iterate() -> None:
    trajectory = [_row(0, 1.0), _row(1, 0.4), _row(2, 0.1)]
    selected, crossed = n10._first_crossing(
        trajectory,
        observable=lambda row: float(row["measured_reprojection_relative_l2"]),
        threshold=0.5,
    )
    assert crossed is True
    assert selected is trajectory[1]


def test_first_crossing_does_not_read_truth_columns() -> None:
    trajectory = [_row(0, 1.0), _row(1, 0.4), _row(2, 0.1)]
    corrupted = copy.deepcopy(trajectory)
    for index, row in enumerate(corrupted):
        row["field_relative_l2"] = str(-1e9 * (index + 1))
        row["clean_reprojection_relative_l2"] = str(1e9 * (index + 1))
    selector = lambda row: float(row["system_residual_fraction"])
    original, _ = n10._first_crossing(
        trajectory, observable=selector, threshold=0.25
    )
    changed, _ = n10._first_crossing(
        corrupted, observable=selector, threshold=0.25
    )
    assert original is trajectory[1]
    assert changed is corrupted[1]


def test_first_crossing_returns_explicit_miss() -> None:
    selected, crossed = n10._first_crossing(
        [_row(0, 1.0), _row(1, 0.8)],
        observable=lambda row: float(row["measured_reprojection_relative_l2"]),
        threshold=0.1,
    )
    assert selected is None
    assert crossed is False


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_first_crossing_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        n10._first_crossing(
            [_row(0, 1.0)],
            observable=lambda row: float(row["measured_reprojection_relative_l2"]),
            threshold=threshold,
        )


def test_group_trajectory_requires_complete_ordered_path() -> None:
    rows = [
        {
            "method": "jacru_m2",
            "model_seed": "17",
            "split": "development",
            "case_id": "case",
            "projection_variant": "oracle",
            "projection_iterations": str(iteration),
            "damping_absolute": "0.0",
            "projection_target_mode": "affine_observation",
            "preconditioner_kind": "dense_exact_camera_block_jacobi_oracle",
        }
        for iteration in (2, 0, 1)
    ]
    grouped = n10._group_trajectory_rows(
        rows,
        expected_iterations=[0, 1, 2],
        expected_variant="oracle",
    )
    assert [
        int(row["projection_iterations"]) for row in next(iter(grouped.values()))
    ] == [0, 1, 2]


def test_candidate_specs_keep_fixed_k_as_comparator() -> None:
    config = {
        "observable_stopping_families": {
            "simulator_noise_floor_multiple": {"multipliers": [1.0]},
            "base_anchor_residual_multiple": {"multipliers": [2.0]},
            "initial_system_residual_fraction": {"maximum_fractions": [0.5]},
        },
        "fixed_iteration_comparators": [0, 1],
    }
    specs = n10._candidate_specs(config, noise_floor=0.02)
    assert [value["candidate_id"] for value in specs] == [
        "noise_floor_x1",
        "base_residual_x2",
        "system_fraction_0.5",
        "fixed_k0",
        "fixed_k1",
    ]
    assert [value["comparator_only"] for value in specs[-2:]] == [True, True]


def test_recover_best_h1_from_gain_identity() -> None:
    assert n10._best_h1_from_row(
        {
            "h1_gain_to_best_matched_classical": "0.25",
            "h1_seminorm_relative_error": "0.75",
        }
    ) == pytest.approx(1.0)
