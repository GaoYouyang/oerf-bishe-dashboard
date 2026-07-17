from __future__ import annotations

import numpy as np
import pytest

from demo_t16_operator.multi_interface_metrics import multi_interface_level_set_metrics


def _coordinate_grid(size: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    axis = np.linspace(-1.0, 1.0, size)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    return xx, yy, zz, float(axis[1] - axis[0])


def _stack(*levels: np.ndarray) -> np.ndarray:
    if not levels:
        return np.empty((11, 11, 11, 0), dtype=np.float64)
    return np.stack(levels, axis=-1)


def test_double_interface_perfect_match_is_permutation_invariant() -> None:
    xx, _, zz, dx = _coordinate_grid()
    left = xx + 0.4
    upper = zz - 0.2
    truth = _stack(left, upper)
    ordered = multi_interface_level_set_metrics(
        truth,
        truth,
        spacing_xyz=(dx, dx, dx),
    )
    permuted = multi_interface_level_set_metrics(
        _stack(upper, left),
        truth,
        spacing_xyz=(dx, dx, dx),
    )

    assert ordered["matched_count"] == 2
    assert ordered["missed_truth_count"] == 0
    assert ordered["false_positive_count"] == 0
    assert ordered["penalized_surface_assd"] == 0.0
    assert ordered["penalized_surface_hd95"] == 0.0
    assert ordered["penalized_surface_f1_at_1dx"] == 1.0
    assert ordered["penalized_surface_f1_at_2dx"] == 1.0
    for key in (
        "assignment_cost_total",
        "penalized_surface_assd",
        "penalized_surface_hd95",
        "penalized_surface_f1_at_1dx",
        "penalized_surface_f1_at_2dx",
        "penalized_normal_angle_unoriented_median_degrees",
    ):
        assert permuted[key] == pytest.approx(ordered[key])
    assert {(row["predicted_index"], row["truth_index"]) for row in permuted["matches"]} == {
        (1, 0),
        (0, 1),
    }


def test_missing_one_of_two_truth_interfaces_is_counted_and_penalized() -> None:
    xx, _, zz, dx = _coordinate_grid()
    result = multi_interface_level_set_metrics(
        _stack(xx + 0.4),
        _stack(xx + 0.4, zz - 0.2),
        spacing_xyz=(dx, dx, dx),
    )

    assert result["matched_count"] == 1
    assert result["missed_truth_count"] == 1
    assert result["false_positive_count"] == 0
    assert result["cardinality_penalty"] == 0.5
    assert result["interface_detection_recall"] == 0.5
    assert result["penalized_surface_f1_at_1dx"] == 0.5
    assert result["penalized_surface_f1_at_2dx"] == 0.5
    assert result["penalized_surface_assd"] == pytest.approx(
        0.5 * result["domain_diagonal"]
    )


def test_extra_prediction_is_false_positive_and_not_best_interface_only() -> None:
    xx, _, zz, dx = _coordinate_grid()
    result = multi_interface_level_set_metrics(
        _stack(xx + 0.4, zz - 0.2),
        _stack(xx + 0.4),
        spacing_xyz=(dx, dx, dx),
    )

    assert result["matched_count"] == 1
    assert result["missed_truth_count"] == 0
    assert result["false_positive_count"] == 1
    assert result["interface_detection_precision"] == 0.5
    assert result["penalized_surface_f1_at_1dx"] == 0.5
    assert result["cardinality_penalty"] == 0.5


def test_zero_truth_and_zero_prediction_is_a_valid_clean_negative() -> None:
    empty = _stack()
    result = multi_interface_level_set_metrics(
        empty,
        empty,
        spacing_xyz=(0.2, 0.2, 0.2),
    )

    assert result["score_valid"] is True
    assert result["matched_count"] == 0
    assert result["false_positive_count"] == 0
    assert result["missed_truth_count"] == 0
    assert result["interface_detection_f1"] == 1.0
    assert result["penalized_surface_assd"] == 0.0
    assert result["penalized_surface_f1_at_1dx"] == 1.0


def test_zero_truth_and_one_surface_prediction_is_false_positive() -> None:
    xx, _, _, dx = _coordinate_grid()
    result = multi_interface_level_set_metrics(
        _stack(xx),
        _stack(),
        spacing_xyz=(dx, dx, dx),
    )

    assert result["truth_surface_count"] == 0
    assert result["predicted_surface_count"] == 1
    assert result["false_positive_count"] == 1
    assert result["cardinality_penalty"] == 1.0
    assert result["interface_detection_precision"] == 0.0
    assert result["penalized_surface_f1_at_1dx"] == 0.0
    assert result["penalized_surface_assd"] == pytest.approx(result["domain_diagonal"])


def test_normal_sign_is_reported_but_unoriented_normal_is_invariant() -> None:
    xx, _, _, dx = _coordinate_grid()
    result = multi_interface_level_set_metrics(
        _stack(-xx),
        _stack(xx),
        spacing_xyz=(dx, dx, dx),
    )

    match = result["matches"][0]
    assert match["normal_angle_median_degrees"] == pytest.approx(180.0)
    assert match["normal_angle_p95_degrees"] == pytest.approx(180.0)
    assert match["normal_angle_unoriented_median_degrees"] == pytest.approx(0.0)
    assert match["normal_angle_unoriented_p95_degrees"] == pytest.approx(0.0)


def test_degenerate_prediction_is_absent_and_degenerate_truth_invalidates_score() -> None:
    xx, _, _, dx = _coordinate_grid()
    constant = np.ones_like(xx)
    missed = multi_interface_level_set_metrics(
        _stack(constant),
        _stack(xx),
        spacing_xyz=(dx, dx, dx),
    )
    assert missed["degenerate_predicted_count"] == 1
    assert missed["predicted_surface_count"] == 0
    assert missed["missed_truth_count"] == 1
    assert missed["penalized_surface_f1_at_1dx"] == 0.0

    invalid = multi_interface_level_set_metrics(
        _stack(),
        _stack(constant),
        spacing_xyz=(dx, dx, dx),
    )
    assert invalid["score_valid"] is False
    assert invalid["status"] == "INVALID_DEGENERATE_TRUTH_SURFACE"
    assert invalid["degenerate_truth_count"] == 1
    assert invalid["penalized_surface_assd"] is None
    assert invalid["penalized_surface_f1_at_1dx"] is None


def test_rejects_more_than_two_interfaces_and_shape_mismatch() -> None:
    xx, yy, zz, dx = _coordinate_grid()
    with pytest.raises(ValueError, match="at most 2"):
        multi_interface_level_set_metrics(
            _stack(xx, yy, zz),
            _stack(),
            spacing_xyz=(dx, dx, dx),
        )
    with pytest.raises(ValueError, match="share one"):
        multi_interface_level_set_metrics(
            np.zeros((9, 9, 9, 0)),
            _stack(),
            spacing_xyz=(dx, dx, dx),
        )
