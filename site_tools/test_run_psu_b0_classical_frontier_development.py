from __future__ import annotations

from site_tools.run_psu_b0_classical_frontier_development import (
    _frontier_comparison,
    select_regularization,
)


def test_select_regularization_uses_declared_tie_breaks() -> None:
    screen = [
        {
            "regularizer": "h1",
            "regularization_lambda": 0.1,
            "mean_combined_loss": 0.4,
            "mean_field_relative_l2": 0.5,
        },
        {
            "regularizer": "h1",
            "regularization_lambda": 0.03,
            "mean_combined_loss": 0.4,
            "mean_field_relative_l2": 0.5,
        },
        {
            "regularizer": "identity",
            "regularization_lambda": 10.0,
            "mean_combined_loss": 0.3,
            "mean_field_relative_l2": 0.4,
        },
    ]
    selected = select_regularization(screen, regularizer="h1")
    assert selected["regularization_lambda"] == 0.03


def test_frontier_comparison_uses_lowest_classical_error() -> None:
    aggregates = [
        {
            "split": "fresh",
            "method": "sobolev_selected",
            "field_relative_l2_mean": 0.50,
        },
        {
            "split": "fresh",
            "method": "cgls_3",
            "field_relative_l2_mean": 0.45,
        },
        {
            "split": "fresh",
            "method": "raw_seed_1",
            "field_relative_l2_mean": 0.44,
        },
        {
            "split": "fresh",
            "method": "gated_seed_1",
            "field_relative_l2_mean": 0.46,
        },
    ]
    result = _frontier_comparison(
        aggregates,
        classical_methods={"sobolev_selected", "cgls_3"},
    )
    assert result[0]["best_classical_method"] == "cgls_3"
    by_method = {
        row["method"]: row for row in result[0]["learned_comparison"]
    }
    assert by_method["raw_seed_1"]["beats_best_classical_mean_field_error"]
    assert not by_method["gated_seed_1"]["beats_best_classical_mean_field_error"]
