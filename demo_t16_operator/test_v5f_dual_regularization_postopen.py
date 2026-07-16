from __future__ import annotations

import pytest

from demo_t16_operator.run_v5f_dual_regularization_postopen import (
    selected_radius_by_block,
    summarize_rows,
)


def test_selected_radius_by_block_rejects_duplicates() -> None:
    row = {"variant_id": "chosen", "block_id": "same"}
    with pytest.raises(ValueError, match="duplicate"):
        selected_radius_by_block([row, dict(row)], "chosen")


def test_summary_applies_route_without_truth_or_audit_routing() -> None:
    rows = [
        {
            "reconstruction_method": "gcv",
            "raw_field_error_reduction_percent": 10.0,
            "raw_audit_error_reduction_percent": -5.0,
            "radius_changed_from_metadata": True,
            "route_ge_0p0pct": True,
        },
        {
            "reconstruction_method": "gcv",
            "raw_field_error_reduction_percent": -20.0,
            "raw_audit_error_reduction_percent": 8.0,
            "radius_changed_from_metadata": True,
            "route_ge_0p0pct": False,
        },
    ]
    summary = summarize_rows(rows, [0.0])[0]
    assert summary["route_ge_0p0pct_coverage"] == pytest.approx(0.5)
    assert summary["route_ge_0p0pct_mean_selected_field_gain_percent"] == pytest.approx(5.0)
    assert summary["route_ge_0p0pct_mean_selected_audit_gain_percent"] == pytest.approx(-2.5)
