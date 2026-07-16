from __future__ import annotations

import pytest

from demo_t16_operator.analyze_v5d_complexity_screening import (
    build_target_rows,
    summarize_targets,
    target_path_diagnostic,
)


def test_target_path_diagnostic_marks_bracketed_and_gap() -> None:
    result = target_path_diagnostic([0.2, 0.7, 1.4], 1.0)
    assert result["target_bracketed"] is True
    assert result["nearest_grid_gap"] == pytest.approx(0.3)


def test_target_path_diagnostic_marks_unreachable() -> None:
    result = target_path_diagnostic([0.2, 0.4, 0.6], 1.0)
    assert result["target_bracketed"] is False
    assert result["path_max"] == pytest.approx(0.6)


def test_build_and_summarize_target_rows() -> None:
    selection = [
        {
            "variant_id": "morozov_1p00",
            "method": "morozov",
            "discrepancy_target": "1.0",
            "effective_df_target": "0.5",
            "rig_id": "rig",
            "block_id": "block",
            "selected_radius": "0.08",
        }
    ]
    surface = [
        {
            "rig_id": "rig",
            "block_id": "block",
            "candidate_aperture_radius": "0.08",
            "validation_camera_index": "0",
            "mean_whitened_discrepancy": value,
            "mean_effective_df_fraction": "0.5",
        }
        for value in ("0.5", "1.2")
    ]
    rows = build_target_rows(surface, selection)
    summary = summarize_targets(rows)
    assert len(rows) == 1
    assert rows[0]["target_bracketed"] is True
    assert summary[0]["target_bracketed_count"] == 1


def test_build_target_rows_rejects_missing_selected_radius_path() -> None:
    selection = [
        {
            "variant_id": "equal_df_0p50",
            "method": "equal_df",
            "discrepancy_target": "1.0",
            "effective_df_target": "0.5",
            "rig_id": "rig",
            "block_id": "block",
            "selected_radius": "0.08",
        }
    ]
    with pytest.raises(ValueError, match="no saved surface path"):
        build_target_rows([], selection)
