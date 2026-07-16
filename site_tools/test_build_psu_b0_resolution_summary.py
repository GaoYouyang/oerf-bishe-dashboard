from __future__ import annotations

import copy

import pytest

from site_tools.build_psu_b0_resolution_summary import build_summary


def _summary(grid: int, residual: float) -> dict:
    rows = [
        {
            "view_id_zero_based": index,
            "ray_count": 10,
            "relative_measurement_l2": residual + 0.001 * index,
        }
        for index in range(9)
    ]
    return {
        "status": "REAL_SUPPORT_B0_CGLS_FIXED_BUDGET_COMPLETE_NO_FIELD_TRUTH",
        "configuration": {
            "grid_shape_zyx": [grid, grid, grid],
            "finite_aperture_sample_count": 16,
            "dtype": "float64",
            "device": "cpu",
        },
        "selection": {
            "total_ray_count": 90,
            "view_rows": [
                {
                    "view_id_zero_based": index,
                    "active_ray_count": 10,
                    "selected_ray_count": 10,
                }
                for index in range(9)
            ],
        },
        "optimization": {
            "fixed_iteration_budget": 4,
            "logical_forward_calls": 4,
            "logical_adjoint_calls": 5,
        },
        "evaluation": {
            "direct_support_relative_measurement_l2": residual,
            "per_view_support_relative_measurement_l2": rows,
        },
        "resource_gate": {
            "full_forward_adjoint_pair_wall_seconds": 1.0,
            "process_max_rss_bytes": 100,
            "server_required_for_current_16_cubed_gate": False,
        },
        "gates": {
            "development_rotation_40_not_accessed": True,
            "final_audit_not_accessed": True,
        },
    }


def test_resolution_summary_requires_same_calls_and_passes_signal() -> None:
    result = build_summary(
        _summary(16, 0.8),
        _summary(32, 0.6),
        practical_signal_minimum=0.02,
    )
    assert result["status"].startswith("SAME_CALL_RESOLUTION_SIGNAL_PASS")
    assert result["aggregate"]["absolute_drop"] == pytest.approx(0.2)
    assert result["aggregate"]["views_improved_count"] == 9
    assert result["decision"]["authoritative_low_resolution_reference"] == "32_cubed"


def test_resolution_summary_rejects_call_mismatch() -> None:
    high = copy.deepcopy(_summary(32, 0.6))
    high["optimization"]["logical_forward_calls"] = 5
    with pytest.raises(ValueError, match="not comparable"):
        build_summary(
            _summary(16, 0.8),
            high,
            practical_signal_minimum=0.02,
        )
