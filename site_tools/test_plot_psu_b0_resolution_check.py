from __future__ import annotations

from pathlib import Path

import numpy as np

from site_tools.plot_psu_b0_resolution_check import render_figure


def _summary(grid: int, residuals: list[float]) -> dict:
    return {
        "optimization": {
            "history": [
                {
                    "iteration": index,
                    "relative_measurement_l2": value,
                }
                for index, value in enumerate(residuals)
            ]
        },
        "configuration": {"grid_shape_zyx": [grid, grid, grid]},
    }


def test_resolution_figure_renders_all_formats(tmp_path: Path) -> None:
    comparison = {
        "status": "SAME_CALL_RESOLUTION_SIGNAL_PASS_NO_FIELD_TRUTH",
        "aggregate": {
            "absolute_drop": 0.16,
            "relative_improvement_fraction": 0.2,
        },
        "per_view": [
            {
                "view_id_zero_based": index,
                "residual_16_cubed": 0.8 + 0.01 * index,
                "residual_32_cubed": 0.6 + 0.01 * index,
            }
            for index in range(9)
        ],
        "resources": {
            "pair_wall_seconds_16_cubed": 50.0,
            "pair_wall_seconds_32_cubed": 51.0,
            "server_required_for_32_cubed_gate": False,
        },
    }
    low = np.zeros((16, 16, 16), dtype=np.float64)
    high = np.zeros((32, 32, 32), dtype=np.float64)
    low[5:10, 5:10, 5:10] = 0.1
    high[10:20, 10:20, 10:20] = -0.08
    manifest = render_figure(
        _summary(16, [1.0, 0.86, 0.82, 0.8, 0.79]),
        _summary(32, [1.0, 0.73, 0.67, 0.65, 0.63]),
        comparison,
        low,
        high,
        tmp_path / "resolution",
    )
    assert manifest["status"].endswith("NO_FIELD_TRUTH")
    for suffix in (".png", ".pdf", ".svg"):
        path = tmp_path / f"resolution{suffix}"
        assert path.exists()
        assert path.stat().st_size > 1000
