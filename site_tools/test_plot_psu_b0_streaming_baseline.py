from __future__ import annotations

from pathlib import Path

import numpy as np

from site_tools.plot_psu_b0_streaming_baseline import render_figure


def test_streaming_baseline_figure_renders_three_formats(tmp_path: Path) -> None:
    summary = {
        "status": "REAL_SUPPORT_B0_CGLS_FIXED_BUDGET_COMPLETE_NO_FIELD_TRUTH",
        "selection": {"total_ray_count": 100},
        "optimization": {
            "history": [
                {
                    "iteration": index,
                    "relative_measurement_l2": 1.0 / (index + 1),
                    "relative_normal_residual_l2": 1.0 / (index + 1) ** 2,
                }
                for index in range(5)
            ],
            "logical_calls": [
                {
                    "operation": "forward",
                    "wall_seconds": 2.0 + index,
                }
                for index in range(4)
            ]
            + [
                {
                    "operation": "adjoint",
                    "wall_seconds": 3.0 + index,
                }
                for index in range(5)
            ],
        },
        "interface_profile": {
            "logical_calls": [
                {"operation": "forward", "wall_seconds": 5.0},
                {"operation": "adjoint", "wall_seconds": 6.0},
            ]
        },
        "evaluation": {
            "per_view_support_relative_measurement_l2": [
                {
                    "view_id_zero_based": index,
                    "relative_measurement_l2": 0.7 + 0.02 * index,
                }
                for index in range(9)
            ],
            "evaluation_logical_calls": [
                {"operation": "forward", "wall_seconds": 4.5}
            ],
        },
        "resource_gate": {
            "full_forward_adjoint_pair_wall_seconds": 11.0,
            "process_max_rss_bytes": 4 * 1024**3,
            "server_required_for_current_16_cubed_gate": False,
        },
    }
    z, y, x = np.mgrid[-1:1:16j, -1:1:16j, -1:1:16j]
    volume = np.exp(-5 * (x * x + y * y + z * z)) * np.sin(4 * x)
    manifest = render_figure(summary, volume, tmp_path / "figure")
    assert manifest["status"].endswith("NO_FIELD_TRUTH")
    for suffix in (".png", ".pdf", ".svg"):
        path = tmp_path / f"figure{suffix}"
        assert path.exists()
        assert path.stat().st_size > 1000
