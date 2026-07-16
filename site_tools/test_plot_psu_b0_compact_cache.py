from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_compact_cache import render_figure


def test_compact_cache_figure_writes_three_formats(tmp_path: Path) -> None:
    benchmark = {
        "status": "COMPACT_CACHE_NUMERICAL_AND_SPEED_GATE_PASS",
        "configuration": {"sample_count": 16},
        "cache": {
            "total_cache_bytes": 5_016_803_984,
            "build_wall_seconds": 14.94,
        },
        "performance": {
            "direct_pair_seconds": [37.2, 38.6],
            "cached_pair_seconds": [17.0, 17.1],
            "median_pair_speedup": 2.22,
        },
        "numerical_equivalence": {
            "forward_relative_difference": 0.0,
            "adjoint_relative_difference": 0.0,
        },
    }
    replay = {
        "status": "CACHED_32_CUBED_REFERENCE_REPLAY_PASS",
        "performance": {
            "reference_optimization_wall_seconds": 218.0,
            "cached_optimization_wall_seconds": 75.0,
            "optimization_speedup": 2.91,
        },
        "numerical_equivalence": {
            "volume_relative_difference": 1.17e-16,
            "support_residual_absolute_difference": 0.0,
            "cached_support_relative_measurement_l2": 0.627132,
        },
    }
    manifest = render_figure(benchmark, replay, tmp_path / "cache_gate")
    assert manifest["status"].startswith("FIGURE_COMPLETE")
    for suffix in (".png", ".pdf", ".svg"):
        assert (tmp_path / f"cache_gate{suffix}").is_file()
