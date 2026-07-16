from __future__ import annotations

import numpy as np

from site_tools.compare_psu_b0_cached_reference import compare_cached_reference


def _report(*, residual: float, wall: float, mode: str) -> dict:
    return {
        "configuration": {"ray_store_mode": mode},
        "optimization": {
            "wall_seconds": wall,
            "logical_forward_calls": 4,
            "logical_adjoint_calls": 5,
        },
        "evaluation": {
            "direct_support_relative_measurement_l2": residual,
            "direct_vs_recurrence_residual_relative_difference": 1e-15,
        },
    }


def test_cached_reference_comparison_passes_equivalent_fast_replay() -> None:
    reference = np.arange(64, dtype=np.float64).reshape(1, 1, 4, 4, 4)
    cached = reference + 1e-14
    report = compare_cached_reference(
        reference_report=_report(
            residual=0.6,
            wall=30.0,
            mode="source_geometry_rebuilt_per_logical_call",
        ),
        cached_report=_report(
            residual=0.6 + 1e-15,
            wall=10.0,
            mode="private_compact_stencil_cache",
        ),
        reference_volume=reference,
        cached_volume=cached,
        residual_difference_maximum=1e-13,
        volume_difference_maximum=1e-12,
        optimization_speedup_minimum=1.5,
        recurrence_difference_maximum=1e-10,
    )
    assert report["status"].endswith("PASS")
    assert report["gates"]["logical_calls_match_frozen_budget"] is True
    assert report["claim_boundary"]["private_volume_voxels_exported"] is False
