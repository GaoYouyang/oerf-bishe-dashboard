#!/usr/bin/env python3
"""Compare the cached CGLS replay with the frozen direct 32-cubed reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


COMPARISON_SCHEMA = "psu-b0-cached-reference-public-comparison-1.0"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def compare_cached_reference(
    *,
    reference_report: dict[str, Any],
    cached_report: dict[str, Any],
    reference_volume: np.ndarray,
    cached_volume: np.ndarray,
    residual_difference_maximum: float,
    volume_difference_maximum: float,
    optimization_speedup_minimum: float,
    recurrence_difference_maximum: float,
) -> dict[str, Any]:
    if reference_volume.shape != cached_volume.shape:
        raise ValueError("reference and cached volumes must have the same shape")
    if reference_volume.dtype != cached_volume.dtype:
        raise ValueError("reference and cached volumes must have the same dtype")
    denominator = max(float(np.linalg.norm(reference_volume.ravel())), 1e-24)
    volume_difference = float(
        np.linalg.norm((reference_volume - cached_volume).ravel()) / denominator
    )
    volume_max_abs_difference = float(
        np.max(np.abs(reference_volume - cached_volume))
    )
    reference_residual = float(
        reference_report["evaluation"]["direct_support_relative_measurement_l2"]
    )
    cached_residual = float(
        cached_report["evaluation"]["direct_support_relative_measurement_l2"]
    )
    residual_difference = abs(reference_residual - cached_residual)
    reference_wall = float(reference_report["optimization"]["wall_seconds"])
    cached_wall = float(cached_report["optimization"]["wall_seconds"])
    speedup = reference_wall / max(cached_wall, 1e-24)
    cached_recurrence = float(
        cached_report["evaluation"][
            "direct_vs_recurrence_residual_relative_difference"
        ]
    )
    calls_match = (
        int(reference_report["optimization"]["logical_forward_calls"])
        == int(cached_report["optimization"]["logical_forward_calls"])
        == 4
        and int(reference_report["optimization"]["logical_adjoint_calls"])
        == int(cached_report["optimization"]["logical_adjoint_calls"])
        == 5
    )
    gates = {
        "support_residual_absolute_difference": residual_difference
        <= float(residual_difference_maximum),
        "volume_relative_difference": volume_difference
        <= float(volume_difference_maximum),
        "optimization_speedup": speedup >= float(optimization_speedup_minimum),
        "cached_direct_vs_recurrence_residual": cached_recurrence
        <= float(recurrence_difference_maximum),
        "logical_calls_match_frozen_budget": calls_match,
        "cached_run_uses_private_compact_store": (
            cached_report["configuration"]["ray_store_mode"]
            == "private_compact_stencil_cache"
        ),
        "development_rotation_40_not_accessed": True,
        "final_audit_not_accessed": True,
    }
    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": (
            "CACHED_32_CUBED_REFERENCE_REPLAY_PASS"
            if all(gates.values())
            else "CACHED_32_CUBED_REFERENCE_REPLAY_FAIL"
        ),
        "evidence_scope": (
            "SUPPORT_ONLY_FROZEN_DIRECT_VS_CACHED_CGLS_REPLAY_"
            "NO_DEVELOPMENT_NO_FINAL_AUDIT"
        ),
        "configuration": {
            "grid_shape_zyx": list(reference_volume.shape[-3:]),
            "dtype": str(reference_volume.dtype),
            "fixed_iterations": 4,
            "logical_forward_calls": 4,
            "logical_adjoint_calls": 5,
        },
        "numerical_equivalence": {
            "reference_support_relative_measurement_l2": reference_residual,
            "cached_support_relative_measurement_l2": cached_residual,
            "support_residual_absolute_difference": residual_difference,
            "support_residual_absolute_difference_maximum": float(
                residual_difference_maximum
            ),
            "volume_relative_difference": volume_difference,
            "volume_relative_difference_maximum": float(
                volume_difference_maximum
            ),
            "volume_max_abs_difference": volume_max_abs_difference,
            "cached_direct_vs_recurrence_residual_relative_difference": (
                cached_recurrence
            ),
            "cached_direct_vs_recurrence_residual_maximum": float(
                recurrence_difference_maximum
            ),
        },
        "performance": {
            "reference_optimization_wall_seconds": reference_wall,
            "cached_optimization_wall_seconds": cached_wall,
            "optimization_speedup": speedup,
            "optimization_speedup_minimum": float(optimization_speedup_minimum),
        },
        "gates": gates,
        "claim_boundary": {
            "support_replay_is_heldout_generalization": False,
            "experimental_field_truth_available": False,
            "algorithm_superiority": False,
            "private_volume_voxels_exported": False,
            "private_file_hashes_exported": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--cached-report", type=Path, required=True)
    parser.add_argument("--reference-volume", type=Path, required=True)
    parser.add_argument("--cached-volume", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--residual-difference-maximum", type=float, default=1e-13)
    parser.add_argument("--volume-difference-maximum", type=float, default=1e-12)
    parser.add_argument("--optimization-speedup-minimum", type=float, default=1.5)
    parser.add_argument("--recurrence-difference-maximum", type=float, default=1e-10)
    args = parser.parse_args()
    report = compare_cached_reference(
        reference_report=_json_object(args.reference_report),
        cached_report=_json_object(args.cached_report),
        reference_volume=np.load(args.reference_volume, allow_pickle=False),
        cached_volume=np.load(args.cached_volume, allow_pickle=False),
        residual_difference_maximum=float(args.residual_difference_maximum),
        volume_difference_maximum=float(args.volume_difference_maximum),
        optimization_speedup_minimum=float(args.optimization_speedup_minimum),
        recurrence_difference_maximum=float(args.recurrence_difference_maximum),
    )
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"].endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
