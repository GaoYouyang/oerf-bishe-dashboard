#!/usr/bin/env python3
"""Build a strict same-rays, same-calls PSU 16³-to-32³ resolution summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "psu-b0-streaming-resolution-public-summary-1.0"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_summary(
    low: dict[str, Any],
    high: dict[str, Any],
    *,
    practical_signal_minimum: float,
) -> dict[str, Any]:
    low_grid = low["configuration"]["grid_shape_zyx"]
    high_grid = high["configuration"]["grid_shape_zyx"]
    comparable = {
        "both_runs_passed_interface_and_inverse_gates": (
            low["status"].endswith("NO_FIELD_TRUTH")
            and high["status"].endswith("NO_FIELD_TRUTH")
        ),
        "same_total_ray_count": (
            low["selection"]["total_ray_count"]
            == high["selection"]["total_ray_count"]
        ),
        "same_view_selection": (
            low["selection"]["view_rows"] == high["selection"]["view_rows"]
        ),
        "same_finite_aperture_sample_count": (
            low["configuration"]["finite_aperture_sample_count"]
            == high["configuration"]["finite_aperture_sample_count"]
        ),
        "same_dtype": (
            low["configuration"]["dtype"] == high["configuration"]["dtype"]
        ),
        "same_device": (
            low["configuration"]["device"] == high["configuration"]["device"]
        ),
        "same_fixed_cgls_iterations": (
            low["optimization"]["fixed_iteration_budget"]
            == high["optimization"]["fixed_iteration_budget"]
        ),
        "same_logical_forward_calls": (
            low["optimization"]["logical_forward_calls"]
            == high["optimization"]["logical_forward_calls"]
        ),
        "same_logical_adjoint_calls": (
            low["optimization"]["logical_adjoint_calls"]
            == high["optimization"]["logical_adjoint_calls"]
        ),
        "expected_grid_pair": low_grid == [16, 16, 16]
        and high_grid == [32, 32, 32],
        "development_rotation_40_not_accessed": (
            low["gates"]["development_rotation_40_not_accessed"]
            and high["gates"]["development_rotation_40_not_accessed"]
        ),
        "final_audit_not_accessed": (
            low["gates"]["final_audit_not_accessed"]
            and high["gates"]["final_audit_not_accessed"]
        ),
    }
    if not all(comparable.values()):
        failed = [name for name, passed in comparable.items() if not passed]
        raise ValueError(f"resolution summaries are not comparable: {failed}")

    low_residual = float(
        low["evaluation"]["direct_support_relative_measurement_l2"]
    )
    high_residual = float(
        high["evaluation"]["direct_support_relative_measurement_l2"]
    )
    absolute_drop = low_residual - high_residual
    relative_improvement = absolute_drop / low_residual
    threshold = float(practical_signal_minimum)
    per_view = []
    low_rows = low["evaluation"]["per_view_support_relative_measurement_l2"]
    high_rows = high["evaluation"]["per_view_support_relative_measurement_l2"]
    for low_row, high_row in zip(low_rows, high_rows, strict=True):
        if low_row["view_id_zero_based"] != high_row["view_id_zero_based"]:
            raise ValueError("per-view rows are not aligned")
        low_value = float(low_row["relative_measurement_l2"])
        high_value = float(high_row["relative_measurement_l2"])
        per_view.append(
            {
                "view_id_zero_based": int(low_row["view_id_zero_based"]),
                "ray_count": int(low_row["ray_count"]),
                "residual_16_cubed": low_value,
                "residual_32_cubed": high_value,
                "absolute_drop": low_value - high_value,
                "improved": high_value < low_value,
            }
        )
    signal = absolute_drop >= threshold
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": (
            "SAME_CALL_RESOLUTION_SIGNAL_PASS_NO_FIELD_TRUTH"
            if signal
            else "SAME_CALL_RESOLUTION_SIGNAL_NOT_ESTABLISHED"
        ),
        "evidence_scope": (
            "REAL_PSU_ALL_SUPPORT_RAYS_QMC16_FLOAT64_FIXED_FOUR_CGLS_ITERATIONS_"
            "16_TO_32_CUBED_COMPARISON_NO_DEVELOPMENT_NO_FINAL_AUDIT_NO_FIELD_"
            "TRUTH"
        ),
        "comparability_gates": comparable,
        "configuration": {
            "ray_count": int(low["selection"]["total_ray_count"]),
            "view_count": 9,
            "finite_aperture_sample_count": int(
                low["configuration"]["finite_aperture_sample_count"]
            ),
            "dtype": low["configuration"]["dtype"],
            "device": low["configuration"]["device"],
            "fixed_cgls_iterations": int(
                low["optimization"]["fixed_iteration_budget"]
            ),
            "logical_forward_calls": int(
                low["optimization"]["logical_forward_calls"]
            ),
            "logical_adjoint_calls": int(
                low["optimization"]["logical_adjoint_calls"]
            ),
        },
        "aggregate": {
            "direct_support_relative_measurement_l2_16_cubed": low_residual,
            "direct_support_relative_measurement_l2_32_cubed": high_residual,
            "absolute_drop": absolute_drop,
            "relative_improvement_fraction": relative_improvement,
            "practical_signal_minimum_absolute_drop": threshold,
            "practical_resolution_signal": signal,
            "views_improved_count": sum(row["improved"] for row in per_view),
            "view_count": len(per_view),
        },
        "per_view": per_view,
        "resources": {
            "pair_wall_seconds_16_cubed": low["resource_gate"][
                "full_forward_adjoint_pair_wall_seconds"
            ],
            "pair_wall_seconds_32_cubed": high["resource_gate"][
                "full_forward_adjoint_pair_wall_seconds"
            ],
            "process_max_rss_bytes_16_cubed": low["resource_gate"][
                "process_max_rss_bytes"
            ],
            "process_max_rss_bytes_32_cubed": high["resource_gate"][
                "process_max_rss_bytes"
            ],
            "server_required_for_32_cubed_gate": high["resource_gate"][
                "server_required_for_current_16_cubed_gate"
            ],
        },
        "decision": {
            "resolution_is_a_material_support_fit_bottleneck": signal,
            "authoritative_low_resolution_reference": (
                "32_cubed" if signal else "16_cubed"
            ),
            "development_rotation_40_may_open_after_report_and_hash_freeze": (
                signal
            ),
            "algorithm_superiority_established": False,
        },
        "claim_boundary": {
            "support_fit_is_heldout_generalization": False,
            "support_fit_is_unique_three_dimensional_truth": False,
            "experimental_field_l2_available": False,
            "development_rotation_40_opened": False,
            "final_audit_opened": False,
        },
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_measurement_values": False,
            "contains_reconstruction_voxels": False,
            "contains_private_hashes": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-summary", type=Path, required=True)
    parser.add_argument("--high-summary", type=Path, required=True)
    parser.add_argument("--practical-signal-minimum", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(
        _load(args.low_summary),
        _load(args.high_summary),
        practical_signal_minimum=args.practical_signal_minimum,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"].endswith("NO_FIELD_TRUTH") else 1


if __name__ == "__main__":
    raise SystemExit(main())
