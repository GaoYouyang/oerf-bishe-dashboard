#!/usr/bin/env python3
"""Validate the public PSU-S16 independent-renderer smoke artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "psu-s16-analytic-renderer-smoke-report-1.0"
EXPECTED_STATUS = "E1_SYNTHETIC_INTERFACE_SMOKE_NOT_SUPERIORITY_EVIDENCE"
EXPECTED_FAMILIES = (
    "smooth_plume",
    "wrinkled_density_interface",
    "oblique_compression_sheet",
    "shock_expansion_pair",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, *, name: str, minimum: float | None = None) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} lies below its allowed minimum")
    return number


def _validate_checksums(output_dir: Path) -> int:
    manifest = output_dir / "checksums.sha256"
    lines = manifest.read_text(encoding="ascii").splitlines()
    expected_names = {
        "README.md",
        "diagnostic.pdf",
        "diagnostic.png",
        "metric_rows.csv",
        "summary.json",
    }
    seen = set()
    for line in lines:
        digest, name = line.split("  ", maxsplit=1)
        path = output_dir / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"checksum mismatch: {name}")
        seen.add(name)
    if seen != expected_names:
        raise ValueError("checksum manifest does not bind the complete artifact")
    return len(seen)


def validate(
    *,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    report = _load_json(output_dir / "summary.json")
    config = _load_json(config_path)
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("unsupported report schema")
    if report.get("status") != EXPECTED_STATUS:
        raise ValueError("report status is not the frozen E1 smoke status")
    if report.get("source_config_sha256") != _sha256(config_path):
        raise ValueError("report is not bound to the current config")
    independence = report["interfaces"]["renderer_independence_audit"]
    if independence.get("passed") is not True or independence.get("present_tokens"):
        raise ValueError("analytic renderer imports the forbidden inverse chain")
    geometry = report["geometry"]
    expected_geometry = config["geometry"]
    for report_key, config_key in (
        ("view_count", "view_count"),
        ("rays_per_view", "rays_per_view"),
        ("grid_size", "grid_size"),
        ("truth_renderer_qmc_samples", "truth_renderer_qmc_samples"),
        ("renderer_audit_qmc_samples", "renderer_audit_qmc_samples"),
        ("inverse_operator_qmc_samples", "inverse_operator_qmc_samples"),
    ):
        if int(geometry[report_key]) != int(expected_geometry[config_key]):
            raise ValueError(f"geometry mismatch: {report_key}")
    if geometry.get("contains_public_ray_coordinates") is not False:
        raise ValueError("public artifact declares ray coordinates")
    operator = report["operator_audit"]
    adjoint_error = _finite(
        operator["adjoint_relative_error"],
        name="adjoint_relative_error",
        minimum=0.0,
    )
    if operator.get("adjoint_gate_passed") is not True or adjoint_error > float(
        operator["adjoint_gate_maximum"]
    ):
        raise ValueError("operator adjoint gate failed")
    stages = int(config["solver"]["fixed_stages"])
    calls = report["solver"]["optimization_calls"]
    if calls != {"forward_calls": stages, "adjoint_calls": stages + 1}:
        raise ValueError("CGLS call accounting does not match the frozen stages")

    rows = report["metric_rows"]
    if tuple(row["family"] for row in rows) != EXPECTED_FAMILIES:
        raise ValueError("metric row family order changed")
    expected_seeds = tuple(int(row["seed"]) for row in config["phantoms"])
    if tuple(int(row["seed"]) for row in rows) != expected_seeds:
        raise ValueError("metric row seeds changed")
    always_finite = (
        "qmc32_vs_qmc64_relative_l2",
        "field_relative_l2",
        "field_rmse",
        "field_nrmse_dynamic_range",
        "field_mean_bias",
        "h1_seminorm_relative_error",
        "support_reprojection_relative_l2",
    )
    for row in rows:
        for name in always_finite:
            _finite(row[name], name=f"{row['family']}.{name}")
        single_interface = row["family"] in {
            "wrinkled_density_interface",
            "oblique_compression_sheet",
        }
        front_names = (
            "surface_assd",
            "surface_hd95",
            "surface_f1_at_1_voxel",
            "surface_f1_at_2_voxels",
            "normal_angle_median_degrees",
            "normal_angle_p95_degrees",
        )
        if single_interface:
            if row["front_metric_status"] != "AVAILABLE_SYNTHETIC_SINGLE_INTERFACE_ONLY":
                raise ValueError("single-interface front metrics are unavailable")
            for name in front_names:
                _finite(row[name], name=f"{row['family']}.{name}")
        elif any(row[name] is not None for name in front_names):
            raise ValueError("non-single-interface row contains unsupported front metrics")

    with (output_dir / "metric_rows.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(rows):
        raise ValueError("CSV and JSON metric row counts disagree")
    aggregate = report["aggregate_diagnostics"]
    if aggregate.get("all_outputs_finite") is not True:
        raise ValueError("artifact reports non-finite outputs")
    qmc_max = max(float(row["qmc32_vs_qmc64_relative_l2"]) for row in rows)
    if not math.isclose(
        qmc_max,
        float(aggregate["qmc32_vs_qmc64_relative_l2_maximum"]),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError("aggregate QMC discrepancy does not match metric rows")
    if any(value is not False for key, value in report["claim_boundary"].items() if key.startswith("is_")):
        raise ValueError("claim boundary promotes the interface smoke")
    policy = report["public_export_policy"]
    if not all(
        policy.get(name) is False
        for name in (
            "contains_local_paths",
            "contains_ray_coordinates_or_indices",
            "contains_measurement_or_volume_arrays",
            "contains_private_manifest_hashes",
        )
    ):
        raise ValueError("public export policy exposes private material")
    checksum_count = _validate_checksums(output_dir)
    return {
        "status": "PASS",
        "metric_row_count": len(rows),
        "checksum_file_count": checksum_count,
        "adjoint_relative_error": adjoint_error,
        "qmc32_vs_qmc64_relative_l2_maximum": qmc_max,
        "claim_level": report["evidence_level"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        config_path=args.config.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
