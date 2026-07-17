#!/usr/bin/env python3
"""Independently validate a rotation-held-out H2 sensitivity report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "psu-rotation40-forward-mismatch-private-report-1.1"
REPORT_STATUS = "H2_FORWARD_MISMATCH_DIAGNOSTIC_COMPLETE_B0_SENSITIVITY_ONLY_NO_FIELD_TRUTH_NO_SUPERIORITY"
VALIDATION_SCHEMA = "psu-rotation40-forward-mismatch-validation-1.1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
METRIC_NAMES = ("measurement", "author_exact_residual", "fixed_domain_residual", "forward_mismatch")
DOMAIN_COUNTS = (
    "total_sample_count", "valid_sample_count", "empty_ray_count", "partial_ray_count", "full_ray_count",
    "centerline_ray_count", "centerline_hit_count", "centerline_miss_count",
)
PROVENANCE_HASHES = (
    "camera_identity_sha256", "bundle_manifest_sha256", "mask_manifest_sha256", "mask_sha256",
    "author_manifest_sha256", "author_forward_sha256", "author_source_sha256", "geometry_sha256", "field_sha256",
)


def _close(left: float, right: float) -> bool:
    return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is non-finite")
    return number


def _metric_invariants(metrics: dict[str, Any], component_count: int, location: str, errors: list[str]) -> None:
    required = {"measurement_squared_l2", "measurement_l2"}
    for name in METRIC_NAMES[1:]:
        required.update({f"{name}_squared_l2", f"{name}_l2", f"{name}_rms", f"{name}_relative_l2"})
    missing = sorted(key for key in required if key not in metrics)
    if missing:
        errors.append(f"{location}: missing metrics {missing}")
        return
    if int(metrics.get("component_count", -1)) != component_count:
        errors.append(f"{location}: component count mismatch")
    try:
        measurement_squared = _finite(metrics["measurement_squared_l2"], f"{location}.measurement_squared_l2")
        measurement_l2 = _finite(metrics["measurement_l2"], f"{location}.measurement_l2")
        if measurement_squared < 0 or not _close(measurement_l2, math.sqrt(measurement_squared)):
            errors.append(f"{location}: measurement norm is inconsistent")
        denominator = max(measurement_squared, 1e-30)
        for name in METRIC_NAMES[1:]:
            squared = _finite(metrics[f"{name}_squared_l2"], f"{location}.{name}_squared_l2")
            l2 = _finite(metrics[f"{name}_l2"], f"{location}.{name}_l2")
            rms = _finite(metrics[f"{name}_rms"], f"{location}.{name}_rms")
            relative = _finite(metrics[f"{name}_relative_l2"], f"{location}.{name}_relative_l2")
            if squared < 0 or not _close(l2, math.sqrt(squared)):
                errors.append(f"{location}: {name} L2 is inconsistent")
            if not _close(rms, math.sqrt(squared / component_count)):
                errors.append(f"{location}: {name} RMS is inconsistent")
            if not _close(relative, math.sqrt(squared / denominator)):
                errors.append(f"{location}: {name} relative L2 is inconsistent")
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(str(exc))


def _domain_invariants(domain: dict[str, Any], rays: int, qmc: int, location: str, errors: list[str]) -> None:
    missing = [key for key in DOMAIN_COUNTS if key not in domain]
    if missing:
        errors.append(f"{location}: missing domain counts {missing}")
        return
    try:
        values = {key: int(domain[key]) for key in DOMAIN_COUNTS}
        total = rays * qmc
        if values["total_sample_count"] != total:
            errors.append(f"{location}: total sample count mismatch")
        if not 0 <= values["valid_sample_count"] <= total:
            errors.append(f"{location}: valid sample count is out of range")
        if sum(values[key] for key in ("empty_ray_count", "partial_ray_count", "full_ray_count")) != rays:
            errors.append(f"{location}: sample-domain ray partition mismatch")
        if values["centerline_ray_count"] != rays or values["centerline_hit_count"] + values["centerline_miss_count"] != rays:
            errors.append(f"{location}: centerline hit/miss partition mismatch")
        valid_fraction = _finite(domain["valid_sample_fraction"], f"{location}.valid_sample_fraction")
        hit_fraction = _finite(domain["centerline_hit_fraction"], f"{location}.centerline_hit_fraction")
        if not _close(valid_fraction, values["valid_sample_count"] / total) or not 0.0 <= valid_fraction <= 1.0:
            errors.append(f"{location}: valid sample fraction mismatch")
        if not _close(hit_fraction, values["centerline_hit_count"] / rays) or not 0.0 <= hit_fraction <= 1.0:
            errors.append(f"{location}: centerline hit fraction mismatch")
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(str(exc))


def _validate_provenance(report: dict[str, Any], records: list[dict[str, Any]], errors: list[str]) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("report provenance is absent")
        return
    for key in ("field_sha256", "field_manifest_sha256", "geometry_sha256", "geometry_manifest_sha256", "generator_source_fingerprint"):
        try:
            _sha(provenance.get(key), f"provenance.{key}")
        except ValueError as exc:
            errors.append(str(exc))
    if not isinstance(provenance.get("generator_version"), str) or not provenance.get("generator_version"):
        errors.append("provenance.generator_version is absent")
    source_hashes = provenance.get("generator_source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        errors.append("provenance.generator_source_sha256 is absent")
    else:
        for path, digest in source_hashes.items():
            try:
                _sha(digest, f"provenance.generator_source_sha256[{path}]")
            except ValueError as exc:
                errors.append(str(exc))
    camera_bindings = provenance.get("camera_bindings")
    if not isinstance(camera_bindings, dict) or set(camera_bindings) != {"2", "3", "4"}:
        errors.append("provenance.camera_bindings is incomplete")
    if provenance.get("field_frozen_before_rotation40_access") is not True or provenance.get("geometry_frozen_before_rotation40_access") is not True:
        errors.append("frozen-input provenance is not attested")
    for index, row in enumerate(records):
        row_provenance = row.get("provenance")
        if not isinstance(row_provenance, dict):
            errors.append(f"per_camera[{index}].provenance is absent")
            continue
        for key in PROVENANCE_HASHES + ("generator_source_fingerprint",):
            try:
                _sha(row_provenance.get(key), f"per_camera[{index}].provenance.{key}")
            except ValueError as exc:
                errors.append(str(exc))
        if row_provenance.get("generator_version") != provenance.get("generator_version"):
            errors.append(f"per_camera[{index}] generator version is inconsistent")
        if row_provenance.get("generator_source_fingerprint") != provenance.get("generator_source_fingerprint"):
            errors.append(f"per_camera[{index}] generator source is inconsistent")
        for key in ("field_sha256", "geometry_sha256"):
            if row_provenance.get(key) != provenance.get(key):
                errors.append(f"per_camera[{index}] {key} is inconsistent")
        binding = camera_bindings.get(str(row.get("camera_id"))) if isinstance(camera_bindings, dict) else None
        if binding != row_provenance:
            errors.append(f"per_camera[{index}] does not match its top-level camera binding")
    by_camera: dict[int, dict[str, Any]] = {}
    for row in records:
        camera = int(row.get("camera_id", -1))
        current = row.get("provenance", {})
        if camera in by_camera and current != by_camera[camera]:
            errors.append(f"camera {camera} provenance changes across QMC sensitivity rows")
        by_camera[camera] = current


def _validate_report(report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if report.get("schema_version") != REPORT_SCHEMA:
        errors.append("unexpected report schema")
    if report.get("status") != REPORT_STATUS:
        errors.append("report status is not the frozen B0 sensitivity-only status")
    if report.get("evidence_scope") != "DEVELOPMENT_ONLY_ROTATION_40_AUTHOR_ARRAY_VERSUS_B0_FIXED_DOMAIN_FINITE_APERTURE_FORWARD_ON_FROZEN_32CUBED_FIELD":
        errors.append("report evidence scope is not rotation-held-out H2")

    configuration = report.get("configuration", {})
    if configuration.get("rotation_degrees") != 40 or configuration.get("camera_ids") != [2, 3, 4]:
        errors.append("report scope is not rotation-40 cameras 2/3/4")
    if configuration.get("qmc_sample_counts") != [8, 16, 32]:
        errors.append("report QMC schedule is not 8/16/32")
    if configuration.get("qmc_interpretation") != "B0_SENSITIVITY_ONLY_AUTHOR_ARRAY_IS_SINGLE_FIXED_ARTIFACT":
        errors.append("report QMC schedule is incorrectly labeled paired convergence")
    if configuration.get("field_grid_shape_zyx") != [32, 32, 32]:
        errors.append("report field is not 32^3")
    try:
        _sha(configuration.get("preregistration_sha256"), "configuration.preregistration_sha256")
    except ValueError as exc:
        errors.append(str(exc))
    boundary = report.get("claim_boundary", {})
    if (
        boundary.get("experimental_field_truth") is not False
        or boundary.get("diagnostic_is_algorithm_superiority") is not False
        or boundary.get("diagnostic_is_reconstruction_quality") is not False
        or boundary.get("rotation_40_is_development_only") is not True
        or boundary.get("camera_held_out") is not False
        or boundary.get("qmc_8_16_32_is_paired_convergence") is not False
        or boundary.get("final_audit_opened") is not False
    ):
        errors.append("report crosses its claim boundary")
    policy = report.get("private_data_policy", {})
    for key in ("contains_private_paths", "contains_source_paths", "contains_field_values", "contains_measurement_values", "contains_prediction_values", "contains_ray_indices"):
        if policy.get(key) is not False:
            errors.append(f"private data policy is not closed for {key}")
    accounting = report.get("domain_and_ood_accounting", {})
    if (
        accounting.get("rotation_40_is_disjoint_from_support_rotations") is not True
        or accounting.get("camera_ids_are_shared_with_support") is not True
        or accounting.get("camera_held_out") is not False
        or accounting.get("held_out_unit") != "rotation_run_not_camera"
        or accounting.get("centerline_misses_accounted") is not True
    ):
        errors.append("domain/OOD accounting is incomplete")

    records = report.get("per_camera", [])
    pooled = report.get("pooled", [])
    expected = {(camera, qmc) for camera in (2, 3, 4) for qmc in (8, 16, 32)}
    if not isinstance(records, list) or {(row.get("camera_id"), row.get("qmc_sample_count")) for row in records if isinstance(row, dict)} != expected or len(records) != len(expected):
        errors.append("per-camera rows are incomplete or duplicated")
        records = [row for row in records if isinstance(row, dict)] if isinstance(records, list) else []
    if not isinstance(pooled, list) or {row.get("qmc_sample_count") for row in pooled if isinstance(row, dict)} != {8, 16, 32} or len(pooled) != 3:
        errors.append("pooled rows are incomplete or duplicated")
        pooled = [row for row in pooled if isinstance(row, dict)] if isinstance(pooled, list) else []
    _validate_provenance(report, records, errors)

    rays = int(configuration.get("rays_per_camera", 0))
    for index, row in enumerate(records):
        location = f"per_camera[{index}]"
        try:
            if int(row.get("qmc_sample_count", -1)) not in (8, 16, 32):
                errors.append(f"{location}: unexpected QMC count")
            if int(row.get("domain", {}).get("ray_count", -1)) != rays:
                errors.append(f"{location}: per-camera ray count mismatch")
            if int(row.get("metrics", {}).get("component_count", -1)) != 2 * rays:
                errors.append(f"{location}: per-camera component count mismatch")
            _metric_invariants(row.get("metrics", {}), 2 * rays, location, errors)
            _domain_invariants(row.get("domain", {}), rays, int(row.get("qmc_sample_count", 0)), location, errors)
        except (TypeError, ValueError) as exc:
            errors.append(f"{location}: {exc}")

    pooled_by_qmc = {row.get("qmc_sample_count"): row for row in pooled if isinstance(row, dict)}
    for qmc in (8, 16, 32):
        rows = [row for row in records if row.get("qmc_sample_count") == qmc]
        aggregate = pooled_by_qmc.get(qmc)
        if aggregate is None or len(rows) != 3:
            continue
        components = sum(int(row.get("metrics", {}).get("component_count", -1)) for row in rows)
        if int(aggregate.get("component_count", -1)) != components or int(aggregate.get("camera_count", -1)) != 3:
            errors.append(f"qmc{qmc}: pooled counts mismatch")
        _metric_invariants(aggregate.get("metrics", aggregate), components, f"pooled[qmc{qmc}]", errors)
        for name in METRIC_NAMES:
            try:
                squared = sum(_finite(row["metrics"][f"{name}_squared_l2"], f"qmc{qmc}.{name}") for row in rows)
                if not _close(aggregate.get(f"{name}_squared_l2", float("nan")), squared):
                    errors.append(f"qmc{qmc}: {name} squared norm mismatch")
                if not _close(aggregate.get(f"{name}_l2", float("nan")), math.sqrt(squared)):
                    errors.append(f"qmc{qmc}: {name} L2 mismatch")
            except (TypeError, ValueError) as exc:
                errors.append(f"qmc{qmc}: {exc}")
        domain = aggregate.get("domain", {})
        _domain_invariants(domain, 3 * rays, qmc, f"pooled[qmc{qmc}]", errors)
        for name in DOMAIN_COUNTS:
            try:
                if int(domain[name]) != sum(int(row["domain"][name]) for row in rows):
                    errors.append(f"qmc{qmc}: domain {name} mismatch")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"qmc{qmc}: {exc}")

    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    try:
        return _validate_report(report)
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        return {
            "schema_version": VALIDATION_SCHEMA,
            "status": "INVALID",
            "errors": [f"validator failed closed: {exc}"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = validate_report(report)
    payload = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
