#!/usr/bin/env python3
"""Independently validate the aggregate-only multiresolution diagnosis pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "demo_t16_operator/results/psu_rotation40_multiresolution_diagnosis_public_v1"
)
EXPECTED_FILES = {
    "README.md",
    "comparison_rows.csv",
    "diagnostic.pdf",
    "diagnostic.png",
    "summary.json",
    "checksums.sha256",
}
EXPECTED_STATUS = "OPENED_BLOCK_FORWARD_GRID_CHANGE_MATERIAL_MECHANISM_UNRESOLVED"
EXPECTED_PROTOCOL_COMMIT = "48e32d780e66926dfb493f28e0375fb83c4057f9"
EXPECTED_TOP_LEVEL = {
    "candidates",
    "claim_boundary",
    "config_sha256",
    "correction_alignment",
    "dataset",
    "direct_forward_runtime",
    "evidence_scope",
    "field_diagnostics",
    "forward",
    "linearity_check",
    "mechanism_decision",
    "protocol_commit",
    "protocol_file_sha256",
    "public_export_policy",
    "schema_version",
    "selection",
    "source_resolution_protocol",
    "status",
    "transforms",
}
DIRECT_IDS = [
    "native_16",
    "prolong_16_to_32",
    "restrict_32_to_16",
    "roundtrip_32_via_16",
    "native_32",
]
LINE_IDS = [
    "line_alpha_0.00",
    "line_alpha_0.25",
    "line_alpha_0.50",
    "line_alpha_0.75",
    "line_alpha_1.00",
]
LINE_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
METRIC_KEYS = {
    "ray_count",
    "vector_relative_l2",
    "component_rmse_px",
    "component_mae_px",
    "residual_magnitude_p95_px",
    "measured_vector_rms_px",
    "predicted_vector_rms_px",
}
EXPECTED_PROTOCOL_FILES = {
    "demo_t16_operator/configs/psu_rotation40_multiresolution_diagnosis_development_v1.json",
    "docs/psu_rotation40_multiresolution_diagnosis_protocol_2026-07-19.md",
    "site_tools/run_psu_rotation40_multiresolution_diagnosis.py",
    "site_tools/test_run_psu_rotation40_multiresolution_diagnosis.py",
    "site_tools/run_psu_rotation40_resolution_transfer.py",
    "site_tools/run_psu_rotation40_b0_reprojection.py",
    "site_tools/psu_rotation40_active_store.py",
    "site_tools/psu_b0_real_support_store.py",
    "site_tools/psu_bost_aperture_domain.py",
    "site_tools/psu_bost_forward_geometry.py",
    "demo_t16_operator/psu_b0_streaming_operator.py",
    "demo_t16_operator/psu_b0_reconstruction_interface.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value


def _assert_close(left: float, right: float, *, label: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"public metric mismatch: {label}")


def _validate_checksums(result_dir: Path) -> None:
    actual = {path.name for path in result_dir.iterdir() if path.is_file()}
    if actual != EXPECTED_FILES:
        raise ValueError("public result file set changed")
    entries: dict[str, str] = {}
    for line in (result_dir / "checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", 1)
        entries[filename] = digest
    if set(entries) != EXPECTED_FILES - {"checksums.sha256"}:
        raise ValueError("checksum coverage changed")
    for filename, digest in entries.items():
        if _sha256(result_dir / filename) != digest:
            raise ValueError(f"checksum mismatch: {filename}")


def _validate_bounded_public_tree(value: Any) -> tuple[int, int]:
    containers = 0
    numeric_leaves = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal containers, numeric_leaves
        if depth > 9:
            raise ValueError("public report nesting exceeds the aggregate schema bound")
        if isinstance(node, Mapping):
            containers += 1
            if len(node) > 24:
                raise ValueError("public object exceeds the aggregate schema bound")
            for key, child in node.items():
                key_text = str(key).lower()
                if key_text in {
                    "data",
                    "values",
                    "array",
                    "arrays",
                    "predictions",
                    "measurements",
                    "observations",
                    "volume",
                    "volumes",
                    "rays",
                    "local_path",
                    "private_path",
                }:
                    raise ValueError("public report contains a non-aggregate payload key")
                walk(child, depth + 1)
        elif isinstance(node, list):
            containers += 1
            if len(node) > 12:
                raise ValueError("public list exceeds the aggregate schema bound")
            for child in node:
                walk(child, depth + 1)
        elif isinstance(node, str):
            if node.startswith("/Users/") or "private_library/" in node:
                raise ValueError("public report contains a private path")
            if "reconstruction_16cubed.npy" in node or "reconstruction_32cubed" in node:
                raise ValueError("public report contains a private filename")
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            numeric_leaves += 1
            if isinstance(node, float) and not math.isfinite(node):
                raise ValueError("public report contains a non-finite scalar")
        else:
            raise ValueError("public report contains an unsupported value type")

    walk(value, 0)
    if containers > 192 or numeric_leaves > 768:
        raise ValueError("public report exceeds the aggregate scalar budget")
    return containers, numeric_leaves


def _camera_rows(candidate: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = candidate.get("per_camera")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("candidate must contain three camera rows")
    mapped = {int(row["camera_id"]): row for row in rows}
    if set(mapped) != {2, 3, 4}:
        raise ValueError("camera identity changed")
    if any(set(row) != METRIC_KEYS | {"camera_id"} for row in rows):
        raise ValueError("camera metric schema changed")
    return mapped


def _validate_line_curve(
    candidates: Mapping[str, Mapping[str, Any]],
    correction: Mapping[str, Any],
) -> None:
    baseline = candidates["prolong_16_to_32"]
    corrected = candidates["native_32"]
    for metric in METRIC_KEYS:
        if metric == "ray_count":
            continue
        _assert_close(
            candidates["line_alpha_0.00"]["aggregate"][metric],
            baseline["aggregate"][metric],
            label=f"line alpha0 pooled {metric}",
        )
        _assert_close(
            candidates["line_alpha_1.00"]["aggregate"][metric],
            corrected["aggregate"][metric],
            label=f"line alpha1 pooled {metric}",
        )
    scopes = [("pooled", correction["pooled"], None)] + [
        (f"camera{row['camera_id']}", row, int(row["camera_id"]))
        for row in correction["per_camera"]
    ]
    for label, alignment, camera_id in scopes:
        residual = float(alignment["residual_l2"])
        update = float(alignment["correction_l2"])
        dot = float(alignment["residual_correction_dot"])
        source = (
            baseline["aggregate"]
            if camera_id is None
            else _camera_rows(baseline)[camera_id]
        )
        measured_norm = residual / float(source["vector_relative_l2"])
        expected_cosine = dot / (residual * update)
        _assert_close(
            alignment["residual_correction_cosine"],
            expected_cosine,
            label=f"{label} correction cosine",
        )
        expected_alpha = dot / (update * update)
        _assert_close(
            alignment["unconstrained_least_squares_alpha_diagnostic_only"],
            expected_alpha,
            label=f"{label} alpha star",
        )
        for candidate_id, alpha in zip(LINE_IDS, LINE_ALPHAS, strict=True):
            squared = residual * residual - 2.0 * alpha * dot + alpha * alpha * update * update
            expected_relative_l2 = math.sqrt(max(squared, 0.0)) / measured_norm
            candidate = candidates[candidate_id]
            actual = (
                candidate["aggregate"]["vector_relative_l2"]
                if camera_id is None
                else _camera_rows(candidate)[camera_id]["vector_relative_l2"]
            )
            _assert_close(
                actual,
                expected_relative_l2,
                label=f"{label} {candidate_id} relative L2",
            )


def _validate_summary(summary: Mapping[str, Any]) -> None:
    if set(summary) != EXPECTED_TOP_LEVEL:
        raise ValueError("public summary top-level schema changed")
    if summary.get("status") != EXPECTED_STATUS:
        raise ValueError("public summary status changed")
    if summary.get("protocol_commit") != EXPECTED_PROTOCOL_COMMIT:
        raise ValueError("protocol commit changed")
    protocol_files = summary.get("protocol_file_sha256")
    if not isinstance(protocol_files, Mapping) or set(protocol_files) != EXPECTED_PROTOCOL_FILES:
        raise ValueError("protocol dependency closure changed")
    for relative, digest in protocol_files.items():
        frozen = subprocess.run(
            ["git", "show", f"{EXPECTED_PROTOCOL_COMMIT}:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if frozen.returncode != 0:
            raise ValueError(f"protocol dependency is absent from Git: {relative}")
        if hashlib.sha256(frozen.stdout).hexdigest() != digest:
            raise ValueError(f"protocol dependency digest changed: {relative}")
    if summary.get("source_resolution_protocol") != {
        "config_path": "demo_t16_operator/configs/psu_rotation40_resolution_transfer_preregistered_v1.json",
        "config_sha256": "9b23fc69951b2ad5c5b7f736738d1fea5937b7da152be4beb0f22b0e45ab2889",
        "result_path": "demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1/summary.json",
        "result_sha256": "34f67c29a3dade7a25b609622472ddd14735f2d8e24fc7ce6aeed3d272eab439",
        "result_commit": "21c6e4a5f57fbed73d216f6d7328c41639554d29",
    }:
        raise ValueError("source resolution evidence binding changed")
    if summary.get("public_export_policy") != {
        "contains_geometry_arrays": False,
        "contains_measurements": False,
        "contains_only_aggregate_metrics": True,
        "contains_predictions": False,
        "contains_volumes": False,
    }:
        raise ValueError("public export policy changed")
    firewall = summary.get("claim_boundary")
    if not isinstance(firewall, Mapping):
        raise ValueError("claim firewall is missing")
    if firewall.get("postopen_development_only") is not True:
        raise ValueError("post-open development boundary changed")
    if any(value is not False for key, value in firewall.items() if key != "postopen_development_only"):
        raise ValueError("a scientific claim firewall opened")
    candidates_list = summary.get("candidates")
    if not isinstance(candidates_list, list) or len(candidates_list) != 10:
        raise ValueError("candidate count changed")
    ids = [str(row.get("candidate_id")) for row in candidates_list]
    if ids != DIRECT_IDS + LINE_IDS:
        raise ValueError("candidate identity or order changed")
    candidates = {str(row["candidate_id"]): row for row in candidates_list}
    for candidate_id, candidate in candidates.items():
        if set(candidate.get("aggregate", {})) != METRIC_KEYS:
            raise ValueError("aggregate metric schema changed")
        _camera_rows(candidate)
        expected_calls = 0 if candidate_id in LINE_IDS else 1
        if int(candidate.get("direct_forward_calls", -1)) != expected_calls:
            raise ValueError("candidate direct-forward provenance changed")
        if candidate.get("postopen_candidate_selection") is not False:
            raise ValueError("candidate-selection boundary changed")
    for candidate_id, alpha in zip(LINE_IDS, LINE_ALPHAS, strict=True):
        _assert_close(candidates[candidate_id]["line_alpha"], alpha, label=candidate_id)
    _assert_close(
        candidates["native_16"]["aggregate"]["vector_relative_l2"],
        0.8432631430215097,
        label="frozen native16 endpoint",
    )
    _assert_close(
        candidates["native_32"]["aggregate"]["vector_relative_l2"],
        0.9595912440290707,
        label="frozen native32 endpoint",
    )
    correction = summary.get("correction_alignment")
    if not isinstance(correction, Mapping):
        raise ValueError("correction alignment is missing")
    if len(correction.get("per_camera", [])) != 3:
        raise ValueError("correction camera rows changed")
    _validate_line_curve(candidates, correction)
    linearity = summary.get("linearity_check", {})
    if linearity.get("passed") is not True:
        raise ValueError("linearity check did not pass")
    if float(linearity["max_abs_direct_minus_derived"]) > float(linearity["tolerance"]):
        raise ValueError("linearity tolerance was exceeded")
    decision = summary.get("mechanism_decision", {})
    native16 = float(candidates["native_16"]["aggregate"]["vector_relative_l2"])
    prolonged = float(
        candidates["prolong_16_to_32"]["aggregate"]["vector_relative_l2"]
    )
    native32 = float(candidates["native_32"]["aggregate"]["vector_relative_l2"])
    grid_gap = abs(native16 - prolonged)
    harm = native32 - prolonged
    camera_cosines = [
        float(row["residual_correction_cosine"])
        for row in correction["per_camera"]
    ]
    all_negative = all(value < 0.0 for value in camera_cosines)
    _assert_close(
        decision["native16_vs_prolonged16_absolute_relative_l2_gap"],
        grid_gap,
        label="grid gap",
    )
    _assert_close(
        decision["native32_harm_vs_prolonged16_absolute_relative_l2"],
        harm,
        label="native32 harm",
    )
    if decision.get("all_three_camera_fine_correction_cosines_negative") is not all_negative:
        raise ValueError("camera correction-cosine decision changed")
    if not (grid_gap > 0.01 and harm >= 0.01 and not all_negative):
        raise ValueError("frozen mechanism inputs no longer imply the published branch")
    if decision.get("machine_diagnosis") != EXPECTED_STATUS:
        raise ValueError("machine diagnosis changed")
    if decision.get("causal_mechanism_proved") is not False:
        raise ValueError("causal claim firewall opened")
    field = summary.get("field_diagnostics")
    expected_field_keys = {
        "native16_vs_prolonged16_prediction_difference_over_measured_l2",
        "native16_vs_prolonged16_prediction_difference_over_native16_prediction_l2",
        "udx32_l2",
        "ux16_gradient_l2_physical_spacing",
        "ux16_l2",
        "x16_l2",
        "x32_gradient_l2_physical_spacing",
        "x32_l2",
        "x32_minus_udx32_l2",
        "x32_minus_udx32_over_x32_l2",
        "x32_minus_ux16_l2",
        "x32_minus_ux16_over_x32_l2",
    }
    if not isinstance(field, Mapping) or set(field) != expected_field_keys:
        raise ValueError("field diagnostic schema changed")
    _assert_close(
        field["native16_vs_prolonged16_prediction_difference_over_measured_l2"],
        0.11183791820497918,
        label="post-open forward-grid prediction gap over measurement",
    )
    _assert_close(
        field[
            "native16_vs_prolonged16_prediction_difference_over_native16_prediction_l2"
        ],
        0.23638117302597667,
        label="post-open forward-grid prediction gap over native prediction",
    )


def _validate_csv(result_dir: Path, summary: Mapping[str, Any]) -> None:
    with (result_dir / "comparison_rows.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 40:
        raise ValueError("public CSV must contain ten pooled and thirty camera rows")
    expected: dict[tuple[str, str, str], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for candidate in summary["candidates"]:
        shared = candidate
        candidate_id = str(candidate["candidate_id"])
        expected[(candidate_id, "pooled", "")] = (candidate["aggregate"], shared)
        for row in candidate["per_camera"]:
            expected[(candidate_id, "camera", str(row["camera_id"]))] = (row, shared)
    for row in rows:
        key = (row["candidate_id"], row["scope"], row["camera_id"])
        if key not in expected:
            raise ValueError("public CSV identity changed")
        source, shared = expected[key]
        if row["construction"] != shared["construction"]:
            raise ValueError("public CSV construction changed")
        if int(row["direct_forward_calls"]) != int(shared["direct_forward_calls"]):
            raise ValueError("public CSV call provenance changed")
        for metric in METRIC_KEYS:
            if metric == "ray_count":
                if int(row[metric]) != int(source[metric]):
                    raise ValueError(f"public CSV mismatch: {key} {metric}")
            else:
                _assert_close(row[metric], source[metric], label=f"CSV {key} {metric}")


def validate_public_result(result_dir: Path) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    _validate_checksums(result_dir)
    summary = _read_json(result_dir / "summary.json")
    containers, numeric_leaves = _validate_bounded_public_tree(summary)
    _validate_summary(summary)
    _validate_csv(result_dir, summary)
    with Image.open(result_dir / "diagnostic.png") as image:
        if image.size != (2700, 1800) or image.mode not in {"RGB", "RGBA"}:
            raise ValueError("public diagnostic PNG shape or mode changed")
    if not (result_dir / "diagnostic.pdf").read_bytes().startswith(b"%PDF-"):
        raise ValueError("public diagnostic PDF header changed")
    return {
        "status": "PUBLIC_AGGREGATE_PACKAGE_VALID",
        "machine_diagnosis": summary["status"],
        "containers": containers,
        "numeric_leaves": numeric_leaves,
        "file_count": len(EXPECTED_FILES),
        "private_payload_detected": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    print(json.dumps(validate_public_result(args.result_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
