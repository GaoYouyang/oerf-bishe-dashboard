#!/usr/bin/env python3
"""Independently validate the aggregate-only rotation-40 public result pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1"
)
EXPECTED_FILES = {
    "README.md",
    "comparison_rows.csv",
    "diagnostic.pdf",
    "diagnostic.png",
    "summary.json",
    "checksums.sha256",
}
EXPECTED_STATUS = "SUPPORT_RESOLUTION_GAIN_DID_NOT_CLEAR_NUMERICAL_TRANSFER_GATE_NO_GO"
EXPECTED_TOP_LEVEL = {
    "attestation_sha256",
    "candidates",
    "claim_boundary",
    "comparison",
    "complexity_context",
    "config_sha256",
    "dataset",
    "evidence_scope",
    "forward",
    "geometry_binding",
    "input_bindings",
    "protocol_commit",
    "public_export_policy",
    "schema_version",
    "selection",
    "status",
    "support_reconstruction_contract",
}
METRIC_KEYS = (
    "ray_count",
    "vector_relative_l2",
    "component_rmse_px",
    "component_mae_px",
    "residual_magnitude_p95_px",
    "measured_vector_rms_px",
    "predicted_vector_rms_px",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value


def _assert_close(left: float, right: float, *, label: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-13, abs_tol=1e-13):
        raise ValueError(f"public metric mismatch: {label}")


def _validate_bounded_public_tree(value: Any) -> tuple[int, int]:
    """Reject path strings and any aggregate tree large enough to hide arrays."""

    containers = 0
    numeric_leaves = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal containers, numeric_leaves
        if depth > 8:
            raise ValueError("public report nesting exceeds the aggregate schema bound")
        if isinstance(node, Mapping):
            containers += 1
            if len(node) > 24:
                raise ValueError("public report object exceeds the aggregate schema bound")
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
            if len(node) > 3:
                raise ValueError("public report list exceeds the aggregate schema bound")
            walk_items = list(node)
            for child in walk_items:
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
    if containers > 96 or numeric_leaves > 192:
        raise ValueError("public report exceeds the aggregate scalar budget")
    return containers, numeric_leaves


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


def _validate_summary(summary: Mapping[str, Any]) -> None:
    if set(summary) != EXPECTED_TOP_LEVEL:
        raise ValueError("public summary top-level schema changed")
    if summary.get("status") != EXPECTED_STATUS:
        raise ValueError("public summary decision changed")
    if summary.get("protocol_commit") != "ba77a17f7c7bb5ba42a9a55a9de776e267d5f9d0":
        raise ValueError("public summary protocol commit changed")
    if summary.get("public_export_policy") != {
        "contains_geometry_arrays": False,
        "contains_measurements": False,
        "contains_only_aggregate_metrics": True,
        "contains_predictions": False,
        "contains_volumes": False,
    }:
        raise ValueError("public export policy changed")
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("public candidate count changed")
    if [row.get("candidate_id") for row in candidates] != [
        "support_cgls4_16cubed",
        "support_cgls4_32cubed",
    ]:
        raise ValueError("public candidate identity changed")
    for candidate in candidates:
        if len(candidate.get("per_camera", [])) != 3:
            raise ValueError("public per-camera row count changed")
        if [row.get("camera_id") for row in candidate["per_camera"]] != [2, 3, 4]:
            raise ValueError("public per-camera identity changed")
        if set(candidate.get("aggregate", {})) != set(METRIC_KEYS):
            raise ValueError("public aggregate metric schema changed")
        if any(set(row) != set(METRIC_KEYS) | {"camera_id"} for row in candidate["per_camera"]):
            raise ValueError("public camera metric schema changed")
    comparison = summary.get("comparison", {})
    rel16 = float(candidates[0]["aggregate"]["vector_relative_l2"])
    rel32 = float(candidates[1]["aggregate"]["vector_relative_l2"])
    _assert_close(comparison["pooled_vector_relative_l2_16"], rel16, label="pooled16")
    _assert_close(comparison["pooled_vector_relative_l2_32"], rel32, label="pooled32")
    _assert_close(
        comparison["pooled_absolute_improvement_16_minus_32"],
        rel16 - rel32,
        label="pooled delta",
    )
    rows16 = {int(row["camera_id"]): row for row in candidates[0]["per_camera"]}
    rows32 = {int(row["camera_id"]): row for row in candidates[1]["per_camera"]}
    comparison_rows = comparison.get("camera_rows", [])
    if len(comparison_rows) != 3:
        raise ValueError("comparison camera row count changed")
    improvements = []
    for row in comparison_rows:
        camera_id = int(row["camera_id"])
        value16 = float(rows16[camera_id]["vector_relative_l2"])
        value32 = float(rows32[camera_id]["vector_relative_l2"])
        improvement = value16 - value32
        improvements.append(improvement)
        _assert_close(row["vector_relative_l2_16"], value16, label=f"camera{camera_id} 16")
        _assert_close(row["vector_relative_l2_32"], value32, label=f"camera{camera_id} 32")
        _assert_close(
            row["absolute_improvement_16_minus_32"],
            improvement,
            label=f"camera{camera_id} delta",
        )
        if row.get("nonworse") is not False:
            raise ValueError("camera nonworse decision changed")
    _assert_close(
        comparison["worst_camera_absolute_improvement_16_minus_32"],
        min(improvements),
        label="worst camera",
    )
    if comparison.get("all_three_cameras_nonworse") is not False:
        raise ValueError("all-camera decision changed")
    if comparison.get("predeclared_numerical_pooled_improvement") is not False:
        raise ValueError("numerical gate decision changed")
    if comparison.get("machine_decision") != EXPECTED_STATUS:
        raise ValueError("comparison machine decision changed")


def _validate_csv(result_dir: Path, summary: Mapping[str, Any]) -> None:
    with (result_dir / "comparison_rows.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8:
        raise ValueError("public CSV must contain two pooled and six camera rows")
    expected: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for candidate in summary["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        expected[(candidate_id, "pooled", "")] = candidate["aggregate"]
        for row in candidate["per_camera"]:
            expected[(candidate_id, "camera", str(row["camera_id"]))] = row
    for row in rows:
        key = (row["candidate_id"], row["scope"], row["camera_id"])
        if key not in expected:
            raise ValueError("public CSV identity changed")
        source = expected[key]
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
        if image.size != (2640, 1720) or image.mode not in {"RGB", "RGBA"}:
            raise ValueError("public diagnostic PNG shape or mode changed")
    if not (result_dir / "diagnostic.pdf").read_bytes().startswith(b"%PDF-"):
        raise ValueError("public diagnostic PDF header changed")
    return {
        "status": "PUBLIC_AGGREGATE_PACKAGE_VALID",
        "machine_decision": summary["status"],
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
