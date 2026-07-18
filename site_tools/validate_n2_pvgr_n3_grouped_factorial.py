#!/usr/bin/env python3
"""Fail-closed validator for the preregistered N2-PVGR-N3 result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "demo_t16_operator/results/n2_pvgr_n3_grouped_factorial_v1"
METHODS = {
    "continuous_affine_n1",
    "operator_consistent_homotopy",
    "picard_1",
    "picard_2",
}
ALLOWED_DECISIONS = {
    "GROUPED_FACTORIAL_FAIL_NO_FORWARD_AUTHORIZATION",
    "PICARD_DOMINATES_OCBH_FORWARD_ROLE_CLOSED_FIELD_VJP_GATE_NEXT",
    "OCBH_NOT_DOMINATED_CONDITIONAL_FIELD_VJP_GATE_NEXT",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{path}.{key}")
        return
    raise TypeError(f"unexpected JSON value at {path}: {type(value).__name__}")


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _validate_manifest(result_dir: Path) -> tuple[dict[str, Any], int]:
    manifest = _read_json(result_dir / "manifest.json")
    _require(
        manifest.get("schema") == "n2-pvgr-n3-grouped-factorial-manifest-1.0",
        "manifest schema mismatch",
    )
    files = manifest.get("files")
    _require(isinstance(files, dict) and len(files) == 41, "manifest file set mismatch")
    for key, entry in files.items():
        path = (ROOT / str(entry["path"])).resolve()
        _require(
            ROOT == path or ROOT in path.parents, f"manifest path escapes root: {key}"
        )
        _require(path.is_file(), f"manifest file missing: {key}")
        _require(_sha256(path) == entry["sha256"], f"manifest hash mismatch: {key}")
        _require(path.stat().st_size == int(entry["bytes"]), f"size mismatch: {key}")
    return manifest, len(files)


def _validate_rows(result: dict[str, Any]) -> dict[str, int]:
    method_rows = result["method_rows"]
    _require(len(method_rows) == 384, "method row count is not 384")
    by_method = {
        method: [row for row in method_rows if row["method_id"] == method]
        for method in METHODS
    }
    _require(all(len(rows) == 96 for rows in by_method.values()), "method imbalance")
    cell_ids = {str(row["cell_id"]) for row in method_rows}
    field_units = {str(row["field_unit_id"]) for row in method_rows}
    _require(len(cell_ids) == 96, "physical cell identifier count is not 96")
    _require(len(field_units) == 8, "field-unit identifier count is not 8")
    for method, rows in by_method.items():
        _require(
            {str(row["cell_id"]) for row in rows} == cell_ids,
            f"cell pairing drift for {method}",
        )

    _require(len(result["teacher_rows"]) == 96, "teacher row count is not 96")
    _require(
        len(result["reference_sentinel_rows"]) == 96,
        "sentinel row count is not 96",
    )
    _require(len(result["timing_rows"]) == 20, "timing row count is not 20")
    _require(
        len(result["query_accounting_rows"]) == 96,
        "query-accounting cell count is not 96",
    )
    _require(
        len(result["field_unit_summary_rows"]) == 32,
        "field-unit summary row count is not 32",
    )
    _require(
        len(result["field_unit_ledger"]) == 8,
        "field-unit ledger row count is not 8",
    )
    return {
        "method_rows": len(method_rows),
        "physical_cells": len(cell_ids),
        "field_units": len(field_units),
        "teacher_rows": len(result["teacher_rows"]),
        "sentinel_rows": len(result["reference_sentinel_rows"]),
        "timing_rows": len(result["timing_rows"]),
    }


def _validate_machine_decision(result: dict[str, Any]) -> str:
    decision = str(result["machine_decision"])
    _require(decision in ALLOWED_DECISIONS, "unregistered machine decision")
    gate_summary = result["ocbh_gate_summary"]
    required_pairs = (
        ("primary", 96),
        ("teacher", 96),
        ("sentinel", 96),
        ("timing", 4),
        ("query", 96),
    )
    all_inherited = True
    for name, required in required_pairs:
        _require(
            int(gate_summary[f"{name}_required_count"]) == required,
            f"{name} required count drifted",
        )
        all_inherited &= int(gate_summary[f"{name}_pass_count"]) == int(
            gate_summary[f"{name}_required_count"]
        )
    _require(
        bool(gate_summary["all_inherited_gates_pass"]) == all_inherited,
        "inherited gate summary is inconsistent",
    )
    dominance_rows = result["picard_forward_dominance_rows"]
    _require(
        {row["method_id"] for row in dominance_rows} == {"picard_1", "picard_2"},
        "dominance method set drifted",
    )
    for row in dominance_rows:
        _require(
            bool(row["dominates_ocbh_forward_role"])
            == all(bool(value) for value in row["dominance_gates"].values()),
            f"dominance gate summary inconsistent for {row['method_id']}",
        )
    any_dominates = any(
        bool(row["dominates_ocbh_forward_role"]) for row in dominance_rows
    )
    expected = (
        "GROUPED_FACTORIAL_FAIL_NO_FORWARD_AUTHORIZATION"
        if not all_inherited
        else (
            "PICARD_DOMINATES_OCBH_FORWARD_ROLE_CLOSED_FIELD_VJP_GATE_NEXT"
            if any_dominates
            else "OCBH_NOT_DOMINATED_CONDITIONAL_FIELD_VJP_GATE_NEXT"
        )
    )
    _require(decision == expected, "machine decision does not follow frozen rules")
    return expected


def _validate_recovery(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    recovery = result.get("analysis_recovery")
    _require(isinstance(recovery, dict), "result lacks analysis-recovery disclosure")
    _require(
        recovery.get("schema") == "n2-pvgr-n3-blind-analysis-recovery-1.0",
        "recovery schema mismatch",
    )
    _require(recovery.get("threshold_seed_cell_bootstrap_changes") is False, "drift")
    _require(recovery.get("physical_cells_rerun") is False, "cell rerun drift")
    _require(int(recovery.get("opaque_checkpoint_count", -1)) == 96, "checkpoint drift")
    _require(
        manifest.get("analysis_recovery") == recovery,
        "manifest and result recovery disclosures differ",
    )
    attestation = ROOT / str(recovery["recovery_attestation"])
    _require(
        _sha256(attestation) == recovery["recovery_attestation_sha256"],
        "attestation drift",
    )


def _validate_csvs_and_figure(result_dir: Path) -> dict[str, int]:
    expected_rows = {
        "metrics.csv": 384,
        "field_unit_summary.csv": 32,
        "field_unit_ledger.csv": 8,
        "picard_forward_dominance.csv": 2,
        "teacher_metrics.csv": 96,
        "reference_sentinel.csv": 96,
        "timing.csv": 20,
        "query_accounting.csv": 480,
    }
    for name, expected in expected_rows.items():
        _require(
            _csv_row_count(result_dir / name) == expected, f"CSV count drift: {name}"
        )
    figure = result_dir / "n2_pvgr_n3_grouped_factorial.png"
    with Image.open(figure) as image:
        _require(
            image.width >= 1600 and image.height >= 800, "figure resolution too low"
        )
        extrema = ImageStat.Stat(image.convert("L")).extrema[0]
        _require(extrema[1] - extrema[0] >= 100, "figure appears visually blank")
    return expected_rows


def validate(result_dir: Path = DEFAULT_RESULT) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    _require(result_dir.is_dir(), "result directory is missing")
    manifest, manifest_count = _validate_manifest(result_dir)
    result = _read_json(result_dir / "result.json")
    _require_finite(result)
    _require(
        result.get("schema") == "n2-pvgr-n3-grouped-factorial-preregistered-1.0",
        "result schema mismatch",
    )
    _require(int(result["physical_cell_count"]) == 96, "result physical-cell drift")
    _require(int(result["field_unit_count"]) == 8, "result field-unit drift")
    _require(
        all(value is False for value in result["claim_authorizations"].values()),
        "broad result claim was authorized",
    )
    row_counts = _validate_rows(result)
    decision = _validate_machine_decision(result)
    _validate_recovery(result, manifest)
    csv_counts = _validate_csvs_and_figure(result_dir)
    config = _read_json(ROOT / manifest["files"]["config"]["path"])
    snapshot = _read_json(result_dir / "config_snapshot.json")
    _require(config == snapshot, "config snapshot is not semantically identical")
    return {
        "status": "PASS_FAIL_CLOSED_BUNDLE_VALIDATION",
        "machine_decision": decision,
        "manifest_file_count": manifest_count,
        "row_counts": row_counts,
        "csv_row_counts": csv_counts,
        "claim_authorizations_all_false": True,
        "analysis_recovery_disclosed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(args.result)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
