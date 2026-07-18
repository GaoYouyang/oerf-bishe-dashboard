#!/usr/bin/env python3
"""Independently validate the N5-D4c MSRA development artifact.

The validator does not import the development runner or its metric module.  It
recomputes gate booleans and headline counts from the committed config and CSV
artifacts, verifies the Cartesian threshold coverage and all file/source
hashes, and preserves the development-only claim boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4c_msra_development_preregistered_v1.json"
)
DEFAULT_RESULT = (
    ROOT / "demo_t16_operator/results/n2_pvgr_n5_d4c_msra_development_v1"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _git_show(commit: str, path: Path) -> bytes:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{_relative(path)}"),
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"committed source missing: {_relative(path)}")
    return completed.stdout


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid CSV boolean: {value!r}")


def _base_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["scenario"],
        row["expected_role"],
        row["trial"],
        row["parameter_name"],
        row["parameter_value"],
        row["probe_count"],
    )


def validate(config_path: Path, result_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    result_dir = result_dir.resolve()
    config = _read_json(config_path)
    result_path = result_dir / "result.json"
    manifest_path = result_dir / "manifest.json"
    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    errors: list[str] = []

    if result.get("machine_decision") != (
        "D4C_MSRA_DEVELOPMENT_CHARACTERIZATION_ONLY_NO_AUTHORIZATION"
    ):
        errors.append("machine decision drifted")
    if result.get("threshold_selected") is not None:
        errors.append("development result selected a gamma threshold")
    if result.get("threshold_selection_forbidden") is not True:
        errors.append("threshold-selection prohibition drifted")
    if any(bool(value) for value in result.get("claim_authorizations", {}).values()):
        errors.append("a development claim was authorized")
    if result.get("config_sha256") != _sha256(config_path):
        errors.append("config hash drifted")
    if manifest.get("config_sha256") != result.get("config_sha256"):
        errors.append("manifest/result config hashes disagree")
    if manifest.get("protocol_commit") != result.get("protocol_commit"):
        errors.append("manifest/result protocol commits disagree")

    for name, record in manifest.get("artifacts", {}).items():
        path = result_dir / name
        if not path.is_file():
            errors.append(f"manifest artifact missing: {name}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"artifact byte count drifted: {name}")
        if _sha256(path) != record["sha256"]:
            errors.append(f"artifact hash drifted: {name}")

    protocol_commit = str(result.get("protocol_commit", ""))
    for relative, expected_hash in result.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            errors.append(f"current source hash drifted: {relative}")
            continue
        try:
            committed = _git_show(protocol_commit, path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if hashlib.sha256(committed).hexdigest() != expected_hash:
            errors.append(f"protocol-commit source hash drifted: {relative}")

    with (result_dir / "synthetic_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        base_rows = list(csv.DictReader(handle))
    with (result_dir / "threshold_probe_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        gate_rows = list(csv.DictReader(handle))
    expected_base_count = int(result["counts"]["synthetic_base_rows"])
    expected_gate_count = int(result["counts"]["threshold_probe_evaluations"])
    if len(base_rows) != expected_base_count:
        errors.append("synthetic base-row count drifted")
    if len(gate_rows) != expected_gate_count:
        errors.append("threshold/probe row count drifted")
    if len(gate_rows) != len(base_rows) * len(config["gamma_threshold_grid"]):
        errors.append("threshold rows do not form base-row x threshold Cartesian size")
    if len({row["scenario"] for row in base_rows}) != int(
        result["counts"]["scenario_count"]
    ):
        errors.append("scenario count drifted")

    thresholds = {float(value) for value in config["gamma_threshold_grid"]}
    fd_threshold = float(config["finite_difference_relative_threshold"])
    structure_threshold = float(config["structure_relative_threshold"])
    coverage: dict[tuple[str, ...], set[float]] = {}
    gate_lookup: dict[tuple[str, ...], dict[str, str]] = {}
    for row in gate_rows:
        key = _base_key(row)
        threshold = float(row["gamma_threshold"])
        coverage.setdefault(key, set()).add(threshold)
        gamma_gate = float(row["maximum_gamma_scaled_normwise_score"]) <= threshold
        fd_gate = float(row["fd_relative_error"]) <= fd_threshold
        structure_gate = float(row["structure_relative_error"]) <= structure_threshold
        branch_gate = not _boolean(row["actual_forward_branch_changed"])
        combined = gamma_gate and fd_gate and structure_gate and branch_gate
        for name, recomputed in (
            ("gamma_gate", gamma_gate),
            ("fd_gate", fd_gate),
            ("structure_gate", structure_gate),
            ("branch_gate", branch_gate),
            ("combined_gate", combined),
        ):
            if _boolean(row[name]) != recomputed:
                errors.append(f"recomputed {name} drifted: {key}/{threshold}")
        gate_lookup[key + (f"{threshold:.17g}",)] = row
    if any(values != thresholds for values in coverage.values()):
        errors.append("one or more base rows lacks the complete threshold grid")
    if len(coverage) != len(base_rows):
        errors.append("base-row identities are not unique or fully covered")

    minimum_threshold = float(config["gamma_threshold_grid"][0])
    low_signal = [
        row
        for row in base_rows
        if row["scenario"] == "clean_low_bilinear_signal"
        and int(row["probe_count"]) == 1
    ]
    self_consistent = [
        row
        for row in gate_rows
        if row["scenario"] == "self_consistent_wrong_derivative"
        and int(row["probe_count"]) == 1
        and math.isclose(float(row["gamma_threshold"]), minimum_threshold)
    ]
    support = [
        row
        for row in gate_rows
        if row["scenario"] == "diagnostic_only_support_flip"
        and int(row["probe_count"]) == 1
        and math.isclose(float(row["gamma_threshold"]), minimum_threshold)
    ]
    hard = [
        row
        for row in gate_rows
        if row["scenario"] == "hard_branch_crossing"
        and int(row["probe_count"]) == 1
        and math.isclose(float(row["gamma_threshold"]), minimum_threshold)
    ]
    recomputed_headline = {
        "low_signal_count": len(low_signal),
        "low_signal_traditional_reject_count": sum(
            not _boolean(row["traditional_dot_gate"]) for row in low_signal
        ),
        "low_signal_maximum_gamma_score": max(
            float(row["maximum_gamma_scaled_normwise_score"]) for row in low_signal
        ),
        "self_consistent_count": len(self_consistent),
        "self_consistent_adjoint_blind_fd_reject_count": sum(
            _boolean(row["gamma_gate"]) and not _boolean(row["fd_gate"])
            for row in self_consistent
        ),
        "diagnostic_support_count": len(support),
        "diagnostic_support_report_only_count": sum(
            _boolean(row["diagnostic_support_changed"])
            and _boolean(row["branch_gate"])
            and _boolean(row["combined_gate"])
            for row in support
        ),
        "hard_branch_count": len(hard),
        "hard_branch_reject_count": sum(
            not _boolean(row["branch_gate"]) for row in hard
        ),
    }
    headline = result.get("headline_diagnostics", {})
    for key, value in recomputed_headline.items():
        observed = headline.get(key)
        equal = (
            math.isclose(float(observed), float(value), rel_tol=1e-14, abs_tol=0.0)
            if isinstance(value, float)
            else observed == value
        )
        if not equal:
            errors.append(f"headline diagnostic drifted: {key}")

    retrospective = result.get("retrospective_d4b", {})
    if retrospective.get("historical_machine_decision") != (
        "D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED"
    ):
        errors.append("retrospective D4b decision drifted")
    if retrospective.get("historical_decision_changed") is not False:
        errors.append("retrospective D4b decision was changed")
    for key, expected_hash in retrospective.get("source_hashes", {}).items():
        source_path = _resolve(config["retrospective_d4b"][key])
        if _sha256(source_path) != expected_hash:
            errors.append(f"retrospective D4b source hash drifted: {key}")

    return {
        "schema": "n2-pvgr-n5-d4c-msra-development-validation-1.0",
        "valid": not errors,
        "errors": errors,
        "machine_decision": result.get("machine_decision"),
        "protocol_commit": protocol_commit,
        "threshold_selected": result.get("threshold_selected"),
        "counts": {
            "synthetic_base_rows": len(base_rows),
            "threshold_probe_rows": len(gate_rows),
            "unique_base_row_keys": len(coverage),
            "threshold_count": len(thresholds),
            "manifest_artifact_count": len(manifest.get("artifacts", {})),
        },
        "recomputed_headline_diagnostics": recomputed_headline,
        "claim_boundary_valid": not any(
            bool(value) for value in result.get("claim_authorizations", {}).values()
        ),
        "historical_d4b_decision_retained": retrospective.get(
            "historical_decision_changed"
        )
        is False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = (
        args.output.resolve()
        if args.output
        else args.result_dir.resolve() / "validation_report.json"
    )
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing to replace D4c validation: {output}")
    report = validate(args.config, args.result_dir)
    _write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
