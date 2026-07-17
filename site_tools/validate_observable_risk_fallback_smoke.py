from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from scipy.stats import beta as beta_distribution

from site_tools.run_observable_risk_fallback_smoke import (
    AUDIT_FIELDS,
    EXPECTED_OUTPUT_FILES,
    GEOMETRY_FIELDS,
    METRIC_FIELDS,
    OUTPUT_PAYLOADS,
    RISK_FIELDS,
    SELECTION_FIELDS,
    SOURCE_RELATIVE_PATHS,
    TRAJECTORY_FIELDS,
    _canonical_json,
    _sha256,
    _source_hashes,
    _strict_json_text,
    load_config,
    reconstruct_evidence,
)


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"{path.name} columns differ from frozen schema")
        return list(reader)


def _compare_scalar(observed: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, bool):
        if observed is not expected:
            raise ValueError(f"{path} mismatch")
    elif isinstance(expected, int):
        if not isinstance(observed, int) or isinstance(observed, bool) or observed != expected:
            raise ValueError(f"{path} mismatch")
    elif isinstance(expected, float):
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise ValueError(f"{path} mismatch")
        if not math.isclose(float(observed), expected, rel_tol=2e-11, abs_tol=2e-12):
            raise ValueError(f"{path} mismatch")
    elif observed != expected:
        raise ValueError(f"{path} mismatch")


def _compare_nested(observed: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise ValueError(f"{path} keys mismatch")
        for key in expected:
            _compare_nested(observed[key], expected[key], path=f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{path} list mismatch")
        for index, value in enumerate(expected):
            _compare_nested(observed[index], value, path=f"{path}[{index}]")
    else:
        _compare_scalar(observed, expected, path=path)


def _compare_csv_rows(
    observed: list[dict[str, str]],
    expected: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    name: str,
) -> None:
    if len(observed) != len(expected):
        raise ValueError(f"{name} row count mismatch")
    for row_index, (actual, rebuilt) in enumerate(zip(observed, expected)):
        for field in fields:
            value = rebuilt[field]
            raw = actual[field]
            path = f"{name}[{row_index}].{field}"
            if isinstance(value, bool):
                if raw not in {"True", "False"} or (raw == "True") is not value:
                    raise ValueError(f"{path} mismatch")
            elif isinstance(value, int):
                try:
                    parsed = int(raw)
                except ValueError as error:
                    raise ValueError(f"{path} mismatch") from error
                if parsed != value:
                    raise ValueError(f"{path} mismatch")
            elif isinstance(value, float):
                try:
                    parsed_float = float(raw)
                except ValueError as error:
                    raise ValueError(f"{path} mismatch") from error
                if not math.isclose(parsed_float, value, rel_tol=2e-11, abs_tol=2e-12):
                    raise ValueError(f"{path} mismatch")
            elif raw != str(value):
                raise ValueError(f"{path} mismatch")


def _verify_checksums(output_dir: Path) -> None:
    lines = (output_dir / "checksums.sha256").read_text(encoding="ascii").splitlines()
    if len(lines) != len(OUTPUT_PAYLOADS):
        raise ValueError("checksum manifest length mismatch")
    observed: dict[str, str] = {}
    for line in lines:
        if "  " not in line:
            raise ValueError("checksum manifest syntax mismatch")
        digest, name = line.split("  ", 1)
        if name in observed or name not in OUTPUT_PAYLOADS:
            raise ValueError("checksum manifest payload mismatch")
        observed[name] = digest
    if set(observed) != set(OUTPUT_PAYLOADS):
        raise ValueError("checksum manifest coverage mismatch")
    for name, digest in observed.items():
        if digest != _sha256(output_dir / name):
            raise ValueError(f"checksum mismatch: {name}")


def _csv_bool(raw: str, *, path: str) -> bool:
    if raw not in {"True", "False"}:
        raise ValueError(f"{path} is not a strict boolean")
    return raw == "True"


def _independent_statistical_cross_check(
    report: dict[str, Any], risk_rows: list[dict[str, str]]
) -> None:
    """Recompute CP bounds and conditional fresh aggregation without runner math."""

    calibration = report["risk_calibration"]
    calibration_rows = [
        row for row in risk_rows if row["split_role"] == "risk_calibration"
    ]
    threshold = float(calibration["acceptance_threshold"])
    accepted = [
        row
        for row in calibration_rows
        if _csv_bool(row["support_gate_passed"], path="calibration.support")
        and row["candidate_partition"] != "paired_cross"
        and float(row["risk_score"]) <= threshold
    ]
    failures = sum(
        _csv_bool(row["harm_failure"], path="calibration.harm_failure")
        for row in accepted
    )
    threshold_count = int(calibration["threshold_candidate_count"])
    risk_alpha = float(calibration["confidence_alpha"]) / threshold_count
    coverage_alpha = (
        float(calibration["coverage_confidence_alpha"]) / threshold_count
    )
    if not accepted or failures == len(accepted):
        risk_upper = 1.0
    else:
        risk_upper = float(
            beta_distribution.ppf(
                1.0 - risk_alpha,
                failures + 1,
                len(accepted) - failures,
            )
        )
    if not accepted:
        coverage_lower = 0.0
    else:
        coverage_lower = float(
            beta_distribution.ppf(
                coverage_alpha,
                len(accepted),
                len(calibration_rows) - len(accepted) + 1,
            )
        )
    _compare_scalar(
        calibration["accepted_count"], len(accepted), path="independent.accepted_count"
    )
    _compare_scalar(
        calibration["failure_count"], failures, path="independent.failure_count"
    )
    _compare_scalar(
        calibration["risk_upper_bound"], risk_upper, path="independent.risk_upper"
    )
    _compare_scalar(
        calibration["takeover_coverage_lower_bound"],
        coverage_lower,
        path="independent.coverage_lower",
    )
    diagnostic_coverage = len(accepted) / len(calibration_rows)
    authorized_coverage = (
        diagnostic_coverage if calibration["development_gate_passed"] else 0.0
    )
    _compare_scalar(
        calibration["authorized_takeover_coverage"],
        authorized_coverage,
        path="independent.authorized_coverage",
    )

    fresh_rows = [
        row for row in risk_rows if row["split_role"] == "fresh_geometry_ood"
    ]
    takeovers = [
        row
        for row in fresh_rows
        if not _csv_bool(row["fallback_used"], path="fresh.fallback_used")
    ]
    fresh_failures = sum(
        _csv_bool(row["harm_failure"], path="fresh.harm_failure")
        for row in takeovers
    )
    conditional_rate = fresh_failures / len(takeovers) if takeovers else None
    worst_field = (
        max(float(row["observed_field_harm_vs_fallback"]) for row in takeovers)
        if takeovers
        else None
    )
    worst_residual = (
        max(float(row["observed_residual_harm_vs_fallback"]) for row in takeovers)
        if takeovers
        else None
    )
    aggregate = report["aggregate"]
    _compare_scalar(
        aggregate["fresh_takeover_count"],
        len(takeovers),
        path="independent.fresh_takeover_count",
    )
    _compare_scalar(
        aggregate["fresh_harm_count"],
        fresh_failures,
        path="independent.fresh_harm_count",
    )
    _compare_scalar(
        aggregate["fresh_selection_conditional_harm_rate"],
        conditional_rate,
        path="independent.fresh_conditional_harm_rate",
    )
    _compare_scalar(
        aggregate["fresh_worst_takeover_field_harm"],
        worst_field,
        path="independent.fresh_worst_field_harm",
    )
    _compare_scalar(
        aggregate["fresh_worst_takeover_residual_harm"],
        worst_residual,
        path="independent.fresh_worst_residual_harm",
    )


def _verify_committed_source(provenance: dict[str, Any], config: dict[str, Any]) -> None:
    root = Path(__file__).resolve().parents[1]
    commit = provenance["source_commit"]
    if provenance["source_worktree_dirty"] is not False:
        raise ValueError("clean-source validation requires a clean entry worktree")
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        capture_output=True,
    )
    if exists.returncode != 0:
        raise ValueError("source commit does not exist in the repository")
    for name, relative_path in SOURCE_RELATIVE_PATHS.items():
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=root,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ValueError(f"source commit lacks {relative_path}")
        digest = hashlib.sha256(result.stdout).hexdigest()
        if provenance["source_sha256"].get(name) != digest:
            raise ValueError(f"source commit blob mismatch: {name}")
    config_result = subprocess.run(
        [
            "git",
            "show",
            f"{commit}:{SOURCE_RELATIVE_PATHS['config_source']}",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    committed_config = _strict_json_text(config_result.stdout)
    committed_config_text = _canonical_json(committed_config) + "\n"
    if hashlib.sha256(committed_config_text.encode("utf-8")).hexdigest() != provenance[
        "config_sha256"
    ]:
        raise ValueError("committed config does not match config snapshot")
    if committed_config != config:
        raise ValueError("committed config semantics differ from config snapshot")


def validate_result_bundle(
    output_dir: Path, *, require_clean_source: bool = False
) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise ValueError("result bundle is not a directory")
    observed_files = {path.name for path in output_dir.iterdir()}
    if observed_files != EXPECTED_OUTPUT_FILES:
        raise ValueError("result bundle file set differs from the frozen schema")
    if any(path.is_symlink() for path in output_dir.iterdir()):
        raise ValueError("result bundle cannot contain symbolic links")
    _verify_checksums(output_dir)

    config = load_config(output_dir / "config_snapshot.json")
    rebuilt = reconstruct_evidence(config)
    csv_contracts = (
        ("geometry_manifest.csv", GEOMETRY_FIELDS, rebuilt["geometry_rows"]),
        ("partition_audit_rows.csv", AUDIT_FIELDS, rebuilt["audit_rows"]),
        ("selection_rows.csv", SELECTION_FIELDS, rebuilt["selection_rows"]),
        ("risk_rows.csv", RISK_FIELDS, rebuilt["risk_rows"]),
        ("metric_rows.csv", METRIC_FIELDS, rebuilt["metric_rows"]),
        ("trajectory_rows.csv", TRAJECTORY_FIELDS, rebuilt["trajectory_rows"]),
    )
    observed_csvs: dict[str, list[dict[str, str]]] = {}
    for filename, fields, expected_rows in csv_contracts:
        observed_rows = _read_csv(output_dir / filename, fields)
        observed_csvs[filename] = observed_rows
        _compare_csv_rows(
            observed_rows,
            expected_rows,
            fields,
            name=filename.removesuffix(".csv"),
        )

    raw_report = _strict_json_text((output_dir / "report.json").read_text(encoding="utf-8"))
    if not isinstance(raw_report, dict):
        raise ValueError("report root must be an object")
    expected_report = rebuilt["report"]
    if set(raw_report) != {*expected_report, "provenance", "runtime"}:
        raise ValueError("report keys mismatch")
    for key, value in expected_report.items():
        _compare_nested(raw_report[key], value, path=f"report.{key}")

    provenance = raw_report["provenance"]
    if set(provenance) != {
        "source_commit",
        "source_worktree_dirty",
        "config_sha256",
        "source_sha256",
        "geometry_manifest_sha256",
    }:
        raise ValueError("provenance keys mismatch")
    commit = provenance["source_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("provenance source commit mismatch")
    if not isinstance(provenance["source_worktree_dirty"], bool):
        raise ValueError("provenance dirty flag mismatch")
    config_text = _canonical_json(config) + "\n"
    if provenance["config_sha256"] != hashlib.sha256(config_text.encode("utf-8")).hexdigest():
        raise ValueError("provenance config hash mismatch")
    if provenance["source_sha256"] != _source_hashes():
        raise ValueError("provenance source hash mismatch")
    if provenance["geometry_manifest_sha256"] != _sha256(output_dir / "geometry_manifest.csv"):
        raise ValueError("provenance geometry hash mismatch")
    if require_clean_source:
        _verify_committed_source(provenance, config)

    runtime = raw_report["runtime"]
    if set(runtime) != {"wall_time_seconds", "wall_time_role", "device", "dtype"}:
        raise ValueError("runtime keys mismatch")
    wall = float(runtime["wall_time_seconds"])
    if not math.isfinite(wall) or wall < 0.0:
        raise ValueError("runtime wall time must be finite and nonnegative")
    if runtime["wall_time_role"] != "MEASURED_SINGLE_RUN_DESCRIPTIVE_NONCOMPARATIVE":
        raise ValueError("runtime wall time role mismatch")
    if runtime["device"] != "cpu" or runtime["dtype"] != "torch.float64":
        raise ValueError("runtime device contract mismatch")

    if any(raw_report["claim_boundary"].values()):
        raise ValueError("claim boundary must remain entirely false")
    if any(
        raw_report["gates"][name]
        for name in (
            "future_paper_gate_passed",
            "research_claim_authorized",
            "real_bost_claim_authorized",
            "generalization_claim_authorized",
            "paper_superiority_claim_authorized",
        )
    ):
        raise ValueError("paper or real-data claim was illegally authorized")
    _independent_statistical_cross_check(
        raw_report, observed_csvs["risk_rows.csv"]
    )
    return raw_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("/tmp/oerf_observable_risk_fallback_smoke"),
    )
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help="bind every source hash and config snapshot to the recorded commit",
    )
    args = parser.parse_args()
    report = validate_result_bundle(
        args.output_dir, require_clean_source=args.require_clean_source
    )
    print(_canonical_json({"aggregate": report["aggregate"], "gates": report["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
