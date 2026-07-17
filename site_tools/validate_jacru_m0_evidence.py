#!/usr/bin/env python3
"""Validate the chained M0/M0.1/M1 JACRU negative-evidence packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = {"cgls", "huber_pdhg", "phase_only", "jacru_no_bias", "jacru_bias"}


class ValidationError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _validate_rows(path: Path, *, expected_pairs: int = 24) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == 20, f"{path}: expected 20 metric rows")
    _require({row["method"] for row in rows} == METHODS, f"{path}: method set drift")
    case_ids = {row["case_id"] for row in rows}
    _require(len(case_ids) == 4, f"{path}: expected four cases")
    for case_id in case_ids:
        selected = [row for row in rows if row["case_id"] == case_id]
        _require(len(selected) == 5, f"{path}: each case needs five methods")
    for row in rows:
        _require(
            int(row["optimization_forward_calls"]) == expected_pairs,
            f"{path}: forward budget drift for {row['method']}",
        )
        _require(
            int(row["optimization_vjp_or_adjoint_calls"]) == expected_pairs,
            f"{path}: reverse budget drift for {row['method']}",
        )
        for key in (
            "field_relative_l2",
            "h1_seminorm_relative_error",
            "measured_reprojection_relative_l2",
        ):
            value = float(row[key])
            _require(value >= 0.0 and value < float("inf"), f"{path}: invalid {key}")
    return {
        "row_count": len(rows),
        "case_count": len(case_ids),
        "methods": sorted(METHODS),
        "fixed_pair_budget": expected_pairs,
    }


def _validate_no_go(report: dict[str, Any], *, label: str) -> None:
    _require(report["decision"]["passed"] is False, f"{label}: pass must remain false")
    _require(
        report["decision"]["status"] == "M0_NO_GO_OR_REVISE",
        f"{label}: status drift",
    )
    boundary = report["claim_boundary"]
    for key in (
        "is_experimental_reconstruction",
        "is_cfd_validation",
        "is_operator_learning",
        "is_neural_operator_superiority",
        "is_confirmatory_or_final",
    ):
        _require(boundary[key] is False, f"{label}: forbidden claim {key}")


def validate_packet(
    *,
    m0_config_path: Path,
    m0_report_path: Path,
    m0_rows_path: Path,
    initialization_report_path: Path,
    m0_1_config_path: Path,
    m0_1_report_path: Path,
    m0_1_rows_path: Path,
    m1_config_path: Path,
    m1_report_path: Path,
    m1_rows_path: Path,
) -> dict[str, Any]:
    m0 = _load_json(m0_report_path)
    initialization = _load_json(initialization_report_path)
    m0_1_config = _load_json(m0_1_config_path)
    m0_1 = _load_json(m0_1_report_path)
    m1_config = _load_json(m1_config_path)
    m1 = _load_json(m1_report_path)
    _validate_no_go(m0, label="M0")
    _validate_no_go(m0_1, label="M0.1")
    _validate_no_go(m1, label="M1")

    _require(
        m0["source_config_sha256"] == _sha256(m0_config_path),
        "M0 source config hash mismatch",
    )
    amendment = m0_1_config["amendment"]
    _require(
        amendment["source_config_sha256"] == _sha256(m0_config_path),
        "M0.1 does not anchor M0 config",
    )
    _require(
        amendment["source_summary_sha256"] == _sha256(m0_report_path),
        "M0.1 does not anchor M0 report",
    )
    m1_amendment = m1_config["amendment"]
    _require(
        m1_amendment["source_m0_1_config_sha256"] == _sha256(m0_1_config_path),
        "M1 does not anchor M0.1 config",
    )
    _require(
        m1_amendment["source_m0_1_summary_sha256"] == _sha256(m0_1_report_path),
        "M1 does not anchor M0.1 report",
    )
    _require(
        m1_amendment["source_initialization_audit_sha256"]
        == _sha256(initialization_report_path),
        "M1 does not anchor the initialization audit",
    )

    _require(
        initialization["status"]
        == "INTERFACE_SCORE_CONFOUNDED_BY_DATA_FREE_INITIALIZATION",
        "initialization confound verdict drift",
    )
    _require(initialization["decision"]["confounded"] is True, "confound must remain true")
    _require(
        initialization["decision"]["interface_gate_credit_authorized"] is False,
        "interface credit must remain closed",
    )
    init_aggregate = initialization["aggregate"]
    _require(
        float(init_aggregate["single_interface_initial_f1_at_1dx_mean"]) == 1.0,
        "data-free initial interface F1 changed",
    )
    _require(
        float(init_aggregate["single_interface_final_minus_initial_f1_mean"]) <= 0.0,
        "final interface unexpectedly improved over data-free initialization",
    )
    _require(
        float(init_aggregate["smooth_initial_false_positive_rate"]) == 1.0,
        "smooth false-positive audit changed",
    )

    m0_field = float(m0["aggregate"]["jacru_bias"]["field_relative_l2_mean"])
    m0_1_field = float(m0_1["aggregate"]["jacru_bias"]["field_relative_l2_mean"])
    scale_repair_gain = (m0_field - m0_1_field) / m0_field
    _require(scale_repair_gain > 0.5, "scale repair did not reproduce its diagnostic gain")
    _require(
        m0["decision"]["diagnostics"]["field_gain_fraction"] < 0.0,
        "M0 field gate unexpectedly passed",
    )
    _require(
        m0_1["decision"]["diagnostics"]["field_gain_fraction"] < 0.0,
        "M0.1 field gate unexpectedly passed",
    )
    _require(
        m1["decision"]["checks"]["reprojection"] is True,
        "M1 should retain only its reprojection check",
    )
    _require(
        sum(bool(value) for value in m1["decision"]["checks"].values()) == 1,
        "M1 passed more than its single diagnostic check",
    )
    m1_field = float(m1["aggregate"]["jacru_no_bias"]["field_relative_l2_mean"])
    cgls_field = float(m1["aggregate"]["cgls"]["field_relative_l2_mean"])
    huber_field = float(m1["aggregate"]["huber_pdhg"]["field_relative_l2_mean"])
    _require(m1_field < cgls_field, "M1 must reproduce its narrow CGLS improvement")
    _require(m1_field > huber_field, "M1 must remain worse than the Huber baseline")

    row_reports = {
        "m0": _validate_rows(m0_rows_path),
        "m0_1": _validate_rows(m0_1_rows_path),
        "m1": _validate_rows(m1_rows_path),
    }
    return {
        "schema_version": "jacru-m0-m1-evidence-validation-1.0",
        "status": "VALIDATED_NEGATIVE_EVIDENCE_PACKET",
        "row_reports": row_reports,
        "diagnostics": {
            "scale_repair_field_gain_fraction": scale_repair_gain,
            "m1_vs_cgls_field_gain_fraction": (cgls_field - m1_field) / cgls_field,
            "m1_vs_huber_field_gain_fraction": (huber_field - m1_field) / huber_field,
        },
        "authorization": {
            "continue_learned_residual_operator": True,
            "claim_jacru_superiority": False,
            "claim_interface_gain": False,
            "open_fresh_or_final_split": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-config", type=Path, required=True)
    parser.add_argument("--m0-report", type=Path, required=True)
    parser.add_argument("--m0-rows", type=Path, required=True)
    parser.add_argument("--initialization-report", type=Path, required=True)
    parser.add_argument("--m0-1-config", type=Path, required=True)
    parser.add_argument("--m0-1-report", type=Path, required=True)
    parser.add_argument("--m0-1-rows", type=Path, required=True)
    parser.add_argument("--m1-config", type=Path, required=True)
    parser.add_argument("--m1-report", type=Path, required=True)
    parser.add_argument("--m1-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_packet(
        m0_config_path=args.m0_config,
        m0_report_path=args.m0_report,
        m0_rows_path=args.m0_rows,
        initialization_report_path=args.initialization_report,
        m0_1_config_path=args.m0_1_config,
        m0_1_report_path=args.m0_1_report,
        m0_1_rows_path=args.m0_1_rows,
        m1_config_path=args.m1_config,
        m1_report_path=args.m1_report,
        m1_rows_path=args.m1_rows,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
