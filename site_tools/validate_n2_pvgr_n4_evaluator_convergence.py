#!/usr/bin/env python3
"""Independently validate the formal N2-PVGR N4 evaluator-audit artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo_t16_operator.run_n2_pvgr_n4_evaluator_convergence as n4  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n4_evaluator_convergence_preregistered_v1.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recompute_h1024_gates(
    cell: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    output = n4._contraction(
        cell["d256_to_d512"]["output_relative_l2"],
        cell["d512_to_d1024"]["output_relative_l2"],
        gates,
    )
    residual = n4._contraction(
        cell["d256_to_d512"]["matched_residual_relative_l2"],
        cell["d512_to_d1024"]["matched_residual_relative_l2"],
        gates,
    )
    tolerance = float(gates["maximum_parent_N3_metric_absolute_difference"])
    return {
        "output_absolute_gate_met": float(cell["d512_to_d1024"]["output_relative_l2"])
        <= float(gates["maximum_h512_to_h1024_output_relative_l2"]),
        "output_contraction_gate_met": output["contraction_gate_met"],
        "matched_residual_absolute_gate_met": float(
            cell["d512_to_d1024"]["matched_residual_relative_l2"]
        )
        <= float(gates["maximum_h512_to_h1024_matched_residual_relative_l2"]),
        "matched_residual_contraction_gate_met": residual["contraction_gate_met"],
        "parent_output_reproduction_gate_met": float(
            cell["parent_reproduction"]["output_absolute_difference"]
        )
        <= tolerance,
        "parent_matched_residual_reproduction_gate_met": float(
            cell["parent_reproduction"]["matched_residual_absolute_difference"]
        )
        <= tolerance,
        **n4._endpoint_integrity_gates(
            {"diagnostics": cell["h512_endpoint_diagnostics"]},
            {"diagnostics": cell["h1024_endpoint_diagnostics"]},
            gates,
        ),
    }


def _recompute_h2048_gates(
    cell: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    output = n4._contraction(
        cell["d512_to_d1024"]["output_relative_l2"],
        cell["d1024_to_d2048"]["output_relative_l2"],
        gates,
    )
    residual = n4._contraction(
        cell["d512_to_d1024"]["matched_residual_relative_l2"],
        cell["d1024_to_d2048"]["matched_residual_relative_l2"],
        gates,
    )
    return {
        "output_absolute_gate_met": float(cell["d1024_to_d2048"]["output_relative_l2"])
        <= float(gates["maximum_h1024_to_h2048_output_relative_l2"]),
        "output_contraction_gate_met": output["contraction_gate_met"],
        "matched_residual_absolute_gate_met": float(
            cell["d1024_to_d2048"]["matched_residual_relative_l2"]
        )
        <= float(gates["maximum_h1024_to_h2048_matched_residual_relative_l2"]),
        "matched_residual_contraction_gate_met": residual["contraction_gate_met"],
        **n4._endpoint_integrity_gates(
            {"diagnostics": cell["h1024_endpoint_diagnostics"]},
            {"diagnostics": cell["h2048_endpoint_diagnostics"]},
            gates,
        ),
    }


def validate(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    parent_config = _read_json(ROOT / config["parent_n3_config"])
    source = _read_json(ROOT / config["source_config"])
    n4._validate_contract(config, parent_config, source)
    attestation = n4._validate_preregistration(config, config_path)
    if output_dir.resolve() != (ROOT / config["formal_output"]).resolve():
        raise ValueError("validator output path drifted from formal N4 output")
    result_path = output_dir / "result.json"
    manifest_path = output_dir / "manifest.json"
    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    if result.get("schema") != "n2-pvgr-n4-evaluator-convergence-result-1.0":
        raise ValueError("N4 result schema drifted")
    if manifest.get("schema") != "n2-pvgr-n4-evaluator-convergence-manifest-1.0":
        raise ValueError("N4 manifest schema drifted")
    if result["protocol_commit"] != attestation["protocol_commit"]:
        raise ValueError("N4 result protocol commit drifted")
    if manifest["protocol_commit"] != attestation["protocol_commit"]:
        raise ValueError("N4 manifest protocol commit drifted")
    for entry in manifest["files"].values():
        path = ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"N4 manifest file missing: {path}")
        if (
            path.stat().st_size != int(entry["bytes"])
            or _sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"N4 manifest hash or size mismatch: {path}")

    cells = result["cells"]
    if len(cells) != 32 or len({cell["cell_id"] for cell in cells}) != 32:
        raise ValueError("N4 result must contain 32 unique cells")
    if sum(cell["role"] == "n3_failure" for cell in cells) != 16:
        raise ValueError("N4 result failure-role count drifted")
    if sum(cell["role"] == "matched_control" for cell in cells) != 16:
        raise ValueError("N4 result control-role count drifted")
    gates = config["convergence_gates"]
    for cell in cells:
        h1024 = _recompute_h1024_gates(cell, gates)
        if h1024 != cell["h1024_gates"]:
            raise ValueError(f"N4 H1024 gate recomputation mismatch: {cell['cell_id']}")
        h1024_pass = all(h1024.values())
        if bool(cell["h1024_all_gates_pass"]) != h1024_pass:
            raise ValueError(f"N4 H1024 aggregate mismatch: {cell['cell_id']}")
        if bool(cell["requires_h2048_escalation"]) != (not h1024_pass):
            raise ValueError(f"N4 escalation rule mismatch: {cell['cell_id']}")
        if h1024_pass:
            if cell["d1024_to_d2048"] is not None or cell["h2048_gates"] is not None:
                raise ValueError(f"N4 opened forbidden H2048 route: {cell['cell_id']}")
            expected_final = True
        else:
            h2048 = _recompute_h2048_gates(cell, gates)
            if h2048 != cell["h2048_gates"]:
                raise ValueError(
                    f"N4 H2048 gate recomputation mismatch: {cell['cell_id']}"
                )
            if bool(cell["h2048_all_gates_pass"]) != all(h2048.values()):
                raise ValueError(f"N4 H2048 aggregate mismatch: {cell['cell_id']}")
            expected_final = all(h2048.values())
        if bool(cell["final_cellwise_reference_authorized"]) != expected_final:
            raise ValueError(f"N4 final cell decision mismatch: {cell['cell_id']}")

    h1024_pass_count = sum(cell["h1024_all_gates_pass"] for cell in cells)
    escalation_count = sum(cell["requires_h2048_escalation"] for cell in cells)
    final_count = sum(cell["final_cellwise_reference_authorized"] for cell in cells)
    counts = result["counts"]
    expected_counts = {
        "physical_cell_count": 32,
        "n3_failure_count": 16,
        "matched_control_count": 16,
        "h1024_pass_count": h1024_pass_count,
        "h2048_escalation_count": escalation_count,
        "final_reference_authorized_count": final_count,
        "level_evaluation_count": 96 + escalation_count,
    }
    if counts != expected_counts:
        raise ValueError("N4 result count ledger drifted")
    expected_decision = (
        "EVALUATOR_CONVERGENCE_CLEARED_FOR_TINY_FIELD_JVP_VJP_GATE"
        if final_count == 32
        else "FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED"
    )
    if result["machine_decision"] != expected_decision:
        raise ValueError("N4 machine decision drifted")
    if result["authorizations"]["uniform_h1024_reference_authorized"] != (
        h1024_pass_count == 32
    ):
        raise ValueError("N4 uniform-reference authorization drifted")
    if result["authorizations"]["tiny_field_jvp_vjp_gate_authorized"] != (
        final_count == 32
    ):
        raise ValueError("N4 tiny-gate authorization drifted")
    broad_keys = (
        "reserved_audit_authorized",
        "real_data_authorized",
        "three_dimensional_reconstruction_authorized",
        "neural_operator_superiority_authorized",
        "paper_claim_authorized",
    )
    if any(result["authorizations"][key] for key in broad_keys):
        raise ValueError("N4 improperly authorized a broad claim")

    metrics = _read_csv(output_dir / "metrics.csv")
    pairs = _read_csv(output_dir / "pair_diagnostics.csv")
    costs = _read_csv(output_dir / "cost_ledger.csv")
    if len(metrics) != 32 or len(pairs) != 16 or len(costs) != 96 + escalation_count:
        raise ValueError("N4 CSV row counts drifted")
    if sum(int(row["step_count"]) == 2048 for row in costs) != escalation_count:
        raise ValueError("N4 H2048 cost rows violate the escalation rule")
    query_sum = sum(int(row["total_logical_point_queries"]) for row in costs)
    if query_sum != int(result["total_logical_point_queries"]):
        raise ValueError("N4 logical-query total drifted")
    if _read_json(output_dir / "config_snapshot.json") != config:
        raise ValueError("N4 config snapshot drifted")

    report = {
        "schema": "n2-pvgr-n4-evaluator-convergence-validation-1.0",
        "valid": True,
        "machine_decision": expected_decision,
        "h1024_pass_count": h1024_pass_count,
        "h2048_escalation_count": escalation_count,
        "final_reference_authorized_count": final_count,
        "manifest_file_count": len(manifest["files"]),
        "claim_boundary_verified": True,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _read_json(args.config.resolve())
    output = (args.output or ROOT / config["formal_output"]).resolve()
    print(json.dumps(validate(args.config.resolve(), output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
