#!/usr/bin/env python3
"""Validate the formal N4.1 execution-amendment result bundle."""

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

import demo_t16_operator.run_n2_pvgr_n4_1_execution_amendment as amendment  # noqa: E402
import site_tools.validate_n2_pvgr_n4_evaluator_convergence as base_validator  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n4_1_execution_amendment_preregistered_v1.json"
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


def validate(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config, scientific, parent, source = amendment._load_contract(config_path)
    base_attestation = amendment._validate_contract(config, scientific, parent, source)
    attestation = amendment._validate_preregistration(config, config_path)
    if output_dir.resolve() != (ROOT / config["formal_output"]).resolve():
        raise ValueError("N4.1 validator output path drifted")
    result = _read_json(output_dir / "result.json")
    manifest = _read_json(output_dir / "manifest.json")
    if result.get("schema") != "n2-pvgr-n4-1-evaluator-convergence-result-1.0":
        raise ValueError("N4.1 result schema drifted")
    if manifest.get("schema") != ("n2-pvgr-n4-1-evaluator-convergence-manifest-1.0"):
        raise ValueError("N4.1 manifest schema drifted")
    if result["protocol_commit"] != attestation["protocol_commit"]:
        raise ValueError("N4.1 result protocol commit drifted")
    if result["base_protocol_commit"] != base_attestation["protocol_commit"]:
        raise ValueError("N4.1 result base protocol commit drifted")
    if manifest["protocol_commit"] != attestation["protocol_commit"]:
        raise ValueError("N4.1 manifest protocol commit drifted")
    if manifest["base_protocol_commit"] != base_attestation["protocol_commit"]:
        raise ValueError("N4.1 manifest base protocol commit drifted")
    for entry in manifest["files"].values():
        path = ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"N4.1 manifest file missing: {path}")
        if (
            path.stat().st_size != int(entry["bytes"])
            or _sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"N4.1 manifest hash or size mismatch: {path}")

    if result["execution_amendment"] != config["implementation_amendment"]:
        raise ValueError("N4.1 result execution amendment drifted")
    if (
        result["known_post_preregistration_observations"]
        != config["known_post_preregistration_observations"]
    ):
        raise ValueError("N4.1 known-observation disclosure drifted")
    cells = result["cells"]
    if len(cells) != 32 or len({cell["cell_id"] for cell in cells}) != 32:
        raise ValueError("N4.1 result must contain 32 unique cells")
    if sum(cell["role"] == "n3_failure" for cell in cells) != 16:
        raise ValueError("N4.1 failure-role count drifted")
    if sum(cell["role"] == "matched_control" for cell in cells) != 16:
        raise ValueError("N4.1 control-role count drifted")
    gates = scientific["convergence_gates"]
    for cell in cells:
        h1024 = base_validator._recompute_h1024_gates(cell, gates)
        if h1024 != cell["h1024_gates"]:
            raise ValueError(f"N4.1 H1024 gate mismatch: {cell['cell_id']}")
        h1024_pass = all(h1024.values())
        if bool(cell["h1024_all_gates_pass"]) != h1024_pass:
            raise ValueError(f"N4.1 H1024 aggregate mismatch: {cell['cell_id']}")
        if bool(cell["requires_h2048_escalation"]) != (not h1024_pass):
            raise ValueError(f"N4.1 escalation rule mismatch: {cell['cell_id']}")
        if h1024_pass:
            if cell["d1024_to_d2048"] is not None or cell["h2048_gates"] is not None:
                raise ValueError(f"N4.1 opened forbidden H2048: {cell['cell_id']}")
            expected_final = True
        else:
            h2048 = base_validator._recompute_h2048_gates(cell, gates)
            if h2048 != cell["h2048_gates"]:
                raise ValueError(f"N4.1 H2048 gate mismatch: {cell['cell_id']}")
            if bool(cell["h2048_all_gates_pass"]) != all(h2048.values()):
                raise ValueError(f"N4.1 H2048 aggregate mismatch: {cell['cell_id']}")
            expected_final = all(h2048.values())
        if bool(cell["final_cellwise_reference_authorized"]) != expected_final:
            raise ValueError(f"N4.1 final cell mismatch: {cell['cell_id']}")

    h1024_pass_count = sum(cell["h1024_all_gates_pass"] for cell in cells)
    escalation_count = sum(cell["requires_h2048_escalation"] for cell in cells)
    final_count = sum(cell["final_cellwise_reference_authorized"] for cell in cells)
    expected_counts = {
        "physical_cell_count": 32,
        "n3_failure_count": 16,
        "matched_control_count": 16,
        "h1024_pass_count": h1024_pass_count,
        "h2048_escalation_count": escalation_count,
        "final_reference_authorized_count": final_count,
        "level_evaluation_count": 96 + escalation_count,
    }
    if result["counts"] != expected_counts:
        raise ValueError("N4.1 result count ledger drifted")
    expected_decision = (
        "EVALUATOR_CONVERGENCE_CLEARED_FOR_TINY_FIELD_JVP_VJP_GATE"
        if final_count == 32
        else "FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED"
    )
    if result["machine_decision"] != expected_decision:
        raise ValueError("N4.1 machine decision drifted")
    if result["authorizations"]["uniform_h1024_reference_authorized"] != (
        h1024_pass_count == 32
    ):
        raise ValueError("N4.1 uniform-reference authorization drifted")
    if result["authorizations"]["tiny_field_jvp_vjp_gate_authorized"] != (
        final_count == 32
    ):
        raise ValueError("N4.1 tiny-gate authorization drifted")
    broad_keys = (
        "reserved_audit_authorized",
        "real_data_authorized",
        "three_dimensional_reconstruction_authorized",
        "neural_operator_superiority_authorized",
        "paper_claim_authorized",
    )
    if any(result["authorizations"][key] for key in broad_keys):
        raise ValueError("N4.1 improperly authorized a broad claim")

    metrics = _read_csv(output_dir / "metrics.csv")
    pairs = _read_csv(output_dir / "pair_diagnostics.csv")
    costs = _read_csv(output_dir / "cost_ledger.csv")
    if len(metrics) != 32 or len(pairs) != 16 or len(costs) != 96 + escalation_count:
        raise ValueError("N4.1 CSV row counts drifted")
    if sum(int(row["step_count"]) == 2048 for row in costs) != escalation_count:
        raise ValueError("N4.1 H2048 cost rows violate escalation rule")
    query_sum = sum(int(row["total_logical_point_queries"]) for row in costs)
    if query_sum != int(result["total_logical_point_queries"]):
        raise ValueError("N4.1 logical-query total drifted")
    if _read_json(output_dir / "amendment_config_snapshot.json") != config:
        raise ValueError("N4.1 amendment config snapshot drifted")
    if _read_json(output_dir / "scientific_protocol_snapshot.json") != scientific:
        raise ValueError("N4.1 scientific protocol snapshot drifted")

    report = {
        "schema": "n2-pvgr-n4-1-evaluator-convergence-validation-1.0",
        "valid": True,
        "machine_decision": expected_decision,
        "h1024_pass_count": h1024_pass_count,
        "h2048_escalation_count": escalation_count,
        "final_reference_authorized_count": final_count,
        "manifest_file_count": len(manifest["files"]),
        "scientific_gate_change": False,
        "aborted_v1_checkpoint_reuse": False,
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
