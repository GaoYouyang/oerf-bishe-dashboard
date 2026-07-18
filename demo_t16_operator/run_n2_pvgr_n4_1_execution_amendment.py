#!/usr/bin/env python3
"""Run the N4.1 execution-only amendment over the frozen N4 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

try:
    from . import run_n2_pvgr_n4_evaluator_convergence as base
except ImportError:
    import run_n2_pvgr_n4_evaluator_convergence as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "demo_t16_operator/configs/"
    "n2_pvgr_n4_1_execution_amendment_preregistered_v1.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _git_text(*args: str) -> str:
    return _git(*args).stdout.decode("utf-8").strip()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _load_contract(
    amendment_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    amendment = _read_json(amendment_path)
    scientific = _read_json(_resolve(str(amendment["base_protocol_config"])))
    parent = _read_json(_resolve(str(scientific["parent_n3_config"])))
    source = _read_json(_resolve(str(scientific["source_config"])))
    return amendment, scientific, parent, source


def _validate_contract(
    amendment: dict[str, Any],
    scientific: dict[str, Any],
    parent: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    if amendment.get("schema") != (
        "n2-pvgr-n4-1-execution-amendment-preregistered-1.0"
    ):
        raise ValueError("N4.1 amendment schema drifted")
    if amendment.get("candidate_id") != "N2-PVGR-N4.1-ECA32":
        raise ValueError("N4.1 candidate identifier drifted")
    if amendment.get("status") != (
        "preregistered_execution_amendment_only_no_scientific_gate_change"
    ):
        raise ValueError("N4.1 amendment status drifted")
    if amendment.get("resume_policy") != (
        "only_hash_validated_level_checkpoints_from_the_N4_1_amendment"
    ):
        raise ValueError("N4.1 resume policy drifted")
    if not all(value is False for value in amendment["claim_authorizations"].values()):
        raise ValueError("N4.1 broad claim authorizations must remain false")
    expected_change = {
        "changed_behavior": (
            "compute_the_complete_H1024_gate_bundle_before_requesting_H2048_"
            "then_use_the_frozen_base_final_cell_decision"
        ),
        "scientific_sample_change": False,
        "threshold_change": False,
        "numerical_route_change": False,
        "metric_change": False,
        "plot_change": False,
        "stopping_rule_change": False,
        "reuse_aborted_v1_checkpoint": False,
    }
    if amendment.get("implementation_amendment") != expected_change:
        raise ValueError("N4.1 implementation amendment drifted")
    if tuple(amendment["known_post_preregistration_observations"]) != (
        "v1_first_failure_role_cell_reported_H1024_PASS",
        "v1_first_matched_control_reached_the_H2048_escalation_branch",
        "no_v1_H2048_result_or_final_result_bundle_exists",
    ):
        raise ValueError("N4.1 known-observation disclosure drifted")
    base_path = _resolve(str(amendment["base_protocol_config"]))
    base_attestation_path = _resolve(str(amendment["base_protocol_attestation"]))
    if base_path != base.DEFAULT_CONFIG.resolve():
        raise ValueError("N4.1 base protocol path drifted")
    if (
        base_attestation_path
        != _resolve(str(scientific["pre_registration_attestation"])).resolve()
    ):
        raise ValueError("N4.1 base attestation path drifted")
    base._validate_contract(scientific, parent, source)
    return base._validate_preregistration(scientific, base_path)


def _validate_preregistration(
    amendment: dict[str, Any], amendment_path: Path
) -> dict[str, Any]:
    attestation_path = _resolve(str(amendment["pre_registration_attestation"]))
    if not attestation_path.is_file():
        raise FileNotFoundError("committed N4.1 attestation is missing")
    attestation = _read_json(attestation_path)
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N4.1 attestation does not prove formal output absence")
    if attestation.get("formal_work_output_absent_at_creation") is not True:
        raise ValueError("N4.1 attestation does not prove formal work absence")
    if attestation.get("config_sha256") != _sha256(amendment_path):
        raise ValueError("N4.1 amendment config does not match attestation")
    protocol_commit = str(attestation["protocol_commit"])
    if _git(
        "merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False
    ).returncode:
        raise ValueError("N4.1 protocol commit is not an ancestor of HEAD")
    if set(attestation["attested_files"]) != set(amendment["attested_files"]):
        raise ValueError("N4.1 attested file key set drifted")
    for key, entry in attestation["attested_files"].items():
        expected = str(amendment["attested_files"][key])
        if entry["path"] != expected:
            raise ValueError(f"N4.1 attested path drifted: {key}")
        path = _resolve(expected)
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"N4.1 current file hash drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{expected}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"N4.1 protocol file hash drifted: {key}")
    if _git(
        "ls-files", "--error-unmatch", _relative(attestation_path), check=False
    ).returncode:
        raise ValueError("N4.1 attestation is not committed")
    paths = [
        _relative(attestation_path),
        *(str(value) for value in amendment["attested_files"].values()),
    ]
    if _git("status", "--porcelain", "--", *paths).stdout.strip():
        raise ValueError("N4.1 preregistered files have uncommitted changes")
    return attestation


def _h1024_predecision(
    cell: dict[str, Any],
    levels: dict[int, dict[str, Any]],
    parent_row: dict[str, str],
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Compute inherited H1024 gates without requiring an H2048 payload."""

    d256_512 = base._adjacent_metrics(levels[256], levels[512])
    d512_1024 = base._adjacent_metrics(levels[512], levels[1024])
    output = base._contraction(
        d256_512["output_relative_l2"], d512_1024["output_relative_l2"], gates
    )
    residual = base._contraction(
        d256_512["matched_residual_relative_l2"],
        d512_1024["matched_residual_relative_l2"],
        gates,
    )
    parent_reproduction = {
        "output_absolute_difference": abs(
            d256_512["output_relative_l2"]
            - float(parent_row["high256_to_high512_output_relative_l2"])
        ),
        "matched_residual_absolute_difference": abs(
            d256_512["matched_residual_relative_l2"]
            - float(parent_row["matched_residual_256_to_512_relative_l2"])
        ),
    }
    tolerance = float(gates["maximum_parent_N3_metric_absolute_difference"])
    gate_bundle = {
        "output_absolute_gate_met": d512_1024["output_relative_l2"]
        <= float(gates["maximum_h512_to_h1024_output_relative_l2"]),
        "output_contraction_gate_met": output["contraction_gate_met"],
        "matched_residual_absolute_gate_met": d512_1024["matched_residual_relative_l2"]
        <= float(gates["maximum_h512_to_h1024_matched_residual_relative_l2"]),
        "matched_residual_contraction_gate_met": residual["contraction_gate_met"],
        "parent_output_reproduction_gate_met": parent_reproduction[
            "output_absolute_difference"
        ]
        <= tolerance,
        "parent_matched_residual_reproduction_gate_met": parent_reproduction[
            "matched_residual_absolute_difference"
        ]
        <= tolerance,
        **base._endpoint_integrity_gates(levels[512], levels[1024], gates),
    }
    return {
        "cell_id": cell["cell_id"],
        "d256_to_d512": d256_512,
        "d512_to_d1024": d512_1024,
        "parent_reproduction": parent_reproduction,
        "h1024_gates": gate_bundle,
        "h1024_all_gates_pass": all(gate_bundle.values()),
        "requires_h2048_escalation": not all(gate_bundle.values()),
    }


def _summary_markdown(result: dict[str, Any]) -> str:
    counts = result["counts"]
    return f"""# N2-PVGR N4.1 evaluator convergence audit

## Machine decision

`{result['machine_decision']}`

## Execution amendment

N4.1 inherits the complete frozen N4 v1 scientific protocol. It changes only the control-flow order used to request H2048 after the complete H1024 gate bundle has been computed. No v1 checkpoint was reused.

## Counts

- H1024 pass: {counts['h1024_pass_count']} / 32
- H2048 escalations: {counts['h2048_escalation_count']} / 32
- Final cellwise references authorized: {counts['final_reference_authorized_count']} / 32
- Uniform H1024 reference authorized: {str(result['authorizations']['uniform_h1024_reference_authorized']).lower()}
- Tiny field-JVP/VJP gate authorized: {str(result['authorizations']['tiny_field_jvp_vjp_gate_authorized']).lower()}

## Claim boundary

This is a post-N3 selected synthetic evaluator audit with a disclosed execution amendment. It cannot establish algorithm superiority, real-BOST validity, three-dimensional reconstruction quality, novelty, or generalization.
"""


def run(
    amendment_path: Path,
    output_dir: Path,
    work_dir: Path,
    *,
    resume: bool,
    enforce_formal_output: bool,
) -> dict[str, Any]:
    amendment_path = amendment_path.resolve()
    amendment, scientific, parent_config, source_raw = _load_contract(amendment_path)
    base_attestation = _validate_contract(
        amendment, scientific, parent_config, source_raw
    )
    attestation = _validate_preregistration(amendment, amendment_path)
    if enforce_formal_output:
        if output_dir != _resolve(str(amendment["formal_output"])).resolve():
            raise ValueError("N4.1 formal output path drifted")
        if work_dir != _resolve(str(amendment["formal_work_output"])).resolve():
            raise ValueError("N4.1 formal work path drifted")
    if output_dir.exists():
        raise FileExistsError(f"N4.1 final output already exists: {output_dir}")
    source = base._source_for_run(source_raw, scientific)
    cells = base.expand_audit_cells(scientific, parent_config)
    parent = base._parent_sentinel_map(scientific)
    gates = scientific["convergence_gates"]
    preregistration_sha256 = _sha256(amendment_path)
    decisions: list[dict[str, Any]] = []
    all_cost_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, cell in enumerate(cells, start=1):
        levels: dict[int, dict[str, Any]] = {}
        for step_count in scientific["base_step_counts"]:
            levels[int(step_count)] = base._load_or_run_level(
                cell,
                source,
                step_count=int(step_count),
                work_dir=work_dir,
                preregistration_sha256=preregistration_sha256,
                resume=resume,
            )
        key = base._cell_key(cell["case_id"], cell["dimensionless_stress_multiplier"])
        preliminary = _h1024_predecision(cell, levels, parent[key], gates)
        if preliminary["requires_h2048_escalation"]:
            levels[2048] = base._load_or_run_level(
                cell,
                source,
                step_count=2048,
                work_dir=work_dir,
                preregistration_sha256=preregistration_sha256,
                resume=resume,
            )
        decision = base._cell_decision(cell, levels, parent[key], gates)
        if decision["h1024_gates"] != preliminary["h1024_gates"]:
            raise RuntimeError("N4.1 predecision drifted from frozen final decision")
        if (
            decision["requires_h2048_escalation"]
            != preliminary["requires_h2048_escalation"]
        ):
            raise RuntimeError("N4.1 escalation request drifted from frozen decision")
        decisions.append(decision)
        all_cost_rows.extend(base._cost_rows(cell, levels))
        print(
            f"N4.1 cell {index:02d}/32 {cell['cell_id']} "
            f"H1024={'PASS' if decision['h1024_all_gates_pass'] else 'FAIL'} "
            f"final={'PASS' if decision['final_cellwise_reference_authorized'] else 'FAIL'}",
            flush=True,
        )

    h1024_pass = sum(item["h1024_all_gates_pass"] for item in decisions)
    escalation_count = sum(item["requires_h2048_escalation"] for item in decisions)
    final_pass = sum(item["final_cellwise_reference_authorized"] for item in decisions)
    cellwise_reference = final_pass == 32
    machine_decision = (
        "EVALUATOR_CONVERGENCE_CLEARED_FOR_TINY_FIELD_JVP_VJP_GATE"
        if cellwise_reference
        else "FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED"
    )
    result = {
        "schema": "n2-pvgr-n4-1-evaluator-convergence-result-1.0",
        "candidate_id": amendment["candidate_id"],
        "protocol_commit": attestation["protocol_commit"],
        "base_protocol_commit": base_attestation["protocol_commit"],
        "run_head_commit": _git_text("rev-parse", "HEAD"),
        "machine_decision": machine_decision,
        "execution_amendment": amendment["implementation_amendment"],
        "known_post_preregistration_observations": amendment[
            "known_post_preregistration_observations"
        ],
        "counts": {
            "physical_cell_count": 32,
            "n3_failure_count": 16,
            "matched_control_count": 16,
            "h1024_pass_count": h1024_pass,
            "h2048_escalation_count": escalation_count,
            "final_reference_authorized_count": final_pass,
            "level_evaluation_count": len(all_cost_rows),
        },
        "authorizations": {
            "uniform_h1024_reference_authorized": h1024_pass == 32,
            "mixed_h1024_h2048_cellwise_reference_authorized": cellwise_reference,
            "tiny_field_jvp_vjp_gate_authorized": cellwise_reference,
            "reserved_audit_authorized": False,
            "real_data_authorized": False,
            "three_dimensional_reconstruction_authorized": False,
            "neural_operator_superiority_authorized": False,
            "paper_claim_authorized": False,
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "total_logical_point_queries": sum(
            int(row["total_logical_point_queries"]) for row in all_cost_rows
        ),
        "cells": decisions,
        "claim_boundary": (
            "Post-N3 selected synthetic evaluator evidence with a disclosed execution "
            "amendment; never algorithm, real-data, reconstruction, novelty, or "
            "generalization success."
        ),
        "figure": "n2_pvgr_n4_1_evaluator_convergence.png",
    }
    staging = base._prepare_staging(work_dir)
    base._atomic_json(staging / "result.json", result)
    base._write_csv(staging / "metrics.csv", base._metric_rows(decisions))
    base._write_csv(staging / "pair_diagnostics.csv", base._pair_rows(decisions))
    base._write_csv(staging / "cost_ledger.csv", all_cost_rows)
    (staging / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    (staging / "amendment_config_snapshot.json").write_text(
        json.dumps(amendment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (staging / "scientific_protocol_snapshot.json").write_text(
        json.dumps(scientific, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    base._plot(staging / result["figure"], decisions, all_cost_rows)
    manifest_inputs = {
        "result": staging / "result.json",
        "metrics": staging / "metrics.csv",
        "pairs": staging / "pair_diagnostics.csv",
        "cost": staging / "cost_ledger.csv",
        "summary": staging / "summary.md",
        "amendment_config_snapshot": staging / "amendment_config_snapshot.json",
        "scientific_protocol_snapshot": staging / "scientific_protocol_snapshot.json",
        "figure": staging / result["figure"],
    }
    manifest = {
        "schema": "n2-pvgr-n4-1-evaluator-convergence-manifest-1.0",
        "protocol_commit": attestation["protocol_commit"],
        "base_protocol_commit": base_attestation["protocol_commit"],
        "run_head_commit": result["run_head_commit"],
        "files": {
            key: {
                "path": f"{_relative(output_dir)}/{path.name}",
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in manifest_inputs.items()
        },
    }
    for key, relative in amendment["attested_files"].items():
        path = _resolve(str(relative))
        manifest["files"][f"attested_{key}"] = {
            "path": str(relative),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    base._atomic_json(staging / "manifest.json", manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-output", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    amendment_path = args.config.resolve()
    amendment, scientific, parent, source = _load_contract(amendment_path)
    base_attestation = _validate_contract(amendment, scientific, parent, source)
    attestation = _validate_preregistration(amendment, amendment_path)
    cells = base.expand_audit_cells(scientific, parent)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "protocol_commit": attestation["protocol_commit"],
                    "base_protocol_commit": base_attestation["protocol_commit"],
                    "physical_cells": len(cells),
                    "scientific_gate_change": False,
                    "aborted_v1_checkpoints_reused": False,
                    "formal_output_exists": _resolve(
                        amendment["formal_output"]
                    ).exists(),
                },
                indent=2,
            )
        )
        return 0
    output = (args.output or _resolve(amendment["formal_output"])).resolve()
    work = (args.work_output or _resolve(amendment["formal_work_output"])).resolve()
    result = run(
        amendment_path,
        output,
        work,
        resume=not args.no_resume,
        enforce_formal_output=True,
    )
    print(json.dumps({"machine_decision": result["machine_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
