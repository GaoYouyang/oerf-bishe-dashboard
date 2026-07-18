#!/usr/bin/env python3
"""Run the preregistered N4 curved-ray evaluator convergence audit.

N4 re-evaluates the 16 N3 sentinel failures and 16 same-field matched
controls at H256/H512/H1024. H2048 is opened only when a preregistered H1024
gate fails. This is an evaluator audit, not a candidate-algorithm experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from . import run_n2_pvgr_n3_grouped_factorial as n3
    from .field_dependent_ray import path_topology_diagnostics, relative_l2
except ImportError:
    import run_n2_pvgr_n3_grouped_factorial as n3
    from field_dependent_ray import path_topology_diagnostics, relative_l2


bridge = n3.bridge
ROOT = Path(__file__).resolve().parents[1]
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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid frozen CSV boolean: {value!r}")
    return value == "True"


def _cell_key(case_id: str, stress: float) -> tuple[str, float]:
    return str(case_id), float(stress)


def _parent_sentinel_map(
    config: dict[str, Any],
) -> dict[tuple[str, float], dict[str, str]]:
    rows = _read_csv(_resolve(str(config["parent_n3_sentinel"])))
    result = {
        _cell_key(row["case_id"], float(row["dimensionless_stress_multiplier"])): row
        for row in rows
    }
    if len(rows) != 96 or len(result) != 96:
        raise ValueError("parent N3 sentinel is not the frozen 96-cell table")
    return result


def _validate_contract(
    config: dict[str, Any],
    parent_config: dict[str, Any],
    source: dict[str, Any],
) -> None:
    if config.get("schema") != "n2-pvgr-n4-evaluator-convergence-preregistered-1.0":
        raise ValueError("N4 schema drifted")
    if config.get("candidate_id") != "N2-PVGR-N4-ECA32":
        raise ValueError("N4 identifier drifted")
    if config.get("status") != (
        "preregistered_evaluator_audit_only_no_algorithm_authorization"
    ):
        raise ValueError("N4 status drifted")
    if (config.get("device"), config.get("dtype")) != ("cpu", "float64"):
        raise ValueError("N4 requires CPU float64")
    if (source.get("device"), source.get("dtype")) != ("cpu", "float64"):
        raise ValueError("source requires CPU float64")
    if int(config["population_count"]) != 256:
        raise ValueError("N4 requires 256 common Sobol rays")
    if tuple(int(value) for value in config["base_step_counts"]) != (256, 512, 1024):
        raise ValueError("N4 base levels must be H256/H512/H1024")
    if int(config["escalation_step_count"]) != 2048:
        raise ValueError("N4 escalation must be H2048")
    if config.get("resume_policy") != (
        "only_hash_validated_level_checkpoints_from_the_same_preregistration"
    ):
        raise ValueError("N4 resume policy drifted")
    if not all(value is False for value in config["claim_authorizations"].values()):
        raise ValueError("all broad N4 claim authorizations must remain false")
    if tuple(config["reserved_audit_families_not_opened"]) != tuple(
        source["reserved_audit_families_not_opened"]
    ):
        raise ValueError("reserved-family declaration drifted")

    n3._validate_contract(parent_config, source)
    cases = {case["id"]: case for case in n3.expand_factorial_cases(parent_config)}
    parent = _parent_sentinel_map(config)
    frozen_failures = {
        key for key, row in parent.items() if not _bool(row["all_gates_pass"])
    }
    pairs = config["audit_pairs"]
    if len(pairs) != 16 or tuple(pair["id"] for pair in pairs) != tuple(
        f"p{index:02d}" for index in range(1, 17)
    ):
        raise ValueError("N4 pair identifiers are not the frozen p01-p16 list")

    declared_failures: set[tuple[str, float]] = set()
    declared_controls: set[tuple[str, float]] = set()
    for pair in pairs:
        failed = pair["failed_cell"]
        control = pair["control_cell"]
        failed_key = _cell_key(failed["case_id"], failed["stress"])
        control_key = _cell_key(control["case_id"], control["stress"])
        if failed_key not in parent or control_key not in parent:
            raise ValueError("N4 pair references a cell absent from N3")
        failed_row = parent[failed_key]
        control_row = parent[control_key]
        if _bool(failed_row["all_gates_pass"]):
            raise ValueError("declared N4 failed cell passed N3")
        if not _bool(control_row["all_gates_pass"]):
            raise ValueError("declared N4 matched control failed N3")
        gate = str(failed["n3_failed_gate"])
        if gate not in {
            "output_convergence_gate_met",
            "matched_residual_convergence_gate_met",
        } or _bool(failed_row[gate]):
            raise ValueError("declared N3 failure mechanism is not false")
        failed_case = cases[str(failed["case_id"])]
        control_case = cases[str(control["case_id"])]
        if failed_case["field_unit_id"] != control_case["field_unit_id"]:
            raise ValueError("matched control changed the field unit")
        if float(failed["stress"]) != float(control["stress"]):
            raise ValueError("matched control changed stress")
        changed = sum(
            failed_case[key] != control_case[key]
            for key in ("orientation_id", "aperture_id")
        )
        if changed != 1:
            raise ValueError("matched control must change exactly one geometry factor")
        declared_failures.add(failed_key)
        declared_controls.add(control_key)
    if declared_failures != frozen_failures or len(declared_controls) != 16:
        raise ValueError("N4 must contain all/only 16 N3 failures and 16 controls")
    if declared_failures.intersection(declared_controls):
        raise ValueError("a cell cannot be both N4 failure and control")

    gates = config["convergence_gates"]
    exact_gate_contract = {
        "scoreability_floor": 1e-12,
        "maximum_unscoreable_adjacent_difference": 1e-12,
        "maximum_h512_to_h1024_output_relative_l2": 2.5e-5,
        "maximum_h512_to_h1024_matched_residual_relative_l2": 2.5e-3,
        "maximum_adjacent_contraction_ratio": 0.5,
        "maximum_h1024_to_h2048_output_relative_l2": 1.25e-5,
        "maximum_h1024_to_h2048_matched_residual_relative_l2": 1.25e-3,
        "minimum_finite_ray_fraction": 1.0,
        "minimum_domain_margin": 0.0,
        "minimum_stencil_margin": 0.0,
        "maximum_direction_norm_error": 1e-12,
        "maximum_frustum_violation_count": 0,
        "maximum_parent_N3_metric_absolute_difference": 1e-12,
    }
    for key, expected in exact_gate_contract.items():
        if float(gates[key]) != expected:
            raise ValueError(f"N4 convergence gate drifted: {key}")
    for key in (
        "require_identical_adjacent_support_crossing_signature",
        "require_identical_adjacent_frustum_violation_signature",
    ):
        if gates.get(key) is not True:
            raise ValueError(f"N4 topology gate drifted: {key}")


def _validate_preregistration(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    if not attestation_path.is_file():
        raise FileNotFoundError("committed N4 preregistration attestation is missing")
    attestation = _read_json(attestation_path)
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N4 attestation does not prove formal output was absent")
    if attestation.get("formal_work_output_absent_at_creation") is not True:
        raise ValueError("N4 attestation does not prove checkpoints were absent")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("N4 config does not match its attestation")
    protocol_commit = str(attestation["protocol_commit"])
    if _git(
        "merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False
    ).returncode:
        raise ValueError("N4 protocol commit is not an ancestor of HEAD")
    if set(attestation["attested_files"]) != set(config["attested_files"]):
        raise ValueError("N4 attested file key set drifted")
    for key, entry in attestation["attested_files"].items():
        expected_path = str(config["attested_files"][key])
        if entry["path"] != expected_path:
            raise ValueError(f"N4 attested path drifted: {key}")
        path = _resolve(expected_path)
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"N4 current file hash drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{expected_path}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"N4 protocol commit hash drifted: {key}")
    tracked = _git(
        "ls-files", "--error-unmatch", _relative(attestation_path), check=False
    )
    if tracked.returncode:
        raise ValueError("N4 attestation is not committed")
    paths = [
        _relative(attestation_path),
        *(str(value) for value in config["attested_files"].values()),
    ]
    if _git("status", "--porcelain", "--", *paths).stdout.strip():
        raise ValueError("N4 preregistered files have uncommitted changes")
    return attestation


def expand_audit_cells(
    config: dict[str, Any], parent_config: dict[str, Any]
) -> list[dict[str, Any]]:
    cases = {case["id"]: case for case in n3.expand_factorial_cases(parent_config)}
    rows: list[dict[str, Any]] = []
    for pair in config["audit_pairs"]:
        failed_case = cases[pair["failed_cell"]["case_id"]]
        control_case = cases[pair["control_cell"]["case_id"]]
        contrast_factor = (
            "orientation"
            if failed_case["orientation_id"] != control_case["orientation_id"]
            else "aperture"
        )
        for role, entry in (
            ("n3_failure", pair["failed_cell"]),
            ("matched_control", pair["control_cell"]),
        ):
            case = copy.deepcopy(cases[entry["case_id"]])
            rows.append(
                {
                    **case,
                    "case_id": str(case["id"]),
                    "pair_id": str(pair["id"]),
                    "role": role,
                    "contrast_factor": contrast_factor,
                    "dimensionless_stress_multiplier": float(entry["stress"]),
                    "n3_failed_gate": entry.get("n3_failed_gate", "none"),
                    "cell_id": f"{case['id']}__stress_{float(entry['stress']):g}",
                }
            )
    if len(rows) != 32 or len({row["cell_id"] for row in rows}) != 32:
        raise ValueError("N4 expansion must produce 32 unique cells")
    return rows


def _source_for_run(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(source)
    adapted["population_count"] = int(config["population_count"])
    return adapted


def _level_path(work_dir: Path, cell_id: str, step_count: int) -> Path:
    return work_dir / "levels" / cell_id / f"H{step_count}.json"


def _level_metadata(cell: dict[str, Any], step_count: int) -> dict[str, Any]:
    return {
        key: cell[key]
        for key in (
            "cell_id",
            "case_id",
            "field_unit_id",
            "family_id",
            "phantom_family",
            "phantom_seed",
            "orientation_id",
            "aperture_id",
            "pair_id",
            "role",
            "contrast_factor",
            "dimensionless_stress_multiplier",
            "n3_failed_gate",
        )
    } | {"step_count": int(step_count)}


def _validate_level_payload(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    preregistration_sha256: str,
) -> None:
    if payload.get("schema") != "n2-pvgr-n4-level-checkpoint-1.0":
        raise ValueError("N4 checkpoint schema drifted")
    if payload.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("N4 checkpoint preregistration hash drifted")
    if payload.get("metadata") != metadata:
        raise ValueError("N4 checkpoint metadata drifted")
    for key in ("high_output_uv", "matched_residual_uv"):
        array = np.asarray(payload[key], dtype=np.float64)
        if array.shape != (256, 2) or not np.all(np.isfinite(array)):
            raise ValueError(f"N4 checkpoint tensor invalid: {key}")
        if payload[f"{key}_sha256"] != _array_sha256(array):
            raise ValueError(f"N4 checkpoint tensor hash drifted: {key}")


def _run_level(
    cell: dict[str, Any],
    source: dict[str, Any],
    *,
    step_count: int,
) -> dict[str, Any]:
    execution_case = n3._execution_case(cell)
    values, states, rig = bridge._build_case_context(execution_case, source)
    scale = float(source["base_refractivity_scale"]) * float(
        cell["dimensionless_stress_multiplier"]
    )
    delta = float(source["difference_step"])
    started = time.perf_counter()
    high, trace = bridge._high_route(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=int(step_count),
        create_graph=False,
    )
    certificate = source["certificate"]
    straight = bridge.build_straight_path_state(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=int(step_count),
        frustum_half_width_u=float(certificate["frustum_half_width_u"]),
        frustum_half_width_v=float(certificate["frustum_half_width_v"]),
    )
    elapsed = time.perf_counter() - started
    matched = high.detach() - straight.projected_outputs.detach()
    support_threshold = float(
        certificate["support_threshold_fraction_of_grid_peak"]
    ) * float(torch.max(torch.abs(values.detach())))
    topology = path_topology_diagnostics(
        values,
        trace,
        support_threshold=support_threshold,
        frustum_half_width_u=float(certificate["frustum_half_width_u"]),
        frustum_half_width_v=float(certificate["frustum_half_width_v"]),
    )
    finite_per_ray = (
        torch.all(torch.isfinite(high.detach()), dim=-1)
        & torch.all(torch.isfinite(straight.projected_outputs.detach()), dim=-1)
        & torch.all(torch.isfinite(trace.positions.detach()), dim=(1, 2))
        & torch.all(torch.isfinite(trace.directions.detach()), dim=(1, 2))
    )
    high_array = high.detach().cpu().numpy()
    matched_array = matched.detach().cpu().numpy()
    return {
        "high_output_uv": high_array.tolist(),
        "high_output_uv_sha256": _array_sha256(high_array),
        "matched_residual_uv": matched_array.tolist(),
        "matched_residual_uv_sha256": _array_sha256(matched_array),
        "diagnostics": {
            "finite_ray_fraction": float(torch.mean(finite_per_ray.to(torch.float64))),
            "minimum_domain_margin": float(trace.minimum_domain_margin),
            "minimum_stencil_margin": float(trace.minimum_stencil_margin),
            "maximum_direction_norm_error": float(trace.maximum_direction_norm_error),
            "support_crossings_per_ray": list(topology.support_crossings_per_ray),
            "frustum_violations_per_ray": list(topology.frustum_violations_per_ray),
            "frustum_violation_count": int(sum(topology.frustum_violations_per_ray)),
            "minimum_frustum_margin": float(topology.minimum_frustum_margin),
        },
        "cost": {
            "wall_seconds": float(elapsed),
            "curved_high_logical_point_queries": 35 * len(states) * int(step_count),
            "straight_logical_point_queries": int(
                straight.query_accounting.total_field_point_queries
            ),
            "total_logical_point_queries": 42 * len(states) * int(step_count),
        },
    }


def _load_or_run_level(
    cell: dict[str, Any],
    source: dict[str, Any],
    *,
    step_count: int,
    work_dir: Path,
    preregistration_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    metadata = _level_metadata(cell, step_count)
    path = _level_path(work_dir, cell["cell_id"], step_count)
    if path.is_file():
        if not resume:
            raise FileExistsError(f"N4 checkpoint exists with resume disabled: {path}")
        payload = _read_json(path)
        _validate_level_payload(payload, metadata, preregistration_sha256)
        return payload
    numerical = _run_level(cell, source, step_count=step_count)
    payload = {
        "schema": "n2-pvgr-n4-level-checkpoint-1.0",
        "preregistration_sha256": preregistration_sha256,
        "metadata": metadata,
        **numerical,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)
    return payload


def _tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    return torch.tensor(payload[key], dtype=torch.float64)


def _adjacent_metrics(lower: dict[str, Any], upper: dict[str, Any]) -> dict[str, float]:
    return {
        "output_relative_l2": relative_l2(
            _tensor(lower, "high_output_uv"), _tensor(upper, "high_output_uv")
        ),
        "matched_residual_relative_l2": relative_l2(
            _tensor(lower, "matched_residual_uv"),
            _tensor(upper, "matched_residual_uv"),
        ),
    }


def _contraction(
    previous: float,
    current: float,
    gates: dict[str, Any],
) -> dict[str, Any]:
    floor = float(gates["scoreability_floor"])
    scoreable = float(previous) >= floor
    ratio = float(current) / max(float(previous), floor)
    contraction_gate = (
        ratio <= float(gates["maximum_adjacent_contraction_ratio"])
        if scoreable
        else float(current) <= float(gates["maximum_unscoreable_adjacent_difference"])
    )
    empirical_order = (
        math.log2(float(previous) / float(current))
        if float(previous) >= floor and float(current) >= floor
        else None
    )
    return {
        "previous_difference": float(previous),
        "current_difference": float(current),
        "scoreable": bool(scoreable),
        "contraction_ratio": float(ratio),
        "empirical_order": empirical_order,
        "contraction_gate_met": bool(contraction_gate),
    }


def _endpoint_integrity_gates(
    lower: dict[str, Any],
    upper: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, bool]:
    diagnostics = upper["diagnostics"]
    lower_diagnostics = lower["diagnostics"]
    return {
        "finite_ray_gate_met": float(diagnostics["finite_ray_fraction"])
        >= float(gates["minimum_finite_ray_fraction"]),
        "domain_margin_gate_met": float(diagnostics["minimum_domain_margin"])
        >= float(gates["minimum_domain_margin"]),
        "stencil_margin_gate_met": float(diagnostics["minimum_stencil_margin"])
        >= float(gates["minimum_stencil_margin"]),
        "direction_norm_gate_met": float(diagnostics["maximum_direction_norm_error"])
        <= float(gates["maximum_direction_norm_error"]),
        "support_topology_gate_met": diagnostics["support_crossings_per_ray"]
        == lower_diagnostics["support_crossings_per_ray"],
        "frustum_topology_gate_met": diagnostics["frustum_violations_per_ray"]
        == lower_diagnostics["frustum_violations_per_ray"],
        "frustum_violation_gate_met": int(diagnostics["frustum_violation_count"])
        <= int(gates["maximum_frustum_violation_count"]),
    }


def _cell_decision(
    cell: dict[str, Any],
    levels: dict[int, dict[str, Any]],
    parent_row: dict[str, str],
    gates: dict[str, Any],
) -> dict[str, Any]:
    d256_512 = _adjacent_metrics(levels[256], levels[512])
    d512_1024 = _adjacent_metrics(levels[512], levels[1024])
    output_contraction = _contraction(
        d256_512["output_relative_l2"], d512_1024["output_relative_l2"], gates
    )
    residual_contraction = _contraction(
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
    h1024_gates = {
        "output_absolute_gate_met": d512_1024["output_relative_l2"]
        <= float(gates["maximum_h512_to_h1024_output_relative_l2"]),
        "output_contraction_gate_met": output_contraction["contraction_gate_met"],
        "matched_residual_absolute_gate_met": d512_1024["matched_residual_relative_l2"]
        <= float(gates["maximum_h512_to_h1024_matched_residual_relative_l2"]),
        "matched_residual_contraction_gate_met": residual_contraction[
            "contraction_gate_met"
        ],
        "parent_output_reproduction_gate_met": parent_reproduction[
            "output_absolute_difference"
        ]
        <= tolerance,
        "parent_matched_residual_reproduction_gate_met": parent_reproduction[
            "matched_residual_absolute_difference"
        ]
        <= tolerance,
        **_endpoint_integrity_gates(levels[512], levels[1024], gates),
    }
    requires_escalation = not all(h1024_gates.values())
    decision: dict[str, Any] = {
        **{
            key: cell[key] for key in _level_metadata(cell, 1024) if key != "step_count"
        },
        "parent_n3_all_gates_pass": _bool(parent_row["all_gates_pass"]),
        "parent_reproduction": parent_reproduction,
        "d256_to_d512": d256_512,
        "d512_to_d1024": d512_1024,
        "h1024_output_contraction": output_contraction,
        "h1024_matched_residual_contraction": residual_contraction,
        "h512_endpoint_diagnostics": levels[512]["diagnostics"],
        "h1024_endpoint_diagnostics": levels[1024]["diagnostics"],
        "h1024_gates": h1024_gates,
        "h1024_all_gates_pass": all(h1024_gates.values()),
        "requires_h2048_escalation": requires_escalation,
    }
    if not requires_escalation:
        decision.update(
            {
                "d1024_to_d2048": None,
                "h2048_gates": None,
                "h2048_all_gates_pass": None,
                "final_reference_step_count": 1024,
                "final_cellwise_reference_authorized": True,
            }
        )
        return decision
    if 2048 not in levels:
        raise ValueError("H2048 escalation payload is missing")
    d1024_2048 = _adjacent_metrics(levels[1024], levels[2048])
    output_2048_contraction = _contraction(
        d512_1024["output_relative_l2"], d1024_2048["output_relative_l2"], gates
    )
    residual_2048_contraction = _contraction(
        d512_1024["matched_residual_relative_l2"],
        d1024_2048["matched_residual_relative_l2"],
        gates,
    )
    h2048_gates = {
        "output_absolute_gate_met": d1024_2048["output_relative_l2"]
        <= float(gates["maximum_h1024_to_h2048_output_relative_l2"]),
        "output_contraction_gate_met": output_2048_contraction["contraction_gate_met"],
        "matched_residual_absolute_gate_met": d1024_2048["matched_residual_relative_l2"]
        <= float(gates["maximum_h1024_to_h2048_matched_residual_relative_l2"]),
        "matched_residual_contraction_gate_met": residual_2048_contraction[
            "contraction_gate_met"
        ],
        **_endpoint_integrity_gates(levels[1024], levels[2048], gates),
    }
    decision.update(
        {
            "d1024_to_d2048": d1024_2048,
            "h2048_output_contraction": output_2048_contraction,
            "h2048_matched_residual_contraction": residual_2048_contraction,
            "h2048_endpoint_diagnostics": levels[2048]["diagnostics"],
            "h2048_gates": h2048_gates,
            "h2048_all_gates_pass": all(h2048_gates.values()),
            "final_reference_step_count": 2048,
            "final_cellwise_reference_authorized": all(h2048_gates.values()),
        }
    )
    return decision


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key)) for key in columns} for row in rows
        )


def _metric_rows(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in decisions:
        rows.append(
            {
                "cell_id": item["cell_id"],
                "pair_id": item["pair_id"],
                "role": item["role"],
                "family_id": item["family_id"],
                "phantom_seed": item["phantom_seed"],
                "orientation_id": item["orientation_id"],
                "aperture_id": item["aperture_id"],
                "dimensionless_stress_multiplier": item[
                    "dimensionless_stress_multiplier"
                ],
                "n3_failed_gate": item["n3_failed_gate"],
                "d256_512_output_relative_l2": item["d256_to_d512"][
                    "output_relative_l2"
                ],
                "d512_1024_output_relative_l2": item["d512_to_d1024"][
                    "output_relative_l2"
                ],
                "d256_512_matched_residual_relative_l2": item["d256_to_d512"][
                    "matched_residual_relative_l2"
                ],
                "d512_1024_matched_residual_relative_l2": item["d512_to_d1024"][
                    "matched_residual_relative_l2"
                ],
                "h1024_output_contraction_ratio": item["h1024_output_contraction"][
                    "contraction_ratio"
                ],
                "h1024_matched_residual_contraction_ratio": item[
                    "h1024_matched_residual_contraction"
                ]["contraction_ratio"],
                "h1024_all_gates_pass": item["h1024_all_gates_pass"],
                "requires_h2048_escalation": item["requires_h2048_escalation"],
                "d1024_2048_output_relative_l2": (
                    item["d1024_to_d2048"]["output_relative_l2"]
                    if item["d1024_to_d2048"]
                    else None
                ),
                "d1024_2048_matched_residual_relative_l2": (
                    item["d1024_to_d2048"]["matched_residual_relative_l2"]
                    if item["d1024_to_d2048"]
                    else None
                ),
                "h2048_all_gates_pass": item["h2048_all_gates_pass"],
                "final_reference_step_count": item["final_reference_step_count"],
                "final_cellwise_reference_authorized": item[
                    "final_cellwise_reference_authorized"
                ],
                "h1024_gates": item["h1024_gates"],
                "h2048_gates": item["h2048_gates"],
            }
        )
    return rows


def _pair_rows(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for item in decisions:
        by_pair.setdefault(item["pair_id"], {})[item["role"]] = item
    rows = []
    for pair_id, values in sorted(by_pair.items()):
        failure = values["n3_failure"]
        control = values["matched_control"]
        rows.append(
            {
                "pair_id": pair_id,
                "contrast_factor": failure["contrast_factor"],
                "failure_cell_id": failure["cell_id"],
                "control_cell_id": control["cell_id"],
                "failure_n3_gate": failure["n3_failed_gate"],
                "failure_h1024_pass": failure["h1024_all_gates_pass"],
                "control_h1024_pass": control["h1024_all_gates_pass"],
                "failure_escalated": failure["requires_h2048_escalation"],
                "control_escalated": control["requires_h2048_escalation"],
                "failure_final_authorized": failure[
                    "final_cellwise_reference_authorized"
                ],
                "control_final_authorized": control[
                    "final_cellwise_reference_authorized"
                ],
                "failure_output_contraction_ratio": failure["h1024_output_contraction"][
                    "contraction_ratio"
                ],
                "control_output_contraction_ratio": control["h1024_output_contraction"][
                    "contraction_ratio"
                ],
                "failure_matched_contraction_ratio": failure[
                    "h1024_matched_residual_contraction"
                ]["contraction_ratio"],
                "control_matched_contraction_ratio": control[
                    "h1024_matched_residual_contraction"
                ]["contraction_ratio"],
            }
        )
    return rows


def _cost_rows(
    cell: dict[str, Any], levels: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": cell["cell_id"],
            "pair_id": cell["pair_id"],
            "role": cell["role"],
            "step_count": step_count,
            **payload["cost"],
        }
        for step_count, payload in sorted(levels.items())
    ]


def _plot(
    path: Path, decisions: list[dict[str, Any]], cost_rows: list[dict[str, Any]]
) -> None:
    failure = [item for item in decisions if item["role"] == "n3_failure"]
    controls = [item for item in decisions if item["role"] == "matched_control"]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for values, label, color, marker in (
        (failure, "N3 failure", "#b33f40", "o"),
        (controls, "matched control", "#286f6b", "s"),
    ):
        axes[0].scatter(
            [item["h1024_output_contraction"]["contraction_ratio"] for item in values],
            [
                item["h1024_matched_residual_contraction"]["contraction_ratio"]
                for item in values
            ],
            label=label,
            color=color,
            marker=marker,
            alpha=0.85,
        )
    axes[0].axvline(0.5, color="#555", linestyle="--", linewidth=1)
    axes[0].axhline(0.5, color="#555", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="output contraction",
        ylabel="matched-residual contraction",
        title="H512 to H1024",
    )
    axes[0].legend(frameon=False)

    counts = {
        "H1024 pass": sum(item["h1024_all_gates_pass"] for item in decisions),
        "H2048 escalated": sum(item["requires_h2048_escalation"] for item in decisions),
        "final authorized": sum(
            item["final_cellwise_reference_authorized"] for item in decisions
        ),
    }
    axes[1].bar(counts, counts.values(), color=["#286f6b", "#c58b2a", "#4267a8"])
    axes[1].set_ylim(0, 32)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Fail-closed cell counts", ylabel="cells")

    by_step: dict[int, list[float]] = {}
    for row in cost_rows:
        by_step.setdefault(int(row["step_count"]), []).append(
            float(row["wall_seconds"])
        )
    steps = sorted(by_step)
    axes[2].plot(
        steps,
        [np.median(by_step[step]) for step in steps],
        marker="o",
        color="#5c4b8a",
    )
    axes[2].set(
        xlabel="RK4 steps H",
        ylabel="median wall seconds",
        title="Observed evaluator cost",
    )
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _summary_markdown(result: dict[str, Any]) -> str:
    counts = result["counts"]
    decision = result["machine_decision"]
    return f"""# N2-PVGR N4 evaluator convergence audit

## Machine decision

`{decision}`

## What was run

- 16 N3 sentinel failures and 16 same-field, same-stress matched controls.
- H256, H512 and H1024 for every cell; H2048 only after a preregistered H1024 failure.
- Curved-ray output, matched curved-minus-straight residual, contraction, topology, domain margins, finite rays, wall time and logical point queries.

## Counts

- H1024 pass: {counts['h1024_pass_count']} / 32
- H2048 escalations: {counts['h2048_escalation_count']} / 32
- Final cellwise references authorized: {counts['final_reference_authorized_count']} / 32
- Uniform H1024 reference authorized: {str(result['authorizations']['uniform_h1024_reference_authorized']).lower()}
- Tiny field-JVP/VJP gate authorized: {str(result['authorizations']['tiny_field_jvp_vjp_gate_authorized']).lower()}

## Claim boundary

This selected synthetic audit can validate or reject the numerical evaluator used by a later tiny reconstruction gate. It cannot establish a new algorithm, neural-operator superiority, real-BOST validity, three-dimensional reconstruction quality, novelty, or generalization.
"""


def _prepare_staging(work_dir: Path) -> Path:
    path = work_dir / "final_artifacts_staging"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def run(
    config_path: Path,
    output_dir: Path,
    work_dir: Path,
    *,
    resume: bool,
    enforce_formal_output: bool,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    parent_config = _read_json(_resolve(str(config["parent_n3_config"])))
    source_raw = _read_json(_resolve(str(config["source_config"])))
    _validate_contract(config, parent_config, source_raw)
    attestation = _validate_preregistration(config, config_path)
    if enforce_formal_output:
        if output_dir != _resolve(str(config["formal_output"])).resolve():
            raise ValueError("N4 formal output path drifted")
        if work_dir != _resolve(str(config["formal_work_output"])).resolve():
            raise ValueError("N4 formal work path drifted")
    if output_dir.exists():
        raise FileExistsError(f"N4 final output already exists: {output_dir}")
    source = _source_for_run(source_raw, config)
    cells = expand_audit_cells(config, parent_config)
    parent = _parent_sentinel_map(config)
    gates = config["convergence_gates"]
    preregistration_sha256 = _sha256(config_path)
    decisions: list[dict[str, Any]] = []
    all_cost_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, cell in enumerate(cells, start=1):
        levels: dict[int, dict[str, Any]] = {}
        for step_count in config["base_step_counts"]:
            levels[int(step_count)] = _load_or_run_level(
                cell,
                source,
                step_count=int(step_count),
                work_dir=work_dir,
                preregistration_sha256=preregistration_sha256,
                resume=resume,
            )
        key = _cell_key(cell["case_id"], cell["dimensionless_stress_multiplier"])
        preliminary = _cell_decision(cell, levels, parent[key], gates)
        if preliminary["requires_h2048_escalation"]:
            levels[2048] = _load_or_run_level(
                cell,
                source,
                step_count=2048,
                work_dir=work_dir,
                preregistration_sha256=preregistration_sha256,
                resume=resume,
            )
        decision = _cell_decision(cell, levels, parent[key], gates)
        decisions.append(decision)
        all_cost_rows.extend(_cost_rows(cell, levels))
        print(
            f"N4 cell {index:02d}/32 {cell['cell_id']} "
            f"H1024={'PASS' if decision['h1024_all_gates_pass'] else 'FAIL'} "
            f"final={'PASS' if decision['final_cellwise_reference_authorized'] else 'FAIL'}",
            flush=True,
        )

    h1024_pass = sum(item["h1024_all_gates_pass"] for item in decisions)
    escalation_count = sum(item["requires_h2048_escalation"] for item in decisions)
    final_pass = sum(item["final_cellwise_reference_authorized"] for item in decisions)
    uniform_h1024 = h1024_pass == 32
    cellwise_reference = final_pass == 32
    machine_decision = (
        "EVALUATOR_CONVERGENCE_CLEARED_FOR_TINY_FIELD_JVP_VJP_GATE"
        if cellwise_reference
        else "FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED"
    )
    result = {
        "schema": "n2-pvgr-n4-evaluator-convergence-result-1.0",
        "candidate_id": config["candidate_id"],
        "protocol_commit": attestation["protocol_commit"],
        "run_head_commit": _git_text("rev-parse", "HEAD"),
        "machine_decision": machine_decision,
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
            "uniform_h1024_reference_authorized": uniform_h1024,
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
            "Selected synthetic evaluator-convergence evidence only; never algorithm, "
            "real-data, reconstruction, novelty, or generalization success."
        ),
        "figure": "n2_pvgr_n4_evaluator_convergence.png",
    }
    staging = _prepare_staging(work_dir)
    _atomic_json(staging / "result.json", result)
    _write_csv(staging / "metrics.csv", _metric_rows(decisions))
    _write_csv(staging / "pair_diagnostics.csv", _pair_rows(decisions))
    _write_csv(staging / "cost_ledger.csv", all_cost_rows)
    (staging / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    (staging / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _plot(staging / result["figure"], decisions, all_cost_rows)
    manifest_inputs = {
        "result": staging / "result.json",
        "metrics": staging / "metrics.csv",
        "pairs": staging / "pair_diagnostics.csv",
        "cost": staging / "cost_ledger.csv",
        "summary": staging / "summary.md",
        "config_snapshot": staging / "config_snapshot.json",
        "figure": staging / result["figure"],
    }
    manifest = {
        "schema": "n2-pvgr-n4-evaluator-convergence-manifest-1.0",
        "protocol_commit": attestation["protocol_commit"],
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
    for key, relative in config["attested_files"].items():
        path = _resolve(str(relative))
        manifest["files"][f"attested_{key}"] = {
            "path": str(relative),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    _atomic_json(staging / "manifest.json", manifest)
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
    config_path = args.config.resolve()
    config = _read_json(config_path)
    parent_config = _read_json(_resolve(str(config["parent_n3_config"])))
    source = _read_json(_resolve(str(config["source_config"])))
    _validate_contract(config, parent_config, source)
    attestation = _validate_preregistration(config, config_path)
    cells = expand_audit_cells(config, parent_config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "protocol_commit": attestation["protocol_commit"],
                    "physical_cells": len(cells),
                    "n3_failures": sum(cell["role"] == "n3_failure" for cell in cells),
                    "matched_controls": sum(
                        cell["role"] == "matched_control" for cell in cells
                    ),
                    "formal_output_exists": _resolve(config["formal_output"]).exists(),
                },
                indent=2,
            )
        )
        return 0
    output = (args.output or _resolve(config["formal_output"])).resolve()
    work = (args.work_output or _resolve(config["formal_work_output"])).resolve()
    result = run(
        config_path,
        output,
        work,
        resume=not args.no_resume,
        enforce_formal_output=True,
    )
    print(json.dumps({"machine_decision": result["machine_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
