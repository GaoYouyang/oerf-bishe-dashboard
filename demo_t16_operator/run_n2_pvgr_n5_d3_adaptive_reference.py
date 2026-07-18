#!/usr/bin/env python3
"""Assemble the preregistered N5-D3 mixed adaptive residual reference pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d3_adaptive_reference_preregistered_v1.json"
)
EXPECTED_ORDER = (
    "smooth-s1729-orientation_58-wide__stress_1",
    "smooth-s1729-orientation_58-narrow__stress_1",
    "smooth-s1729-orientation_58-wide__stress_3",
    "smooth-s1729-orientation_58-narrow__stress_3",
    "smooth-s1729-orientation_58-wide__stress_10",
    "smooth-s1729-orientation_58-narrow__stress_10",
    "smooth-s1871-orientation_58-wide__stress_1",
    "smooth-s1871-orientation_58-narrow__stress_1",
    "smooth-s1871-orientation_58-wide__stress_3",
    "smooth-s1871-orientation_58-narrow__stress_3",
    "smooth-s1871-orientation_58-wide__stress_10",
    "smooth-s1871-orientation_58-narrow__stress_10",
    "smooth-s1987-orientation_22-narrow__stress_1",
    "smooth-s1987-orientation_22-wide__stress_1",
    "smooth-s2131-orientation_58-wide__stress_1",
    "smooth-s2131-orientation_58-narrow__stress_1",
    "smooth-s2131-orientation_58-wide__stress_3",
    "smooth-s2131-orientation_58-narrow__stress_3",
    "smooth-s2131-orientation_58-wide__stress_10",
    "smooth-s2131-orientation_58-narrow__stress_10",
    "wrinkled-s3163-orientation_22-narrow__stress_1",
    "wrinkled-s3163-orientation_58-narrow__stress_1",
    "wrinkled-s3163-orientation_22-narrow__stress_3",
    "wrinkled-s3163-orientation_58-narrow__stress_3",
    "wrinkled-s3163-orientation_22-narrow__stress_10",
    "wrinkled-s3163-orientation_58-narrow__stress_10",
    "wrinkled-s3163-orientation_22-wide__stress_1",
    "wrinkled-s3163-orientation_58-wide__stress_1",
    "wrinkled-s3163-orientation_22-wide__stress_3",
    "wrinkled-s3163-orientation_58-wide__stress_3",
    "wrinkled-s3163-orientation_22-wide__stress_10",
    "wrinkled-s3163-orientation_58-wide__stress_10",
)
TAIL_CELLS = (
    "smooth-s1871-orientation_58-narrow__stress_1",
    "smooth-s1871-orientation_58-narrow__stress_3",
)
D2_SELECTED_CELLS = (
    "smooth-s1871-orientation_58-narrow__stress_1",
    "smooth-s1871-orientation_58-wide__stress_1",
    "smooth-s1871-orientation_58-narrow__stress_3",
    "smooth-s1871-orientation_58-wide__stress_3",
)


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"float64_le")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _identity_record(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": cell["cell_id"],
        "case_id": cell["case_id"],
        "field_unit_id": cell["field_unit_id"],
        "family_id": cell["family_id"],
        "phantom_family": cell["phantom_family"],
        "phantom_seed": int(cell["phantom_seed"]),
        "orientation_id": cell["orientation_id"],
        "aperture_id": cell["aperture_id"],
        "pair_id": cell["pair_id"],
        "role": cell["role"],
        "contrast_factor": cell["contrast_factor"],
        "dimensionless_stress_multiplier": float(
            cell["dimensionless_stress_multiplier"]
        ),
        "n3_failed_gate": cell["n3_failed_gate"],
    }


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _checkpoint_merkle_root(paths: list[Path], work: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(work).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_contract(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n5-d3-adaptive-reference-preregistered-1.0":
        raise ValueError("N5-D3 schema drifted")
    if config.get("candidate_id") != "N2-PVGR-N5-D3-ADAPTIVE32":
        raise ValueError("N5-D3 identifier drifted")
    if config.get("status") != (
        "preregistered_post_n4_d1_d2_mechanical_reconciliation_no_algorithm_authorization"
    ):
        raise ValueError("N5-D3 status drifted")
    if (config.get("device"), config.get("dtype"), config.get("array_encoding")) != (
        "cpu",
        "float64",
        "float64_little_endian_c_order",
    ):
        raise ValueError("N5-D3 numerical encoding drifted")
    if int(config.get("population_count", 0)) != 256:
        raise ValueError("N5-D3 population count drifted")
    if int(config.get("expected_physical_cell_count", 0)) != 32:
        raise ValueError("N5-D3 physical cell count drifted")
    if tuple(config.get("expected_cell_order", ())) != EXPECTED_ORDER:
        raise ValueError("N5-D3 cell order drifted")
    if tuple(config.get("tail_cell_ids", ())) != TAIL_CELLS:
        raise ValueError("N5-D3 tail cells drifted")
    expected_allocation = {
        "H1024_raw_separate_subtraction": 23,
        "H2048_raw_separate_subtraction": 7,
        "H8192_paired_neumaier": 2,
    }
    if config.get("expected_allocation") != expected_allocation:
        raise ValueError("N5-D3 allocation drifted")
    expected_policy = {
        "n4_authorized_cells_use_parent_final_step_and_matched_residual_uv": True,
        "only_two_n4_unauthorized_tail_cells_use_d2_H8192_paired_neumaier": True,
        "no_cell_reselection": True,
        "no_numerical_forward_rerun": True,
        "mixed_reference_pack": True,
        "uniform_paired_reference": False,
        "paired_equivalence_coverage_count": 4,
        "decomposed_curved_and_straight_outputs_available_for_all_cells": False,
    }
    if config.get("reference_policy") != expected_policy:
        raise ValueError("N5-D3 reference policy drifted")
    expected_observable = {
        "observable_id": "matched_curved_minus_straight_detector_deflection_uv",
        "coordinate_order": ["u", "v"],
        "units": "synthetic_internal_deflection_units_not_calibrated_to_pixel_or_angle",
        "ray_order": "frozen_parent_common_sobol_order_256",
        "array_shape_per_cell": [256, 2],
        "semantics": "residual_reference_only_not_detector_output_and_not_reconstructed_field",
    }
    if config.get("observable_contract") != expected_observable:
        raise ValueError("N5-D3 observable contract drifted")
    expected_gates = {
        "required_cell_count": 32,
        "required_unique_cell_count": 32,
        "required_h1024_count": 23,
        "required_h2048_count": 7,
        "required_h8192_count": 2,
        "required_finite_fraction": 1.0,
        "maximum_d1_route_equivalence_relative_l2": 5e-12,
        "required_d2_machine_decision": "D2_SELECTED_TAIL_RESOLVED_AT_H8192",
        "required_n4_checkpoint_count": 105,
        "required_n4_h2048_checkpoint_count": 9,
    }
    if config.get("gates") != expected_gates:
        raise ValueError("N5-D3 gates drifted")
    expected_decision = {
        "valid_requires_every_gate_and_every_source_hash": True,
        "valid_decision": "D3_VALID_MIXED_RESIDUAL_REFERENCE_ONLY",
        "failure_decision": "D3_FAIL_CLOSED",
        "uniform_paired_reference_must_remain_false": True,
        "assembly_adds_zero_field_point_queries": True,
        "d3_does_not_recompute_or_upgrade_any_reference": True,
        "d3_does_not_authorize_field_derivatives_reconstruction_or_training": True,
    }
    if config.get("decision_contract") != expected_decision:
        raise ValueError("N5-D3 decision contract drifted")
    expected_authorizations = {
        "fresh_reference": False,
        "uniform_paired_reference": False,
        "field_jvp_vjp": False,
        "three_dimensional_reconstruction": False,
        "neural_operator_training": False,
        "neural_operator_superiority": False,
        "real_data": False,
        "generalization": False,
        "paper_claim": False,
    }
    if config.get("claim_authorizations") != expected_authorizations:
        raise ValueError("N5-D3 claim authorization contract drifted")


def _manifest_entry(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    matches = [entry for entry in manifest.get("files", {}).values() if entry.get("path") == path]
    if len(matches) != 1:
        raise ValueError(f"N5-D3 manifest does not bind exactly one path: {path}")
    entry = matches[0]
    source = _resolve(path)
    if not source.is_file() or _sha256(source) != entry.get("sha256"):
        raise ValueError(f"N5-D3 manifest hash drifted: {path}")
    return entry


def _validate_preregistration(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    if not attestation_path.is_file():
        raise FileNotFoundError("committed N5-D3 attestation is missing")
    attestation = _read_json(attestation_path)
    if not isinstance(attestation, dict) or attestation.get("schema") != (
        "n2-pvgr-n5-d3-adaptive-reference-attestation-1.0"
    ):
        raise ValueError("N5-D3 attestation schema drifted")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D3 attestation does not prove output absence")
    if attestation.get("formal_output") != config["formal_output"]:
        raise ValueError("N5-D3 attestation output path drifted")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("N5-D3 config does not match its attestation")
    protocol_commit = str(attestation["protocol_commit"])
    if attestation.get("repository_head_at_creation") != protocol_commit:
        raise ValueError("N5-D3 attestation was not created at its protocol commit")
    if _git("merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False).returncode:
        raise ValueError("N5-D3 protocol commit is not an ancestor of HEAD")
    if set(attestation.get("attested_files", {})) != set(config["attested_files"]):
        raise ValueError("N5-D3 attested file set drifted")
    for key, entry in attestation["attested_files"].items():
        expected = str(config["attested_files"][key])
        if entry.get("path") != expected or _sha256(_resolve(expected)) != entry.get("sha256"):
            raise ValueError(f"N5-D3 attested file drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{expected}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"N5-D3 protocol file hash drifted: {key}")
    tracked = _git("ls-files", "--error-unmatch", _relative(attestation_path), check=False)
    if tracked.returncode:
        raise ValueError("N5-D3 attestation is not committed")
    watched = [_relative(attestation_path), *config["attested_files"].values()]
    if _git("status", "--porcelain", "--", *watched).stdout.strip():
        raise ValueError("N5-D3 preregistered files have uncommitted changes")
    return attestation


def _validate_parents(config: dict[str, Any]) -> dict[str, Any]:
    n4_result = _read_json(_resolve(str(config["parent_n4_result"])))
    n4_manifest = _read_json(_resolve(str(config["parent_n4_manifest"])))
    n4_validation = _read_json(_resolve(str(config["parent_n4_validation"])))
    recovery_config = _read_json(_resolve(str(config["parent_n4_recovery_config"])))
    recovery_attestation = _read_json(
        _resolve(str(config["parent_n4_recovery_attestation"]))
    )
    recovery_validation = _read_json(
        _resolve(str(config["parent_n4_recovery_validation"]))
    )
    d1_result = _read_json(_resolve(str(config["parent_d1_result"])))
    d1_levels = _read_json(_resolve(str(config["parent_d1_levels"])))
    d1_manifest = _read_json(_resolve(str(config["parent_d1_manifest"])))
    d1_validation = _read_json(_resolve(str(config["parent_d1_validation"])))
    d2_result = _read_json(_resolve(str(config["parent_d2_result"])))
    d2_levels = _read_json(_resolve(str(config["parent_d2_levels"])))
    d2_manifest = _read_json(_resolve(str(config["parent_d2_manifest"])))
    d2_validation = _read_json(_resolve(str(config["parent_d2_validation"])))
    d2_config = _read_json(_resolve(str(config["parent_d2_config"])))
    d2_attestation = _read_json(_resolve(str(config["parent_d2_attestation"])))
    dictionaries = (
        n4_result,
        n4_manifest,
        n4_validation,
        recovery_config,
        recovery_attestation,
        recovery_validation,
        d1_result,
        d1_manifest,
        d1_validation,
        d2_result,
        d2_manifest,
        d2_validation,
        d2_config,
        d2_attestation,
    )
    if any(not isinstance(item, dict) for item in dictionaries):
        raise ValueError("N5-D3 parent bundle contains an invalid JSON object")
    if not isinstance(d1_levels, list) or not isinstance(d2_levels, list):
        raise ValueError("N5-D3 parent level arrays are invalid")
    _manifest_entry(n4_manifest, str(config["parent_n4_result"]))
    _manifest_entry(d1_manifest, str(config["parent_d1_result"]))
    _manifest_entry(d1_manifest, str(config["parent_d1_levels"]))
    _manifest_entry(d2_manifest, str(config["parent_d2_result"]))
    _manifest_entry(d2_manifest, str(config["parent_d2_levels"]))
    _manifest_entry(d2_manifest, str(config["parent_d2_config"]))
    _manifest_entry(d2_manifest, str(config["parent_d2_attestation"]))
    d2_runner_path = str(d2_config.get("attested_files", {}).get("runner", ""))
    d2_runner_manifest = _manifest_entry(d2_manifest, d2_runner_path)
    if d2_config.get("selected_cells") != list(D2_SELECTED_CELLS):
        raise ValueError("N5-D3 parent D2 config selected cells drifted")
    if d2_config.get("reference_method") != "paired_neumaier":
        raise ValueError("N5-D3 parent D2 config reference method drifted")
    if d2_attestation.get("config_sha256") != _sha256(
        _resolve(str(config["parent_d2_config"]))
    ):
        raise ValueError("N5-D3 parent D2 attestation config hash drifted")
    if d2_attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D3 parent D2 attestation output-absence claim drifted")
    d2_attested_runner = d2_attestation.get("attested_files", {}).get("runner", {})
    if d2_attested_runner.get("path") != d2_runner_path or d2_attested_runner.get(
        "sha256"
    ) != d2_runner_manifest.get("sha256"):
        raise ValueError("N5-D3 parent D2 runner chain drifted")
    if n4_result.get("machine_decision") != "FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED":
        raise ValueError("N5-D3 parent N4.1 decision drifted")
    if n4_validation.get("valid") is not True or recovery_validation.get("valid") is not True:
        raise ValueError("N5-D3 parent N4.1 validation is not valid")
    if n4_result.get("counts", {}).get("physical_cell_count") != 32:
        raise ValueError("N5-D3 parent N4.1 cell count drifted")
    if [row.get("cell_id") for row in n4_result.get("cells", [])] != list(EXPECTED_ORDER):
        raise ValueError("N5-D3 parent N4.1 cell order drifted")
    if d1_result.get("machine_decision") != (
        "D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR"
    ) or d1_result.get("contract_gates_pass") is not True:
        raise ValueError("N5-D3 parent D1 decision drifted")
    if d1_validation.get("valid") is not True:
        raise ValueError("N5-D3 parent D1 validation is not valid")
    if float(d1_result.get("maximum_route_equivalence_relative_l2", float("inf"))) > float(
        config["gates"]["maximum_d1_route_equivalence_relative_l2"]
    ):
        raise ValueError("N5-D3 parent D1 route-equivalence gate drifted")
    if int(d1_result.get("selected_cell_count", 0)) != int(
        config["reference_policy"]["paired_equivalence_coverage_count"]
    ):
        raise ValueError("N5-D3 parent D1 coverage drifted")
    if d2_result.get("machine_decision") != config["gates"]["required_d2_machine_decision"]:
        raise ValueError("N5-D3 parent D2 decision drifted")
    if d2_result.get("protocol_commit") != d2_attestation.get("protocol_commit"):
        raise ValueError("N5-D3 parent D2 protocol commit binding drifted")
    if int(d2_result.get("selected_cell_count", 0)) != 4 or int(
        d2_result.get("new_level_evaluation_count", 0)
    ) != 8:
        raise ValueError("N5-D3 parent D2 population contract drifted")
    if [row.get("cell_id") for row in d2_result.get("cells", [])] != list(
        D2_SELECTED_CELLS
    ):
        raise ValueError("N5-D3 parent D2 selected cell order drifted")
    expected_level_order = [
        (cell_id, step)
        for cell_id in D2_SELECTED_CELLS
        for step in (4096, 8192)
    ]
    if [
        (row.get("cell_id"), int(row.get("step_count", 0))) for row in d2_levels
    ] != expected_level_order:
        raise ValueError("N5-D3 parent D2 level order drifted")
    if d2_validation.get("valid") is not True or not all(
        row.get("all_gates_pass") is True for row in d2_result.get("cells", [])
    ):
        raise ValueError("N5-D3 parent D2 validation is not valid")
    work = _resolve(str(config["parent_n4_checkpoint_work"]))
    paths = sorted(work.glob(str(config["parent_n4_checkpoint_glob"])))
    h2048_count = sum(path.name == "H2048.json" for path in paths)
    if len(paths) != int(config["gates"]["required_n4_checkpoint_count"]):
        raise ValueError("N5-D3 N4 checkpoint count drifted")
    if h2048_count != int(config["gates"]["required_n4_h2048_checkpoint_count"]):
        raise ValueError("N5-D3 N4 H2048 checkpoint count drifted")
    merkle = _checkpoint_merkle_root(paths, work)
    if merkle != recovery_attestation.get("opaque_checkpoint_merkle_root"):
        raise ValueError("N5-D3 N4 checkpoint Merkle root drifted")
    if merkle != recovery_validation.get("opaque_checkpoint_merkle_root"):
        raise ValueError("N5-D3 N4 recovery validation Merkle root drifted")
    return {
        "n4_result": n4_result,
        "d1_result": d1_result,
        "d2_result": d2_result,
        "d2_levels": d2_levels,
        "n4_checkpoint_merkle_root": merkle,
        "n4_checkpoint_count": len(paths),
        "n4_h2048_checkpoint_count": h2048_count,
    }


def _allocation(n4_result: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in n4_result["cells"]:
        cell_id = str(cell["cell_id"])
        if cell.get("final_cellwise_reference_authorized") is True:
            step = int(cell["final_reference_step_count"])
            if step not in (1024, 2048):
                raise ValueError(f"N5-D3 invalid N4 final step: {cell_id}")
            rows.append(
                {
                    "cell": cell,
                    "step_count": step,
                    "reference_method": "raw_separate_subtraction",
                    "source_kind": "n4_checkpoint",
                }
            )
        else:
            if cell_id not in TAIL_CELLS:
                raise ValueError(f"N5-D3 unexpected unauthorized N4 cell: {cell_id}")
            rows.append(
                {
                    "cell": cell,
                    "step_count": 8192,
                    "reference_method": "paired_neumaier",
                    "source_kind": "n5_d2_level",
                }
            )
    if [row["cell"]["cell_id"] for row in rows] != list(config["expected_cell_order"]):
        raise ValueError("N5-D3 derived cell order drifted")
    counts = {
        "H1024_raw_separate_subtraction": sum(
            row["step_count"] == 1024 for row in rows
        ),
        "H2048_raw_separate_subtraction": sum(
            row["step_count"] == 2048 for row in rows
        ),
        "H8192_paired_neumaier": sum(row["step_count"] == 8192 for row in rows),
    }
    if counts != config["expected_allocation"]:
        raise ValueError("N5-D3 mechanically derived allocation drifted")
    return rows


def _valid_array(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (256, 2) or not np.all(np.isfinite(array)):
        raise ValueError(f"N5-D3 invalid source array: {label}")
    return array


def _compact_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "finite_ray_fraction",
        "minimum_domain_margin",
        "minimum_stencil_margin",
        "maximum_direction_norm_error",
        "frustum_violation_count",
        "minimum_frustum_margin",
    )
    return {key: value[key] for key in keys if key in value}


def _n4_reference(
    allocation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    cell = allocation["cell"]
    step = int(allocation["step_count"])
    path = (
        _resolve(str(config["parent_n4_checkpoint_work"]))
        / "levels"
        / str(cell["cell_id"])
        / f"H{step}.json"
    )
    checkpoint = _read_json(path)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != (
        "n2-pvgr-n4-level-checkpoint-1.0"
    ):
        raise ValueError(f"N5-D3 N4 checkpoint schema drifted: {cell['cell_id']}")
    metadata = checkpoint.get("metadata", {})
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
    ):
        if metadata.get(key) != cell.get(key):
            raise ValueError(f"N5-D3 N4 metadata drifted for {cell['cell_id']}: {key}")
    if int(metadata.get("step_count", -1)) != step:
        raise ValueError(f"N5-D3 N4 step metadata drifted: {cell['cell_id']}")
    array = _valid_array(checkpoint.get("matched_residual_uv"), label=str(cell["cell_id"]))
    if _legacy_array_sha256(array) != checkpoint.get("matched_residual_uv_sha256"):
        raise ValueError(f"N5-D3 N4 source array hash drifted: {cell['cell_id']}")
    source_record = {
        "schema": checkpoint["schema"],
        "metadata": checkpoint["metadata"],
        "matched_residual_uv_sha256": checkpoint["matched_residual_uv_sha256"],
        "high_output_uv_sha256": checkpoint["high_output_uv_sha256"],
        "diagnostics": checkpoint["diagnostics"],
        "cost": checkpoint["cost"],
        "preregistration_sha256": checkpoint["preregistration_sha256"],
    }
    return array, {
        "source_path": _relative(path),
        "source_file_sha256": _sha256(path),
        "source_record_sha256": _canonical_json_sha256(source_record),
        "source_array_sha256_legacy": checkpoint["matched_residual_uv_sha256"],
        "source_array_hash_schema": "legacy_float64_shape_and_c_bytes",
        "source_array_selector": "matched_residual_uv",
        "source_logical_point_queries": int(checkpoint["cost"]["total_logical_point_queries"]),
        "source_wall_seconds": float(checkpoint["cost"]["wall_seconds"]),
        "source_diagnostics": _compact_diagnostics(checkpoint["diagnostics"]),
        "available_companion_hashes": {
            "high_output_uv": checkpoint["high_output_uv_sha256"],
            "straight_output_uv": None,
        },
    }


def _d2_reference(
    allocation: dict[str, Any],
    config: dict[str, Any],
    d2_levels: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    cell = allocation["cell"]
    matches = [
        row
        for row in d2_levels
        if row.get("cell_id") == cell["cell_id"] and int(row.get("step_count", 0)) == 8192
    ]
    if len(matches) != 1:
        raise ValueError(f"N5-D3 D2 H8192 source is not unique: {cell['cell_id']}")
    row = matches[0]
    if row.get("pair_id") != cell.get("pair_id") or row.get("role") != cell.get("role"):
        raise ValueError(f"N5-D3 D2 metadata drifted: {cell['cell_id']}")
    entry = row.get("methods", {}).get("paired_neumaier", {})
    array = _valid_array(entry.get("values"), label=str(cell["cell_id"]))
    if _legacy_array_sha256(array) != entry.get("sha256"):
        raise ValueError(f"N5-D3 D2 source array hash drifted: {cell['cell_id']}")
    path = _resolve(str(config["parent_d2_levels"]))
    monitor_hashes = {
        method: details["sha256"] for method, details in row.get("methods", {}).items()
    }
    return array, {
        "source_path": _relative(path),
        "source_file_sha256": _sha256(path),
        "source_record_sha256": _canonical_json_sha256(row),
        "source_array_sha256_legacy": entry["sha256"],
        "source_array_hash_schema": "legacy_float64_shape_and_c_bytes",
        "source_array_selector": (
            f"cell_id={cell['cell_id']};step_count=8192;methods.paired_neumaier.values"
        ),
        "source_logical_point_queries": int(row["cost"]["logical_point_queries"]),
        "source_wall_seconds": float(row["cost"]["wall_seconds"]),
        "source_diagnostics": _compact_diagnostics(row["diagnostics"]),
        "available_monitor_hashes": monitor_hashes,
        "source_identity_sha256_from_parent_n4": _canonical_json_sha256(
            _identity_record(cell)
        ),
        "d2_available_identity_fields": ["cell_id", "pair_id", "role", "step_count"],
        "d2_full_identity_bound_via_attested_runner_and_parent_n4": True,
        "available_companion_hashes": {
            "curved_output_uv": None,
            "straight_output_uv": None,
        },
    }


def _pack_cells(
    allocation: list[dict[str, Any]],
    config: dict[str, Any],
    d2_levels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observable = config["observable_contract"]
    for index, item in enumerate(allocation):
        cell = item["cell"]
        if item["source_kind"] == "n4_checkpoint":
            array, provenance = _n4_reference(item, config)
            selection_basis = "N4_1_final_cellwise_reference_authorized"
        else:
            array, provenance = _d2_reference(item, config, d2_levels)
            selection_basis = "N4_1_failed_then_D2_selected_tail_resolved"
        identity = _identity_record(cell)
        rows.append(
            {
                "pack_index": index,
                **identity,
                "identity_sha256": _canonical_json_sha256(identity),
                "step_count": int(item["step_count"]),
                "reference_method": item["reference_method"],
                "source_kind": item["source_kind"],
                "selection_basis": selection_basis,
                "observable_id": observable["observable_id"],
                "units": observable["units"],
                "coordinate_order": observable["coordinate_order"],
                "array_shape": list(array.shape),
                "array_encoding": config["array_encoding"],
                "pack_array_sha256_float64_le_c_order": _array_sha256(array),
                "l2_norm": float(np.linalg.norm(array)),
                "finite_fraction": float(np.mean(np.isfinite(array))),
                "reference_values": array.tolist(),
                "provenance": provenance,
            }
        )
    return rows


def _gate_results(
    cells: list[dict[str, Any]],
    config: dict[str, Any],
    parents: dict[str, Any],
) -> dict[str, bool]:
    steps = [int(row["step_count"]) for row in cells]
    methods = [str(row["reference_method"]) for row in cells]
    gates = config["gates"]
    return {
        "cell_count_gate_met": len(cells) == int(gates["required_cell_count"]),
        "unique_cell_gate_met": len({row["cell_id"] for row in cells})
        == int(gates["required_unique_cell_count"]),
        "cell_order_gate_met": [row["cell_id"] for row in cells]
        == list(config["expected_cell_order"]),
        "h1024_count_gate_met": sum(step == 1024 for step in steps)
        == int(gates["required_h1024_count"]),
        "h2048_count_gate_met": sum(step == 2048 for step in steps)
        == int(gates["required_h2048_count"]),
        "h8192_count_gate_met": sum(step == 8192 for step in steps)
        == int(gates["required_h8192_count"]),
        "method_mapping_gate_met": all(
            (step in (1024, 2048) and method == "raw_separate_subtraction")
            or (step == 8192 and method == "paired_neumaier")
            for step, method in zip(steps, methods, strict=True)
        ),
        "finite_gate_met": min(row["finite_fraction"] for row in cells)
        >= float(gates["required_finite_fraction"]),
        "shape_gate_met": all(row["array_shape"] == [256, 2] for row in cells),
        "n4_merkle_gate_met": parents["n4_checkpoint_count"]
        == int(gates["required_n4_checkpoint_count"]),
        "d1_route_gate_met": float(
            parents["d1_result"]["maximum_route_equivalence_relative_l2"]
        )
        <= float(gates["maximum_d1_route_equivalence_relative_l2"]),
        "d2_tail_gate_met": parents["d2_result"]["machine_decision"]
        == gates["required_d2_machine_decision"],
        "mixed_semantics_disclosed_gate_met": config["reference_policy"][
            "mixed_reference_pack"
        ]
        is True
        and config["reference_policy"]["uniform_paired_reference"] is False,
        "assembly_zero_query_gate_met": True,
    }


def _write_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    fields = (
        "pack_index",
        "cell_id",
        "case_id",
        "pair_id",
        "role",
        "field_unit_id",
        "family_id",
        "phantom_family",
        "phantom_seed",
        "orientation_id",
        "aperture_id",
        "contrast_factor",
        "dimensionless_stress_multiplier",
        "n3_failed_gate",
        "step_count",
        "reference_method",
        "source_kind",
        "identity_sha256",
        "pack_array_sha256_float64_le_c_order",
        "l2_norm",
        "finite_fraction",
        "source_logical_point_queries",
        "source_path",
        "source_file_sha256",
        "source_record_sha256",
        "source_array_sha256_legacy",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            row = {key: cell.get(key) for key in fields}
            for key in (
                "source_logical_point_queries",
                "source_path",
                "source_file_sha256",
                "source_record_sha256",
                "source_array_sha256_legacy",
            ):
                row[key] = cell["provenance"][key]
            writer.writerow(row)


def _plot(result: dict[str, Any], cells: list[dict[str, Any]], path: Path) -> None:
    x = np.arange(len(cells))
    colors = ["#397f76" if row["step_count"] < 8192 else "#a44a3f" for row in cells]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    allocation = result["allocation_counts"]
    labels = ["H1024 raw", "H2048 raw", "H8192 paired"]
    values = [
        allocation["H1024_raw_separate_subtraction"],
        allocation["H2048_raw_separate_subtraction"],
        allocation["H8192_paired_neumaier"],
    ]
    axes[0].bar(labels, values, color=["#397f76", "#4c78a8", "#a44a3f"])
    axes[0].set_ylabel("cell count")
    axes[0].set_title("Frozen 23 / 7 / 2 allocation")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.22)
    for index, value in enumerate(values):
        axes[0].text(index, value, str(value), ha="center", va="bottom")

    axes[1].scatter(x, [row["l2_norm"] for row in cells], c=colors, s=34)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("frozen N4.1 cell order")
    axes[1].set_ylabel("residual reference L2 norm")
    axes[1].set_title("Reference magnitude by cell")
    axes[1].grid(alpha=0.22)

    axes[2].scatter(
        x,
        [row["provenance"]["source_logical_point_queries"] for row in cells],
        c=colors,
        s=34,
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("frozen N4.1 cell order")
    axes[2].set_ylabel("source logical point queries")
    axes[2].set_title("Cost of selected source level")
    axes[2].grid(alpha=0.22)
    figure.suptitle(
        "N5-D3 mixed adaptive residual reference pack (infrastructure, not performance)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _summary(result: dict[str, Any]) -> str:
    counts = result["allocation_counts"]
    return "\n".join(
        (
            "# N2-PVGR N5-D3 mixed adaptive residual reference pack",
            "",
            f"Machine decision: `{result['machine_decision']}`",
            "",
            "This is a post-N4/D1/D2 synthetic residual-reference reconciliation. It is mixed-method infrastructure, not a fresh set, uniform paired reference, reconstruction result, model result or paper claim.",
            "",
            f"- Cells: {result['cell_count']}",
            f"- Allocation: H1024 raw={counts['H1024_raw_separate_subtraction']}, H2048 raw={counts['H2048_raw_separate_subtraction']}, H8192 paired-Neumaier={counts['H8192_paired_neumaier']}",
            f"- Stacked shape: {result['stacked_array_shape']}",
            f"- Minimum finite fraction: {result['minimum_finite_fraction']:.1f}",
            f"- Selected source queries: {result['selected_reference_source_logical_point_queries']:,}",
            f"- New assembly field queries: {result['assembly_logical_point_queries']}",
            f"- Paired equivalence coverage: {result['paired_equivalence_coverage_count']}/32",
            "",
            "All broad claim authorizations remain false. The next derivative experiment requires its own preregistration and must keep detector-output and residual-reference derivatives separate.",
            "",
        )
    )


def _manifest(
    staging: Path,
    output: Path,
    sources: dict[str, Path],
    files: list[str],
) -> dict[str, Any]:
    return {
        "schema": "n2-pvgr-n5-d3-adaptive-reference-manifest-1.0",
        "files": {
            key: {"path": _relative(path), "sha256": _sha256(path)}
            for key, path in sources.items()
        }
        | {
            name: {
                "path": _relative(output / name),
                "sha256": _sha256(staging / name),
            }
            for name in files
        },
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("N5-D3 config is invalid")
    _validate_contract(config)
    attestation = _validate_preregistration(config, config_path)
    parents = _validate_parents(config)
    allocation = _allocation(parents["n4_result"], config)
    cells = _pack_cells(allocation, config, parents["d2_levels"])
    gate_results = _gate_results(cells, config, parents)
    decision = (
        config["decision_contract"]["valid_decision"]
        if all(gate_results.values())
        else config["decision_contract"]["failure_decision"]
    )
    stacked = np.stack(
        [np.asarray(row["reference_values"], dtype=np.float64) for row in cells], axis=0
    )
    allocation_counts = {
        "H1024_raw_separate_subtraction": sum(row["step_count"] == 1024 for row in cells),
        "H2048_raw_separate_subtraction": sum(row["step_count"] == 2048 for row in cells),
        "H8192_paired_neumaier": sum(row["step_count"] == 8192 for row in cells),
    }
    cell_order_sha256 = _canonical_json_sha256([row["cell_id"] for row in cells])
    result = {
        "schema": "n2-pvgr-n5-d3-adaptive-reference-result-1.0",
        "candidate_id": config["candidate_id"],
        "protocol_commit": attestation["protocol_commit"],
        "machine_decision": decision,
        "post_selected_synthetic_reconciliation": True,
        "mixed_reference_pack": True,
        "uniform_paired_reference": False,
        "cell_count": len(cells),
        "unique_cell_count": len({row["cell_id"] for row in cells}),
        "allocation_counts": allocation_counts,
        "cell_order_sha256": cell_order_sha256,
        "stacked_array_shape": list(stacked.shape),
        "stacked_array_sha256_float64_le_c_order": _array_sha256(stacked),
        "array_encoding": config["array_encoding"],
        "minimum_finite_fraction": min(row["finite_fraction"] for row in cells),
        "paired_equivalence_coverage_count": int(
            config["reference_policy"]["paired_equivalence_coverage_count"]
        ),
        "selected_reference_source_logical_point_queries": sum(
            row["provenance"]["source_logical_point_queries"] for row in cells
        ),
        "assembly_logical_point_queries": 0,
        "n4_checkpoint_merkle_root": parents["n4_checkpoint_merkle_root"],
        "gate_results": gate_results,
        "all_gates_pass": all(gate_results.values()),
        "authorizations": config["claim_authorizations"],
        "limitations": config["known_limitations"],
        "next_protocol_boundary": (
            "A separately preregistered tiny-field derivative audit may be designed, but D3 "
            "does not authorize a field derivative, reconstruction or training claim."
        ),
    }
    pack = {
        "schema": "n2-pvgr-n5-d3-adaptive-reference-pack-1.0",
        "candidate_id": config["candidate_id"],
        "protocol_commit": attestation["protocol_commit"],
        "machine_decision": decision,
        "observable_contract": config["observable_contract"],
        "reference_policy": config["reference_policy"],
        "cell_order_sha256": cell_order_sha256,
        "stacked_array_sha256_float64_le_c_order": result[
            "stacked_array_sha256_float64_le_c_order"
        ],
        "cells": cells,
    }
    output = output.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise FileExistsError("N5-D3 output or staging directory already exists")
    staging.mkdir(parents=True)
    (staging / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (staging / "reference_pack.json").write_text(
        json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (staging / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(staging / "cell_ledger.csv", cells)
    (staging / "summary.md").write_text(_summary(result), encoding="utf-8")
    _plot(result, cells, staging / "n2_pvgr_n5_d3_adaptive_reference.png")
    files = [
        "cell_ledger.csv",
        "config_snapshot.json",
        "n2_pvgr_n5_d3_adaptive_reference.png",
        "reference_pack.json",
        "result.json",
        "summary.md",
    ]
    sources = {
        "config": config_path,
        "attestation": _resolve(str(config["pre_registration_attestation"])),
        "runner": Path(__file__).resolve(),
        "parent_n4_result": _resolve(str(config["parent_n4_result"])),
        "parent_n4_recovery_attestation": _resolve(
            str(config["parent_n4_recovery_attestation"])
        ),
        "parent_d1_result": _resolve(str(config["parent_d1_result"])),
        "parent_d2_result": _resolve(str(config["parent_d2_result"])),
        "parent_d2_levels": _resolve(str(config["parent_d2_levels"])),
        "parent_d2_config": _resolve(str(config["parent_d2_config"])),
        "parent_d2_attestation": _resolve(str(config["parent_d2_attestation"])),
    }
    (staging / "manifest.json").write_text(
        json.dumps(_manifest(staging, output, sources, files), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    staging.replace(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _read_json(args.config.resolve())
    if not isinstance(config, dict):
        raise ValueError("N5-D3 config is invalid")
    output = args.output or _resolve(str(config["formal_output"]))
    result = run(args.config.resolve(), output.resolve())
    print(
        json.dumps(
            {
                "machine_decision": result["machine_decision"],
                "allocation_counts": result["allocation_counts"],
                "stacked_array_sha256_float64_le_c_order": result[
                    "stacked_array_sha256_float64_le_c_order"
                ],
                "selected_reference_source_logical_point_queries": result[
                    "selected_reference_source_logical_point_queries"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
