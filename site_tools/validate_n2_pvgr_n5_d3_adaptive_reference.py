#!/usr/bin/env python3
"""Independently validate the formal N5-D3 mixed adaptive reference pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image


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
TAIL_CELLS = {
    "smooth-s1871-orientation_58-narrow__stress_1",
    "smooth-s1871-orientation_58-narrow__stress_3",
}
D2_SELECTED_CELLS = (
    "smooth-s1871-orientation_58-narrow__stress_1",
    "smooth-s1871-orientation_58-wide__stress_1",
    "smooth-s1871-orientation_58-narrow__stress_3",
    "smooth-s1871-orientation_58-wide__stress_3",
)
RESULT_FILES = (
    "cell_ledger.csv",
    "config_snapshot.json",
    "n2_pvgr_n5_d3_adaptive_reference.png",
    "reference_pack.json",
    "result.json",
    "summary.md",
)
SOURCE_KEYS = (
    "config",
    "attestation",
    "runner",
    "parent_n4_result",
    "parent_n4_recovery_attestation",
    "parent_d1_result",
    "parent_d2_result",
    "parent_d2_levels",
    "parent_d2_config",
    "parent_d2_attestation",
)
EXPECTED_AUTHORIZATIONS = {
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


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
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


def _same(left: float, right: float, *, label: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=2e-13, abs_tol=2e-15):
        raise ValueError(f"N5-D3 numeric mismatch for {label}: {left} != {right}")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _checkpoint_merkle_root(paths: list[Path], work: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(work).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_entry(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    matches = [entry for entry in manifest.get("files", {}).values() if entry.get("path") == path]
    if len(matches) != 1:
        raise ValueError(f"N5-D3 parent manifest binding drifted: {path}")
    source = _resolve(path)
    if not source.is_file() or _sha256(source) != matches[0].get("sha256"):
        raise ValueError(f"N5-D3 parent manifest hash mismatch: {path}")
    return matches[0]


def _verify_attestation(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    path = _resolve(str(config["pre_registration_attestation"]))
    attestation = _read_json(path)
    if not isinstance(attestation, dict) or attestation.get("schema") != (
        "n2-pvgr-n5-d3-adaptive-reference-attestation-1.0"
    ):
        raise ValueError("N5-D3 attestation schema drifted")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D3 attestation output-absence claim drifted")
    if attestation.get("formal_output") != config["formal_output"]:
        raise ValueError("N5-D3 attestation formal output drifted")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("N5-D3 attestation config hash drifted")
    protocol_commit = str(attestation["protocol_commit"])
    if attestation.get("repository_head_at_creation") != protocol_commit:
        raise ValueError("N5-D3 attestation commit binding drifted")
    if _git("merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False).returncode:
        raise ValueError("N5-D3 protocol commit is not an ancestor")
    if set(attestation.get("attested_files", {})) != set(config["attested_files"]):
        raise ValueError("N5-D3 attested file set drifted")
    for key, entry in attestation["attested_files"].items():
        expected = str(config["attested_files"][key])
        current = _resolve(expected)
        if entry.get("path") != expected or _sha256(current) != entry.get("sha256"):
            raise ValueError(f"N5-D3 attested current file drifted: {key}")
        frozen = _git("show", f"{protocol_commit}:{expected}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"N5-D3 attested committed file drifted: {key}")
    relative = path.resolve().relative_to(ROOT).as_posix()
    if _git("ls-files", "--error-unmatch", relative, check=False).returncode:
        raise ValueError("N5-D3 attestation is not Git-tracked")
    if _git("status", "--porcelain", "--", relative).stdout.strip():
        raise ValueError("N5-D3 attestation has uncommitted changes")
    committed = _git("show", f"HEAD:{relative}").stdout
    if hashlib.sha256(committed).hexdigest() != _sha256(path):
        raise ValueError("N5-D3 attestation bytes are not bound by HEAD")
    return attestation


def _verify_result_manifest(
    manifest: dict[str, Any],
    output: Path,
    config_path: Path,
    config: dict[str, Any],
) -> None:
    if manifest.get("schema") != "n2-pvgr-n5-d3-adaptive-reference-manifest-1.0":
        raise ValueError("N5-D3 result manifest schema drifted")
    if set(manifest.get("files", {})) != set(RESULT_FILES) | set(SOURCE_KEYS):
        raise ValueError("N5-D3 result manifest file set drifted")
    expected_sources = {
        "config": config_path.resolve().relative_to(ROOT).as_posix(),
        "attestation": str(config["pre_registration_attestation"]),
        "runner": str(config["attested_files"]["runner"]),
        "parent_n4_result": str(config["parent_n4_result"]),
        "parent_n4_recovery_attestation": str(
            config["parent_n4_recovery_attestation"]
        ),
        "parent_d1_result": str(config["parent_d1_result"]),
        "parent_d2_result": str(config["parent_d2_result"]),
        "parent_d2_levels": str(config["parent_d2_levels"]),
        "parent_d2_config": str(config["parent_d2_config"]),
        "parent_d2_attestation": str(config["parent_d2_attestation"]),
    }
    for key, entry in manifest["files"].items():
        path = _resolve(str(entry["path"]))
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise ValueError(f"N5-D3 result manifest hash mismatch: {key}")
        if key in RESULT_FILES and path.resolve() != (output / key).resolve():
            raise ValueError(f"N5-D3 result manifest output path drifted: {key}")
        if key in SOURCE_KEYS and entry["path"] != expected_sources[key]:
            raise ValueError(f"N5-D3 result manifest source path drifted: {key}")


def _verify_figure(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
        width, height = image.size
    sampled = rgb[::8, ::8]
    if width < 1200 or height < 400 or float(np.std(rgb)) < 3.0:
        raise ValueError("N5-D3 figure is too small or blank")
    if np.unique(sampled.reshape(-1, 3), axis=0).shape[0] < 64:
        raise ValueError("N5-D3 figure has insufficient visual content")
    return {"width": int(width), "height": int(height), "pixel_std": float(np.std(rgb))}


def _verify_claim_structures(
    config: dict[str, Any],
    result: dict[str, Any],
    pack: dict[str, Any],
) -> None:
    if config.get("claim_authorizations") != EXPECTED_AUTHORIZATIONS:
        raise ValueError("N5-D3 config authorization structure drifted")
    if result.get("authorizations") != EXPECTED_AUTHORIZATIONS:
        raise ValueError("N5-D3 result authorization structure drifted")
    if pack.get("reference_policy") != config.get("reference_policy"):
        raise ValueError("N5-D3 pack reference policy drifted")
    if pack.get("observable_contract") != config.get("observable_contract"):
        raise ValueError("N5-D3 pack observable contract drifted")


def _validate_config_contract(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n5-d3-adaptive-reference-preregistered-1.0":
        raise ValueError("N5-D3 config schema drifted")
    if config.get("candidate_id") != "N2-PVGR-N5-D3-ADAPTIVE32":
        raise ValueError("N5-D3 config candidate drifted")
    if config.get("status") != (
        "preregistered_post_n4_d1_d2_mechanical_reconciliation_no_algorithm_authorization"
    ):
        raise ValueError("N5-D3 config status drifted")
    if (config.get("device"), config.get("dtype"), config.get("array_encoding")) != (
        "cpu",
        "float64",
        "float64_little_endian_c_order",
    ):
        raise ValueError("N5-D3 config numerical encoding drifted")
    if int(config.get("population_count", 0)) != 256 or int(
        config.get("expected_physical_cell_count", 0)
    ) != 32:
        raise ValueError("N5-D3 config population contract drifted")
    if tuple(config.get("expected_cell_order", ())) != EXPECTED_ORDER:
        raise ValueError("N5-D3 config cell order drifted")
    if tuple(config.get("tail_cell_ids", ())) != tuple(
        cell_id for cell_id in EXPECTED_ORDER if cell_id in TAIL_CELLS
    ):
        raise ValueError("N5-D3 config tail cells drifted")
    expected_allocation = {
        "H1024_raw_separate_subtraction": 23,
        "H2048_raw_separate_subtraction": 7,
        "H8192_paired_neumaier": 2,
    }
    if config.get("expected_allocation") != expected_allocation:
        raise ValueError("N5-D3 config allocation drifted")
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
        raise ValueError("N5-D3 config reference policy drifted")
    expected_observable = {
        "observable_id": "matched_curved_minus_straight_detector_deflection_uv",
        "coordinate_order": ["u", "v"],
        "units": "synthetic_internal_deflection_units_not_calibrated_to_pixel_or_angle",
        "ray_order": "frozen_parent_common_sobol_order_256",
        "array_shape_per_cell": [256, 2],
        "semantics": "residual_reference_only_not_detector_output_and_not_reconstructed_field",
    }
    if config.get("observable_contract") != expected_observable:
        raise ValueError("N5-D3 config observable contract drifted")
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
        raise ValueError("N5-D3 config gates drifted")
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
        raise ValueError("N5-D3 config decision contract drifted")
    if config.get("claim_authorizations") != EXPECTED_AUTHORIZATIONS:
        raise ValueError("N5-D3 config authorization structure drifted")


def _expected_ledger_row(pack_cell: dict[str, Any], index: int) -> dict[str, str]:
    provenance = pack_cell["provenance"]
    fields = (
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
    )
    expected = {key: str(pack_cell[key]) for key in fields}
    expected["pack_index"] = str(index)
    for key in (
        "source_logical_point_queries",
        "source_path",
        "source_file_sha256",
        "source_record_sha256",
        "source_array_sha256_legacy",
    ):
        expected[key] = str(provenance[key])
    return expected


def _verify_ledger_row(
    pack_cell: dict[str, Any],
    ledger_row: dict[str, str],
    index: int,
) -> None:
    expected = _expected_ledger_row(pack_cell, index)
    if set(ledger_row) != set(expected):
        raise ValueError(f"N5-D3 ledger column set drifted: {pack_cell['cell_id']}")
    for key, value in expected.items():
        if ledger_row.get(key) != value:
            raise ValueError(f"N5-D3 ledger field drifted for {pack_cell['cell_id']}: {key}")


def _source_record_n4(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": checkpoint["schema"],
        "metadata": checkpoint["metadata"],
        "matched_residual_uv_sha256": checkpoint["matched_residual_uv_sha256"],
        "high_output_uv_sha256": checkpoint["high_output_uv_sha256"],
        "diagnostics": checkpoint["diagnostics"],
        "cost": checkpoint["cost"],
        "preregistration_sha256": checkpoint["preregistration_sha256"],
    }


def _expected_source(
    parent_cell: dict[str, Any],
    pack_cell: dict[str, Any],
    config: dict[str, Any],
    d2_levels: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    cell_id = str(parent_cell["cell_id"])
    if parent_cell.get("final_cellwise_reference_authorized") is True:
        step = int(parent_cell["final_reference_step_count"])
        path = (
            _resolve(str(config["parent_n4_checkpoint_work"]))
            / "levels"
            / cell_id
            / f"H{step}.json"
        )
        checkpoint = _read_json(path)
        if not isinstance(checkpoint, dict) or checkpoint.get("schema") != (
            "n2-pvgr-n4-level-checkpoint-1.0"
        ):
            raise ValueError(f"N5-D3 invalid N4 checkpoint: {cell_id}")
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
            if metadata.get(key) != parent_cell.get(key):
                raise ValueError(f"N5-D3 N4 metadata drifted for {cell_id}: {key}")
        if int(metadata.get("step_count", -1)) != step:
            raise ValueError(f"N5-D3 N4 step metadata drifted: {cell_id}")
        array = np.asarray(checkpoint["matched_residual_uv"], dtype=np.float64)
        if array.shape != (256, 2) or not np.all(np.isfinite(array)):
            raise ValueError(f"N5-D3 invalid N4 source array: {cell_id}")
        if _legacy_array_sha256(array) != checkpoint["matched_residual_uv_sha256"]:
            raise ValueError(f"N5-D3 N4 source array hash mismatch: {cell_id}")
        expected = {
            "step_count": step,
            "reference_method": "raw_separate_subtraction",
            "source_kind": "n4_checkpoint",
            "source_path": path.resolve().relative_to(ROOT).as_posix(),
            "source_file_sha256": _sha256(path),
            "source_record_sha256": _canonical_json_sha256(_source_record_n4(checkpoint)),
            "source_array_sha256_legacy": checkpoint["matched_residual_uv_sha256"],
            "source_logical_point_queries": int(
                checkpoint["cost"]["total_logical_point_queries"]
            ),
        }
    else:
        if cell_id not in TAIL_CELLS:
            raise ValueError(f"N5-D3 unexpected unauthorized N4 source: {cell_id}")
        matches = [
            row
            for row in d2_levels
            if row.get("cell_id") == cell_id and int(row.get("step_count", 0)) == 8192
        ]
        if len(matches) != 1:
            raise ValueError(f"N5-D3 D2 H8192 source is not unique: {cell_id}")
        row = matches[0]
        if row.get("pair_id") != parent_cell.get("pair_id") or row.get(
            "role"
        ) != parent_cell.get("role"):
            raise ValueError(f"N5-D3 D2 metadata drifted: {cell_id}")
        entry = row["methods"]["paired_neumaier"]
        array = np.asarray(entry["values"], dtype=np.float64)
        if array.shape != (256, 2) or not np.all(np.isfinite(array)):
            raise ValueError(f"N5-D3 invalid D2 source array: {cell_id}")
        if _legacy_array_sha256(array) != entry["sha256"]:
            raise ValueError(f"N5-D3 D2 source array hash mismatch: {cell_id}")
        path = _resolve(str(config["parent_d2_levels"]))
        expected = {
            "step_count": 8192,
            "reference_method": "paired_neumaier",
            "source_kind": "n5_d2_level",
            "source_path": path.resolve().relative_to(ROOT).as_posix(),
            "source_file_sha256": _sha256(path),
            "source_record_sha256": _canonical_json_sha256(row),
            "source_array_sha256_legacy": entry["sha256"],
            "source_logical_point_queries": int(row["cost"]["logical_point_queries"]),
            "source_identity_sha256_from_parent_n4": _canonical_json_sha256(
                _identity_record(parent_cell)
            ),
            "d2_available_identity_fields": [
                "cell_id",
                "pair_id",
                "role",
                "step_count",
            ],
            "d2_full_identity_bound_via_attested_runner_and_parent_n4": True,
        }
    if int(pack_cell.get("step_count", -1)) != expected["step_count"]:
        raise ValueError(f"N5-D3 selected step drifted: {cell_id}")
    if pack_cell.get("reference_method") != expected["reference_method"]:
        raise ValueError(f"N5-D3 reference method drifted: {cell_id}")
    if pack_cell.get("source_kind") != expected["source_kind"]:
        raise ValueError(f"N5-D3 source kind drifted: {cell_id}")
    provenance = pack_cell.get("provenance", {})
    for key in (
        "source_path",
        "source_file_sha256",
        "source_record_sha256",
        "source_array_sha256_legacy",
        "source_logical_point_queries",
    ):
        if provenance.get(key) != expected[key]:
            raise ValueError(f"N5-D3 provenance drifted for {cell_id}: {key}")
    if expected["source_kind"] == "n5_d2_level":
        for key in (
            "source_identity_sha256_from_parent_n4",
            "d2_available_identity_fields",
            "d2_full_identity_bound_via_attested_runner_and_parent_n4",
        ):
            if provenance.get(key) != expected[key]:
                raise ValueError(f"N5-D3 D2 identity binding drifted for {cell_id}: {key}")
    return array, expected


def validate(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    output = output.resolve()
    config = _read_json(config_path)
    result = _read_json(output / "result.json")
    pack = _read_json(output / "reference_pack.json")
    manifest = _read_json(output / "manifest.json")
    snapshot = _read_json(output / "config_snapshot.json")
    ledger = _read_csv(output / "cell_ledger.csv")
    if not all(isinstance(value, dict) for value in (config, result, pack, manifest, snapshot)):
        raise ValueError("N5-D3 formal bundle contains an invalid JSON object")
    _validate_config_contract(config)
    if snapshot != config:
        raise ValueError("N5-D3 config snapshot drifted")
    _verify_claim_structures(config, result, pack)
    attestation = _verify_attestation(config, config_path)
    _verify_result_manifest(manifest, output, config_path, config)

    n4_result = _read_json(_resolve(str(config["parent_n4_result"])))
    n4_manifest = _read_json(_resolve(str(config["parent_n4_manifest"])))
    n4_validation = _read_json(_resolve(str(config["parent_n4_validation"])))
    recovery_attestation = _read_json(
        _resolve(str(config["parent_n4_recovery_attestation"]))
    )
    recovery_validation = _read_json(
        _resolve(str(config["parent_n4_recovery_validation"]))
    )
    d1_result = _read_json(_resolve(str(config["parent_d1_result"])))
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
    if any(not isinstance(item, dict) for item in dictionaries) or not isinstance(
        d2_levels, list
    ):
        raise ValueError("N5-D3 parent evidence JSON drifted")
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
        raise ValueError("N5-D3 D2 config selected cells drifted")
    if d2_config.get("reference_method") != "paired_neumaier":
        raise ValueError("N5-D3 D2 config reference method drifted")
    if d2_attestation.get("config_sha256") != _sha256(
        _resolve(str(config["parent_d2_config"]))
    ):
        raise ValueError("N5-D3 D2 attestation config hash drifted")
    if d2_attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("N5-D3 D2 attestation output-absence claim drifted")
    d2_attested_runner = d2_attestation.get("attested_files", {}).get("runner", {})
    if d2_attested_runner.get("path") != d2_runner_path or d2_attested_runner.get(
        "sha256"
    ) != d2_runner_manifest.get("sha256"):
        raise ValueError("N5-D3 D2 attested runner chain drifted")
    if n4_validation.get("valid") is not True or recovery_validation.get("valid") is not True:
        raise ValueError("N5-D3 N4 parent validation drifted")
    if d1_validation.get("valid") is not True or d2_validation.get("valid") is not True:
        raise ValueError("N5-D3 D1/D2 parent validation drifted")
    if d1_result.get("machine_decision") != (
        "D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR"
    ):
        raise ValueError("N5-D3 D1 decision drifted")
    if float(d1_result["maximum_route_equivalence_relative_l2"]) > 5e-12:
        raise ValueError("N5-D3 D1 route-equivalence gate drifted")
    if d2_result.get("machine_decision") != "D2_SELECTED_TAIL_RESOLVED_AT_H8192":
        raise ValueError("N5-D3 D2 decision drifted")
    if d2_result.get("protocol_commit") != d2_attestation.get("protocol_commit"):
        raise ValueError("N5-D3 D2 protocol commit binding drifted")
    if int(d2_result.get("selected_cell_count", 0)) != 4 or int(
        d2_result.get("new_level_evaluation_count", 0)
    ) != 8:
        raise ValueError("N5-D3 D2 population contract drifted")
    if [row.get("cell_id") for row in d2_result.get("cells", [])] != list(
        D2_SELECTED_CELLS
    ):
        raise ValueError("N5-D3 D2 selected cell order drifted")
    expected_level_order = [
        (cell_id, step)
        for cell_id in D2_SELECTED_CELLS
        for step in (4096, 8192)
    ]
    if [
        (row.get("cell_id"), int(row.get("step_count", 0))) for row in d2_levels
    ] != expected_level_order:
        raise ValueError("N5-D3 D2 level order drifted")
    if not all(row.get("all_gates_pass") is True for row in d2_result.get("cells", [])):
        raise ValueError("N5-D3 D2 cell gates drifted")

    work = _resolve(str(config["parent_n4_checkpoint_work"]))
    checkpoints = sorted(work.glob(str(config["parent_n4_checkpoint_glob"])))
    merkle = _checkpoint_merkle_root(checkpoints, work)
    if len(checkpoints) != 105 or sum(path.name == "H2048.json" for path in checkpoints) != 9:
        raise ValueError("N5-D3 checkpoint inventory count drifted")
    if merkle != recovery_attestation.get("opaque_checkpoint_merkle_root"):
        raise ValueError("N5-D3 checkpoint Merkle root drifted")
    if merkle != recovery_validation.get("opaque_checkpoint_merkle_root"):
        raise ValueError("N5-D3 recovery validation Merkle root drifted")

    parent_cells = n4_result.get("cells", [])
    pack_cells = pack.get("cells", [])
    if [row.get("cell_id") for row in parent_cells] != list(EXPECTED_ORDER):
        raise ValueError("N5-D3 N4 parent cell order drifted")
    if [row.get("cell_id") for row in pack_cells] != list(EXPECTED_ORDER):
        raise ValueError("N5-D3 pack cell order drifted")
    if len(pack_cells) != 32 or len(ledger) != 32:
        raise ValueError("N5-D3 pack or ledger cell count drifted")
    arrays: list[np.ndarray] = []
    selected_queries = 0
    steps: list[int] = []
    for index, (parent_cell, pack_cell, ledger_row) in enumerate(
        zip(parent_cells, pack_cells, ledger, strict=True)
    ):
        if int(pack_cell.get("pack_index", -1)) != index:
            raise ValueError(f"N5-D3 pack index drifted: {index}")
        identity = _identity_record(parent_cell)
        packed_identity = {key: pack_cell.get(key) for key in identity}
        if packed_identity != identity:
            raise ValueError(f"N5-D3 packed identity drifted: {pack_cell['cell_id']}")
        identity_sha = _canonical_json_sha256(identity)
        if pack_cell.get("identity_sha256") != identity_sha:
            raise ValueError(f"N5-D3 identity hash drifted: {pack_cell['cell_id']}")
        observable = config["observable_contract"]
        if pack_cell.get("observable_id") != observable["observable_id"]:
            raise ValueError(f"N5-D3 observable ID drifted: {pack_cell['cell_id']}")
        if pack_cell.get("units") != observable["units"]:
            raise ValueError(f"N5-D3 units drifted: {pack_cell['cell_id']}")
        if pack_cell.get("coordinate_order") != observable["coordinate_order"]:
            raise ValueError(f"N5-D3 coordinate order drifted: {pack_cell['cell_id']}")
        if pack_cell.get("array_encoding") != config["array_encoding"]:
            raise ValueError(f"N5-D3 array encoding drifted: {pack_cell['cell_id']}")
        expected_selection_basis = (
            "N4_1_final_cellwise_reference_authorized"
            if parent_cell.get("final_cellwise_reference_authorized") is True
            else "N4_1_failed_then_D2_selected_tail_resolved"
        )
        if pack_cell.get("selection_basis") != expected_selection_basis:
            raise ValueError(f"N5-D3 selection basis drifted: {pack_cell['cell_id']}")
        expected_array, source = _expected_source(parent_cell, pack_cell, config, d2_levels)
        packed_array = np.asarray(pack_cell.get("reference_values"), dtype=np.float64)
        if packed_array.shape != (256, 2) or not np.all(np.isfinite(packed_array)):
            raise ValueError(f"N5-D3 invalid packed array: {pack_cell['cell_id']}")
        if not np.array_equal(packed_array, expected_array):
            raise ValueError(f"N5-D3 packed values differ from source: {pack_cell['cell_id']}")
        if pack_cell.get("pack_array_sha256_float64_le_c_order") != _array_sha256(
            packed_array
        ):
            raise ValueError(f"N5-D3 packed array hash drifted: {pack_cell['cell_id']}")
        _same(pack_cell["l2_norm"], np.linalg.norm(packed_array), label=pack_cell["cell_id"])
        _same(pack_cell["finite_fraction"], 1.0, label=f"{pack_cell['cell_id']} finite")
        _verify_ledger_row(pack_cell, ledger_row, index)
        if pack_cell["source_kind"] == "n4_checkpoint":
            checkpoint = _read_json(_resolve(pack_cell["provenance"]["source_path"]))
            expected_companions = {
                "high_output_uv": checkpoint["high_output_uv_sha256"],
                "straight_output_uv": None,
            }
            if pack_cell["provenance"].get("available_companion_hashes") != expected_companions:
                raise ValueError(f"N5-D3 N4 companion disclosure drifted: {pack_cell['cell_id']}")
        else:
            source_rows = [
                row
                for row in d2_levels
                if row.get("cell_id") == pack_cell["cell_id"]
                and int(row.get("step_count", 0)) == 8192
            ]
            expected_monitors = {
                method: details["sha256"]
                for method, details in source_rows[0]["methods"].items()
            }
            if pack_cell["provenance"].get("available_monitor_hashes") != expected_monitors:
                raise ValueError(f"N5-D3 D2 monitor disclosure drifted: {pack_cell['cell_id']}")
        selected_queries += source["source_logical_point_queries"]
        steps.append(source["step_count"])
        arrays.append(packed_array)

    stacked = np.stack(arrays, axis=0)
    allocation = {
        "H1024_raw_separate_subtraction": sum(step == 1024 for step in steps),
        "H2048_raw_separate_subtraction": sum(step == 2048 for step in steps),
        "H8192_paired_neumaier": sum(step == 8192 for step in steps),
    }
    expected_allocation = {
        "H1024_raw_separate_subtraction": 23,
        "H2048_raw_separate_subtraction": 7,
        "H8192_paired_neumaier": 2,
    }
    if allocation != expected_allocation or result.get("allocation_counts") != allocation:
        raise ValueError("N5-D3 allocation summary drifted")
    cell_order_sha = _canonical_json_sha256(list(EXPECTED_ORDER))
    stacked_sha = _array_sha256(stacked)
    if pack.get("cell_order_sha256") != cell_order_sha or result.get(
        "cell_order_sha256"
    ) != cell_order_sha:
        raise ValueError("N5-D3 cell-order hash drifted")
    if pack.get("stacked_array_sha256_float64_le_c_order") != stacked_sha or result.get(
        "stacked_array_sha256_float64_le_c_order"
    ) != stacked_sha:
        raise ValueError("N5-D3 stacked array hash drifted")
    if result.get("stacked_array_shape") != [32, 256, 2]:
        raise ValueError("N5-D3 stacked shape drifted")
    if int(result.get("cell_count", -1)) != 32 or int(
        result.get("unique_cell_count", -1)
    ) != 32:
        raise ValueError("N5-D3 result cell count drifted")
    if int(result.get("selected_reference_source_logical_point_queries", -1)) != selected_queries:
        raise ValueError("N5-D3 selected source query ledger drifted")
    if int(result.get("assembly_logical_point_queries", -1)) != 0:
        raise ValueError("N5-D3 assembly query ledger drifted")
    if result.get("mixed_reference_pack") is not True or result.get(
        "uniform_paired_reference"
    ) is not False:
        raise ValueError("N5-D3 mixed semantic disclosure drifted")
    if int(result.get("paired_equivalence_coverage_count", -1)) != 4:
        raise ValueError("N5-D3 paired equivalence coverage drifted")
    if result.get("n4_checkpoint_merkle_root") != merkle:
        raise ValueError("N5-D3 result checkpoint Merkle root drifted")
    if result.get("candidate_id") != config["candidate_id"] or pack.get(
        "candidate_id"
    ) != config["candidate_id"]:
        raise ValueError("N5-D3 candidate ID drifted")
    if result.get("protocol_commit") != attestation["protocol_commit"] or pack.get(
        "protocol_commit"
    ) != attestation["protocol_commit"]:
        raise ValueError("N5-D3 protocol commit drifted")
    if result.get("machine_decision") != "D3_VALID_MIXED_RESIDUAL_REFERENCE_ONLY":
        raise ValueError("N5-D3 machine decision drifted")
    expected_gate_keys = {
        "cell_count_gate_met",
        "unique_cell_gate_met",
        "cell_order_gate_met",
        "h1024_count_gate_met",
        "h2048_count_gate_met",
        "h8192_count_gate_met",
        "method_mapping_gate_met",
        "finite_gate_met",
        "shape_gate_met",
        "n4_merkle_gate_met",
        "d1_route_gate_met",
        "d2_tail_gate_met",
        "mixed_semantics_disclosed_gate_met",
        "assembly_zero_query_gate_met",
    }
    if set(result.get("gate_results", {})) != expected_gate_keys:
        raise ValueError("N5-D3 gate key set drifted")
    if result.get("all_gates_pass") is not True or not all(
        result["gate_results"].values()
    ):
        raise ValueError("N5-D3 gate summary drifted")
    if pack.get("machine_decision") != result["machine_decision"]:
        raise ValueError("N5-D3 pack/result decision drifted")
    figure = _verify_figure(output / "n2_pvgr_n5_d3_adaptive_reference.png")
    summary = (output / "summary.md").read_text(encoding="utf-8")
    if "mixed-method infrastructure" not in summary or "All broad claim authorizations remain false" not in summary:
        raise ValueError("N5-D3 summary claim boundary drifted")
    return {
        "schema": "n2-pvgr-n5-d3-adaptive-reference-validation-1.0",
        "valid": True,
        "machine_decision": result["machine_decision"],
        "protocol_commit": attestation["protocol_commit"],
        "independent_validator_imports_runner": False,
        "parent_manifests_and_validations_verified": True,
        "n4_checkpoint_inventory_recomputed": True,
        "n4_checkpoint_merkle_root": merkle,
        "source_mapping_and_hashes_recomputed": True,
        "array_hashes_and_stacked_hash_recomputed": True,
        "cell_order_and_23_7_2_allocation_recomputed": True,
        "query_ledger_recomputed": True,
        "mixed_semantics_and_claim_boundary_verified": True,
        "cell_count": len(pack_cells),
        "allocation_counts": allocation,
        "stacked_array_sha256_float64_le_c_order": stacked_sha,
        "selected_reference_source_logical_point_queries": selected_queries,
        "figure": figure,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _read_json(args.config.resolve())
    if not isinstance(config, dict):
        raise ValueError("N5-D3 config is invalid")
    output = args.output or _resolve(str(config["formal_output"]))
    report = args.report or output / "validation_report.json"
    validation = validate(args.config.resolve(), output.resolve())
    report.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
