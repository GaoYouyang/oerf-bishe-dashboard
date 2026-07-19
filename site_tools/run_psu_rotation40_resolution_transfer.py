#!/usr/bin/env python3
"""Audit whether a frozen 16-to-32-cubed support gain transfers to rotation 40."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo_t16_operator.psu_b0_streaming_operator import (
    PSUB0StreamingOperator,
    zero_outer_boundary_support,
)
from site_tools.psu_rotation40_active_store import PSURotation40ActiveRayStore
from site_tools.run_psu_rotation40_b0_reprojection import metric_row


CONFIG_SCHEMA = "psu-rotation40-resolution-transfer-preregistered-1.0"
CONFIG_STATUS = "FROZEN_BEFORE_16_CUBED_ROTATION40_SCORE"
REPORT_SCHEMA = "psu-rotation40-resolution-transfer-public-report-1.0"
ATTESTATION_SCHEMA = "psu-rotation40-resolution-transfer-attestation-1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _git(*args: str, binary: bool = False, check: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
    )
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8").strip()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("resolution-transfer schema changed")
    if config.get("status") != CONFIG_STATUS:
        raise ValueError("resolution-transfer config was not frozen before scoring")
    if config.get("scope") != (
        "ALREADY_OPENED_ROTATION40_DEVELOPMENT_RESOLUTION_TRANSFER_AUDIT_"
        "NO_FIELD_TRUTH_NO_FINAL_ROTATION"
    ):
        raise ValueError("resolution-transfer evidence scope changed")
    expected_attested_files = {
        "runner": "site_tools/run_psu_rotation40_resolution_transfer.py",
        "attestation_creator": "site_tools/create_psu_rotation40_resolution_transfer_attestation.py",
        "test": "site_tools/test_run_psu_rotation40_resolution_transfer.py",
        "active_store": "site_tools/psu_rotation40_active_store.py",
        "forward_operator": "demo_t16_operator/psu_b0_streaming_operator.py",
        "metric_implementation": "site_tools/run_psu_rotation40_b0_reprojection.py",
        "preregistration_note": "docs/psu_rotation40_resolution_transfer_prereg_2026-07-19.md",
    }
    if config.get("pre_registration_attestation") != (
        "demo_t16_operator/configs/psu_rotation40_resolution_transfer_attestation_v1.json"
    ):
        raise ValueError("resolution-transfer attestation path changed")
    if config.get("formal_output") != (
        "demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1"
    ):
        raise ValueError("resolution-transfer formal output changed")
    if config.get("attested_files") != expected_attested_files:
        raise ValueError("resolution-transfer attested file set changed")
    dataset = config.get("dataset")
    expected_dataset = {
        "doi": "10.26208/1VE2-5C19",
        "rotation_degrees": 40,
        "camera_ids": [2, 3, 4],
        "support_camera_ids": [2, 3, 4],
        "support_rotation_degrees": [0, 50, 90],
        "held_out_unit": "ROTATION_RUN_NOT_CAMERA",
        "independent_rotation_block_count": 1,
        "camera_rows_are_not_independent_repeats": True,
        "selection_mode": "all_rotation40_active_rows",
        "row_order": "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON",
    }
    if not isinstance(dataset, Mapping) or dict(dataset) != expected_dataset:
        raise ValueError("rotation-40 dataset contract changed")
    expected_inputs = {
        "rotation40_payload_private_report_sha256": "4a90fdbe17bd5dc168679f36e003f51eacb336bce382356242cf6b2c4cf11b79",
        "rotation40_payload_public_summary_sha256": "3ee408c5c61ab2ad14431b5ff40fd9c7c179d415d2692a7533fb0cf43b13c8b5",
        "resolution_comparison_public_summary_sha256": "b23305f0f1436dd4fa49d9359be8d4916f7eca701708bac862fb0a00f0d6408a",
    }
    if config.get("input_bindings") != expected_inputs:
        raise ValueError("rotation-40 input bindings changed")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("exactly two frozen resolution candidates are required")
    expected_candidates = (
        {
            "candidate_id": "support_cgls4_16cubed",
            "grid_shape_zyx": [16, 16, 16],
            "volume_sha256": "3b2e87b53707475ea1667ab55288f22d8b24c355cadd4e9d24ec651a84fd2e9e",
            "volume_dtype": "float64",
            "support_relative_l2": 0.7877106877179901,
            "support_fit_role": "FROZEN_SAME_CALL_LOW_RESOLUTION_REFERENCE",
            "support_score_known_before_rotation40_protocol_freeze": True,
            "nominal_grid_value_count": 4096,
            "free_interior_value_count": 2744,
            "grid_node_spacing_xyz_m": [0.014666666666666666] * 3,
            "private_report_sha256": "71c9bed2f4f62b9621a6e47368c5bab8754bb067bbb0cb0d88aea0bdd72b249a",
            "public_summary_sha256": "daebe042ec3cb9eaa9ec9c4c348754043854a5aeda52e10693f6e8864bce0bb3",
        },
        {
            "candidate_id": "support_cgls4_32cubed",
            "grid_shape_zyx": [32, 32, 32],
            "volume_sha256": "e72a8709d12329a96d6f9a012019f09fe3ceebc11e9c9a3b2daa5bea3835df71",
            "volume_dtype": "float64",
            "support_relative_l2": 0.6271324683999563,
            "support_fit_role": "FROZEN_CACHED_REFERENCE_NUMERICALLY_EQUIVALENT_TO_SAME_CALL_32_CUBED",
            "support_score_known_before_rotation40_protocol_freeze": True,
            "nominal_grid_value_count": 32768,
            "free_interior_value_count": 27000,
            "grid_node_spacing_xyz_m": [0.0070967741935483875] * 3,
            "private_report_sha256": "e0caa152eba4bede08821ac18a35f8da79a396e22365ec929297076c6172d5d3",
            "public_summary_sha256": "a81340366a2f47196c0b25816538bd7767f848fd54ca2e75db660491be58b70d",
        },
    )
    for candidate, expected in zip(candidates, expected_candidates, strict=True):
        if not isinstance(candidate, Mapping) or dict(candidate) != expected:
            raise ValueError("candidate identity, score, complexity, or provenance changed")
    geometry = config.get("geometry_binding")
    expected_geometry = {
        "private_report_sha256": "9b426dbec49d898267714749a952f25355e6c84bd5d8e97782f80283b6244a90",
        "public_summary_sha256": "f7fcbc55004a40829fab4781a7ae7167d17f55ce0147fa8313cf0ba1a8008d98",
    }
    if not isinstance(geometry, Mapping) or dict(geometry) != expected_geometry:
        raise ValueError("rotation-40 geometry binding changed")
    expected_support_contract = {
        "support_view_count": 9,
        "support_camera_ids": [2, 3, 4],
        "support_rotation_degrees": [0, 50, 90],
        "selection_mode": "all_active_rows",
        "measurement_weighting": "UNWEIGHTED_VECTOR_L2_OVER_ALL_SELECTED_SUPPORT_ROWS",
        "solver": "CGLS",
        "start": "zero_field",
        "fixed_iterations": 4,
        "logical_forward_calls": 4,
        "logical_adjoint_calls": 5,
        "iteration_budget_selected_on_support_or_development": False,
        "positivity": False,
        "finite_aperture_sample_count": 16,
        "gauge": "zero_one_grid_node_outer_boundary",
        "grid_minimum_xyz_m": [-0.11, -0.11, -0.11],
        "grid_maximum_xyz_m": [0.11, 0.11, 0.11],
    }
    if config.get("support_reconstruction_contract") != expected_support_contract:
        raise ValueError("support reconstruction provenance changed")
    forward = config.get("forward")
    expected_forward = {
        "grid_minimum_xyz_m": [-0.11, -0.11, -0.11],
        "grid_maximum_xyz_m": [0.11, 0.11, 0.11],
        "gauge": "zero_one_grid_node_outer_boundary",
        "finite_aperture_sample_count": 16,
        "chunk_rays": 32768,
        "dtype": "float64",
        "device": "cpu",
        "torch_threads": 8,
    }
    if not isinstance(forward, Mapping) or any(
        forward.get(key) != value for key, value in expected_forward.items()
    ):
        raise ValueError("frozen forward contract changed")
    expected_metrics = [
        "vector_relative_l2",
        "component_rmse_px",
        "component_mae_px",
        "residual_magnitude_p95_px",
        "measured_vector_rms_px",
        "predicted_vector_rms_px",
    ]
    if config.get("metrics") != expected_metrics:
        raise ValueError("metric list or order changed")
    decision = config.get("decision")
    expected_decision = {
        "primary_metric": "pooled_vector_relative_l2_absolute_improvement_16_minus_32",
        "pooled_weighting": "RAY_COUNT_WEIGHTED_OVER_ALL_SELECTED_ROWS",
        "minimum_predeclared_numerical_absolute_improvement": 0.01,
        "threshold_interpretation": "NUMERICAL_SCREEN_ONLY_NOT_A_PHYSICAL_OR_PRACTICAL_SIGNIFICANCE_BOUND",
        "require_all_three_cameras_nonworse": True,
        "camera_nonworse_tolerance": 0.0,
        "report_equal_camera_macro_average": True,
        "report_worst_camera_delta": True,
        "no_amplitude_rescaling": True,
        "no_parameter_or_iteration_selection": True,
        "no_new_candidate_comparison": True,
        "not_an_algorithm_or_compute_fairness_comparison": True,
    }
    if not isinstance(decision, Mapping) or any(
        decision.get(key) != value for key, value in expected_decision.items()
    ):
        raise ValueError("result-before-frozen decision contract changed")
    expected_firewall = {
        "algorithm_superiority": False,
        "experimental_field_truth_available": False,
        "field_relative_l2_available": False,
        "final_rotations_opened": False,
        "cross_session_generalization": False,
        "independent_camera_replication": False,
        "practical_significance_established": False,
        "compute_fair_comparison": False,
        "neural_operator_trained": False,
        "publish_predictions_or_measurement_arrays": False,
        "paper_claim_authorized": False,
    }
    firewall = config.get("claim_firewall")
    if not isinstance(firewall, Mapping) or dict(firewall) != expected_firewall:
        raise ValueError("claim firewall must remain complete and false")


def _attested_external_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_bindings": dict(config["input_bindings"]),
        "geometry_binding": dict(config["geometry_binding"]),
        "candidate_bindings": [
            {
                key: candidate[key]
                for key in (
                    "candidate_id",
                    "volume_sha256",
                    "private_report_sha256",
                    "public_summary_sha256",
                    "support_relative_l2",
                )
            }
            for candidate in config["candidates"]
        ],
        "support_reconstruction_contract": dict(
            config["support_reconstruction_contract"]
        ),
    }


def verify_attestation(
    config: Mapping[str, Any],
    config_path: Path,
    attestation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    expected_attestation = (ROOT / str(config["pre_registration_attestation"])).resolve()
    if attestation_path.resolve() != expected_attestation:
        raise ValueError("attestation path does not match the frozen config")
    expected_output = (ROOT / str(config["formal_output"])).resolve()
    if output_dir.resolve() != expected_output:
        raise ValueError("output path does not match the frozen config")
    attestation = _read_json(attestation_path)
    if attestation.get("schema_version") != ATTESTATION_SCHEMA:
        raise ValueError("resolution-transfer attestation schema changed")
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("attestation does not prove pre-result creation")
    if attestation.get("formal_output") != config["formal_output"]:
        raise ValueError("attestation formal output changed")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("attestation config hash changed")
    if attestation.get("external_input_bindings") != _attested_external_inputs(config):
        raise ValueError("attestation external input bindings changed")
    expected_holdout = {
        "support_camera_ids": [2, 3, 4],
        "support_rotation_degrees": [0, 50, 90],
        "scored_camera_ids": [2, 3, 4],
        "scored_rotation_degrees": 40,
        "held_out_unit": "ROTATION_RUN_NOT_CAMERA",
        "independent_rotation_block_count": 1,
    }
    if attestation.get("held_out_design") != expected_holdout:
        raise ValueError("attestation holdout identity changed")
    protocol_commit = str(attestation.get("protocol_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_commit):
        raise ValueError("attestation protocol commit is invalid")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("protocol commit is not an ancestor of HEAD")
    config_relative = _repo_relative(config_path)
    frozen_config = _git("show", f"{protocol_commit}:{config_relative}", binary=True)
    if hashlib.sha256(frozen_config).hexdigest() != _sha256(config_path):
        raise ValueError("current config differs from the protocol commit")
    attested_files = attestation.get("attested_files")
    if not isinstance(attested_files, Mapping) or set(attested_files) != set(
        config["attested_files"]
    ):
        raise ValueError("attestation file set changed")
    for key, relative_value in config["attested_files"].items():
        relative = str(relative_value)
        entry = attested_files[key]
        if not isinstance(entry, Mapping) or entry.get("path") != relative:
            raise ValueError(f"attested path changed: {key}")
        current = (ROOT / relative).resolve()
        if ROOT not in current.parents or not current.is_file():
            raise ValueError(f"attested file is absent or escapes repository: {key}")
        current_digest = _sha256(current)
        frozen = _git("show", f"{protocol_commit}:{relative}", binary=True)
        frozen_digest = hashlib.sha256(frozen).hexdigest()
        if current_digest != entry.get("sha256") or frozen_digest != current_digest:
            raise ValueError(f"attested file changed after protocol freeze: {key}")
    attestation_relative = _repo_relative(attestation_path)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", attestation_relative],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if tracked.returncode != 0:
        raise ValueError("attestation is not Git-tracked")
    if str(_git("status", "--porcelain", "--", attestation_relative)):
        raise ValueError("attestation has uncommitted changes")
    committed_attestation = _git("show", f"HEAD:{attestation_relative}", binary=True)
    if hashlib.sha256(committed_attestation).hexdigest() != _sha256(attestation_path):
        raise ValueError("attestation bytes are not bound by HEAD")
    return attestation


def _verify_npy(path: Path, manifest_entry: Mapping[str, Any]) -> None:
    if not path.is_file() or _sha256(path) != manifest_entry.get("sha256"):
        raise ValueError(f"private array checksum changed: {path.name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != manifest_entry.get("shape"):
        raise ValueError(f"private array shape changed: {path.name}")
    if str(array.dtype) != manifest_entry.get("dtype"):
        raise ValueError(f"private array dtype changed: {path.name}")


def _verify_candidate_provenance(
    candidate: Mapping[str, Any],
    volume_path: Path,
    private_report_path: Path,
    public_summary_path: Path,
) -> None:
    if _sha256(volume_path) != candidate["volume_sha256"]:
        raise ValueError(f"volume checksum changed: {candidate['candidate_id']}")
    if _sha256(private_report_path) != candidate["private_report_sha256"]:
        raise ValueError(f"private provenance changed: {candidate['candidate_id']}")
    if _sha256(public_summary_path) != candidate["public_summary_sha256"]:
        raise ValueError(f"public provenance changed: {candidate['candidate_id']}")
    report = _read_json(private_report_path)
    shape = list(candidate["grid_shape_zyx"])
    expected_volume = {
        "filename": volume_path.name,
        "sha256": candidate["volume_sha256"],
        "shape": [1, 1, *shape],
        "dtype": "float64",
    }
    if report.get("private_volume") != expected_volume:
        raise ValueError(f"private volume provenance mismatch: {candidate['candidate_id']}")
    configuration = report.get("configuration", {})
    expected_configuration = {
        "grid_shape_zyx": shape,
        "dtype": "float64",
        "device": "cpu",
        "gauge": "zero_one_voxel_outer_boundary",
        "finite_aperture_sample_count": 16,
        "cgls_fixed_iterations": 4,
        "positivity": False,
    }
    if any(configuration.get(key) != value for key, value in expected_configuration.items()):
        raise ValueError(f"support reconstruction configuration changed: {candidate['candidate_id']}")
    optimization = report.get("optimization", {})
    expected_optimization = {
        "solver": "CGLS",
        "start": "zero_field",
        "fixed_iteration_budget": 4,
        "logical_forward_calls": 4,
        "logical_adjoint_calls": 5,
        "iteration_budget_selected_on_support_or_development": False,
    }
    if any(optimization.get(key) != value for key, value in expected_optimization.items()):
        raise ValueError(f"support solver provenance changed: {candidate['candidate_id']}")
    if report.get("selection", {}).get("selection_mode") != "all_active_rows":
        raise ValueError(f"support row selection changed: {candidate['candidate_id']}")
    score = report.get("evaluation", {}).get("direct_support_relative_measurement_l2")
    if score != candidate["support_relative_l2"]:
        raise ValueError(f"support score provenance changed: {candidate['candidate_id']}")


def verify_private_inputs(
    *,
    config: Mapping[str, Any],
    geometry_root: Path,
    geometry_private_report_path: Path,
    geometry_public_summary_path: Path,
    payload_root: Path,
    payload_private_report_path: Path,
    payload_public_summary_path: Path,
    resolution_public_summary_path: Path,
    volume_paths: Mapping[str, Path],
    candidate_private_report_paths: Mapping[str, Path],
    candidate_public_summary_paths: Mapping[str, Path],
) -> None:
    geometry_binding = config["geometry_binding"]
    if _sha256(geometry_private_report_path) != geometry_binding["private_report_sha256"]:
        raise ValueError("private geometry report checksum changed")
    if _sha256(geometry_public_summary_path) != geometry_binding["public_summary_sha256"]:
        raise ValueError("public geometry summary checksum changed")
    inputs = config["input_bindings"]
    if _sha256(payload_private_report_path) != inputs[
        "rotation40_payload_private_report_sha256"
    ]:
        raise ValueError("private rotation-40 payload report checksum changed")
    if _sha256(payload_public_summary_path) != inputs[
        "rotation40_payload_public_summary_sha256"
    ]:
        raise ValueError("public rotation-40 payload summary checksum changed")
    if _sha256(resolution_public_summary_path) != inputs[
        "resolution_comparison_public_summary_sha256"
    ]:
        raise ValueError("support resolution comparison summary checksum changed")
    geometry_report = _read_json(geometry_private_report_path)
    payload_report = _read_json(payload_private_report_path)
    geometry_rows = {
        int(row["camera_id"]): row for row in geometry_report.get("camera_manifests", [])
    }
    payload_rows = {
        int(row["camera_id"]): row for row in payload_report.get("shard_manifests", [])
    }
    if set(geometry_rows) != {2, 3, 4} or set(payload_rows) != {2, 3, 4}:
        raise ValueError("private manifest camera set changed")
    for camera_id in (2, 3, 4):
        geometry_directory = geometry_root / f"camera_{camera_id:02d}"
        geometry_manifest_path = geometry_directory / "geometry_manifest.json"
        geometry_manifest = _read_json(geometry_manifest_path)
        if geometry_manifest != geometry_rows[camera_id]:
            raise ValueError(f"geometry manifest changed: camera {camera_id}")
        payload_directory = payload_root / f"camera_{camera_id:02d}"
        payload_manifest_path = payload_directory / "shard_manifest.json"
        payload_manifest = _read_json(payload_manifest_path)
        if payload_manifest != payload_rows[camera_id]:
            raise ValueError(f"payload manifest changed: camera {camera_id}")
        if _sha256(payload_manifest_path) != geometry_manifest[
            "observation_manifest_sha256"
        ]:
            raise ValueError(f"payload-to-geometry binding changed: camera {camera_id}")
        for filename, entry in geometry_manifest["files"].items():
            _verify_npy(geometry_directory / filename, entry)
        for filename, entry in payload_manifest["files"].items():
            _verify_npy(payload_directory / filename, entry)
    for candidate in config["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id not in volume_paths:
            raise ValueError(f"missing volume path: {candidate_id}")
        if candidate_id not in candidate_private_report_paths:
            raise ValueError(f"missing private provenance path: {candidate_id}")
        if candidate_id not in candidate_public_summary_paths:
            raise ValueError(f"missing public provenance path: {candidate_id}")
        _verify_candidate_provenance(
            candidate,
            volume_paths[candidate_id],
            candidate_private_report_paths[candidate_id],
            candidate_public_summary_paths[candidate_id],
        )


def validate_public_report(
    report: Mapping[str, Any], *, forbidden_path_tokens: tuple[str, ...] = ()
) -> None:
    policy = report.get("public_export_policy")
    expected_policy = {
        "contains_predictions": False,
        "contains_measurements": False,
        "contains_geometry_arrays": False,
        "contains_volumes": False,
        "contains_only_aggregate_metrics": True,
    }
    if policy != expected_policy:
        raise ValueError("public export policy changed")
    forbidden_keys = {
        "predictions",
        "prediction_values",
        "measurements",
        "measurement_values",
        "observations",
        "ray_geometry",
        "geometry_values",
        "volume_values",
        "local_paths",
        "private_paths",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys.intersection(str(key) for key in value):
                raise ValueError("public report contains a forbidden payload key")
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            if value.startswith("/Users/") or "private_library/" in value:
                raise ValueError("public report contains a private path")
            if any(token and token in value for token in forbidden_path_tokens):
                raise ValueError("public report contains a supplied private path")
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            raise ValueError("public report contains a numeric array")

    walk(report)


def compare_resolution_metrics(
    metrics_16: Mapping[str, Any],
    metrics_32: Mapping[str, Any],
    *,
    minimum_predeclared_numerical_absolute_improvement: float,
    camera_nonworse_tolerance: float,
) -> dict[str, Any]:
    aggregate_16 = metrics_16["aggregate"]
    aggregate_32 = metrics_32["aggregate"]
    pooled_improvement = float(aggregate_16["vector_relative_l2"]) - float(
        aggregate_32["vector_relative_l2"]
    )
    rows_16 = {int(row["camera_id"]): row for row in metrics_16["per_camera"]}
    rows_32 = {int(row["camera_id"]): row for row in metrics_32["per_camera"]}
    if set(rows_16) != {2, 3, 4} or set(rows_32) != {2, 3, 4}:
        raise ValueError("both candidates must contain cameras 2, 3, and 4")
    camera_rows = []
    for camera_id in (2, 3, 4):
        value_16 = float(rows_16[camera_id]["vector_relative_l2"])
        value_32 = float(rows_32[camera_id]["vector_relative_l2"])
        improvement = value_16 - value_32
        nonworse = value_32 <= value_16 + float(camera_nonworse_tolerance)
        camera_rows.append(
            {
                "camera_id": camera_id,
                "vector_relative_l2_16": value_16,
                "vector_relative_l2_32": value_32,
                "absolute_improvement_16_minus_32": improvement,
                "nonworse": bool(nonworse),
            }
        )
    macro_16 = float(np.mean([row["vector_relative_l2_16"] for row in camera_rows]))
    macro_32 = float(np.mean([row["vector_relative_l2_32"] for row in camera_rows]))
    macro_improvement = macro_16 - macro_32
    worst_camera_improvement = float(
        min(row["absolute_improvement_16_minus_32"] for row in camera_rows)
    )
    numerical_gate = pooled_improvement >= float(
        minimum_predeclared_numerical_absolute_improvement
    )
    all_camera_nonworse = all(row["nonworse"] for row in camera_rows)
    if numerical_gate and all_camera_nonworse:
        decision = "RESOLUTION_TRANSFER_SIGNAL_PASS_NO_FIELD_TRUTH"
    elif numerical_gate:
        decision = "POOLED_TRANSFER_WITH_CAMERA_HARM_NO_GO"
    else:
        decision = "SUPPORT_RESOLUTION_GAIN_DID_NOT_CLEAR_NUMERICAL_TRANSFER_GATE_NO_GO"
    return {
        "pooled_vector_relative_l2_16": float(
            aggregate_16["vector_relative_l2"]
        ),
        "pooled_vector_relative_l2_32": float(
            aggregate_32["vector_relative_l2"]
        ),
        "pooled_absolute_improvement_16_minus_32": pooled_improvement,
        "pooled_weighting": "RAY_COUNT_WEIGHTED_OVER_ALL_SELECTED_ROWS",
        "equal_camera_macro_relative_l2_16": macro_16,
        "equal_camera_macro_relative_l2_32": macro_32,
        "equal_camera_macro_absolute_improvement_16_minus_32": macro_improvement,
        "worst_camera_absolute_improvement_16_minus_32": worst_camera_improvement,
        "minimum_predeclared_numerical_absolute_improvement": float(
            minimum_predeclared_numerical_absolute_improvement
        ),
        "predeclared_numerical_pooled_improvement": bool(numerical_gate),
        "practical_significance_established": False,
        "all_three_cameras_nonworse": bool(all_camera_nonworse),
        "independent_rotation_block_count": 1,
        "camera_rows": camera_rows,
        "machine_decision": decision,
    }


def _candidate_metrics(
    *,
    candidate: Mapping[str, Any],
    volume_path: Path,
    store: PSURotation40ActiveRayStore,
    measured: torch.Tensor,
    forward: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    if _sha256(volume_path) != candidate["volume_sha256"]:
        raise ValueError(f"volume checksum changed: {candidate['candidate_id']}")
    volume_numpy = np.load(volume_path, allow_pickle=False)
    shape = tuple(int(value) for value in candidate["grid_shape_zyx"])
    if volume_numpy.shape != (1, 1, *shape) or str(volume_numpy.dtype) != "float64":
        raise ValueError(f"volume shape or dtype changed: {candidate['candidate_id']}")
    support = zero_outer_boundary_support(shape, dtype=torch.float64)
    operator = PSUB0StreamingOperator(
        ray_store=store,
        grid_shape=shape,
        grid_minimum_xyz=tuple(forward["grid_minimum_xyz_m"]),
        grid_maximum_xyz=tuple(forward["grid_maximum_xyz_m"]),
        support=support,
        dtype=torch.float64,
    )
    volume = torch.from_numpy(volume_numpy).to(dtype=torch.float64)
    with torch.no_grad():
        predicted = operator(volume)
    measured_numpy = measured[0].cpu().numpy()
    predicted_numpy = predicted[0].cpu().numpy()
    aggregate = metric_row(measured_numpy, predicted_numpy)
    per_camera = []
    start = 0
    for camera in store.cameras:
        stop = start + camera.ray_count
        row = metric_row(
            measured_numpy[start:stop],
            predicted_numpy[start:stop],
        )
        row["camera_id"] = int(camera.camera_id)
        per_camera.append(row)
        start = stop
    calls = operator.call_report()
    hit_count = int(sum(record["b0_hit_count"] for record in calls["records"]))
    gates = {
        "all_selected_active_rays_hit_b0": hit_count == store.ray_count,
        "prediction_finite": bool(np.all(np.isfinite(predicted_numpy))),
        "single_logical_forward_call": calls["forward_calls"] == 1,
        "outer_boundary_zero": float(
            torch.max(torch.abs(volume * (1.0 - support))).item()
        )
        == 0.0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"candidate mechanical gate failed: {candidate['candidate_id']}")
    records = calls.get("records", [])
    forward_wall_seconds = float(
        sum(float(record.get("wall_seconds", 0.0)) for record in records)
    )
    peak_rss_bytes = int(
        max((int(record.get("max_rss_bytes_after_call", 0)) for record in records), default=0)
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "grid_shape_zyx": list(shape),
        "volume_sha256": candidate["volume_sha256"],
        "support_relative_l2": float(candidate["support_relative_l2"]),
        "support_score_known_before_rotation40_protocol_freeze": True,
        "nominal_grid_value_count": int(candidate["nominal_grid_value_count"]),
        "free_interior_value_count": int(candidate["free_interior_value_count"]),
        "grid_node_spacing_xyz_m": list(candidate["grid_node_spacing_xyz_m"]),
        "aggregate": aggregate,
        "per_camera": per_camera,
        "runtime": {
            "logical_forward_calls": int(calls["forward_calls"]),
            "logical_adjoint_calls": int(calls["adjoint_calls"]),
            "forward_wall_seconds": forward_wall_seconds,
            "candidate_end_to_end_wall_seconds": float(time.perf_counter() - started),
            "peak_rss_bytes_after_call": peak_rss_bytes,
        },
        "gates": gates,
    }


def _write_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "scope",
        "camera_id",
        "ray_count",
        "vector_relative_l2",
        "component_rmse_px",
        "component_mae_px",
        "residual_magnitude_p95_px",
        "measured_vector_rms_px",
        "predicted_vector_rms_px",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            aggregate = dict(candidate["aggregate"])
            writer.writerow(
                {
                    "candidate_id": candidate["candidate_id"],
                    "scope": "pooled",
                    "camera_id": "",
                    **aggregate,
                }
            )
            for camera in candidate["per_camera"]:
                row = dict(camera)
                camera_id = row.pop("camera_id")
                writer.writerow(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "scope": "camera",
                        "camera_id": camera_id,
                        **row,
                    }
                )


def _plot(report: Mapping[str, Any], png: Path, pdf: Path) -> None:
    candidates = report["candidates"]
    labels = ["pooled", "cam 2", "cam 3", "cam 4"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    colors = ("#3e6b89", "#d97745")
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.34
    for index, candidate in enumerate(candidates):
        values = [candidate["aggregate"]["vector_relative_l2"]] + [
            row["vector_relative_l2"] for row in candidate["per_camera"]
        ]
        axes[0, 0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=candidate["candidate_id"].replace("support_cgls4_", ""),
            color=colors[index],
        )
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("vector relative L2")
    axes[0, 0].set_title("Already-opened rotation-40 development observations")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.2)

    support = [candidate["support_relative_l2"] for candidate in candidates]
    rotation = [candidate["aggregate"]["vector_relative_l2"] for candidate in candidates]
    axes[0, 1].plot((16, 32), support, "o-", color="#2f7d6d", label="nine support views")
    axes[0, 1].plot((16, 32), rotation, "s-", color="#b24c63", label="rotation 40")
    axes[0, 1].set_xticks((16, 32), ("16 cubed", "32 cubed"))
    axes[0, 1].set_ylabel("vector relative L2")
    axes[0, 1].set_title("Support fit versus held-out rotation-run transfer")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    for index, candidate in enumerate(candidates):
        measured = [candidate["aggregate"]["measured_vector_rms_px"]] + [
            row["measured_vector_rms_px"] for row in candidate["per_camera"]
        ]
        predicted = [candidate["aggregate"]["predicted_vector_rms_px"]] + [
            row["predicted_vector_rms_px"] for row in candidate["per_camera"]
        ]
        if index == 0:
            axes[1, 0].plot(x, measured, "o--", color="#444444", label="measured")
        axes[1, 0].plot(
            x,
            predicted,
            "o-",
            color=colors[index],
            label=candidate["candidate_id"].replace("support_cgls4_", ""),
        )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("vector RMS / px")
    axes[1, 0].set_title("Prediction amplitude without post-hoc scaling")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].axis("off")
    comparison = report["comparison"]
    lines = [
        "PSU ROTATION-40 RESOLUTION TRANSFER",
        "",
        f"Decision: {comparison['machine_decision']}",
        f"Pooled 16 - 32: {comparison['pooled_absolute_improvement_16_minus_32']:+.6f}",
        "Numerical screen: >= "
        f"{comparison['minimum_predeclared_numerical_absolute_improvement']:.3f}",
        f"All cameras nonworse: {comparison['all_three_cameras_nonworse']}",
        "Same cameras; one held-out rotation block, not 3 repeats.",
        "",
        "Real observations, one opened rotation, no volumetric truth.",
        "Not an algorithm/compute-fairness comparison.",
        "No amplitude fit, iteration selection, or new algorithm.",
        "Final rotations remain sealed.",
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    fig.suptitle(
        "Does a same-call grid-resolution gain transfer to a held-out rotation run?",
        fontsize=14,
    )
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def run_resolution_transfer(
    *,
    config_path: Path,
    attestation_path: Path,
    geometry_root: Path,
    geometry_private_report_path: Path,
    geometry_public_summary_path: Path,
    payload_root: Path,
    payload_private_report_path: Path,
    payload_public_summary_path: Path,
    resolution_public_summary_path: Path,
    volume_paths: Mapping[str, Path],
    candidate_private_report_paths: Mapping[str, Path],
    candidate_public_summary_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    validate_config(config)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    attestation = verify_attestation(
        config,
        config_path,
        attestation_path.resolve(),
        output_dir.resolve(),
    )
    verify_private_inputs(
        config=config,
        geometry_root=geometry_root.resolve(),
        geometry_private_report_path=geometry_private_report_path.resolve(),
        geometry_public_summary_path=geometry_public_summary_path.resolve(),
        payload_root=payload_root.resolve(),
        payload_private_report_path=payload_private_report_path.resolve(),
        payload_public_summary_path=payload_public_summary_path.resolve(),
        resolution_public_summary_path=resolution_public_summary_path.resolve(),
        volume_paths={key: path.resolve() for key, path in volume_paths.items()},
        candidate_private_report_paths={
            key: path.resolve() for key, path in candidate_private_report_paths.items()
        },
        candidate_public_summary_paths={
            key: path.resolve() for key, path in candidate_public_summary_paths.items()
        },
    )
    forward = config["forward"]
    torch.set_num_threads(int(forward["torch_threads"]))
    store = PSURotation40ActiveRayStore(
        geometry_root,
        rays_per_camera=None,
        sample_count=int(forward["finite_aperture_sample_count"]),
        chunk_rays=int(forward["chunk_rays"]),
        grid_minimum_xyz=tuple(forward["grid_minimum_xyz_m"]),
        grid_maximum_xyz=tuple(forward["grid_maximum_xyz_m"]),
    )
    loader = PSUB0StreamingOperator(
        ray_store=store,
        grid_shape=(16, 16, 16),
        grid_minimum_xyz=tuple(forward["grid_minimum_xyz_m"]),
        grid_maximum_xyz=tuple(forward["grid_maximum_xyz_m"]),
        support=zero_outer_boundary_support((16, 16, 16), dtype=torch.float64),
        dtype=torch.float64,
    )
    measured = loader.load_observations()
    candidates = []
    for candidate in config["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id not in volume_paths:
            raise ValueError(f"missing volume path: {candidate_id}")
        candidates.append(
            _candidate_metrics(
                candidate=candidate,
                volume_path=volume_paths[candidate_id].resolve(),
                store=store,
                measured=measured,
                forward=forward,
            )
        )
    comparison = compare_resolution_metrics(
        candidates[0],
        candidates[1],
        minimum_predeclared_numerical_absolute_improvement=float(
            config["decision"]["minimum_predeclared_numerical_absolute_improvement"]
        ),
        camera_nonworse_tolerance=float(
            config["decision"]["camera_nonworse_tolerance"]
        ),
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": comparison["machine_decision"],
        "evidence_scope": config["scope"],
        "config_sha256": _sha256(config_path),
        "attestation_sha256": _sha256(attestation_path),
        "protocol_commit": attestation["protocol_commit"],
        "dataset": dict(config["dataset"]),
        "input_bindings": dict(config["input_bindings"]),
        "geometry_binding": dict(config["geometry_binding"]),
        "support_reconstruction_contract": dict(
            config["support_reconstruction_contract"]
        ),
        "selection": store.selection_summary(),
        "forward": dict(forward),
        "candidates": candidates,
        "complexity_context": {
            "nominal_grid_value_ratio_32_over_16": 8.0,
            "free_interior_value_ratio_32_over_16": 27000.0 / 2744.0,
            "same_solver_call_budget": True,
            "equal_parameter_count": False,
            "equal_compute_cost": False,
            "algorithm_comparison": False,
        },
        "comparison": comparison,
        "claim_boundary": {
            "rotation40_development_scored": True,
            "real_measurements_used": True,
            "same_physical_cameras_as_support": True,
            "held_out_rotation_run_not_camera": True,
            "independent_rotation_block_count": 1,
            "experimental_field_truth_available": False,
            "field_relative_l2_available": False,
            "algorithm_superiority": False,
            "compute_fair_comparison": False,
            "practical_significance_established": False,
            "neural_operator_trained": False,
            "final_rotations_opened": False,
            "cross_session_generalization": False,
            "paper_claim_authorized": False,
        },
        "public_export_policy": {
            "contains_predictions": False,
            "contains_measurements": False,
            "contains_geometry_arrays": False,
            "contains_volumes": False,
            "contains_only_aggregate_metrics": True,
        },
    }
    private_tokens = tuple(
        str(path.resolve())
        for path in (
            geometry_root,
            geometry_private_report_path,
            payload_root,
            payload_private_report_path,
            *volume_paths.values(),
            *candidate_private_report_paths.values(),
        )
    )
    validate_public_report(report, forbidden_path_tokens=private_tokens)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "summary.json", report)
    _write_csv(output_dir / "comparison_rows.csv", candidates)
    _plot(report, output_dir / "diagnostic.png", output_dir / "diagnostic.pdf")
    (output_dir / "README.md").write_text(
        "# PSU rotation-40 resolution-transfer public audit\n\n"
        f"Machine decision: `{comparison['machine_decision']}`.\n\n"
        "This package contains aggregate metrics and figures only. It contains no "
        "raw measurements, predictions, ray geometry, private volumes, final rotations, "
        "experimental volumetric truth, or algorithm-superiority claim. The same physical "
        "cameras appear in support and scoring; only the rotation run is held out.\n",
        encoding="utf-8",
    )
    _write_checksums(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--geometry-private-report", type=Path, required=True)
    parser.add_argument("--geometry-public-summary", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--payload-private-report", type=Path, required=True)
    parser.add_argument("--payload-public-summary", type=Path, required=True)
    parser.add_argument("--resolution-public-summary", type=Path, required=True)
    parser.add_argument("--volume-16", type=Path, required=True)
    parser.add_argument("--volume-16-private-report", type=Path, required=True)
    parser.add_argument("--volume-16-public-summary", type=Path, required=True)
    parser.add_argument("--volume-32", type=Path, required=True)
    parser.add_argument("--volume-32-private-report", type=Path, required=True)
    parser.add_argument("--volume-32-public-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_resolution_transfer(
        config_path=args.config,
        attestation_path=args.attestation,
        geometry_root=args.geometry_root,
        geometry_private_report_path=args.geometry_private_report,
        geometry_public_summary_path=args.geometry_public_summary,
        payload_root=args.payload_root,
        payload_private_report_path=args.payload_private_report,
        payload_public_summary_path=args.payload_public_summary,
        resolution_public_summary_path=args.resolution_public_summary,
        volume_paths={
            "support_cgls4_16cubed": args.volume_16,
            "support_cgls4_32cubed": args.volume_32,
        },
        candidate_private_report_paths={
            "support_cgls4_16cubed": args.volume_16_private_report,
            "support_cgls4_32cubed": args.volume_32_private_report,
        },
        candidate_public_summary_paths={
            "support_cgls4_16cubed": args.volume_16_public_summary,
            "support_cgls4_32cubed": args.volume_32_public_summary,
        },
        output_dir=args.output_dir,
    )
    print(json.dumps(report["comparison"], ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
