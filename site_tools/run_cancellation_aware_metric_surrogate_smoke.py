#!/usr/bin/env python3
"""Run the audited Metric-A synthetic CPU smoke and write traceable evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from demo_t16_operator.cancellation_aware_metric_surrogate import (
    SCHEMA_VERSION as INTERFACE_SCHEMA_VERSION,
    STATUS,
    ExactMassInstrumentation,
    apply_calibration_envelope,
    audit_schur_safety,
    build_diagonal_metric,
    exact_mass_access_scope,
    exact_masses_from_signed_matrix,
    fit_calibration_envelope,
    fit_metric_surrogate,
    generate_tiny_rigs,
    inference_features_from_rig,
    predict_metric_masses,
    run_signed_residual_trajectory,
    split_rigs_three_way,
)


CONFIG_SCHEMA_VERSION = "cancellation-aware-metric-surrogate-smoke-config-2.0"
REPORT_SCHEMA_VERSION = "cancellation-aware-metric-surrogate-smoke-report-2.0"
EVIDENCE_SCOPE = "SYNTHETIC_TINY_SIGNED_MATRIX_THREE_WAY_GEOMETRY_SMOKE_ONLY"
METHODS = (
    "factor",
    "exact_oracle",
    "scalar_factor_train_selected",
    "exact_factor_interpolation_oracle",
    "learned_oracle_free",
    "calibrated_envelope",
)
CLAIM_BOUNDARY = {
    "new_algorithm_claimed": False,
    "real_data_used": False,
    "generalization_claimed": False,
    "superiority_claimed": False,
}
SOURCE_FILES = (
    "demo_t16_operator/cancellation_aware_metric_surrogate.py",
    "demo_t16_operator/test_cancellation_aware_metric_surrogate.py",
    "site_tools/run_cancellation_aware_metric_surrogate_smoke.py",
    "site_tools/test_run_cancellation_aware_metric_surrogate_smoke.py",
)


def _reject_constant(raw: str) -> None:
    raise ValueError(f"invalid JSON constant: {raw}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _require_exact_keys(name: str, value: Mapping[str, Any], expected: set[str]) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(f"{name} keys differ: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        "config",
        config,
        {"schema_version", "status", "evidence_scope", "seeds", "rigs", "estimator", "calibration", "comparators", "solver", "runtime", "claim_boundary"},
    )
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("unexpected metric-surrogate config schema")
    if config["status"] != STATUS:
        raise ValueError("config must retain development-only status")
    if config["evidence_scope"] != EVIDENCE_SCOPE:
        raise ValueError("config evidence_scope must remain the frozen synthetic scope")
    if config["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("all development-only claim flags must remain false")

    objects = ("seeds", "rigs", "estimator", "calibration", "comparators", "solver", "runtime")
    if any(not isinstance(config[name], Mapping) for name in objects):
        raise ValueError("all config sections must be objects")
    seeds, rigs, estimator = config["seeds"], config["rigs"], config["estimator"]
    calibration, comparators = config["calibration"], config["comparators"]
    solver, runtime = config["solver"], config["runtime"]
    _require_exact_keys("seeds", seeds, {"geometry", "noise", "training"})
    _require_exact_keys("rigs", rigs, {"row_count", "column_count", "split_unit", "random_ray_split_used", "assignments"})
    _require_exact_keys("estimator", estimator, {"hidden_dim", "steps", "learning_rate"})
    _require_exact_keys("calibration", calibration, {"envelope_margin", "fresh_exact_access_forbidden"})
    _require_exact_keys("comparators", comparators, {"scalar_factor_grid", "exact_factor_alpha_grid", "selection_metric"})
    _require_exact_keys("solver", solver, {"eta", "theta", "checkpoints"})
    _require_exact_keys("runtime", runtime, {"device", "dtype", "timing_role"})
    if runtime != {"device": "cpu", "dtype": "torch.float64", "timing_role": "MEASURED_SINGLE_RUN_NONCOMPARATIVE"}:
        raise ValueError("runtime must be the frozen Mac-sized CPU float64 contract")
    if any(int(seeds[name]) < 0 for name in ("geometry", "noise", "training")):
        raise ValueError("seeds must be nonnegative")
    if int(seeds["geometry"]) == int(seeds["noise"]):
        raise ValueError("geometry and noise seeds must differ")
    if rigs["split_unit"] != "COMPLETE_RIG" or rigs["random_ray_split_used"] is not False:
        raise ValueError("only complete-rig splitting is permitted")
    assignments = rigs["assignments"]
    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError("rig assignments must be a nonempty object")
    roles = list(assignments.values())
    expected_roles = {"train", "safety_calibration", "fresh_geometry_ood"}
    if set(roles) != expected_roles or any(roles.count(role) < 2 for role in expected_roles):
        raise ValueError("each of train/calibration/fresh-OOD requires at least two rigs")
    if int(rigs["row_count"]) < 3 or int(rigs["column_count"]) < 3:
        raise ValueError("matrix dimensions must be at least 3x3")
    if int(estimator["hidden_dim"]) < 2 or int(estimator["steps"]) < 1:
        raise ValueError("estimator size/steps must be positive")
    if not math.isfinite(float(estimator["learning_rate"])) or float(estimator["learning_rate"]) <= 0:
        raise ValueError("learning rate must be finite and positive")
    if calibration["fresh_exact_access_forbidden"] is not True or float(calibration["envelope_margin"]) < 1.0:
        raise ValueError("calibration must forbid fresh exact access and use margin >= 1")
    if comparators["selection_metric"] != "FINAL_FIELD_RELATIVE_L2_TRAIN_MEAN":
        raise ValueError("comparator selection metric must remain frozen")
    for grid_name in ("scalar_factor_grid", "exact_factor_alpha_grid"):
        grid = comparators[grid_name]
        if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or not grid:
            raise ValueError(f"{grid_name} must be a nonempty sequence")
        values = [float(value) for value in grid]
        if values != sorted(set(values)) or any(not math.isfinite(value) for value in values):
            raise ValueError(f"{grid_name} must be finite, sorted, and unique")
    if any(float(value) <= 0 for value in comparators["scalar_factor_grid"]):
        raise ValueError("scalar factor candidates must be positive")
    if any(not 0.0 <= float(value) <= 1.0 for value in comparators["exact_factor_alpha_grid"]):
        raise ValueError("exact-factor alpha candidates must lie in [0,1]")
    checkpoints = solver["checkpoints"]
    if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, (str, bytes)):
        raise ValueError("solver checkpoints must be a sequence")
    checkpoint_values = [int(value) for value in checkpoints]
    if checkpoint_values != sorted(set(checkpoint_values)) or not checkpoint_values or checkpoint_values[0] != 0 or checkpoint_values[-1] < 1:
        raise ValueError("checkpoints must be sorted/unique and include zero")
    if not 0.0 < float(solver["eta"]) < 1.0 or not 0.0 <= float(solver["theta"]) <= 1.0:
        raise ValueError("eta/theta outside frozen domains")
    return json.loads(_canonical_json(config))


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("config root must be an object")
    return _validate_config(value)


def _relative_l2(observed: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(observed - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-30))


def _p95_relative_entry_error(observed: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.quantile(torch.abs(observed - expected) / expected.clamp_min(1e-30), 0.95))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    clean = not bool(git("status", "--porcelain"))
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree_clean": clean,
        "source_snapshot_status": (
            "COMMITTED_CLEAN_REPRODUCIBLE_FROM_COMMIT"
            if clean
            else "UNCOMMITTED_SOURCE_SNAPSHOT_NOT_REPRODUCIBLE_FROM_COMMIT_ALONE"
        ),
        "clean_rerun_required_after_commit": not clean,
        "source_file_sha256": {name: _sha256(REPOSITORY_ROOT / name) for name in SOURCE_FILES},
    }


def _model_parameter_rows(estimator: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, tensor in sorted(estimator.state_dict().items()):
        flat = tensor.detach().cpu().reshape(-1)
        rows.extend({"tensor": name, "index": index, "value": float(value)} for index, value in enumerate(flat.tolist()))
    return rows


def _trajectory_final_field(rig: Any, row: torch.Tensor, column: torch.Tensor, solver: Mapping[str, Any]) -> float:
    metric = build_diagonal_metric(row, column, eta=float(solver["eta"]))
    trajectory = run_signed_residual_trajectory(
        rig.signed_matrix, rig.target, rig.truth, metric,
        checkpoints=(0, int(solver["checkpoints"][-1])), theta=float(solver["theta"]),
    )
    return float(trajectory.rows[-1]["field_relative_l2"])


def _select_simple_controls(
    train_rigs: Sequence[Any],
    comparators: Mapping[str, Any],
    solver: Mapping[str, Any],
    instrumentation: ExactMassInstrumentation,
) -> tuple[dict[str, Any], dict[str, int]]:
    scalar_scores: dict[float, float] = {}
    interpolation_scores: dict[float, float] = {}
    ledger = {
        "signed_forward_solver_calls": 0,
        "signed_transpose_solver_calls": 0,
        "signed_forward_evaluation_calls": 0,
        "field_error_evaluation_calls": 0,
        "candidate_rig_evaluations": 0,
        "exact_mass_materializations": 0,
        "factor_mass_accesses": 0,
    }
    endpoint = int(solver["checkpoints"][-1])
    for scalar in (float(value) for value in comparators["scalar_factor_grid"]):
        scalar_scores[scalar] = sum(_trajectory_final_field(rig, scalar * rig.factor_row_mass, scalar * rig.factor_column_mass, solver) for rig in train_rigs) / len(train_rigs)
        ledger["candidate_rig_evaluations"] += len(train_rigs)
        ledger["factor_mass_accesses"] += len(train_rigs)
        ledger["signed_forward_solver_calls"] += endpoint * len(train_rigs)
        ledger["signed_transpose_solver_calls"] += endpoint * len(train_rigs)
        ledger["signed_forward_evaluation_calls"] += 2 * len(train_rigs)
        ledger["field_error_evaluation_calls"] += 2 * len(train_rigs)
    for alpha in (float(value) for value in comparators["exact_factor_alpha_grid"]):
        values = []
        for rig in train_rigs:
            with exact_mass_access_scope(
                instrumentation,
                rig_id=rig.rig_id,
                split_role=rig.split_role,
                phase="train_exact_factor_control_selection",
                fresh_access_allowed=False,
            ):
                exact_row, exact_column = exact_masses_from_signed_matrix(
                    rig.signed_matrix
                )
            values.append(_trajectory_final_field(rig, alpha * exact_row + (1.0 - alpha) * rig.factor_row_mass, alpha * exact_column + (1.0 - alpha) * rig.factor_column_mass, solver))
        interpolation_scores[alpha] = sum(values) / len(values)
        ledger["candidate_rig_evaluations"] += len(train_rigs)
        ledger["exact_mass_materializations"] += len(train_rigs)
        ledger["factor_mass_accesses"] += len(train_rigs)
        ledger["signed_forward_solver_calls"] += endpoint * len(train_rigs)
        ledger["signed_transpose_solver_calls"] += endpoint * len(train_rigs)
        ledger["signed_forward_evaluation_calls"] += 2 * len(train_rigs)
        ledger["field_error_evaluation_calls"] += 2 * len(train_rigs)
    selected_scalar = min(scalar_scores, key=lambda value: (scalar_scores[value], value))
    selected_alpha = min(interpolation_scores, key=lambda value: (interpolation_scores[value], value))
    interpolation_duplicates_exact = selected_alpha == 1.0
    return {
        "selection_metric": comparators["selection_metric"],
        "scalar_factor_scores": {str(k): v for k, v in scalar_scores.items()},
        "selected_scalar_factor": selected_scalar,
        "exact_factor_interpolation_scores": {str(k): v for k, v in interpolation_scores.items()},
        "selected_exact_factor_alpha": selected_alpha,
        "exact_factor_interpolation_is_oracle": True,
        "selected_exact_factor_duplicate_of_exact_oracle": interpolation_duplicates_exact,
    }, ledger


def run_smoke(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Execute one deterministic synthetic run; timing remains descriptive."""

    entry_git_state = _git_state()
    frozen = _validate_config(config)
    total_started = time.perf_counter()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    timing: dict[str, Any] = {"role": "MEASURED_SINGLE_RUN_NONCOMPARATIVE", "clock": "time.perf_counter"}
    try:
        started = time.perf_counter()
        rigs = generate_tiny_rigs(
            split_assignments=frozen["rigs"]["assignments"],
            geometry_seed=int(frozen["seeds"]["geometry"]),
            noise_seed=int(frozen["seeds"]["noise"]),
            row_count=int(frozen["rigs"]["row_count"]),
            column_count=int(frozen["rigs"]["column_count"]),
            dtype=torch.float64,
        )
        train_rigs, calibration_rigs, fresh_rigs, split_contract = split_rigs_three_way(rigs)
        geometry_rows = [
            {
                "rig_id": rig.rig_id,
                "split_role": rig.split_role,
                "geometry_seed": rig.geometry_seed,
                "noise_seed": rig.noise_seed,
                "angle": rig.geometry_parameters[0],
                "aperture": rig.geometry_parameters[1],
                "shear": rig.geometry_parameters[2],
                "cancellation": rig.geometry_parameters[3],
                "geometry_parameters_sha256": rig.geometry_parameters_sha256,
            }
            for rig in rigs
        ]
        timing["data_generation_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        estimator, training = fit_metric_surrogate(
            train_rigs,
            hidden_dim=int(frozen["estimator"]["hidden_dim"]),
            steps=int(frozen["estimator"]["steps"]),
            learning_rate=float(frozen["estimator"]["learning_rate"]),
            seed=int(frozen["seeds"]["training"]),
        )
        timing["training_seconds"] = time.perf_counter() - started

        exact_instrumentation = ExactMassInstrumentation.empty()
        started = time.perf_counter()
        controls, control_ledger = _select_simple_controls(
            train_rigs,
            frozen["comparators"],
            frozen["solver"],
            exact_instrumentation,
        )
        timing["simple_control_selection_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        envelope = fit_calibration_envelope(
            estimator,
            calibration_rigs,
            margin=float(frozen["calibration"]["envelope_margin"]),
            instrumentation=exact_instrumentation,
        )
        timing["safety_calibration_seconds"] = time.perf_counter() - started

        metric_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        method_timing = {method: {"setup_seconds": 0.0, "factor_feature_construction_seconds": 0.0, "audit_seconds": 0.0, "iteration_seconds": 0.0} for method in METHODS}
        method_ledger = {method: {"setup_exact_mass_materializations": 0, "audit_exact_mass_recomputations": 0, "evaluation_exact_mass_recomputations": 0, "factor_mass_vector_accesses": 0, "factor_feature_construction_calls": 0, "estimator_head_forward_calls": 0, "schur_audit_calls": 0, "signed_forward_solver_calls": 0, "signed_transpose_solver_calls": 0, "signed_forward_evaluation_calls": 0, "field_error_evaluation_calls": 0} for method in METHODS}
        total_violations = 0
        eta = float(frozen["solver"]["eta"])
        for rig in fresh_rigs:
            for method in METHODS:
                setup_started = time.perf_counter()
                phase, fresh_allowed = {
                    "factor": ("factor_setup", False),
                    "exact_oracle": ("exact_oracle_setup", True),
                    "scalar_factor_train_selected": ("scalar_factor_setup", False),
                    "exact_factor_interpolation_oracle": (
                        "exact_factor_interpolation_oracle_setup",
                        True,
                    ),
                    "learned_oracle_free": ("learned_prediction_setup", False),
                    "calibrated_envelope": (
                        "calibrated_envelope_prediction_setup",
                        False,
                    ),
                }[method]
                with exact_mass_access_scope(
                    exact_instrumentation,
                    rig_id=rig.rig_id,
                    split_role=rig.split_role,
                    phase=phase,
                    fresh_access_allowed=fresh_allowed,
                ):
                    if method == "factor":
                        row_mass, column_mass = rig.factor_row_mass, rig.factor_column_mass
                        method_ledger[method]["factor_mass_vector_accesses"] += 2
                    elif method == "exact_oracle":
                        row_mass, column_mass = exact_masses_from_signed_matrix(rig.signed_matrix)
                        method_ledger[method]["setup_exact_mass_materializations"] += 1
                    elif method == "scalar_factor_train_selected":
                        scalar = float(controls["selected_scalar_factor"])
                        row_mass, column_mass = scalar * rig.factor_row_mass, scalar * rig.factor_column_mass
                        method_ledger[method]["factor_mass_vector_accesses"] += 2
                    elif method == "exact_factor_interpolation_oracle":
                        alpha = float(controls["selected_exact_factor_alpha"])
                        exact_row, exact_column = exact_masses_from_signed_matrix(rig.signed_matrix)
                        row_mass = alpha * exact_row + (1.0 - alpha) * rig.factor_row_mass
                        column_mass = alpha * exact_column + (1.0 - alpha) * rig.factor_column_mass
                        method_ledger[method]["setup_exact_mass_materializations"] += 1
                        method_ledger[method]["factor_mass_vector_accesses"] += 2
                    else:
                        feature_started = time.perf_counter()
                        features = inference_features_from_rig(rig)
                        method_timing[method]["factor_feature_construction_seconds"] += time.perf_counter() - feature_started
                        method_ledger[method]["factor_mass_vector_accesses"] += 2
                        method_ledger[method]["factor_feature_construction_calls"] += 1
                        predicted_row, predicted_column = predict_metric_masses(estimator, features)
                        method_ledger[method]["estimator_head_forward_calls"] += 2
                        if method == "learned_oracle_free":
                            row_mass, column_mass = predicted_row, predicted_column
                        else:
                            row_mass, column_mass = apply_calibration_envelope(predicted_row, predicted_column, envelope)
                    metric = build_diagonal_metric(row_mass, column_mass, eta=eta)
                method_timing[method]["setup_seconds"] += time.perf_counter() - setup_started
                for axis, values in (("row", row_mass), ("column", column_mass)):
                    prediction_rows.extend({"rig_id": rig.rig_id, "method": method, "axis": axis, "index": index, "mass": float(value)} for index, value in enumerate(values.tolist()))

                audit_started = time.perf_counter()
                with exact_mass_access_scope(
                    exact_instrumentation,
                    rig_id=rig.rig_id,
                    split_role=rig.split_role,
                    phase="posthoc_schur_audit",
                    fresh_access_allowed=True,
                ):
                    audit = audit_schur_safety(rig.signed_matrix, metric, eta=eta)
                method_timing[method]["audit_seconds"] += time.perf_counter() - audit_started
                method_ledger[method]["schur_audit_calls"] += 1
                method_ledger[method]["audit_exact_mass_recomputations"] += 1
                total_violations += int(audit["total_violation_count"])

                iteration_started = time.perf_counter()
                trajectory = run_signed_residual_trajectory(
                    rig.signed_matrix, rig.target, rig.truth, metric,
                    checkpoints=tuple(int(v) for v in frozen["solver"]["checkpoints"]),
                    theta=float(frozen["solver"]["theta"]),
                )
                method_timing[method]["iteration_seconds"] += time.perf_counter() - iteration_started
                for key in trajectory.ledger:
                    if key in method_ledger[method]:
                        method_ledger[method][key] += int(trajectory.ledger[key])
                trajectory_rows.extend({"rig_id": rig.rig_id, "split_role": rig.split_role, "method": method, **row} for row in trajectory.rows)
                with exact_mass_access_scope(
                    exact_instrumentation,
                    rig_id=rig.rig_id,
                    split_role=rig.split_role,
                    phase="posthoc_metric_evaluation",
                    fresh_access_allowed=True,
                ):
                    exact_row, exact_column = exact_masses_from_signed_matrix(
                        rig.signed_matrix
                    )
                method_ledger[method]["evaluation_exact_mass_recomputations"] += 1
                metric_rows.append({
                    "rig_id": rig.rig_id,
                    "split_role": rig.split_role,
                    "method": method,
                    "row_relative_l2_vs_exact": _relative_l2(row_mass, exact_row),
                    "column_relative_l2_vs_exact": _relative_l2(column_mass, exact_column),
                    "row_p95_relative_entry_error": _p95_relative_entry_error(row_mass, exact_row),
                    "column_p95_relative_entry_error": _p95_relative_entry_error(column_mass, exact_column),
                    "final_normalized_residual_l2": trajectory.rows[-1]["normalized_residual_l2"],
                    "final_field_relative_l2": trajectory.rows[-1]["field_relative_l2"],
                    **audit,
                })
    finally:
        torch.use_deterministic_algorithms(previous_determinism)

    final_by_method = {
        method: [row for row in metric_rows if row["method"] == method]
        for method in METHODS
    }
    aggregate = {
        method: {
            "fresh_rig_count": len(rows),
            "mean_final_field_relative_l2": sum(float(row["final_field_relative_l2"]) for row in rows) / len(rows),
            "mean_final_normalized_residual_l2": sum(float(row["final_normalized_residual_l2"]) for row in rows) / len(rows),
            "fresh_rigs_with_any_schur_violation": sum(int(row["total_violation_count"] > 0) for row in rows),
        }
        for method, rows in final_by_method.items()
    }
    candidate = "calibrated_envelope"
    stable_wins = []
    for rig in fresh_rigs:
        values = {row["method"]: float(row["final_field_relative_l2"]) for row in metric_rows if row["rig_id"] == rig.rig_id}
        stable_wins.append(values[candidate] < values["factor"] and values[candidate] < values["scalar_factor_train_selected"])
    candidate_safe = aggregate[candidate]["fresh_rigs_with_any_schur_violation"] == 0
    substitution_authorized = bool(stable_wins) and all(stable_wins) and candidate_safe

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.iterdir():
        if stale.is_file():
            stale.unlink()
    model_path = output_dir / "model_parameters.csv"
    geometry_path = output_dir / "geometry_manifest.csv"
    prediction_path = output_dir / "predictions.csv"
    metric_path = output_dir / "metric_rows.csv"
    trajectory_path = output_dir / "trajectory_rows.csv"
    _write_csv(model_path, _model_parameter_rows(estimator), ("tensor", "index", "value"))
    _write_csv(
        geometry_path,
        geometry_rows,
        (
            "rig_id",
            "split_role",
            "geometry_seed",
            "noise_seed",
            "angle",
            "aperture",
            "shear",
            "cancellation",
            "geometry_parameters_sha256",
        ),
    )
    _write_csv(prediction_path, prediction_rows, ("rig_id", "method", "axis", "index", "mass"))
    _write_csv(metric_path, metric_rows, ("rig_id", "split_role", "method", "row_relative_l2_vs_exact", "column_relative_l2_vs_exact", "row_p95_relative_entry_error", "column_p95_relative_entry_error", "final_normalized_residual_l2", "final_field_relative_l2", "row_violation_count", "column_violation_count", "spectral_violation_count", "total_violation_count", "maximum_row_product", "maximum_column_product", "dense_normalized_spectral_norm_squared", "schur_squared_upper_bound", "exact_masses_recomputed_from_signed_a"))
    _write_csv(trajectory_path, trajectory_rows, ("rig_id", "split_role", "method", "iteration", "normalized_residual_l2", "field_relative_l2", "solution_l2"))
    timing["fresh_method_seconds"] = {
        method: {
            **values,
            "total_seconds": values["setup_seconds"]
            + values["audit_seconds"]
            + values["iteration_seconds"],
            "factor_feature_construction_is_setup_subcomponent": True,
        }
        for method, values in method_timing.items()
    }
    timing["total_seconds"] = time.perf_counter() - total_started
    config_hash = hashlib.sha256((_canonical_json(frozen) + "\n").encode()).hexdigest()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "interface_schema_version": INTERFACE_SCHEMA_VERSION,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "provenance": {
            **entry_git_state,
            "config_sha256": config_hash,
            "geometry_manifest_sha256": _sha256(geometry_path),
            "model_parameters_sha256": _sha256(model_path),
            "fresh_predictions_sha256": _sha256(prediction_path),
        },
        "split_contract": split_contract,
        "data_contract": {
            "operator": "DETERMINISTIC_TINY_MULTI_GEOMETRY_SIGNED_DENSE_MATRIX",
            "geometry_sampling": "INDEPENDENT_PARAMETERS_WITH_SEPARATE_FRESH_OOD_SUPPORT",
            "per_rig_seed_derivation": "SHA256_CANONICAL_JSON_ARRAY_OF_BASE_SEED_RIG_ID_SPLIT_ROLE_FIRST_63_BITS",
            "geometry_fixture_role": "FIXED_SYNTHETIC_REGRESSION_FIXTURE_NOT_AN_INDEPENDENT_STATISTICAL_SEED_SWEEP",
            "geometry_base_seed_selection": "SELECTED_TO_RETAIN_THE_39_VIOLATION_REGRESSION_TARGET_AFTER_SHA256_SEED_MIGRATION",
            "inference_type": "InferenceRigFeatures_WITHOUT_A_EXACT_TRUTH_TARGET",
            "primary_metric": "FINAL_FIELD_RELATIVE_L2",
            "experimental_reaction_flow_data": False,
        },
        "config": frozen,
        "training": training,
        "simple_control_selection": controls,
        "calibration_envelope": {
            "row_scale": envelope.row_scale,
            "column_scale": envelope.column_scale,
            "calibration_rig_ids": list(envelope.calibration_rig_ids),
            "exact_mass_materializations": envelope.exact_mass_materializations,
            "factor_mass_vector_accesses": envelope.factor_mass_vector_accesses,
            "factor_feature_construction_calls": envelope.factor_feature_construction_calls,
            "estimator_head_forward_calls": envelope.estimator_head_forward_calls,
            "fresh_exact_access": exact_instrumentation.summary()[
                "fresh_candidate_exact_access"
            ],
            "cost_role": "ONE_TIME_OFFLINE_SAFETY_CALIBRATION_WITH_EXACT_TEACHERS",
        },
        "fresh_exact_access_instrumentation": exact_instrumentation.summary(),
        "feature_cost_contract": {
            "learned_features_require_factor_row_and_column_mass": True,
            "cold_start": "MUST_CONSTRUCT_FACTOR_MAJORIZER_AND_BOTH_FACTOR_MASS_VECTORS_BEFORE_LEARNED_INFERENCE",
            "precomputed": "MAY_REUSE_CACHED_FACTOR_MASS_VECTORS_BUT_STILL_COUNTS_TWO_VECTOR_ACCESSES_PER_RIG_AND_METHOD",
            "measured_smoke_mode": "PRECOMPUTED_SYNTHETIC_FACTOR_MASSES_WITH_FEATURE_CONSTRUCTION_CHARGED_TO_SETUP",
            "end_to_end_cost_reduction_claimed": False,
        },
        "method_contracts": {
            "factor": "DEPLOYABLE_FACTOR_MAJORIZER_BASELINE",
            "exact_oracle": "NONDEPLOYABLE_EXACT_ABS_A_ORACLE",
            "scalar_factor_train_selected": "DEPLOYABLE_TRAIN_SELECTED_SCALAR_TIMES_FACTOR_BASELINE",
            "exact_factor_interpolation_oracle": "NONDEPLOYABLE_TRAIN_SELECTED_EXACT_FACTOR_ORACLE",
            "learned_oracle_free": "DEPLOYABLE_INPUT_ONLY_UNCLIPPED_ESTIMATOR",
            "calibrated_envelope": "DEPLOYABLE_INPUT_ONLY_AFTER_EXACT_CALIBRATION_COST_NO_FRESH_EXACT",
        },
        "evidence_counting": {
            "raw_method_count": len(METHODS),
            "independent_method_count": len(METHODS)
            - int(
                controls[
                    "selected_exact_factor_duplicate_of_exact_oracle"
                ]
            ),
            "duplicate_methods": (
                {
                    "exact_factor_interpolation_oracle": {
                        "duplicate_of_exact_oracle": True,
                        "reason": "TRAIN_SELECTED_ALPHA_EQUALS_1.0",
                    }
                }
                if controls[
                    "selected_exact_factor_duplicate_of_exact_oracle"
                ]
                else {}
            ),
            "independent_method_names": [
                method
                for method in METHODS
                if not (
                    method == "exact_factor_interpolation_oracle"
                    and controls[
                        "selected_exact_factor_duplicate_of_exact_oracle"
                    ]
                )
            ],
        },
        "aggregate_fresh_ood": aggregate,
        "call_ledger": {
            "data_generation": {"exact_teacher_materializations": len(rigs), "factor_majorizer_materializations": len(rigs)},
            "training": {"exact_teacher_rigs": len(train_rigs), "full_batch_head_forward_calls": training["full_batch_head_forward_calls"]},
            "simple_control_selection": control_ledger,
            "safety_calibration": {
                "exact_mass_materializations": envelope.exact_mass_materializations,
                "factor_mass_vector_accesses": envelope.factor_mass_vector_accesses,
                "factor_feature_construction_calls": envelope.factor_feature_construction_calls,
                "estimator_head_forward_calls": envelope.estimator_head_forward_calls,
            },
            "fresh_by_method": method_ledger,
        },
        "timing": timing,
        "environment": {"device": "cpu", "dtype": "torch.float64", "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "deterministic_algorithms_during_run": True},
        "decision": {
            "status": "SYNTHETIC_SMOKE_EXECUTED_NO_SCIENTIFIC_GATE_OPENED",
            "all_methods_total_schur_violation_count": total_violations,
            "calibrated_envelope_all_fresh_schur_safe": candidate_safe,
            "calibrated_envelope_beats_factor_and_simple_on_each_fresh_rig": all(stable_wins),
            "per_fresh_rig_stable_win_flags": {rig.rig_id: flag for rig, flag in zip(fresh_rigs, stable_wins, strict=True)},
            "research_claim_authorized": False,
            "metric_substitution_authorized": substitution_authorized,
            "authorization_rule": "TRUE_ONLY_IF_CALIBRATED_ENVELOPE_IS_SCHUR_SAFE_AND_BEATS_FACTOR_AND_TRAIN_SELECTED_SCALAR_FACTOR_ON_EVERY_FRESH_OOD_RIG",
            "next_gate": "SUPPORT_OOD_DETECTION_PLUS_FACTOR_FALLBACK_AND_STRUCTURED_SAFE_PARAMETERIZATION",
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(_canonical_json(report) + "\n", encoding="utf-8")
    checksum_path = output_dir / "checksums.sha256"
    payloads = (
        geometry_path,
        metric_path,
        model_path,
        prediction_path,
        report_path,
        trajectory_path,
    )
    checksum_path.write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in payloads), encoding="ascii")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "demo_t16_operator/configs/cancellation_aware_metric_surrogate_smoke_v1.json")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "demo_t16_operator/results/cancellation_aware_metric_surrogate_smoke")
    args = parser.parse_args()
    report = run_smoke(load_config(args.config), output_dir=args.output_dir)
    print(_canonical_json(report["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
