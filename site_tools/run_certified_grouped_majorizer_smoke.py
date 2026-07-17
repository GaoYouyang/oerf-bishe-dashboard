from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import torch

from demo_t16_operator.certified_grouped_majorizer import (
    PRIMITIVE_COUNT,
    SCHEMA_VERSION as INTERFACE_SCHEMA_VERSION,
    STATUS,
    DeploymentGeometry,
    GeometryPartitionStump,
    TinyPrimitiveRig,
    audit_partition_safety,
    build_diagonal_metric,
    build_grouped_majorizer,
    deployment_geometry_from_rig,
    fit_geometry_partition_stump,
    generate_tiny_primitive_rigs,
    predefined_partitions,
    run_signed_pdhg_trajectory,
    select_global_fixed_partition,
    select_partition,
    split_rigs_three_way,
)


CONFIG_SCHEMA_VERSION = "certified-grouped-majorizer-smoke-config-1.0"
REPORT_SCHEMA_VERSION = "certified-grouped-majorizer-smoke-report-1.0"
EVIDENCE_SCOPE = "SYNTHETIC_MULTIPRIMITIVE_CPU_SMOKE_ONLY"
METHODS = (
    "singleton_factor",
    "fixed_paired_local",
    "fixed_paired_cross",
    "fixed_triad_bridge",
    "train_selected_fixed",
    "geometry_conditioned_selector",
    "all_in_one_exact_oracle",
)
FIXED_METHOD_PARTITIONS = {
    "singleton_factor": "singleton_factor",
    "fixed_paired_local": "paired_local",
    "fixed_paired_cross": "paired_cross",
    "fixed_triad_bridge": "triad_bridge",
    "all_in_one_exact_oracle": "all_in_one_exact",
}
OUTPUT_PAYLOADS = (
    "config_snapshot.json",
    "construction_cost_rows.csv",
    "geometry_manifest.csv",
    "metric_rows.csv",
    "partition_audit_rows.csv",
    "report.json",
    "selection_rows.csv",
    "trajectory_rows.csv",
)
EXPECTED_OUTPUT_FILES = frozenset((*OUTPUT_PAYLOADS, "checksums.sha256"))

METRIC_FIELDS = (
    "rig_id",
    "split_role",
    "method",
    "partition_name",
    "oracle_only",
    "final_normalized_residual_l2",
    "final_field_relative_l2",
    "harm_vs_train_selected_fixed_field_l2",
    "pointwise_violation_count",
    "row_violation_count",
    "column_violation_count",
    "spectral_violation_count",
    "total_violation_count",
    "signed_forward_solver_calls",
    "signed_transpose_solver_calls",
    "signed_forward_evaluation_calls",
    "field_error_evaluation_calls",
    "iteration_budget",
    "construction_wall_time_seconds",
    "audit_wall_time_seconds",
    "solver_wall_time_seconds",
    "cost_proxy_units",
    "primitive_component_accumulation_entries",
    "signed_group_addition_entries",
    "group_absolute_value_entries",
    "materialized_signed_group_count",
    "materialized_group_component_entries",
    "largest_fused_group_component_entries",
    "full_signed_operator_materialized",
    "cost_proxy_definition",
)
TRAJECTORY_FIELDS = (
    "rig_id",
    "split_role",
    "method",
    "partition_name",
    "iteration",
    "normalized_residual_l2",
    "field_relative_l2",
    "solution_l2",
)
AUDIT_FIELDS = (
    "rig_id",
    "split_role",
    "partition_name",
    "oracle_only",
    "pointwise_violation_count",
    "row_mass_mismatch_count",
    "column_mass_mismatch_count",
    "support_violation_count",
    "row_violation_count",
    "column_violation_count",
    "spectral_violation_count",
    "total_violation_count",
    "maximum_entrywise_slack",
    "minimum_entrywise_slack",
    "maximum_row_product",
    "maximum_column_product",
    "dense_normalized_spectral_norm_squared",
    "schur_squared_upper_bound",
    "active_row_count",
    "active_column_count",
    "exact_masses_recomputed_from_signed_primitives",
    "triangle_inequality_certificate",
)
SELECTION_FIELDS = (
    "rig_id",
    "split_role",
    "selection_role",
    "selected_partition",
    "oracle_only",
    "geometry_feature_0",
    "geometry_feature_1",
    "geometry_feature_2",
    "geometry_feature_3",
    "geometry_feature_4",
    "geometry_feature_5",
)
COST_FIELDS = (
    "rig_id",
    "method",
    "partition_name",
    "definition",
    "primitive_component_accumulation_entries",
    "signed_group_addition_entries",
    "group_absolute_value_entries",
    "materialized_signed_group_count",
    "materialized_group_component_entries",
    "largest_fused_group_component_entries",
    "full_signed_operator_materialized",
    "cost_proxy_units",
    "construction_wall_time_seconds",
)
GEOMETRY_FIELDS = (
    "rig_id",
    "split_role",
    "geometry_seed_sha256",
    "noise_seed_sha256",
    "geometry_feature_sha256",
    "geometry_feature_0",
    "geometry_feature_1",
    "geometry_feature_2",
    "geometry_feature_3",
    "geometry_feature_4",
    "geometry_feature_5",
)


def _reject_constant(raw: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {raw}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _strict_json_text(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def _require_exact_keys(name: str, value: Mapping[str, Any], expected: set[str]) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
        )


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "evidence_scope",
        "seeds",
        "rigs",
        "partitions",
        "selector",
        "solver",
        "runtime",
        "claim_boundary",
    }
    _require_exact_keys("config", config, expected_top)
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("schema_version is not frozen")
    if config["status"] != STATUS:
        raise ValueError("status is not development-only")
    if config["evidence_scope"] != EVIDENCE_SCOPE:
        raise ValueError("evidence_scope must remain synthetic")

    seeds = config["seeds"]
    _require_exact_keys("seeds", seeds, {"geometry", "noise"})
    if any(not isinstance(seeds[name], int) or isinstance(seeds[name], bool) for name in seeds):
        raise ValueError("seeds must be integers")
    if seeds["geometry"] == seeds["noise"]:
        raise ValueError("geometry and noise seeds must differ")

    rigs = config["rigs"]
    _require_exact_keys(
        "rigs",
        rigs,
        {
            "row_count",
            "column_count",
            "primitive_count",
            "split_unit",
            "random_ray_split_used",
            "assignments",
        },
    )
    if rigs["split_unit"] != "COMPLETE_RIG" or rigs["random_ray_split_used"] is not False:
        raise ValueError("complete-rig splitting is mandatory")
    if rigs["primitive_count"] != PRIMITIVE_COUNT:
        raise ValueError(f"primitive_count must equal {PRIMITIVE_COUNT}")
    if not isinstance(rigs["row_count"], int) or not isinstance(rigs["column_count"], int):
        raise ValueError("matrix dimensions must be integers")
    if rigs["row_count"] < 4 or rigs["column_count"] < 4:
        raise ValueError("matrix dimensions must be at least four")
    assignments = rigs["assignments"]
    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError("rig assignments must be a nonempty object")
    allowed_roles = {"train", "safety_calibration", "fresh_geometry_ood"}
    if any(not isinstance(key, str) or not key for key in assignments):
        raise ValueError("rig ids must be nonempty strings")
    if set(assignments.values()).difference(allowed_roles):
        raise ValueError("unknown split role")
    minimums = {"train": 4, "safety_calibration": 2, "fresh_geometry_ood": 3}
    for role, minimum in minimums.items():
        if sum(value == role for value in assignments.values()) < minimum:
            raise ValueError(f"{role} requires at least {minimum} complete rigs")

    specs = predefined_partitions(PRIMITIVE_COUNT)
    partitions = config["partitions"]
    _require_exact_keys(
        "partitions",
        partitions,
        {
            "fixed_partition_names",
            "global_candidate_names",
            "selector_candidate_names",
            "exact_oracle_name",
            "all_in_one_for_selector_forbidden",
            "cost_proxy_role",
        },
    )
    expected_fixed = ["paired_local", "paired_cross", "triad_bridge"]
    expected_candidates = ["singleton_factor", *expected_fixed]
    if partitions["fixed_partition_names"] != expected_fixed:
        raise ValueError("fixed partition names differ from the frozen catalogue")
    if partitions["global_candidate_names"] != expected_candidates:
        raise ValueError("global candidate names differ from the frozen catalogue")
    if partitions["selector_candidate_names"] != expected_candidates:
        raise ValueError("selector candidate names differ from the frozen catalogue")
    if partitions["exact_oracle_name"] != "all_in_one_exact":
        raise ValueError("exact oracle name differs from the frozen catalogue")
    if partitions["all_in_one_for_selector_forbidden"] is not True:
        raise ValueError("all-in-one must be forbidden to the selector")
    if partitions["cost_proxy_role"] != "ANALYTIC_PROXY_NOT_WALL_TIME":
        raise ValueError("construction cost role is not frozen")
    if set(expected_candidates + [partitions["exact_oracle_name"]]) != set(specs):
        raise ValueError("partition catalogue is incomplete")

    selector = config["selector"]
    _require_exact_keys(
        "selector",
        selector,
        {
            "model_class",
            "train_top_k",
            "fresh_exact_truth_target_access_forbidden",
        },
    )
    if selector["model_class"] != "DEPTH_ONE_GEOMETRY_STUMP":
        raise ValueError("selector model class is not frozen")
    if not isinstance(selector["train_top_k"], int) or selector["train_top_k"] < 1:
        raise ValueError("selector train_top_k must be a positive integer")
    if selector["fresh_exact_truth_target_access_forbidden"] is not True:
        raise ValueError("fresh exact/truth/target access must remain forbidden")

    solver = config["solver"]
    _require_exact_keys("solver", solver, {"eta", "theta", "checkpoints"})
    eta, theta = float(solver["eta"]), float(solver["theta"])
    if not math.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("solver eta must lie in (0,1)")
    if not math.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError("solver theta must lie in [0,1]")
    checkpoints = solver["checkpoints"]
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[0] != 0
        or checkpoints[-1] < 1
        or any(not isinstance(value, int) or value < 0 for value in checkpoints)
    ):
        raise ValueError("solver checkpoints must be sorted unique nonnegative integers including zero")

    runtime = config["runtime"]
    _require_exact_keys("runtime", runtime, {"device", "dtype", "timing_role"})
    if runtime != {
        "device": "cpu",
        "dtype": "torch.float64",
        "timing_role": "MEASURED_SINGLE_RUN_NONCOMPARATIVE",
    }:
        raise ValueError("runtime must remain the frozen CPU float64 smoke")

    claim = config["claim_boundary"]
    _require_exact_keys(
        "claim_boundary",
        claim,
        {
            "real_bost_claimed",
            "generalization_claimed",
            "paper_superiority_claimed",
            "exact_oracle_is_deployable",
        },
    )
    if any(value is not False for value in claim.values()):
        raise ValueError("claim boundary must remain entirely false")
    return json.loads(_canonical_json(config))


def load_config(path: Path) -> dict[str, Any]:
    value = _strict_json_text(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("config root must be an object")
    return _validate_config(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"{path.name} columns differ from frozen schema")
        return list(reader)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"source_commit": commit, "source_worktree_dirty": dirty}


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "algorithm_source": root / "demo_t16_operator" / "certified_grouped_majorizer.py",
        "runner_source": Path(__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _generate_from_config(config: Mapping[str, Any]) -> list[TinyPrimitiveRig]:
    rigs = config["rigs"]
    return generate_tiny_primitive_rigs(
        split_assignments=rigs["assignments"],
        geometry_seed=int(config["seeds"]["geometry"]),
        noise_seed=int(config["seeds"]["noise"]),
        row_count=int(rigs["row_count"]),
        column_count=int(rigs["column_count"]),
        primitive_count=int(rigs["primitive_count"]),
        dtype=torch.float64,
    )


def _score_table(
    rigs: Sequence[TinyPrimitiveRig],
    *,
    candidate_names: Sequence[str],
    solver: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], float], dict[str, int | float]]:
    specs = predefined_partitions()
    table: dict[tuple[str, str], float] = {}
    ledger: dict[str, int | float] = {
        "signed_forward_solver_calls": 0,
        "signed_transpose_solver_calls": 0,
        "signed_forward_evaluation_calls": 0,
        "field_error_evaluation_calls": 0,
        "trajectory_count": 0,
        "wall_time_seconds": 0.0,
    }
    for rig in rigs:
        for name in candidate_names:
            majorizer = build_grouped_majorizer(rig.primitives, specs[name])
            metric = build_diagonal_metric(majorizer, eta=float(solver["eta"]))
            trajectory = run_signed_pdhg_trajectory(
                rig.signed_matrix,
                rig.target,
                rig.truth,
                metric,
                checkpoints=solver["checkpoints"],
                theta=float(solver["theta"]),
            )
            table[(rig.rig_id, name)] = float(trajectory.rows[-1]["field_relative_l2"])
            for key in (
                "signed_forward_solver_calls",
                "signed_transpose_solver_calls",
                "signed_forward_evaluation_calls",
                "field_error_evaluation_calls",
            ):
                ledger[key] = int(ledger[key]) + int(trajectory.ledger[key])
            ledger["trajectory_count"] = int(ledger["trajectory_count"]) + 1
            ledger["wall_time_seconds"] = float(ledger["wall_time_seconds"]) + trajectory.wall_time_seconds
    return table, ledger


def _fit_selection_contract(
    train_rigs: Sequence[TinyPrimitiveRig],
    safety_rigs: Sequence[TinyPrimitiveRig],
    config: Mapping[str, Any],
) -> tuple[str, dict[str, float], GeometryPartitionStump, dict[str, Any], dict[tuple[str, str], float], dict[str, int | float]]:
    candidates = config["partitions"]["global_candidate_names"]
    table, ledger = _score_table(
        [*train_rigs, *safety_rigs],
        candidate_names=candidates,
        solver=config["solver"],
    )
    selected_fixed, fixed_scores = select_global_fixed_partition(
        train_rigs, table, candidates
    )
    selector, selector_report = fit_geometry_partition_stump(
        train_rigs,
        safety_rigs,
        table,
        candidate_names=config["partitions"]["selector_candidate_names"],
        exact_oracle_partition=config["partitions"]["exact_oracle_name"],
        train_top_k=int(config["selector"]["train_top_k"]),
    )
    return selected_fixed, fixed_scores, selector, selector_report, table, ledger


def _method_partition_map(
    deployment_geometry: DeploymentGeometry,
    *,
    selected_fixed: str,
    selector: GeometryPartitionStump,
) -> dict[str, str]:
    if not isinstance(deployment_geometry, DeploymentGeometry):
        raise TypeError("fresh method partitioning requires DeploymentGeometry")
    selected = select_partition(selector, deployment_geometry)
    return {
        **FIXED_METHOD_PARTITIONS,
        "train_selected_fixed": selected_fixed,
        "geometry_conditioned_selector": selected,
    }


def _prepare_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    observed = {path.name for path in output_dir.iterdir()}
    unexpected = observed.difference(EXPECTED_OUTPUT_FILES)
    if unexpected:
        raise ValueError(f"output directory contains unexpected files: {sorted(unexpected)}")
    for name in EXPECTED_OUTPUT_FILES:
        path = output_dir / name
        if path.is_symlink():
            raise ValueError("output payloads must not be symbolic links")
        if path.exists():
            path.unlink()


def run_smoke(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    entry_git_state = _git_state()
    frozen = _validate_config(config)
    _prepare_output(output_dir)
    config_path = output_dir / "config_snapshot.json"
    config_path.write_text(_canonical_json(frozen) + "\n", encoding="utf-8")

    rigs = _generate_from_config(frozen)
    train_rigs, safety_rigs, fresh_rigs, split_contract = split_rigs_three_way(rigs)
    specs = predefined_partitions()
    geometry_rows = [
        {
            "rig_id": rig.rig_id,
            "split_role": rig.split_role,
            "geometry_seed_sha256": rig.geometry_seed_sha256,
            "noise_seed_sha256": rig.noise_seed_sha256,
            "geometry_feature_sha256": rig.geometry_feature_sha256,
            **{
                f"geometry_feature_{index}": float(value)
                for index, value in enumerate(rig.geometry_features)
            },
        }
        for rig in sorted(rigs, key=lambda item: item.rig_id)
    ]
    geometry_path = output_dir / "geometry_manifest.csv"
    _write_csv(geometry_path, geometry_rows, GEOMETRY_FIELDS)
    (
        selected_fixed,
        fixed_scores,
        selector,
        selector_report,
        selection_score_table,
        selection_ledger,
    ) = _fit_selection_contract(train_rigs, safety_rigs, frozen)

    audit_rows: list[dict[str, Any]] = []
    for rig in rigs:
        for partition_name, spec in specs.items():
            majorizer = build_grouped_majorizer(rig.primitives, spec)
            metric = build_diagonal_metric(majorizer, eta=float(frozen["solver"]["eta"]))
            audit = audit_partition_safety(
                rig.primitives, spec, metric, eta=float(frozen["solver"]["eta"])
            )
            audit_rows.append(
                {
                    "rig_id": rig.rig_id,
                    "split_role": rig.split_role,
                    "partition_name": partition_name,
                    "oracle_only": spec.oracle_only,
                    **audit,
                }
            )
    if any(int(row["total_violation_count"]) != 0 for row in audit_rows):
        raise RuntimeError("certified partition safety failed; smoke stopped before claims")

    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for rig in fresh_rigs:
        deployment_geometry = deployment_geometry_from_rig(rig)
        method_partitions = _method_partition_map(
            deployment_geometry, selected_fixed=selected_fixed, selector=selector
        )
        geometry = [float(value) for value in rig.geometry_features]
        for role, partition_name in (
            ("TRAIN_SELECTED_GLOBAL_FIXED", selected_fixed),
            ("GEOMETRY_CONDITIONED_SELECTOR", method_partitions["geometry_conditioned_selector"]),
        ):
            selection_rows.append(
                {
                    "rig_id": rig.rig_id,
                    "split_role": rig.split_role,
                    "selection_role": role,
                    "selected_partition": partition_name,
                    "oracle_only": specs[partition_name].oracle_only,
                    **{f"geometry_feature_{index}": value for index, value in enumerate(geometry)},
                }
            )

        rig_metric_rows: list[dict[str, Any]] = []
        for method in METHODS:
            partition_name = method_partitions[method]
            spec = specs[partition_name]
            construction_started = perf_counter()
            majorizer = build_grouped_majorizer(rig.primitives, spec)
            metric = build_diagonal_metric(majorizer, eta=float(frozen["solver"]["eta"]))
            construction_wall = perf_counter() - construction_started
            audit_started = perf_counter()
            audit = audit_partition_safety(
                rig.primitives, spec, metric, eta=float(frozen["solver"]["eta"])
            )
            audit_wall = perf_counter() - audit_started
            if int(audit["total_violation_count"]) != 0:
                raise RuntimeError(f"fresh partition {partition_name} failed safety")
            trajectory = run_signed_pdhg_trajectory(
                rig.signed_matrix,
                rig.target,
                rig.truth,
                metric,
                checkpoints=frozen["solver"]["checkpoints"],
                theta=float(frozen["solver"]["theta"]),
            )
            final = trajectory.rows[-1]
            cost = majorizer.construction_cost
            row = {
                "rig_id": rig.rig_id,
                "split_role": rig.split_role,
                "method": method,
                "partition_name": partition_name,
                "oracle_only": spec.oracle_only,
                "final_normalized_residual_l2": final["normalized_residual_l2"],
                "final_field_relative_l2": final["field_relative_l2"],
                "harm_vs_train_selected_fixed_field_l2": 0.0,
                "pointwise_violation_count": audit["pointwise_violation_count"],
                "row_violation_count": audit["row_violation_count"],
                "column_violation_count": audit["column_violation_count"],
                "spectral_violation_count": audit["spectral_violation_count"],
                "total_violation_count": audit["total_violation_count"],
                **trajectory.ledger,
                "construction_wall_time_seconds": construction_wall,
                "audit_wall_time_seconds": audit_wall,
                "solver_wall_time_seconds": trajectory.wall_time_seconds,
                **{
                    key: cost[key]
                    for key in (
                        "cost_proxy_units",
                        "primitive_component_accumulation_entries",
                        "signed_group_addition_entries",
                        "group_absolute_value_entries",
                        "materialized_signed_group_count",
                        "materialized_group_component_entries",
                        "largest_fused_group_component_entries",
                        "full_signed_operator_materialized",
                    )
                },
                "cost_proxy_definition": cost["definition"],
            }
            rig_metric_rows.append(row)
            trajectory_rows.extend(
                {
                    "rig_id": rig.rig_id,
                    "split_role": rig.split_role,
                    "method": method,
                    "partition_name": partition_name,
                    **checkpoint,
                }
                for checkpoint in trajectory.rows
            )
            cost_rows.append(
                {
                    "rig_id": rig.rig_id,
                    "method": method,
                    "partition_name": partition_name,
                    **cost,
                    "construction_wall_time_seconds": construction_wall,
                }
            )
        baseline = next(
            float(row["final_field_relative_l2"])
            for row in rig_metric_rows
            if row["method"] == "train_selected_fixed"
        )
        for row in rig_metric_rows:
            row["harm_vs_train_selected_fixed_field_l2"] = (
                float(row["final_field_relative_l2"]) - baseline
            )
        metric_rows.extend(rig_metric_rows)

    aggregate: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        rows = [row for row in metric_rows if row["method"] == method]
        aggregate[method] = {
            "fresh_rig_count": len(rows),
            "mean_final_field_relative_l2": sum(
                float(row["final_field_relative_l2"]) for row in rows
            )
            / len(rows),
            "mean_final_normalized_residual_l2": sum(
                float(row["final_normalized_residual_l2"]) for row in rows
            )
            / len(rows),
            "mean_harm_vs_train_selected_fixed_field_l2": sum(
                float(row["harm_vs_train_selected_fixed_field_l2"]) for row in rows
            )
            / len(rows),
            "maximum_harm_vs_train_selected_fixed_field_l2": max(
                float(row["harm_vs_train_selected_fixed_field_l2"]) for row in rows
            ),
            "total_schur_violation_count": sum(
                int(row["total_violation_count"]) for row in rows
            ),
            "mean_cost_proxy_units": sum(int(row["cost_proxy_units"]) for row in rows)
            / len(rows),
        }

    selector_rows = {
        row["rig_id"]: row
        for row in metric_rows
        if row["method"] == "geometry_conditioned_selector"
    }
    fixed_rows = {
        row["rig_id"]: row
        for row in metric_rows
        if row["method"] == "train_selected_fixed"
    }
    per_fresh_harm = {
        rig_id: float(selector_rows[rig_id]["final_field_relative_l2"])
        - float(fixed_rows[rig_id]["final_field_relative_l2"])
        for rig_id in sorted(selector_rows)
    }
    per_fresh_win = {rig_id: harm < -1e-12 for rig_id, harm in per_fresh_harm.items()}
    safety_harm: dict[str, float] = {}
    for rig in safety_rigs:
        selected = select_partition(selector, deployment_geometry_from_rig(rig))
        safety_harm[rig.rig_id] = (
            float(selection_score_table[(rig.rig_id, selected)])
            - float(selection_score_table[(rig.rig_id, selected_fixed)])
        )
    selected_fresh_partitions = {
        row["rig_id"]: row["selected_partition"]
        for row in selection_rows
        if row["selection_role"] == "GEOMETRY_CONDITIONED_SELECTOR"
    }
    exact_name = frozen["partitions"]["exact_oracle_name"]
    exact_selection_count = sum(
        partition == exact_name for partition in selected_fresh_partitions.values()
    )
    call_tuples = {
        rig.rig_id: {
            (
                int(row["signed_forward_solver_calls"]),
                int(row["signed_transpose_solver_calls"]),
                int(row["signed_forward_evaluation_calls"]),
                int(row["field_error_evaluation_calls"]),
            )
            for row in metric_rows
            if row["rig_id"] == rig.rig_id
        }
        for rig in fresh_rigs
    }
    fair_calls = all(len(values) == 1 for values in call_tuples.values())
    all_audits_safe = all(int(row["total_violation_count"]) == 0 for row in audit_rows)
    selector_fresh_safe = all(
        int(row["total_violation_count"]) == 0 for row in selector_rows.values()
    )
    selector_beats_every_fresh = bool(per_fresh_win) and all(per_fresh_win.values())
    selector_beats_every_safety = bool(safety_harm) and all(
        value < -1e-12 for value in safety_harm.values()
    )
    geometry_adaptation_observed = (
        not bool(selector_report["is_constant"])
        and len(set(selected_fresh_partitions.values())) >= 2
    )
    advantage_not_from_exact = (
        exact_name not in selector.allowed_partitions and exact_selection_count == 0
    )
    smoke_gate = (
        all_audits_safe
        and selector_fresh_safe
        and selector_beats_every_fresh
        and selector_beats_every_safety
        and geometry_adaptation_observed
        and advantage_not_from_exact
        and fair_calls
    )

    metric_path = output_dir / "metric_rows.csv"
    trajectory_path = output_dir / "trajectory_rows.csv"
    audit_path = output_dir / "partition_audit_rows.csv"
    selection_path = output_dir / "selection_rows.csv"
    cost_path = output_dir / "construction_cost_rows.csv"
    _write_csv(metric_path, metric_rows, METRIC_FIELDS)
    _write_csv(trajectory_path, trajectory_rows, TRAJECTORY_FIELDS)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    _write_csv(selection_path, selection_rows, SELECTION_FIELDS)
    _write_csv(cost_path, cost_rows, COST_FIELDS)

    config_hash = hashlib.sha256((_canonical_json(frozen) + "\n").encode("utf-8")).hexdigest()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "interface_schema_version": INTERFACE_SCHEMA_VERSION,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "mathematical_contract": {
            "operator_decomposition": "A=sum_l C_l",
            "partition_majorizer": "M_P=sum_G abs(sum_{l in G} C_l)",
            "certificate": "M_P >= abs(A) entrywise by triangle inequality",
            "singleton_role": "FACTOR_MAJORIZER",
            "all_in_one_role": "EXACT_ABS_A_NONDEPLOYABLE_ORACLE",
            "zero_mass_support_handling": "ZERO_STEP_ON_ZERO_MAJORZER_MASS; DOMINANCE_IMPLIES_ZERO_EXACT_MASS",
        },
        "claim_boundary": frozen["claim_boundary"],
        "provenance": {
            **entry_git_state,
            "config_sha256": config_hash,
            "source_sha256": _source_hashes(),
            "geometry_manifest_sha256": _sha256(geometry_path),
        },
        "split_contract": split_contract,
        "partition_catalogue": {
            name: {
                "groups": [list(group) for group in spec.groups],
                "oracle_only": spec.oracle_only,
                "construction_cost_proxy": build_grouped_majorizer(
                    rigs[0].primitives, spec
                ).construction_cost,
            }
            for name, spec in specs.items()
        },
        "selection": {
            "global_fixed_partition": selected_fixed,
            "global_train_mean_scores": fixed_scores,
            "selector_model": selector_report,
            "selector_allowed_partitions": list(selector.allowed_partitions),
            "fresh_selected_partitions": selected_fresh_partitions,
            "fresh_exact_truth_target_access": False,
            "fresh_sensitive_access_scope": "FORBIDDEN_TO_SELECTOR; OFFLINE_CONSTRUCTOR_USES_PRIMITIVES_AND_OFFLINE_EVALUATOR_USES_TRUTH_AFTER_SELECTION",
            "all_in_one_exact_available_to_selector": False,
        },
        "method_contracts": {
            "singleton_factor": "DEPLOYABLE_SAFE_SINGLETON_PARTITION",
            "fixed_paired_local": "DEPLOYABLE_SAFE_FIXED_GROUPING",
            "fixed_paired_cross": "DEPLOYABLE_SAFE_FIXED_GROUPING",
            "fixed_triad_bridge": "DEPLOYABLE_SAFE_FIXED_GROUPING",
            "train_selected_fixed": "DEPLOYABLE_SAFE_TRAIN_SELECTED_GLOBAL_PARTITION",
            "geometry_conditioned_selector": "DEPLOYABLE_GEOMETRY_ONLY_SAFE_PARTITION_SELECTION",
            "all_in_one_exact_oracle": "NONDEPLOYABLE_EXACT_ABS_A_ORACLE_ONLY",
        },
        "aggregate_fresh_geometry_ood": aggregate,
        "selection_setup_ledger": selection_ledger,
        "fresh_call_contract": {
            "same_signed_A_and_A_transpose_budget_per_method": fair_calls,
            "per_rig_call_tuples": {
                rig_id: [list(value) for value in sorted(values)]
                for rig_id, values in call_tuples.items()
            },
            "wall_time_role": "MEASURED_SINGLE_RUN_NONCOMPARATIVE",
        },
        "decision": {
            "all_partition_audits_zero_violation": all_audits_safe,
            "selector_all_fresh_schur_safe": selector_fresh_safe,
            "per_fresh_rig_harm_vs_train_selected_fixed": per_fresh_harm,
            "per_fresh_rig_strict_win_flags": per_fresh_win,
            "selector_beats_train_selected_fixed_on_every_fresh_rig": selector_beats_every_fresh,
            "per_safety_rig_harm_vs_train_selected_fixed": safety_harm,
            "selector_beats_train_selected_fixed_on_every_safety_rig": selector_beats_every_safety,
            "selector_is_nonconstant": not bool(selector_report["is_constant"]),
            "fresh_unique_selected_partition_count": len(set(selected_fresh_partitions.values())),
            "geometry_adaptation_observed_on_fresh": geometry_adaptation_observed,
            "selector_exact_all_in_one_selection_count": exact_selection_count,
            "advantage_not_due_to_all_in_one_exact": advantage_not_from_exact,
            "equal_fresh_A_A_transpose_call_budget": fair_calls,
            "synthetic_algorithmic_smoke_gate_passed": smoke_gate,
            "research_claim_authorized": smoke_gate,
            "real_bost_claim_authorized": False,
            "claim_scope_if_authorized": "SYNTHETIC_FOLLOWUP_CANDIDATE_ONLY_NOT_PAPER_SUPERIORITY",
        },
        "limitations": [
            "Synthetic dense multi-primitive matrices are not a validated BOST forward model.",
            "Fresh geometry is synthetic OOD, not laboratory generalization evidence.",
            "The construction cost is an analytic proxy; wall times are single-run noncomparative measurements.",
            "Primitive accumulation entries count each primitive once and are not reported as measured hardware cost.",
            "The all-in-one partition uses exact abs(A) and is a nondeployable oracle comparator.",
            "Passing the smoke gate would authorize only a larger preregistered experiment, not a paper claim.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{_sha256(output_dir / name)}  {name}\n" for name in OUTPUT_PAYLOADS),
        encoding="ascii",
    )
    validate_result_bundle(output_dir)
    return report


def _bool_csv(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid CSV boolean: {value}")


def _assert_close(observed: float, expected: float, *, name: str) -> None:
    if not math.isclose(observed, expected, rel_tol=2e-11, abs_tol=2e-12):
        raise ValueError(f"{name} arithmetic mismatch: {observed} != {expected}")


def _assert_nonnegative_finite(value: str, *, name: str) -> None:
    """Check timing syntax without treating a wall clock as reproducible evidence."""

    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def validate_result_bundle(output_dir: Path) -> dict[str, Any]:
    observed_files = {path.name for path in output_dir.iterdir()}
    if observed_files != EXPECTED_OUTPUT_FILES:
        raise ValueError("result file set differs from the frozen manifest")
    checksum_lines = (output_dir / "checksums.sha256").read_text(encoding="ascii").splitlines()
    expected_names = list(OUTPUT_PAYLOADS)
    if len(checksum_lines) != len(expected_names):
        raise ValueError("checksum manifest length differs")
    checksum_map: dict[str, str] = {}
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        if name in checksum_map or name not in OUTPUT_PAYLOADS:
            raise ValueError("checksum manifest contains duplicate or unexpected payload")
        if len(digest) != 64 or digest != _sha256(output_dir / name):
            raise ValueError(f"checksum mismatch for {name}")
        checksum_map[name] = digest
    if set(checksum_map) != set(OUTPUT_PAYLOADS):
        raise ValueError("checksum manifest is incomplete")

    config = _strict_json_text((output_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    config = _validate_config(config)
    report = _strict_json_text((output_dir / "report.json").read_text(encoding="utf-8"))
    if report["schema_version"] != REPORT_SCHEMA_VERSION or report["status"] != STATUS:
        raise ValueError("report schema/status mismatch")
    canonical_config_hash = hashlib.sha256(
        (_canonical_json(config) + "\n").encode("utf-8")
    ).hexdigest()
    if report["provenance"]["config_sha256"] != canonical_config_hash:
        raise ValueError("report config hash mismatch")
    if report["provenance"]["source_sha256"] != _source_hashes():
        raise ValueError("report source hashes differ from current implementation")

    audit_rows = _read_csv(output_dir / "partition_audit_rows.csv", AUDIT_FIELDS)
    metric_rows = _read_csv(output_dir / "metric_rows.csv", METRIC_FIELDS)
    selection_rows = _read_csv(output_dir / "selection_rows.csv", SELECTION_FIELDS)
    geometry_rows = _read_csv(output_dir / "geometry_manifest.csv", GEOMETRY_FIELDS)
    trajectory_rows = _read_csv(output_dir / "trajectory_rows.csv", TRAJECTORY_FIELDS)
    cost_rows = _read_csv(output_dir / "construction_cost_rows.csv", COST_FIELDS)
    rigs = _generate_from_config(config)
    train_rigs, safety_rigs, fresh_rigs, _ = split_rigs_three_way(rigs)
    specs = predefined_partitions()
    geometry_index = {row["rig_id"]: row for row in geometry_rows}
    if set(geometry_index) != {rig.rig_id for rig in rigs} or len(geometry_index) != len(geometry_rows):
        raise ValueError("geometry manifest coverage differs")
    for rig in rigs:
        row = geometry_index[rig.rig_id]
        if row["split_role"] != rig.split_role:
            raise ValueError("geometry manifest split role mismatch")
        for field, expected in (
            ("geometry_seed_sha256", rig.geometry_seed_sha256),
            ("noise_seed_sha256", rig.noise_seed_sha256),
            ("geometry_feature_sha256", rig.geometry_feature_sha256),
        ):
            if row[field] != expected:
                raise ValueError(f"geometry manifest {field} mismatch")
        for index, expected in enumerate(rig.geometry_features):
            _assert_close(
                float(row[f"geometry_feature_{index}"]),
                float(expected),
                name=f"geometry manifest feature {index}",
            )
    if report["provenance"]["geometry_manifest_sha256"] != _sha256(
        output_dir / "geometry_manifest.csv"
    ):
        raise ValueError("geometry manifest provenance hash mismatch")
    expected_audit_keys = {(rig.rig_id, name) for rig in rigs for name in specs}
    observed_audit = {(row["rig_id"], row["partition_name"]): row for row in audit_rows}
    if set(observed_audit) != expected_audit_keys or len(observed_audit) != len(audit_rows):
        raise ValueError("partition audit coverage differs")
    for rig in rigs:
        for name, spec in specs.items():
            majorizer = build_grouped_majorizer(rig.primitives, spec)
            metric = build_diagonal_metric(majorizer, eta=float(config["solver"]["eta"]))
            expected = audit_partition_safety(
                rig.primitives, spec, metric, eta=float(config["solver"]["eta"])
            )
            row = observed_audit[(rig.rig_id, name)]
            for field in (
                "pointwise_violation_count",
                "row_mass_mismatch_count",
                "column_mass_mismatch_count",
                "support_violation_count",
                "row_violation_count",
                "column_violation_count",
                "spectral_violation_count",
                "total_violation_count",
                "active_row_count",
                "active_column_count",
            ):
                if int(row[field]) != int(expected[field]):
                    raise ValueError(f"independent safety reconstruction mismatch: {rig.rig_id}/{name}/{field}")
            for field in (
                "maximum_entrywise_slack",
                "minimum_entrywise_slack",
                "maximum_row_product",
                "maximum_column_product",
                "dense_normalized_spectral_norm_squared",
                "schur_squared_upper_bound",
            ):
                _assert_close(float(row[field]), float(expected[field]), name=f"audit {field}")
            if _bool_csv(row["exact_masses_recomputed_from_signed_primitives"]) is not True:
                raise ValueError("audit did not recompute exact masses")

    selected_fixed, _, selector, selector_report, score_table, _ = _fit_selection_contract(
        train_rigs, safety_rigs, config
    )
    if report["selection"]["global_fixed_partition"] != selected_fixed:
        raise ValueError("global fixed selection mismatch")
    if set(report["selection"]["selector_model"]) != set(selector_report):
        raise ValueError("selector report fields differ")
    for field, expected in selector_report.items():
        observed = report["selection"]["selector_model"][field]
        if isinstance(expected, float):
            _assert_close(float(observed), expected, name=f"selector {field}")
        elif observed != expected:
            raise ValueError(f"selector {field} mismatch")
    if report["selection"]["selector_allowed_partitions"] != list(selector.allowed_partitions):
        raise ValueError("selector allowed partitions mismatch")
    if report["selection"]["fresh_exact_truth_target_access"] is not False:
        raise ValueError("fresh selector truth access flag is invalid")
    if report["selection"]["all_in_one_exact_available_to_selector"] is not False:
        raise ValueError("selector exact availability flag is invalid")

    metric_index = {(row["rig_id"], row["method"]): row for row in metric_rows}
    expected_metric_keys = {(rig.rig_id, method) for rig in fresh_rigs for method in METHODS}
    if set(metric_index) != expected_metric_keys or len(metric_index) != len(metric_rows):
        raise ValueError("fresh metric coverage differs")
    selection_index = {
        (row["rig_id"], row["selection_role"]): row for row in selection_rows
    }
    if len(selection_index) != 2 * len(fresh_rigs):
        raise ValueError("selection row coverage differs")
    trajectory_index = {
        (row["rig_id"], row["method"], int(row["iteration"])): row
        for row in trajectory_rows
    }
    expected_trajectory_keys = {
        (rig.rig_id, method, int(iteration))
        for rig in fresh_rigs
        for method in METHODS
        for iteration in config["solver"]["checkpoints"]
    }
    if set(trajectory_index) != expected_trajectory_keys or len(trajectory_index) != len(trajectory_rows):
        raise ValueError("fresh trajectory coverage differs")
    cost_index = {(row["rig_id"], row["method"]): row for row in cost_rows}
    if set(cost_index) != expected_metric_keys or len(cost_index) != len(cost_rows):
        raise ValueError("construction cost coverage differs")

    expected_metric_rows: dict[tuple[str, str], dict[str, Any]] = {}
    per_harm: dict[str, float] = {}
    per_win: dict[str, bool] = {}
    selected_fresh: dict[str, str] = {}
    call_sets: dict[str, set[tuple[int, int, int, int]]] = {}
    for rig in fresh_rigs:
        expected_selected = select_partition(selector, deployment_geometry_from_rig(rig))
        expected_method_partitions = _method_partition_map(
            deployment_geometry_from_rig(rig), selected_fixed=selected_fixed, selector=selector
        )
        fixed_selection_row = selection_index[(rig.rig_id, "TRAIN_SELECTED_GLOBAL_FIXED")]
        selected_row = selection_index[(rig.rig_id, "GEOMETRY_CONDITIONED_SELECTOR")]
        for row, role, partition_name in (
            (fixed_selection_row, "TRAIN_SELECTED_GLOBAL_FIXED", selected_fixed),
            (selected_row, "GEOMETRY_CONDITIONED_SELECTOR", expected_selected),
        ):
            if row["split_role"] != rig.split_role or row["selection_role"] != role:
                raise ValueError("fresh selection metadata mismatch")
            if row["selected_partition"] != partition_name:
                raise ValueError("fresh selector partition mismatch")
            if _bool_csv(row["oracle_only"]) != specs[partition_name].oracle_only:
                raise ValueError("fresh selection oracle flag mismatch")
            for index, value in enumerate(rig.geometry_features):
                _assert_close(
                    float(row[f"geometry_feature_{index}"]),
                    float(value),
                    name=f"fresh selection geometry {rig.rig_id}/{role}/{index}",
                )
        if _bool_csv(selected_row["oracle_only"]):
            raise ValueError("selector illegally selected an oracle-only partition")
        selected_fresh[rig.rig_id] = expected_selected

        for method in METHODS:
            partition_name = expected_method_partitions[method]
            spec = specs[partition_name]
            majorizer = build_grouped_majorizer(rig.primitives, spec)
            metric = build_diagonal_metric(majorizer, eta=float(config["solver"]["eta"]))
            safety = audit_partition_safety(
                rig.primitives, spec, metric, eta=float(config["solver"]["eta"])
            )
            trajectory = run_signed_pdhg_trajectory(
                rig.signed_matrix,
                rig.target,
                rig.truth,
                metric,
                checkpoints=config["solver"]["checkpoints"],
                theta=float(config["solver"]["theta"]),
            )
            observed_metric = metric_index[(rig.rig_id, method)]
            expected_metric_rows[(rig.rig_id, method)] = {
                "partition_name": partition_name,
                "oracle_only": spec.oracle_only,
                "final": trajectory.rows[-1],
                "safety": safety,
                "ledger": trajectory.ledger,
                "cost": majorizer.construction_cost,
            }
            if observed_metric["split_role"] != rig.split_role:
                raise ValueError("metric split role mismatch")
            if observed_metric["partition_name"] != partition_name:
                raise ValueError("metric partition mismatch")
            if _bool_csv(observed_metric["oracle_only"]) != spec.oracle_only:
                raise ValueError("metric oracle flag mismatch")
            for field in ("normalized_residual_l2", "field_relative_l2"):
                _assert_close(
                    float(observed_metric[f"final_{field}"]),
                    float(trajectory.rows[-1][field]),
                    name=f"metric trajectory mismatch: {rig.rig_id}/{method}/{field}",
                )
            for field in (
                "pointwise_violation_count",
                "row_violation_count",
                "column_violation_count",
                "spectral_violation_count",
                "total_violation_count",
            ):
                if int(observed_metric[field]) != int(safety[field]):
                    raise ValueError(f"metric safety mismatch: {rig.rig_id}/{method}/{field}")
            for field in (
                "signed_forward_solver_calls",
                "signed_transpose_solver_calls",
                "signed_forward_evaluation_calls",
                "field_error_evaluation_calls",
                "iteration_budget",
            ):
                if int(observed_metric[field]) != int(trajectory.ledger[field]):
                    raise ValueError(f"metric call mismatch: {rig.rig_id}/{method}/{field}")
            for field in (
                "cost_proxy_units",
                "primitive_component_accumulation_entries",
                "signed_group_addition_entries",
                "group_absolute_value_entries",
                "materialized_signed_group_count",
                "materialized_group_component_entries",
                "largest_fused_group_component_entries",
                "full_signed_operator_materialized",
            ):
                if int(observed_metric[field]) != int(majorizer.construction_cost[field]):
                    raise ValueError(f"metric construction cost mismatch: {rig.rig_id}/{method}/{field}")
            if observed_metric["cost_proxy_definition"] != majorizer.construction_cost["definition"]:
                raise ValueError(f"metric construction cost definition mismatch: {rig.rig_id}/{method}")
            for field in (
                "construction_wall_time_seconds",
                "audit_wall_time_seconds",
                "solver_wall_time_seconds",
            ):
                _assert_nonnegative_finite(observed_metric[field], name=f"metric {field}")
            observed_cost = cost_index[(rig.rig_id, method)]
            if (
                observed_cost["partition_name"] != partition_name
                or observed_cost["definition"] != majorizer.construction_cost["definition"]
            ):
                raise ValueError(f"construction cost metadata mismatch: {rig.rig_id}/{method}")
            for field, value in majorizer.construction_cost.items():
                if field == "definition":
                    continue
                if int(observed_cost[field]) != int(value):
                    raise ValueError(f"construction cost mismatch: {rig.rig_id}/{method}/{field}")
            _assert_nonnegative_finite(
                observed_cost["construction_wall_time_seconds"],
                name=f"construction cost wall time {rig.rig_id}/{method}",
            )
            for checkpoint in trajectory.rows:
                observed_trajectory = trajectory_index[(rig.rig_id, method, int(checkpoint["iteration"]))]
                if (
                    observed_trajectory["split_role"] != rig.split_role
                    or observed_trajectory["partition_name"] != partition_name
                ):
                    raise ValueError(f"trajectory metadata mismatch: {rig.rig_id}/{method}")
                for field in ("normalized_residual_l2", "field_relative_l2", "solution_l2"):
                    _assert_close(
                        float(observed_trajectory[field]),
                        float(checkpoint[field]),
                        name=f"trajectory replay mismatch: {rig.rig_id}/{method}/{checkpoint['iteration']}/{field}",
                    )

        expected_baseline = float(
            expected_metric_rows[(rig.rig_id, "train_selected_fixed")]["final"]["field_relative_l2"]
        )
        for method in METHODS:
            expected_harm = float(
                expected_metric_rows[(rig.rig_id, method)]["final"]["field_relative_l2"]
            ) - expected_baseline
            _assert_close(
                float(metric_index[(rig.rig_id, method)]["harm_vs_train_selected_fixed_field_l2"]),
                expected_harm,
                name=f"metric harm mismatch: {rig.rig_id}/{method}",
            )
        selector_metric = metric_index[(rig.rig_id, "geometry_conditioned_selector")]
        fixed_metric = metric_index[(rig.rig_id, "train_selected_fixed")]
        harm = float(selector_metric["final_field_relative_l2"]) - float(fixed_metric["final_field_relative_l2"])
        _assert_close(float(selector_metric["harm_vs_train_selected_fixed_field_l2"]), harm, name="selector harm")
        per_harm[rig.rig_id] = harm
        per_win[rig.rig_id] = harm < -1e-12
        call_sets[rig.rig_id] = {
            (
                int(metric_index[(rig.rig_id, method)]["signed_forward_solver_calls"]),
                int(metric_index[(rig.rig_id, method)]["signed_transpose_solver_calls"]),
                int(metric_index[(rig.rig_id, method)]["signed_forward_evaluation_calls"]),
                int(metric_index[(rig.rig_id, method)]["field_error_evaluation_calls"]),
            )
            for method in METHODS
        }

    aggregate = report["aggregate_fresh_geometry_ood"]
    expected_aggregate_fields = {
        "fresh_rig_count",
        "mean_final_field_relative_l2",
        "mean_final_normalized_residual_l2",
        "mean_harm_vs_train_selected_fixed_field_l2",
        "maximum_harm_vs_train_selected_fixed_field_l2",
        "total_schur_violation_count",
        "mean_cost_proxy_units",
    }
    if set(aggregate) != set(METHODS):
        raise ValueError("aggregate methods differ")
    for method in METHODS:
        observed = aggregate[method]
        if set(observed) != expected_aggregate_fields:
            raise ValueError(f"aggregate fields differ: {method}")
        rows = [metric_index[(rig.rig_id, method)] for rig in fresh_rigs]
        expected = {
            "fresh_rig_count": len(rows),
            "mean_final_field_relative_l2": sum(float(row["final_field_relative_l2"]) for row in rows) / len(rows),
            "mean_final_normalized_residual_l2": sum(float(row["final_normalized_residual_l2"]) for row in rows) / len(rows),
            "mean_harm_vs_train_selected_fixed_field_l2": sum(float(row["harm_vs_train_selected_fixed_field_l2"]) for row in rows) / len(rows),
            "maximum_harm_vs_train_selected_fixed_field_l2": max(float(row["harm_vs_train_selected_fixed_field_l2"]) for row in rows),
            "total_schur_violation_count": sum(int(row["total_violation_count"]) for row in rows),
            "mean_cost_proxy_units": sum(int(row["cost_proxy_units"]) for row in rows) / len(rows),
        }
        for field, value in expected.items():
            if isinstance(value, float):
                _assert_close(float(observed[field]), value, name=f"aggregate mismatch: {method}/{field}")
            elif int(observed[field]) != value:
                raise ValueError(f"aggregate mismatch: {method}/{field}")
    safety_harm = {
        rig.rig_id: float(
            score_table[
                (
                    rig.rig_id,
                    select_partition(selector, deployment_geometry_from_rig(rig)),
                )
            ]
        )
        - float(score_table[(rig.rig_id, selected_fixed)])
        for rig in safety_rigs
    }
    all_safe = all(int(row["total_violation_count"]) == 0 for row in audit_rows)
    selector_safe = all(
        int(metric_index[(rig.rig_id, "geometry_conditioned_selector")]["total_violation_count"])
        == 0
        for rig in fresh_rigs
    )
    every_fresh = all(per_win.values())
    every_safety = all(value < -1e-12 for value in safety_harm.values())
    geometry_adaptation = (
        not bool(selector_report["is_constant"]) and len(set(selected_fresh.values())) >= 2
    )
    advantage_not_exact = (
        config["partitions"]["exact_oracle_name"] not in selector.allowed_partitions
        and all(value != config["partitions"]["exact_oracle_name"] for value in selected_fresh.values())
    )
    fair_calls = all(len(values) == 1 for values in call_sets.values())
    gate = (
        all_safe
        and selector_safe
        and every_fresh
        and every_safety
        and geometry_adaptation
        and advantage_not_exact
        and fair_calls
    )
    decision = report["decision"]
    expected_decision_values = {
        "all_partition_audits_zero_violation": all_safe,
        "selector_all_fresh_schur_safe": selector_safe,
        "per_fresh_rig_harm_vs_train_selected_fixed": per_harm,
        "per_fresh_rig_strict_win_flags": per_win,
        "selector_beats_train_selected_fixed_on_every_fresh_rig": every_fresh,
        "per_safety_rig_harm_vs_train_selected_fixed": safety_harm,
        "selector_beats_train_selected_fixed_on_every_safety_rig": every_safety,
        "selector_is_nonconstant": not bool(selector_report["is_constant"]),
        "fresh_unique_selected_partition_count": len(set(selected_fresh.values())),
        "geometry_adaptation_observed_on_fresh": geometry_adaptation,
        "selector_exact_all_in_one_selection_count": 0,
        "advantage_not_due_to_all_in_one_exact": advantage_not_exact,
        "equal_fresh_A_A_transpose_call_budget": fair_calls,
        "synthetic_algorithmic_smoke_gate_passed": gate,
        "research_claim_authorized": gate,
        "real_bost_claim_authorized": False,
    }
    for key, expected in expected_decision_values.items():
        observed = decision[key]
        if isinstance(expected, dict) and expected and isinstance(next(iter(expected.values())), float):
            if set(observed) != set(expected):
                raise ValueError(f"decision {key} keys mismatch")
            for item_key, item_value in expected.items():
                _assert_close(float(observed[item_key]), item_value, name=f"decision {key}/{item_key}")
        elif observed != expected:
            raise ValueError(f"decision {key} mismatch")
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "demo_t16_operator" / "configs" / "certified_grouped_majorizer_smoke_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "demo_t16_operator" / "results" / "certified_grouped_majorizer_smoke",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        report = validate_result_bundle(args.output_dir)
    else:
        report = run_smoke(load_config(args.config), output_dir=args.output_dir)
    print(_canonical_json(report["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
