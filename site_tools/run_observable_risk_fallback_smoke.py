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
    audit_partition_safety,
    build_diagonal_metric,
    build_grouped_majorizer,
    deployment_geometry_from_rig,
    predefined_partitions,
    run_signed_pdhg_trajectory,
)
from demo_t16_operator.observable_risk_fallback import (
    ALLOWED_CANDIDATES,
    FALLBACK_PARTITION,
    FIELD_ENDPOINT,
    FEATURE_NAMES,
    FORBIDDEN_ORACLE,
    RESIDUAL_ENDPOINT,
    SCHEMA_VERSION as INTERFACE_SCHEMA_VERSION,
    SPLIT_ROLES,
    STATUS,
    audit_operator_decomposition,
    calibration_policy_contract_sha256,
    calibrate_acceptance_threshold,
    feature_schema_sha256,
    fit_observable_risk_rule,
    generate_four_way_rigs,
    selection_conditional_harm_rate,
    select_with_risk_fallback,
    split_rigs_four_way,
)


CONFIG_SCHEMA_VERSION = "observable-risk-fallback-smoke-config-1.4"
REPORT_SCHEMA_VERSION = "observable-risk-fallback-smoke-report-1.4"
EVIDENCE_SCOPE = "SYNTHETIC_FOUR_SPLIT_RCCF_CPU_MICRO_SMOKE_ONLY"
METHODS = ("fallback", "candidate", "selected")
OUTPUT_PAYLOADS = (
    "config_snapshot.json",
    "geometry_manifest.csv",
    "partition_audit_rows.csv",
    "selection_rows.csv",
    "risk_rows.csv",
    "metric_rows.csv",
    "trajectory_rows.csv",
    "report.json",
)
EXPECTED_OUTPUT_FILES = frozenset((*OUTPUT_PAYLOADS, "checksums.sha256"))
SOURCE_RELATIVE_PATHS = {
    "v3_algorithm_source": "demo_t16_operator/certified_grouped_majorizer.py",
    "rccf_algorithm_source": "demo_t16_operator/observable_risk_fallback.py",
    "runner_source": "site_tools/run_observable_risk_fallback_smoke.py",
    "validator_source": "site_tools/validate_observable_risk_fallback_smoke.py",
    "config_source": "demo_t16_operator/configs/observable_risk_fallback_smoke_v1.json",
}

GEOMETRY_FIELDS = (
    "rig_id",
    "split_role",
    "geometry_seed_sha256",
    "noise_seed_sha256",
    "geometry_feature_sha256",
    *(f"geometry_feature_{index}" for index in range(len(FEATURE_NAMES))),
)
AUDIT_FIELDS = (
    "rig_id",
    "split_role",
    "partition_name",
    "operator_decomposition_mismatch_count",
    "operator_decomposition_max_abs_error",
    "operator_decomposition_verified",
    "pointwise_violation_count",
    "row_violation_count",
    "column_violation_count",
    "spectral_violation_count",
    "total_violation_count",
    "maximum_row_product",
    "maximum_column_product",
    "dense_normalized_spectral_norm_squared",
    "schur_squared_upper_bound",
)
SELECTION_FIELDS = (
    "rig_id",
    "split_role",
    "candidate_partition",
    "selected_partition",
    "fallback_partition",
    "fallback_used",
    "fallback_reason",
    "risk_score",
    "acceptance_threshold",
    "risk_upper_bound",
    "support_gate_passed",
    "observable_feature_sha256",
    "observable_feature_schema_sha256",
    "uses_truth",
    "uses_target",
    "uses_primitives",
    "uses_signed_matrix",
    "uses_exact_abs_operator",
    "uses_solver_trajectory",
)
RISK_FIELDS = (
    "rig_id",
    "split_role",
    "candidate_partition",
    "risk_score",
    "acceptance_threshold",
    "support_gate_passed",
    "observed_field_harm_vs_fallback",
    "observed_residual_harm_vs_fallback",
    "harm_failure",
    "fallback_used",
    "selection_frozen_before_offline_evaluation",
)
METRIC_FIELDS = (
    "rig_id",
    "split_role",
    "method",
    "partition_name",
    "final_normalized_residual_l2",
    "final_field_relative_l2",
    "harm_vs_fallback_field_l2",
    "harm_vs_fallback_residual_l2",
    "total_violation_count",
    "signed_forward_solver_calls",
    "signed_transpose_solver_calls",
    "signed_forward_evaluation_calls",
    "field_error_evaluation_calls",
    "iteration_budget",
    "cost_proxy_units",
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
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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
    _require_exact_keys(
        "config",
        config,
        {
            "schema_version",
            "status",
            "evidence_scope",
            "seeds",
            "rigs",
            "partitions",
            "model",
            "solver",
            "risk_calibration",
            "development_gate",
            "future_paper_gate",
            "runtime",
            "claim_boundary",
        },
    )
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("config schema version is not frozen")
    if config["status"] != STATUS or config["evidence_scope"] != EVIDENCE_SCOPE:
        raise ValueError("config must remain development-only synthetic evidence")

    seeds = config["seeds"]
    _require_exact_keys("seeds", seeds, {"geometry", "noise"})
    if any(not isinstance(value, int) or isinstance(value, bool) for value in seeds.values()):
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
            "random_ray_or_pixel_split_used",
            "assignments",
        },
    )
    if rigs["split_unit"] != "COMPLETE_RIG" or rigs["random_ray_or_pixel_split_used"] is not False:
        raise ValueError("four-way splitting must use complete rigs")
    if rigs["primitive_count"] != PRIMITIVE_COUNT:
        raise ValueError("primitive count differs from the frozen v3 interface")
    if any(not isinstance(rigs[name], int) or rigs[name] < 4 for name in ("row_count", "column_count")):
        raise ValueError("matrix dimensions must be integers at least four")
    assignments = rigs["assignments"]
    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError("rig assignments must be a nonempty object")
    if set(assignments.values()).difference(SPLIT_ROLES):
        raise ValueError("rig assignment contains an unknown split role")
    minimums = {
        "train": 8,
        "model_selection": 6,
        "risk_calibration": 12,
        "fresh_geometry_ood": 8,
    }
    for role, minimum in minimums.items():
        if sum(value == role for value in assignments.values()) < minimum:
            raise ValueError(f"{role} requires at least {minimum} complete rigs")

    partitions = config["partitions"]
    _require_exact_keys(
        "partitions",
        partitions,
        {
            "candidate_names",
            "fallback_partition",
            "forbidden_oracle",
            "all_in_one_for_selector_forbidden",
        },
    )
    if partitions["candidate_names"] != list(ALLOWED_CANDIDATES):
        raise ValueError("selector candidate catalogue is not frozen")
    if partitions["fallback_partition"] != FALLBACK_PARTITION:
        raise ValueError("fallback must remain paired_cross")
    if partitions["forbidden_oracle"] != FORBIDDEN_ORACLE:
        raise ValueError("forbidden oracle must remain all_in_one_exact")
    if partitions["all_in_one_for_selector_forbidden"] is not True:
        raise ValueError("all-in-one exact must remain forbidden")

    model = config["model"]
    _require_exact_keys(
        "model",
        model,
        {
            "class",
            "train_top_k",
            "failure_penalty",
            "field_harm_tolerance",
            "residual_harm_tolerance",
            "fresh_sensitive_access_forbidden",
        },
    )
    if model["class"] != "DEPTH_ONE_OBSERVABLE_GEOMETRY_RISK_RULE":
        raise ValueError("risk model class is not frozen")
    if not isinstance(model["train_top_k"], int) or model["train_top_k"] < 1:
        raise ValueError("train_top_k must be positive")
    if (
        float(model["failure_penalty"]) < 0.0
        or float(model["field_harm_tolerance"]) < 0.0
        or float(model["residual_harm_tolerance"]) < 0.0
    ):
        raise ValueError("model penalties must be nonnegative")
    if model["fresh_sensitive_access_forbidden"] is not True:
        raise ValueError("fresh sensitive access must remain forbidden")

    solver = config["solver"]
    _require_exact_keys("solver", solver, {"eta", "theta", "checkpoints"})
    if not 0.0 < float(solver["eta"]) < 1.0 or not 0.0 <= float(solver["theta"]) <= 1.0:
        raise ValueError("solver parameters are outside the frozen domain")
    checkpoints = solver["checkpoints"]
    if (
        not isinstance(checkpoints, list)
        or checkpoints != sorted(set(checkpoints))
        or not checkpoints
        or checkpoints[0] != 0
        or checkpoints[-1] < 1
        or any(not isinstance(value, int) or value < 0 for value in checkpoints)
    ):
        raise ValueError("solver checkpoints are invalid")

    risk = config["risk_calibration"]
    _require_exact_keys(
        "risk_calibration",
        risk,
        {
            "method",
            "confidence_alpha",
            "coverage_confidence_alpha",
            "multiplicity_correction",
        },
    )
    if (
        risk["method"] != "ONE_SIDED_CLOPPER_PEARSON"
        or risk["multiplicity_correction"]
        != "BONFERRONI_FROZEN_FINITE_GRID"
        or not 0.0 < float(risk["confidence_alpha"]) < 1.0
        or not 0.0 < float(risk["coverage_confidence_alpha"]) < 1.0
    ):
        raise ValueError("risk calibration contract is invalid")
    if (
        float(risk["confidence_alpha"])
        + float(risk["coverage_confidence_alpha"])
        > 0.05
    ):
        raise ValueError("joint risk and coverage confidence budget exceeds 0.05")

    development = config["development_gate"]
    _require_exact_keys(
        "development_gate",
        development,
        {
            "minimum_calibration_rigs",
            "minimum_fresh_rigs",
            "maximum_risk_upper",
            "minimum_takeover_coverage",
            "maximum_fresh_conditional_harm_rate",
            "maximum_worst_field_harm",
            "maximum_worst_residual_harm",
        },
    )
    if development["minimum_calibration_rigs"] < 12 or development["minimum_fresh_rigs"] < 8:
        raise ValueError("development sample floors cannot be weakened")
    for name in (
        "maximum_risk_upper",
        "minimum_takeover_coverage",
        "maximum_fresh_conditional_harm_rate",
    ):
        if not 0.0 <= float(development[name]) <= 1.0:
            raise ValueError(f"development {name} must lie in [0,1]")
    if float(development["maximum_risk_upper"]) > 0.5:
        raise ValueError("development risk upper limit cannot be weakened")
    if float(development["minimum_takeover_coverage"]) < 0.25:
        raise ValueError("development takeover coverage cannot be weakened")
    if float(development["maximum_fresh_conditional_harm_rate"]) > 0.0:
        raise ValueError("development fresh harm-rate limit cannot be weakened")
    for endpoint in ("field", "residual"):
        name = f"maximum_worst_{endpoint}_harm"
        if float(development[name]) < 0.0:
            raise ValueError("development worst-harm limit must be nonnegative")
        if float(development[name]) > 0.02:
            raise ValueError("development worst-harm limit cannot be weakened")

    future = config["future_paper_gate"]
    _require_exact_keys(
        "future_paper_gate",
        future,
        {
            "minimum_calibration_rigs",
            "minimum_fresh_rigs",
            "maximum_risk_upper",
            "minimum_takeover_coverage",
            "maximum_fresh_conditional_harm_rate",
            "maximum_worst_field_harm",
            "maximum_worst_residual_harm",
            "requires_real_bost_signed_primitive_interface",
            "requires_independent_geometry_clusters",
            "requires_measured_cost_pareto",
        },
    )
    if future["minimum_calibration_rigs"] < 128 or future["minimum_fresh_rigs"] < 48:
        raise ValueError("future paper sample floors cannot be weakened")
    if (
        float(future["maximum_risk_upper"]) > 0.05
        or float(future["maximum_fresh_conditional_harm_rate"]) > 0.05
    ):
        raise ValueError("future paper risk limits cannot be weakened")
    if float(future["minimum_takeover_coverage"]) < 0.3:
        raise ValueError("future paper takeover coverage cannot be weakened")
    if (
        float(future["maximum_worst_field_harm"]) > 0.02
        or float(future["maximum_worst_residual_harm"]) > 0.02
    ):
        raise ValueError("future paper worst-harm limit cannot be weakened")
    if any(future[name] is not True for name in (
        "requires_real_bost_signed_primitive_interface",
        "requires_independent_geometry_clusters",
        "requires_measured_cost_pareto",
    )):
        raise ValueError("future paper physical and cost gates are mandatory")

    runtime = config["runtime"]
    if runtime != {
        "device": "cpu",
        "dtype": "torch.float64",
        "timing_role": "MEASURED_SINGLE_RUN_DESCRIPTIVE_NONCOMPARATIVE",
    }:
        raise ValueError("runtime must remain the frozen CPU smoke")
    claims = config["claim_boundary"]
    _require_exact_keys(
        "claim_boundary",
        claims,
        {
            "synthetic_interface_success_claimed",
            "real_bost_claimed",
            "generalization_claimed",
            "paper_superiority_claimed",
            "deeponet_fno_nerif_superiority_claimed",
        },
    )
    if any(value is not False for value in claims.values()):
        raise ValueError("all claim boundaries must remain false")
    return json.loads(_canonical_json(config))


def load_config(path: Path) -> dict[str, Any]:
    value = _strict_json_text(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("config root must be an object")
    return _validate_config(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"source_commit": commit, "source_worktree_dirty": dirty}


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        name: _sha256(root / relative_path)
        for name, relative_path in SOURCE_RELATIVE_PATHS.items()
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


def _generate(config: Mapping[str, Any]):
    rigs = config["rigs"]
    return generate_four_way_rigs(
        split_assignments=rigs["assignments"],
        geometry_seed=int(config["seeds"]["geometry"]),
        noise_seed=int(config["seeds"]["noise"]),
        row_count=int(rigs["row_count"]),
        column_count=int(rigs["column_count"]),
    )


def _offline_score_table(
    rigs, config: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, float]]:
    specs = predefined_partitions()
    output: dict[tuple[str, str], dict[str, float]] = {}
    for rig in rigs:
        for partition_name in ALLOWED_CANDIDATES:
            majorizer = build_grouped_majorizer(rig.primitives, specs[partition_name])
            metric = build_diagonal_metric(majorizer, eta=float(config["solver"]["eta"]))
            trajectory = run_signed_pdhg_trajectory(
                rig.signed_matrix,
                rig.target,
                rig.truth,
                metric,
                checkpoints=config["solver"]["checkpoints"],
                theta=float(config["solver"]["theta"]),
            )
            output[(rig.rig_id, partition_name)] = {
                FIELD_ENDPOINT: float(trajectory.rows[-1][FIELD_ENDPOINT]),
                RESIDUAL_ENDPOINT: float(trajectory.rows[-1][RESIDUAL_ENDPOINT]),
            }
    return output


def reconstruct_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically rebuild every scientific field from the frozen config."""

    frozen = _validate_config(config)
    rigs = _generate(frozen)
    grouped, split_contract = split_rigs_four_way(rigs)
    specs = predefined_partitions()
    decomposition_by_rig = {
        rig.rig_id: audit_operator_decomposition(rig.primitives, rig.signed_matrix)
        for rig in rigs
    }
    if any(
        not bool(audit["operator_decomposition_verified"])
        for audit in decomposition_by_rig.values()
    ):
        raise RuntimeError("actual solver operator differs from signed primitive sum")
    offline_rigs = [
        *grouped["train"],
        *grouped["model_selection"],
        *grouped["risk_calibration"],
    ]
    score_table = _offline_score_table(offline_rigs, frozen)
    harm_tolerances = {
        FIELD_ENDPOINT: float(frozen["model"]["field_harm_tolerance"]),
        RESIDUAL_ENDPOINT: float(frozen["model"]["residual_harm_tolerance"]),
    }
    rule, model_report = fit_observable_risk_rule(
        grouped["train"],
        grouped["model_selection"],
        score_table,
        candidate_names=frozen["partitions"]["candidate_names"],
        train_top_k=int(frozen["model"]["train_top_k"]),
        harm_tolerances=harm_tolerances,
        failure_penalty=float(frozen["model"]["failure_penalty"]),
    )
    expected_policy_contract_sha256 = calibration_policy_contract_sha256(
        rule,
        harm_tolerances=harm_tolerances,
        confidence_alpha=float(frozen["risk_calibration"]["confidence_alpha"]),
        coverage_confidence_alpha=float(
            frozen["risk_calibration"]["coverage_confidence_alpha"]
        ),
        maximum_risk_upper=float(frozen["development_gate"]["maximum_risk_upper"]),
        minimum_takeover_coverage=float(
            frozen["development_gate"]["minimum_takeover_coverage"]
        ),
    )
    calibration, calibration_records = calibrate_acceptance_threshold(
        rule,
        grouped["risk_calibration"],
        score_table,
        harm_tolerances=harm_tolerances,
        confidence_alpha=float(frozen["risk_calibration"]["confidence_alpha"]),
        coverage_confidence_alpha=float(
            frozen["risk_calibration"]["coverage_confidence_alpha"]
        ),
        maximum_risk_upper=float(frozen["development_gate"]["maximum_risk_upper"]),
        minimum_takeover_coverage=float(frozen["development_gate"]["minimum_takeover_coverage"]),
    )

    # Freeze every fresh decision before any fresh truth, target, primitive, or
    # solver trajectory is read by the offline evaluator below.
    fresh_decisions = {
        rig.rig_id: select_with_risk_fallback(
            deployment_geometry_from_rig(rig),
            rule,
            calibration,
            expected_policy_contract_sha256=expected_policy_contract_sha256,
        )
        for rig in grouped["fresh_geometry_ood"]
    }

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

    audit_rows: list[dict[str, Any]] = []
    for rig in rigs:
        decomposition = decomposition_by_rig[rig.rig_id]
        for partition_name in ALLOWED_CANDIDATES:
            spec = specs[partition_name]
            majorizer = build_grouped_majorizer(rig.primitives, spec)
            metric = build_diagonal_metric(majorizer, eta=float(frozen["solver"]["eta"]))
            audit = audit_partition_safety(
                rig.primitives, spec, metric, eta=float(frozen["solver"]["eta"])
            )
            combined_audit = {**decomposition, **audit}
            audit_rows.append(
                {
                    "rig_id": rig.rig_id,
                    "split_role": rig.split_role,
                    "partition_name": partition_name,
                    **{field: combined_audit[field] for field in AUDIT_FIELDS[3:]},
                }
            )
    if any(int(row["total_violation_count"]) != 0 for row in audit_rows):
        raise RuntimeError("deterministic grouped-majorizer certificate failed")

    selection_rows = [
        {
            "rig_id": decision.rig_id,
            "split_role": "fresh_geometry_ood",
            **{
                field: getattr(decision, field)
                for field in SELECTION_FIELDS
                if field not in {"rig_id", "split_role"}
            },
        }
        for decision in (fresh_decisions[rig.rig_id] for rig in grouped["fresh_geometry_ood"])
    ]

    risk_rows: list[dict[str, Any]] = []
    for row in calibration_records:
        fallback_used = not (
            calibration.development_gate_passed
            and bool(row["support_gate_passed"])
            and row["candidate_partition"] != FALLBACK_PARTITION
            and float(row["risk_score"]) <= calibration.acceptance_threshold
        )
        risk_rows.append(
            {
                "rig_id": row["rig_id"],
                "split_role": "risk_calibration",
                "candidate_partition": row["candidate_partition"],
                "risk_score": row["risk_score"],
                "acceptance_threshold": calibration.acceptance_threshold,
                "support_gate_passed": row["support_gate_passed"],
                "observed_field_harm_vs_fallback": row[
                    "observed_field_harm_vs_fallback"
                ],
                "observed_residual_harm_vs_fallback": row[
                    "observed_residual_harm_vs_fallback"
                ],
                "harm_failure": row["harm_failure"],
                "fallback_used": fallback_used,
                "selection_frozen_before_offline_evaluation": True,
            }
        )

    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    fresh_selected_field_harms: list[float] = []
    fresh_selected_residual_harms: list[float] = []
    fresh_joint_failures: list[bool] = []
    takeover_field_harms: list[float] = []
    takeover_residual_harms: list[float] = []
    for rig in grouped["fresh_geometry_ood"]:
        decision = fresh_decisions[rig.rig_id]
        method_partitions = {
            "fallback": FALLBACK_PARTITION,
            "candidate": decision.candidate_partition,
            "selected": decision.selected_partition,
        }
        rig_rows: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            partition_name = method_partitions[method]
            spec = specs[partition_name]
            majorizer = build_grouped_majorizer(rig.primitives, spec)
            metric = build_diagonal_metric(majorizer, eta=float(frozen["solver"]["eta"]))
            audit = audit_partition_safety(
                rig.primitives, spec, metric, eta=float(frozen["solver"]["eta"])
            )
            trajectory = run_signed_pdhg_trajectory(
                rig.signed_matrix,
                rig.target,
                rig.truth,
                metric,
                checkpoints=frozen["solver"]["checkpoints"],
                theta=float(frozen["solver"]["theta"]),
            )
            final = trajectory.rows[-1]
            row = {
                "rig_id": rig.rig_id,
                "split_role": rig.split_role,
                "method": method,
                "partition_name": partition_name,
                "final_normalized_residual_l2": final["normalized_residual_l2"],
                "final_field_relative_l2": final["field_relative_l2"],
                "harm_vs_fallback_field_l2": 0.0,
                "harm_vs_fallback_residual_l2": 0.0,
                "total_violation_count": audit["total_violation_count"],
                **trajectory.ledger,
                "cost_proxy_units": majorizer.construction_cost["cost_proxy_units"],
                "cost_proxy_definition": majorizer.construction_cost["definition"],
            }
            rig_rows[method] = row
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
        fallback_error = float(rig_rows["fallback"]["final_field_relative_l2"])
        fallback_residual = float(
            rig_rows["fallback"]["final_normalized_residual_l2"]
        )
        for method in METHODS:
            rig_rows[method]["harm_vs_fallback_field_l2"] = (
                float(rig_rows[method]["final_field_relative_l2"]) - fallback_error
            )
            rig_rows[method]["harm_vs_fallback_residual_l2"] = (
                float(rig_rows[method]["final_normalized_residual_l2"])
                - fallback_residual
            )
            metric_rows.append(rig_rows[method])
        selected_field_harm = float(
            rig_rows["selected"]["harm_vs_fallback_field_l2"]
        )
        selected_residual_harm = float(
            rig_rows["selected"]["harm_vs_fallback_residual_l2"]
        )
        joint_failure = (
            selected_field_harm > harm_tolerances[FIELD_ENDPOINT]
            or selected_residual_harm > harm_tolerances[RESIDUAL_ENDPOINT]
        )
        fresh_selected_field_harms.append(selected_field_harm)
        fresh_selected_residual_harms.append(selected_residual_harm)
        fresh_joint_failures.append(joint_failure)
        if not decision.fallback_used:
            takeover_field_harms.append(selected_field_harm)
            takeover_residual_harms.append(selected_residual_harm)
        risk_rows.append(
            {
                "rig_id": rig.rig_id,
                "split_role": rig.split_role,
                "candidate_partition": decision.candidate_partition,
                "risk_score": decision.risk_score,
                "acceptance_threshold": decision.acceptance_threshold,
                "support_gate_passed": decision.support_gate_passed,
                "observed_field_harm_vs_fallback": selected_field_harm,
                "observed_residual_harm_vs_fallback": selected_residual_harm,
                "harm_failure": joint_failure,
                "fallback_used": decision.fallback_used,
                "selection_frozen_before_offline_evaluation": True,
            }
        )

    fresh_count = len(grouped["fresh_geometry_ood"])
    takeover_count = sum(not decision.fallback_used for decision in fresh_decisions.values())
    fresh_harm_count = sum(fresh_joint_failures)
    fresh_conditional_harm_rate = selection_conditional_harm_rate(
        fresh_harm_count, takeover_count
    )
    worst_takeover_field_harm = (
        max(takeover_field_harms) if takeover_field_harms else None
    )
    worst_takeover_residual_harm = (
        max(takeover_residual_harms) if takeover_residual_harms else None
    )
    all_audits_safe = all(
        int(row["total_violation_count"]) == 0
        and int(row["operator_decomposition_mismatch_count"]) == 0
        for row in audit_rows
    )
    development = frozen["development_gate"]
    micro_gate = (
        all_audits_safe
        and calibration.development_gate_passed
        and calibration.calibration_count >= int(development["minimum_calibration_rigs"])
        and fresh_count >= int(development["minimum_fresh_rigs"])
        and takeover_count / fresh_count
        >= float(development["minimum_takeover_coverage"])
        and fresh_conditional_harm_rate is not None
        and fresh_conditional_harm_rate
        <= float(development["maximum_fresh_conditional_harm_rate"])
        and worst_takeover_field_harm is not None
        and worst_takeover_field_harm
        <= float(development["maximum_worst_field_harm"])
        and worst_takeover_residual_harm is not None
        and worst_takeover_residual_harm
        <= float(development["maximum_worst_residual_harm"])
    )
    fallback_metric = [row for row in metric_rows if row["method"] == "fallback"]
    selected_metric = [row for row in metric_rows if row["method"] == "selected"]
    aggregate = {
        "calibration_count": calibration.calibration_count,
        "calibration_accepted_count": calibration.accepted_count,
        "calibration_failure_count": calibration.failure_count,
        "calibration_failure_fraction": (
            calibration.failure_count / calibration.accepted_count
            if calibration.accepted_count
            else 0.0
        ),
        "calibration_risk_upper_bound": calibration.risk_upper_bound,
        "calibration_takeover_coverage": calibration.takeover_coverage,
        "calibration_takeover_coverage_lower_bound": calibration.takeover_coverage_lower_bound,
        "calibration_authorized_takeover_coverage": calibration.authorized_takeover_coverage,
        "calibration_authorized_takeover_coverage_lower_bound": calibration.authorized_takeover_coverage_lower_bound,
        "fresh_count": fresh_count,
        "fresh_takeover_count": takeover_count,
        "fresh_takeover_coverage": takeover_count / fresh_count,
        "fresh_fallback_rate": 1.0 - takeover_count / fresh_count,
        "fresh_harm_count": fresh_harm_count,
        "fresh_policy_harm_rate": fresh_harm_count / fresh_count,
        "fresh_selection_conditional_harm_rate": fresh_conditional_harm_rate,
        "fresh_worst_takeover_field_harm": worst_takeover_field_harm,
        "fresh_worst_takeover_residual_harm": worst_takeover_residual_harm,
        "fresh_mean_selected_field_relative_l2": sum(
            float(row["final_field_relative_l2"]) for row in selected_metric
        )
        / fresh_count,
        "fresh_mean_fallback_field_relative_l2": sum(
            float(row["final_field_relative_l2"]) for row in fallback_metric
        )
        / fresh_count,
        "partition_audit_count": len(audit_rows),
        "partition_audit_violation_count": sum(
            int(row["total_violation_count"]) for row in audit_rows
        ),
        "operator_decomposition_mismatch_count": sum(
            int(row["operator_decomposition_mismatch_count"])
            for row in audit_rows
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "interface_schema_version": INTERFACE_SCHEMA_VERSION,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "split_contract": split_contract,
        "mathematical_contract": {
            "operator_decomposition": "A=sum_l C_l",
            "actual_solver_operator_decomposition_audited": True,
            "partition_majorizer": "M_P=sum_G abs(sum_l_in_G C_l)",
            "deterministic_certificate": "M_P >= abs(A) entrywise; Schur audit required for every rig and candidate",
            "statistical_risk_does_not_replace_certificate": True,
        },
        "model_selection_contract": {
            "train_role": "GENERATE_FINITE_RULE_CANDIDATES_AND_OFFLINE_HARM_LABELS",
            "model_selection_role": "SELECT_MODEL_FEATURE_AND_RULE_ONLY",
            "risk_calibration_role": "FREEZE_ACCEPTANCE_THRESHOLD_AND_RISK_BOUND_ONLY",
            "fresh_role": "ONE_PASS_OBSERVABLE_SELECTION_THEN_OFFLINE_EVALUATION",
            "model": model_report,
            "joint_harm_endpoints": [FIELD_ENDPOINT, RESIDUAL_ENDPOINT],
            "front_endpoint_role": "REQUIRED_AT_SPATIAL_TOMOGRAPHY_STAGE_NOT_DEFINED_ON_TINY_VECTOR_SMOKE",
        },
        "risk_calibration": {
            "method": "ONE_SIDED_CLOPPER_PEARSON",
            "confidence_alpha": calibration.confidence_alpha,
            "coverage_confidence_alpha": calibration.coverage_confidence_alpha,
            "multiplicity_correction": calibration.multiplicity_correction,
            "threshold_candidate_count": calibration.threshold_candidate_count,
            "threshold_grid": list(calibration.threshold_grid),
            "rule_contract_sha256": calibration.rule_contract_sha256,
            "policy_contract_sha256": calibration.policy_contract_sha256,
            "independently_expected_policy_contract_sha256": expected_policy_contract_sha256,
            "joint_harm_endpoints": list(calibration.joint_harm_endpoints),
            "harm_tolerances": dict(calibration.harm_tolerances),
            "corrected_risk_alpha": calibration.corrected_risk_alpha,
            "corrected_coverage_alpha": calibration.corrected_coverage_alpha,
            "maximum_risk_upper": calibration.maximum_risk_upper,
            "minimum_takeover_coverage": calibration.minimum_takeover_coverage,
            "simultaneous_family_confidence_lower_bound": 1.0
            - calibration.confidence_alpha
            - calibration.coverage_confidence_alpha,
            "acceptance_threshold": calibration.acceptance_threshold,
            "accepted_count": calibration.accepted_count,
            "failure_count": calibration.failure_count,
            "risk_upper_bound": calibration.risk_upper_bound,
            "takeover_coverage": calibration.takeover_coverage,
            "takeover_coverage_lower_bound": calibration.takeover_coverage_lower_bound,
            "authorized_takeover_coverage": calibration.authorized_takeover_coverage,
            "authorized_takeover_coverage_lower_bound": calibration.authorized_takeover_coverage_lower_bound,
            "development_gate_passed": calibration.development_gate_passed,
        },
        "observable_only_contract": {
            "feature_names": list(FEATURE_NAMES),
            "feature_schema_sha256": feature_schema_sha256(),
            "support_gate": "TRAIN_PLUS_MODEL_SELECTION_AXIS_ALIGNED_FEATURE_ENVELOPE",
            "fresh_selector_input_type": "DeploymentGeometry",
            "uses_truth": False,
            "uses_target": False,
            "uses_primitives": False,
            "uses_signed_matrix": False,
            "uses_exact_abs_operator": False,
            "uses_solver_trajectory": False,
            "fallback_partition": FALLBACK_PARTITION,
            "forbidden_oracle": FORBIDDEN_ORACLE,
        },
        "validation_contract": {
            "deterministic_full_replay": True,
            "independent_scipy_cp_and_conditional_aggregation_cross_check": True,
            "independent_solver_implementation": False,
        },
        "aggregate": aggregate,
        "gates": {
            "synthetic_micro_interface_gate_passed": micro_gate,
            "future_paper_gate_passed": False,
            "future_paper_gate_failure_reasons": [
                "MICRO_SAMPLE_COUNTS_BELOW_FUTURE_PAPER_FLOORS",
                "REAL_BOST_SIGNED_PRIMITIVE_INTERFACE_NOT_VALIDATED",
                "INDEPENDENT_REAL_GEOMETRY_CLUSTERS_NOT_EVALUATED",
                "MEASURED_COST_PARETO_NOT_EVALUATED",
            ],
            "research_claim_authorized": False,
            "real_bost_claim_authorized": False,
            "generalization_claim_authorized": False,
            "paper_superiority_claim_authorized": False,
        },
        "claim_boundary": frozen["claim_boundary"],
        "limitations": [
            "This is a small synthetic interface gate, not a validated BOST forward model.",
            "Offline train, model-selection, calibration, and evaluation stages use synthetic truth; fresh selection does not.",
            "Clopper-Pearson calibration uses too few rigs for a paper claim.",
            "The tiny vector smoke has no defensible spatial front metric; front harm becomes mandatory at the spatial tomography stage.",
            "Context and sampling-manifest mismatch gates require the real BOST interface and are not authorized by this toy.",
            "The validator independently recomputes CP bounds and conditional aggregates, but shares the synthetic solver implementation.",
            "Wall time is descriptive and noncomparative.",
            "No DeepONet, FNO, NeRIF, real-data, generalization, or superiority claim is authorized.",
        ],
    }
    return {
        "config": frozen,
        "geometry_rows": geometry_rows,
        "audit_rows": sorted(audit_rows, key=lambda row: (row["rig_id"], row["partition_name"])),
        "selection_rows": sorted(selection_rows, key=lambda row: row["rig_id"]),
        "risk_rows": sorted(risk_rows, key=lambda row: (row["split_role"], row["rig_id"])),
        "metric_rows": sorted(metric_rows, key=lambda row: (row["rig_id"], METHODS.index(row["method"]))),
        "trajectory_rows": sorted(
            trajectory_rows,
            key=lambda row: (row["rig_id"], METHODS.index(row["method"]), int(row["iteration"])),
        ),
        "report": report,
    }


def run_smoke(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    entry_git_state = _git_state()
    started = perf_counter()
    evidence = reconstruct_evidence(config)
    _prepare_output(output_dir)
    config_text = _canonical_json(evidence["config"]) + "\n"
    (output_dir / "config_snapshot.json").write_text(config_text, encoding="utf-8")
    _write_csv(output_dir / "geometry_manifest.csv", evidence["geometry_rows"], GEOMETRY_FIELDS)
    _write_csv(output_dir / "partition_audit_rows.csv", evidence["audit_rows"], AUDIT_FIELDS)
    _write_csv(output_dir / "selection_rows.csv", evidence["selection_rows"], SELECTION_FIELDS)
    _write_csv(output_dir / "risk_rows.csv", evidence["risk_rows"], RISK_FIELDS)
    _write_csv(output_dir / "metric_rows.csv", evidence["metric_rows"], METRIC_FIELDS)
    _write_csv(output_dir / "trajectory_rows.csv", evidence["trajectory_rows"], TRAJECTORY_FIELDS)
    report = {
        **evidence["report"],
        "provenance": {
            **entry_git_state,
            "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
            "source_sha256": _source_hashes(),
            "geometry_manifest_sha256": _sha256(output_dir / "geometry_manifest.csv"),
        },
        "runtime": {
            "wall_time_seconds": perf_counter() - started,
            "wall_time_role": "MEASURED_SINGLE_RUN_DESCRIPTIVE_NONCOMPARATIVE",
            "device": "cpu",
            "dtype": "torch.float64",
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256(output_dir / name)}  {name}\n" for name in OUTPUT_PAYLOADS),
        encoding="ascii",
    )
    from site_tools.validate_observable_risk_fallback_smoke import validate_result_bundle

    validate_result_bundle(output_dir)
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "demo_t16_operator" / "configs" / "observable_risk_fallback_smoke_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/oerf_observable_risk_fallback_smoke"),
    )
    args = parser.parse_args()
    report = run_smoke(load_config(args.config), output_dir=args.output_dir)
    print(_canonical_json({"aggregate": report["aggregate"], "gates": report["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
