#!/usr/bin/env python3
"""Run the frozen N1.9 camera-contrast/global-K screen on opened development."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.jacru_n1_8_camera_ray_hybrid import (  # noqa: E402
    visible_total_correction_radius,
)
from demo_t16_operator.jacru_n1_9_global_contrast import (  # noqa: E402
    SUPPORTED_CONTRAST_BASIS_KINDS,
    build_global_contrast_basis,
)
from demo_t16_operator.psu_b0_streaming_operator import (  # noqa: E402
    zero_outer_boundary_support,
)
from site_tools import run_jacru_n1_5_approximation_error_headroom as n15a  # noqa: E402
from site_tools import run_jacru_n1_6_adjoint_low_rank as n16  # noqa: E402
from site_tools import run_jacru_n1_7_geometry_krylov_oracle as n17  # noqa: E402
from site_tools import run_jacru_n1_8_hybrid_design_screen as n18  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/jacru_n1_9_global_contrast_screen_postopen_v1.json"
)
DEFAULT_OUTPUT = ROOT / "demo_t16_operator/results/jacru_n1_9_global_contrast_scratch"
REPORT_SCHEMA = "jacru-n1-9-global-contrast-screen-report-1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _validate_config(config: Mapping[str, Any], *, seed_limit: int | None) -> None:
    if config.get("schema_version") != "jacru-n1-9-global-contrast-screen-1.0":
        raise ValueError("unexpected N1.9 design config schema")
    if config.get("status") != "POSTOPEN_DESIGN_SCREEN_NO_NEW_GEOMETRY":
        raise ValueError("N1.9 must remain post-open and nonconfirmatory")
    kinds = tuple(str(value) for value in config["candidate_basis_kinds"])
    if kinds != SUPPORTED_CONTRAST_BASIS_KINDS:
        raise ValueError("N1.9 candidate set or frozen ordering drifted")

    expected_paths = {
        "source_t0_config": "demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json",
        "source_n1_5_a_config": "demo_t16_operator/configs/jacru_n1_5_approximation_error_headroom_development_v1.json",
        "source_n1_7_frozen_result": "demo_t16_operator/results/jacru_n1_7_geometry_krylov_postopen_full1",
        "source_n1_8_config": "demo_t16_operator/configs/jacru_n1_8_hybrid_design_screen_postopen_v1.json",
        "source_n1_8_result": "demo_t16_operator/results/jacru_n1_8_hybrid_design_screen_postopen_audit_amended_full1",
        "design_freeze_document": "docs/jacru_n1_9_global_contrast_freeze_2026-07-18.md",
    }
    for key, expected in expected_paths.items():
        if config.get(key) != expected:
            raise ValueError(f"N1.9 frozen source path drifted: {key}")
    if config.get("evaluated_partitions") != ["development"]:
        raise ValueError("N1.9 may evaluate only opened development")

    integrity_files = {
        "source_t0_config_sha256": ROOT / expected_paths["source_t0_config"],
        "source_n1_5_a_config_sha256": ROOT
        / expected_paths["source_n1_5_a_config"],
        "source_n1_7_summary_sha256": ROOT
        / expected_paths["source_n1_7_frozen_result"]
        / "summary.json",
        "source_n1_7_manifest_sha256": ROOT
        / expected_paths["source_n1_7_frozen_result"]
        / "case_manifest.csv",
        "source_n1_8_config_sha256": ROOT / expected_paths["source_n1_8_config"],
        "source_n1_8_summary_sha256": ROOT
        / expected_paths["source_n1_8_result"]
        / "summary.json",
        "source_n1_8_manifest_sha256": ROOT
        / expected_paths["source_n1_8_result"]
        / "case_manifest.csv",
        "source_n1_8_model_sha256": ROOT
        / "demo_t16_operator/jacru_n1_8_camera_ray_hybrid.py",
        "source_n1_8_runner_sha256": ROOT
        / "site_tools/run_jacru_n1_8_hybrid_design_screen.py",
        "source_n1_7_model_sha256": ROOT
        / "demo_t16_operator/jacru_n1_7_krylov_correction.py",
        "source_n1_7_runner_sha256": ROOT
        / "site_tools/run_jacru_n1_7_geometry_krylov_oracle.py",
        "source_n1_6_runner_sha256": ROOT
        / "site_tools/run_jacru_n1_6_adjoint_low_rank.py",
        "source_n1_5_runner_sha256": ROOT
        / "site_tools/run_jacru_n1_5_approximation_error_headroom.py",
        "source_n1_5_teacher_sha256": ROOT
        / "demo_t16_operator/jacru_n1_5_high_order_correction.py",
        "source_streaming_operator_sha256": ROOT
        / "demo_t16_operator/psu_b0_streaming_operator.py",
        "source_synthetic_fixture_sha256": ROOT
        / "demo_t16_operator/jacru_synthetic_fixture.py",
    }
    integrity = config.get("source_integrity")
    if not isinstance(integrity, Mapping) or set(integrity) != set(integrity_files):
        raise ValueError("N1.9 source-integrity manifest drifted")
    for key, path in integrity_files.items():
        if str(integrity[key]) != n16._sha256(path):
            raise ValueError(f"N1.9 frozen source hash drifted: {key}")

    source_n18 = n16._read_json(ROOT / str(config["source_n1_8_config"]))
    for key in (
        "total_correction_trust_region",
        "budget",
        "oracle_screen",
        "design_selection_gate",
    ):
        if config.get(key) != source_n18.get(key):
            raise ValueError(f"N1.9 must inherit the frozen N1.8 {key}")

    cost = config.get("operational_cost_gate")
    if cost != {
        "reference_candidate_id": "component_damping",
        "metric": "paired_oracle_excluded_solver_path_seconds_ratio",
        "median_ratio_maximum": 1.25,
        "p90_ratio_maximum": 1.5,
        "included_timing": "shared_warm_plus_basis_setup_plus_refinement",
        "excluded_timing": "evaluator_oracle_coefficient_projection",
        "timing_device": "current_local_machine",
        "timing_is_portable_hardware_claim": False,
        "timing_is_deployable_method_end_to_end_claim": False,
    }:
        raise ValueError("N1.9 operational cost gate drifted")
    if config.get("numerical_applicability") != {
        "schur_gate": "NOT_APPLICABLE_NO_COVARIANCE_OR_MAJORIZER",
        "schur_violation_must_not_be_reported_as_zero": True,
    }:
        raise ValueError("N1.9 Schur applicability contract drifted")

    basis = config["basis"]
    expected_rows = np.asarray(
        (
            (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0),
            (1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)),
        )
    )
    actual_rows = np.asarray(basis["camera_contrast_rows"], dtype=np.float64)
    if basis.get("normal_operator") != "A_P_At" or basis.get(
        "camera_contrast"
    ) != "three_camera_helmert_unit_l2":
        raise ValueError("N1.9 normal operator or contrast family drifted")
    if actual_rows.shape != (2, 3) or not np.allclose(
        actual_rows, expected_rows, rtol=0.0, atol=1e-15
    ):
        raise ValueError("N1.9 frozen Helmert contrast values drifted")
    expected_orders = {
        "residual_contrast_vector_order": [
            "damping",
            "warm_residual",
            "camera_contrast_1_warm_residual",
            "camera_contrast_2_warm_residual",
            "normal_damping",
            "normal_warm_residual",
        ],
        "damping_contrast_vector_order": [
            "damping",
            "warm_residual",
            "camera_contrast_1_damping",
            "camera_contrast_2_damping",
            "normal_damping",
            "normal_warm_residual",
        ],
    }
    for key, expected in expected_orders.items():
        if basis.get(key) != expected:
            raise ValueError(f"N1.9 frozen vector order drifted: {key}")
    if basis.get("evaluated_case_truth_is_forbidden") is not True:
        raise ValueError("evaluated-case truth must be forbidden")
    if (
        float(basis["dependence_tolerance"]) != 1e-10
        or basis.get("orthonormalization")
        != "ordered_two_pass_modified_gram_schmidt"
        or float(basis["maximum_orthonormality_defect"]) != 1e-10
    ):
        raise ValueError("N1.9 frozen orthonormalization contract drifted")
    if int(basis["minimum_accepted_rank"]) != 6 or int(
        basis["maximum_accepted_rank"]
    ) != 6:
        raise ValueError("N1.9 candidates require exact rank six")

    budget = config["budget"]
    expected_total = (
        int(budget["deployable_total_low_forward_calls"]),
        int(budget["deployable_total_low_adjoint_calls"]),
    )
    expected_schedule = (2, 2, 10, 6)
    schedules = config.get("candidate_schedules")
    if not isinstance(schedules, Mapping) or set(schedules) != set(kinds):
        raise ValueError("N1.9 schedules or candidate set drifted")
    for kind in kinds:
        schedule = schedules[kind]
        actual_schedule = (
            int(schedule["basis_setup_forward_calls"]),
            int(schedule["basis_setup_adjoint_calls"]),
            int(schedule["refine_iterations"]),
            int(schedule["required_rank"]),
        )
        if actual_schedule != expected_schedule:
            raise ValueError(f"N1.9 frozen schedule drifted: {kind}")
        setup_f, setup_at, refine, _ = actual_schedule
        actual_total = (
            int(budget["warm_cgls_iterations"])
            + int(budget["warm_projection_forward_calls"])
            + setup_f
            + refine,
            int(budget["warm_cgls_iterations"]) + setup_at + refine,
        )
        if actual_total != expected_total or expected_total != (25, 24):
            raise ValueError(f"candidate does not match 25F/24AT: {kind}")

    selection = config["selection_rule"]
    if (
        selection.get("eligible_projection_oracle") != "measurement"
        or selection.get("eligible_representation_roles")
        != [
            "SOLVER_AWARE_REPRESENTATION_ELIGIBLE",
            "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE",
        ]
        or selection.get("if_no_candidate_passes")
        != "CLOSE_RANK6_CAMERA_GLOBAL_K_BRANCH"
    ):
        raise ValueError("N1.9 branch-closure rule drifted")
    claim = config["claim_boundary"]
    for key in (
        "opens_new_geometry",
        "may_change_n1_7_verdict",
        "may_change_n1_8_verdict",
        "may_claim_algorithm_gain",
        "may_claim_confirmed_gain",
        "may_train_a_learner",
        "is_real_bost_evidence",
        "opens_ood_fresh_or_final",
    ):
        if claim.get(key) is not False:
            raise ValueError(f"N1.9 claim boundary drifted: {key}")
    if claim.get("close_rank6_camera_global_k_branch_if_both_fail") is not True:
        raise ValueError("N1.9 must fail closed when both candidates fail")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")


def _assert_frozen_checkout(paths: list[Path]) -> None:
    """Require decisive runs to use committed, unmodified N1.9 sources."""

    relative_paths = []
    for path in paths:
        try:
            relative = str(path.resolve().relative_to(ROOT))
        except ValueError as error:
            raise RuntimeError("N1.9 protected source escaped the repository") from error
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"N1.9 decisive source is not committed: {relative}")
        relative_paths.append(relative)
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *relative_paths],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if clean.returncode != 0:
        raise RuntimeError("N1.9 decisive sources differ from the recorded Git commit")


def _role_claim_label(role: str) -> str:
    if role == "SOLVER_AWARE_REPRESENTATION_ELIGIBLE":
        return "NONNEGATIVE_SUPPORT_ADJOINT_SCREEN_ONLY"
    if role == "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE":
        return "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE"
    return "REPRESENTATION_NO_GO"


def _candidate_states(
    prepared: list[n16.PreparedCase], config: Mapping[str, Any]
) -> dict[str, list[n17.BasisState]]:
    output = {kind: [] for kind in config["candidate_basis_kinds"]}
    trust = config["total_correction_trust_region"]
    partitions = set(str(value) for value in config["evaluated_partitions"])
    for state in prepared:
        if state.record.partition not in partitions:
            continue
        view = n17.DeployableCaseView(
            measured_observation=state.measured_observation,
            signal_scale=state.signal_scale,
            warm_field=state.warm_field,
            warm_projection=state.warm_projection,
            damping_normalized=state.damping_normalized,
            shared_warm_seconds=state.shared_warm_seconds,
            operator=state.record.case.inference.operator,
        )
        warm_residual = (view.measured_observation - view.warm_projection) / view.signal_scale
        radius = visible_total_correction_radius(
            view.damping_normalized,
            warm_residual,
            damping_floor_multiplier=float(trust["damping_norm_floor_multiplier"]),
            warm_residual_multiplier=float(trust["warm_residual_norm_multiplier"]),
            damping_cap_multiplier=float(trust["damping_norm_cap_multiplier"]),
        )
        for kind in config["candidate_basis_kinds"]:
            schedule = config["candidate_schedules"][str(kind)]
            operator = view.operator
            forward, adjoint = n15a._operator_maps(operator)
            support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
            before = operator.call_report()
            started = time.perf_counter()
            basis = build_global_contrast_basis(
                kind=str(kind),
                damping=view.damping_normalized,
                warm_residual=warm_residual,
                forward=forward,
                adjoint=adjoint,
                support=support,
                geometry=state.record.case.inference.geometry,
                dependence_tolerance=float(config["basis"]["dependence_tolerance"]),
            )
            elapsed = time.perf_counter() - started
            delta = n17._call_delta(before, operator.call_report())
            expected_delta = {
                "forward_calls": int(schedule["basis_setup_forward_calls"]),
                "adjoint_calls": int(schedule["basis_setup_adjoint_calls"]),
            }
            if delta != expected_delta or (
                basis.setup_forward_calls,
                basis.setup_adjoint_calls,
            ) != (
                expected_delta["forward_calls"],
                expected_delta["adjoint_calls"],
            ):
                raise RuntimeError("N1.9 basis setup call ledger drifted")
            if basis.rank != int(schedule["required_rank"]):
                raise RuntimeError("N1.9 basis rank is not the frozen exact rank six")
            expected_names = config["basis"][
                "residual_contrast_vector_order"
                if kind == "residual_contrast_global_k6_total"
                else "damping_contrast_vector_order"
            ]
            if list(basis.names) != expected_names:
                raise RuntimeError("N1.9 basis vector order or independence drifted")
            output[str(kind)].append(
                n17.BasisState(
                    deployable=view,
                    evaluator=state.record,
                    basis=basis,
                    warm_residual_normalized=warm_residual,
                    coefficient_radius=radius,
                    basis_setup_seconds=elapsed,
                    setup_forward_calls=basis.setup_forward_calls,
                    setup_adjoint_calls=basis.setup_adjoint_calls,
                )
            )
    return output


def _operational_cost_gate(
    case_rows: list[dict[str, Any]],
    *,
    candidate_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    policy = config["operational_cost_gate"]
    candidate = n16._index_rows(case_rows, candidate_id)
    reference = n16._index_rows(case_rows, str(policy["reference_candidate_id"]))
    if not candidate or set(candidate) != set(reference):
        raise RuntimeError("N1.9 paired timing case sets drifted")
    ratios = []
    for key in sorted(candidate):
        candidate_seconds = float(candidate[key]["end_to_end_seconds"])
        reference_seconds = float(reference[key]["end_to_end_seconds"])
        if (
            not np.isfinite(candidate_seconds)
            or not np.isfinite(reference_seconds)
            or candidate_seconds <= 0.0
            or reference_seconds <= 0.0
        ):
            raise RuntimeError("N1.9 timing ledger contains a nonpositive value")
        ratios.append(candidate_seconds / reference_seconds)
    median_ratio = float(np.median(ratios))
    p90_ratio = float(np.quantile(ratios, 0.9))
    checks = {
        "median_oracle_excluded_solver_path_ratio": median_ratio
        <= float(policy["median_ratio_maximum"]),
        "p90_oracle_excluded_solver_path_ratio": p90_ratio
        <= float(policy["p90_ratio_maximum"]),
    }
    return {
        "reference_candidate_id": str(policy["reference_candidate_id"]),
        "paired_case_count": len(ratios),
        "median_oracle_excluded_solver_path_seconds_ratio": median_ratio,
        "p90_oracle_excluded_solver_path_seconds_ratio": p90_ratio,
        "checks": checks,
        "passed": all(checks.values()),
        "portable_hardware_claim": False,
        "deployable_method_end_to_end_claim": False,
        "excluded_timing": str(policy["excluded_timing"]),
    }


def _select(
    gates: list[dict[str, Any]],
    aggregates: Mapping[str, Mapping[str, Any]],
    *,
    decisive: bool,
    eligible_roles: tuple[str, ...],
) -> dict[str, Any]:
    if not decisive:
        return {
            "authorized": False,
            "selected_candidate_id": None,
            "rank6_camera_global_k_branch_closed": False,
            "status": "N1_9_SMOKE_NONDECISIVE",
        }
    eligible = [
        row
        for row in gates
        if row["reconstruction_passed"]
        and row["operational_cost_passed"]
        and row["role"] in eligible_roles
    ]
    if not eligible:
        return {
            "authorized": False,
            "selected_candidate_id": None,
            "rank6_camera_global_k_branch_closed": True,
            "status": "N1_9_RANK6_CAMERA_GLOBAL_K_BRANCH_CLOSED",
        }
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(aggregates[row["candidate_id"]]["mean_field_gain_over_low_cgls24"]),
            -float(row["extra_headroom_retention_over_component_damping"]),
            -float(row["physics_fidelity_adjoint_gain"]),
            str(row["candidate_id"]),
        ),
    )
    selected = ranked[0]
    return {
        "authorized": True,
        "selected_candidate_id": selected["candidate_id"],
        "selected_internal_role": selected["role"],
        "selected_claim_label": selected["role_claim_label"],
        "rank6_camera_global_k_branch_closed": False,
        "status": "N1_9_NEW_SPLIT_PREREGISTRATION_AUTHORIZED",
    }


def _plot(
    aggregates: list[dict[str, Any]], gates: list[dict[str, Any]], output: Path
) -> None:
    candidates = [
        row
        for row in aggregates
        if str(row["candidate_id"]).endswith("_measurement_oracle")
    ]
    gate_map = {str(row["candidate_id"]): row for row in gates}
    labels = [
        str(row["candidate_id"])
        .replace("_contrast_global_k6_total_measurement_oracle", "")
        .title()
        for row in candidates
    ]
    fields = [100.0 * float(row["mean_field_gain_over_low_cgls24"]) for row in candidates]
    extras = [
        100.0
        * float(
            gate_map[str(row["candidate_id"])][
                "extra_headroom_retention_over_component_damping"
            ]
        )
        for row in candidates
    ]
    adjoints = [
        100.0
        * float(row["evaluator_mean_adjoint_residual_gain_over_component_damping"])
        for row in candidates
    ]
    costs = [
        100.0
        * float(
            gate_map[str(row["candidate_id"])]["operational_cost"][
                "p90_oracle_excluded_solver_path_seconds_ratio"
            ]
        )
        for row in candidates
    ]
    colors = ["#28766f", "#9b6045"]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.5), constrained_layout=True)
    for axis, values, title, threshold in (
        (axes[0], fields, "Field gain vs CGLS-24", 5.0),
        (axes[1], extras, "Extra headroom after damping", 60.0),
        (axes[2], adjoints, "Support-adjoint gain vs damping", 50.0),
        (axes[3], costs, "P90 solver-path ratio (oracle excluded)", 150.0),
    ):
        axis.bar(np.arange(len(labels)), values, color=colors)
        axis.axhline(threshold, color="#8b3d32", linestyle="--", linewidth=1.4)
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_ylabel("percent")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("N1.9 frozen camera-contrast/global-K design screen", fontsize=14)
    fig.savefig(output / "diagnostic.png", dpi=180)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = n16._read_json(config_path)
    _validate_config(config, seed_limit=args.seed_limit)
    freeze_path = ROOT / config["design_freeze_document"]
    if not freeze_path.is_file():
        raise FileNotFoundError(f"missing N1.9 design freeze: {freeze_path}")
    if args.seed_limit is None:
        _assert_frozen_checkout(
            [
                config_path,
                freeze_path,
                Path(__file__),
                ROOT / "demo_t16_operator/jacru_n1_9_global_contrast.py",
                ROOT / "demo_t16_operator/test_jacru_n1_9_global_contrast.py",
                ROOT / "site_tools/test_run_jacru_n1_9_global_contrast_screen.py",
            ]
        )

    n15_path = ROOT / config["source_n1_5_a_config"]
    source_path = ROOT / config["source_t0_config"]
    n18_config_path = ROOT / config["source_n1_8_config"]
    n18_result_path = ROOT / config["source_n1_8_result"]
    n15_config = n16._read_json(n15_path)
    source = n16._read_json(source_path)
    prepared, manifest = n16._prepare_cases(
        config, n15_config, source, seed_limit=args.seed_limit
    )
    by_kind = _candidate_states(prepared, config)
    case_rows, diagnostic_rows, basis_rows = n18._evaluate(by_kind, config)
    aggregates = n17._aggregate_partition(
        case_rows, diagnostic_rows, partition="development"
    )
    aggregate_map = {str(row["candidate_id"]): row for row in aggregates}

    n17_result_path = ROOT / config["source_n1_7_frozen_result"]
    n17_summary = n16._read_json(n17_result_path / "summary.json")
    n17_manifest_path = n17_result_path / "case_manifest.csv"
    with n17_manifest_path.open(encoding="utf-8", newline="") as handle:
        n17_manifest = list(csv.DictReader(handle))
    n18_summary = n16._read_json(n18_result_path / "summary.json")
    n18_manifest_path = n18_result_path / "case_manifest.csv"
    with n18_manifest_path.open(encoding="utf-8", newline="") as handle:
        n18_manifest = list(csv.DictReader(handle))
    allow_subset = args.seed_limit is not None
    n18._assert_same_development_cases(
        manifest, n17_manifest, allow_current_subset=allow_subset
    )
    n18._assert_same_development_cases(
        manifest, n18_manifest, allow_current_subset=allow_subset
    )
    if n18_summary.get("status") != "NO_N1_8_CONFIRMATION_AUTHORIZATION":
        raise RuntimeError("N1.9 source N1.8 status drifted")

    frozen_field = float(
        n17_summary["primary_development_aggregate"][
            "mean_field_gain_over_low_cgls24"
        ]
    )
    gates = []
    for kind in config["candidate_basis_kinds"]:
        candidate_id = f"{kind}_measurement_oracle"
        gate = n18._gate(
            aggregate_map[candidate_id],
            extra_headroom_retention=n18._extra_headroom_retention(
                case_rows, candidate_id=candidate_id
            ),
            frozen_n17_field_gain=frozen_field,
            basis_rows=basis_rows,
            config=config,
        )
        gate["operational_cost"] = _operational_cost_gate(
            case_rows, candidate_id=candidate_id, config=config
        )
        gate["operational_cost_passed"] = gate["operational_cost"]["passed"]
        gate["role_claim_label"] = _role_claim_label(str(gate["role"]))
        gates.append(gate)
    selection = _select(
        gates,
        aggregate_map,
        decisive=args.seed_limit is None,
        eligible_roles=tuple(config["selection_rule"]["eligible_representation_roles"]),
    )
    summary = {
        "schema": REPORT_SCHEMA,
        "status": selection["status"],
        "evidence_level": config["evidence_level"],
        "runtime_seconds": time.perf_counter() - started,
        "seed_limit": args.seed_limit,
        "opened_geometry_cluster_count": len({row["base_seed"] for row in basis_rows}),
        "opened_case_count": len(
            {(row["base_seed"], row["family"]) for row in basis_rows}
        ),
        "candidate_basis_kinds": config["candidate_basis_kinds"],
        "radius_applies_to_entire_correction": True,
        "finite_k_truth_search_was_run": False,
        "learner_was_trained": False,
        "opens_new_geometry": False,
        "may_claim_algorithm_gain": False,
        "n1_7_development_case_identity_verified": True,
        "n1_8_development_case_identity_verified": True,
        "source_n1_8_no_auth_status_verified": True,
        "schur_gate_status": config["numerical_applicability"]["schur_gate"],
        "schur_violation_was_not_reported_as_zero": True,
        "operational_cost_scope": config["operational_cost_gate"]["metric"],
        "deployable_method_end_to_end_cost_was_not_claimed": True,
        "selection": selection,
        "candidate_gates": gates,
        "development_aggregates": aggregates,
        "claim_boundary": config["claim_boundary"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    n16._write_csv(output / "case_manifest.csv", manifest)
    n16._write_csv(output / "case_metrics.csv", case_rows)
    n16._write_csv(output / "target_diagnostics.csv", diagnostic_rows)
    n16._write_csv(output / "basis_diagnostics.csv", basis_rows)
    n16._write_csv(output / "aggregate_metrics.csv", aggregates)
    _plot(aggregates, gates, output)

    provenance = {
        "schema": "jacru-n1-9-global-contrast-provenance-1.0",
        "git_commit": n16._git_commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": n16._sha256(config_path),
        "design_freeze_path": str(freeze_path.relative_to(ROOT)),
        "design_freeze_sha256": n16._sha256(freeze_path),
        "source_n1_5_a_config_sha256": n16._sha256(n15_path),
        "source_t0_config_sha256": n16._sha256(source_path),
        "source_n1_7_summary_sha256": n16._sha256(n17_result_path / "summary.json"),
        "source_n1_7_manifest_sha256": n16._sha256(n17_manifest_path),
        "source_n1_8_config_path": str(n18_config_path.relative_to(ROOT)),
        "source_n1_8_config_sha256": n16._sha256(n18_config_path),
        "source_n1_8_summary_path": str((n18_result_path / "summary.json").relative_to(ROOT)),
        "source_n1_8_summary_sha256": n16._sha256(n18_result_path / "summary.json"),
        "source_n1_8_manifest_sha256": n16._sha256(n18_manifest_path),
        "runner_path": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": n16._sha256(Path(__file__)),
        "model_module_path": "demo_t16_operator/jacru_n1_9_global_contrast.py",
        "model_module_sha256": n16._sha256(
            ROOT / "demo_t16_operator/jacru_n1_9_global_contrast.py"
        ),
        "source_integrity_manifest_verified_before_case_preparation": True,
        "decisive_sources_matched_recorded_git_commit": args.seed_limit is None,
        "development_was_already_opened": True,
        "new_geometry_opened": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# N1.9 camera-contrast/global-K design screen\n\n"
        f"Status: **{selection['status']}**.\n\n"
        "This package reuses already-opened synthetic development only. It cannot "
        "establish algorithm gain, confirmation, OOD transfer, or real-BOST evidence.\n",
        encoding="utf-8",
    )
    n16._write_checksums(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
