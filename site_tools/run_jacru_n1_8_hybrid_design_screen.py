#!/usr/bin/env python3
"""Run the post-open N1.8 hybrid-frame design screen on frozen old data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.jacru_n1_5_high_order_correction import HighOrderTeacherMaps  # noqa: E402
from demo_t16_operator.jacru_n1_6_adjoint_low_rank import fit_measurement_pca  # noqa: E402
from demo_t16_operator.jacru_n1_7_krylov_correction import (  # noqa: E402
    adjoint_projection_oracle,
    measurement_projection_oracle,
)
from demo_t16_operator.jacru_n1_8_camera_ray_hybrid import (  # noqa: E402
    SUPPORTED_BASIS_KINDS,
    build_camera_ray_hybrid_basis,
    visible_total_correction_radius,
)
from demo_t16_operator.psu_b0_streaming_operator import (  # noqa: E402
    zero_outer_boundary_support,
)
from site_tools import run_jacru_n1_5_approximation_error_headroom as n15a  # noqa: E402
from site_tools import run_jacru_n1_6_adjoint_low_rank as n16  # noqa: E402
from site_tools import run_jacru_n1_7_geometry_krylov_oracle as n17  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/jacru_n1_8_hybrid_design_screen_postopen_v1.json"
)
DEFAULT_OUTPUT = ROOT / "demo_t16_operator/results/jacru_n1_8_hybrid_design_scratch"
REPORT_SCHEMA = "jacru-n1-8-hybrid-design-screen-report-1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _validate_config(config: Mapping[str, Any], *, seed_limit: int | None) -> None:
    if config.get("schema_version") != "jacru-n1-8-hybrid-design-screen-1.0":
        raise ValueError("unexpected N1.8 design config schema")
    if config.get("status") != "POSTOPEN_DESIGN_SCREEN_NO_NEW_GEOMETRY":
        raise ValueError("N1.8 design screen must remain post-open and nonconfirmatory")
    kinds = tuple(str(value) for value in config["candidate_basis_kinds"])
    if kinds != SUPPORTED_BASIS_KINDS:
        raise ValueError("candidate basis kinds or frozen ordering drifted")
    fit_policy = config["fit_mode_policy"]
    if (
        fit_policy.get("fit_partition") != "fit"
        or fit_policy.get("target") != "exact_mismatch_minus_component_damping"
        or int(fit_policy["centered_pca_rank"]) != 2
        or fit_policy.get("include_fit_mean_in_basis") is not False
    ):
        raise ValueError("fit-only centered PCA contract drifted")
    basis = config["basis"]
    if basis.get("normal_operator") != "A_P_At":
        raise ValueError("hybrid bases must use solver-consistent A P A^T")
    if basis.get("evaluated_case_truth_is_forbidden") is not True:
        raise ValueError("evaluated-case truth must be forbidden at basis construction")
    trust = config["total_correction_trust_region"]
    if trust.get("applies_to_entire_correction") is not True:
        raise ValueError("N1.8 radius must constrain the entire correction")
    if trust.get("uses_exact_target") is not False:
        raise ValueError("N1.8 radius must not read exact target")
    budget = config["budget"]
    warm = int(budget["warm_cgls_iterations"])
    projection = int(budget["warm_projection_forward_calls"])
    probe_f = int(budget["basis_probe_forward_calls"])
    probe_at = int(budget["basis_probe_adjoint_calls"])
    refine = int(budget["candidate_refine_iterations"])
    expected = (
        int(budget["deployable_total_low_forward_calls"]),
        int(budget["deployable_total_low_adjoint_calls"]),
    )
    if (warm + projection + probe_f + refine, warm + probe_at + refine) != expected:
        raise ValueError("hybrid candidate call budget drifted")
    if (probe_f, probe_at) != (2, 2):
        raise ValueError("N1.8 matched oracle controls require two sequential probes")
    expected_schedules = {
        "krylov4_total": (2, 2, 10, 4),
        "fit_pca2_krylov6_total": (2, 2, 10, 6),
        "camera_block6_total": (0, 0, 12, 6),
        "pose_fourier_krylov6_total": (2, 2, 10, 6),
        "detector_moment_krylov6_total": (2, 2, 10, 6),
    }
    schedules = config.get("candidate_schedules")
    if not isinstance(schedules, Mapping) or set(schedules) != set(kinds):
        raise ValueError("candidate schedules or candidate set drifted")
    for kind, expected_schedule in expected_schedules.items():
        schedule = schedules[kind]
        actual_schedule = (
            int(schedule["basis_setup_forward_calls"]),
            int(schedule["basis_setup_adjoint_calls"]),
            int(schedule["refine_iterations"]),
            int(schedule["required_rank"]),
        )
        if actual_schedule != expected_schedule:
            raise ValueError(f"frozen candidate schedule drifted: {kind}")
        setup_f, setup_at, candidate_refine, _ = actual_schedule
        if (
            warm + projection + setup_f + candidate_refine,
            warm + setup_at + candidate_refine,
        ) != expected:
            raise ValueError(f"candidate does not match the 25F/24AT budget: {kind}")
    gates = config["design_selection_gate"]
    if gates.get("extra_headroom_retention_definition") != (
        "sum(E_damping-E_candidate)/sum(E_damping-E_exact)"
    ) or float(gates["extra_headroom_retention_over_component_damping_minimum"]) != 0.6:
        raise ValueError("N1.8 extra-headroom definition or frozen threshold drifted")
    if (
        float(gates["solver_aware_role_adjoint_gain_minimum"]) != 0.0
        or float(gates["forward_correction_role_adjoint_gain_minimum"]) != 0.5
    ):
        raise ValueError("N1.8 role-classification thresholds drifted")
    claim = config["claim_boundary"]
    for key in (
        "opens_new_geometry",
        "may_change_n1_7_verdict",
        "may_claim_algorithm_gain",
        "may_claim_confirmed_gain",
        "may_train_a_learner",
        "is_real_bost_evidence",
        "opens_ood_fresh_or_final",
    ):
        if claim.get(key) is not False:
            raise ValueError(f"claim boundary drifted: {key}")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")


def _fit_centered_modes(states: list[n16.PreparedCase], config: Mapping[str, Any]) -> torch.Tensor:
    fit = [state for state in states if state.record.partition == "fit"]
    if len(fit) < 3:
        raise ValueError("fit-only PCA requires at least three fit cases")
    targets = torch.stack(
        [
            (state.record.mismatch_normalized - state.damping_normalized).reshape(-1)
            for state in fit
        ]
    )
    rank = int(config["fit_mode_policy"]["centered_pca_rank"])
    return fit_measurement_pca(targets, rank=rank).vectors


def _candidate_states(
    prepared: list[n16.PreparedCase],
    fit_modes: torch.Tensor,
    config: Mapping[str, Any],
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
            basis = build_camera_ray_hybrid_basis(
                kind=str(kind),
                damping=view.damping_normalized,
                warm_residual=warm_residual,
                forward=forward,
                adjoint=adjoint,
                support=support,
                geometry=state.record.case.inference.geometry,
                fit_modes=fit_modes if kind == "fit_pca2_krylov6_total" else None,
                dependence_tolerance=float(config["basis"]["dependence_tolerance"]),
            )
            elapsed = time.perf_counter() - started
            delta = n17._call_delta(before, operator.call_report())
            expected_delta = {
                "forward_calls": basis.setup_forward_calls,
                "adjoint_calls": basis.setup_adjoint_calls,
            }
            if delta != expected_delta:
                raise RuntimeError("hybrid basis setup call ledger drifted")
            if (
                basis.setup_forward_calls,
                basis.setup_adjoint_calls,
            ) != (
                int(schedule["basis_setup_forward_calls"]),
                int(schedule["basis_setup_adjoint_calls"]),
            ):
                raise RuntimeError("basis implementation disagrees with frozen schedule")
            if basis.rank != int(schedule["required_rank"]):
                raise RuntimeError("basis rank does not match the frozen candidate design")
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


def _basis_row(state: n17.BasisState, measurement: Any, adjoint: Any) -> dict[str, Any]:
    basis = state.basis
    record = state.evaluator
    return {
        "partition": record.partition,
        "base_seed": record.base_seed,
        "family": record.family,
        "case_id": record.case.inference.case_id,
        "geometry_digest": record.case.inference.geometry.digest,
        "basis_kind": basis.kind,
        "basis_names": "|".join(basis.names),
        "basis_rank": basis.rank,
        "dropped_names": "|".join(basis.dropped_names),
        "raw_norms_json": json.dumps(list(basis.raw_norms)),
        "orthonormality_defect": basis.orthonormality_defect,
        "fit_mode_count": basis.fit_mode_count,
        "total_correction_radius": state.coefficient_radius,
        "measurement_coefficient_norm": measurement.coefficient_norm,
        "measurement_coefficient_clipped": measurement.clipped,
        "measurement_target_residual_ratio": measurement.residual_ratio,
        "adjoint_coefficient_norm": adjoint.coefficient_norm,
        "adjoint_coefficient_clipped": adjoint.clipped,
        "adjoint_target_residual_ratio": adjoint.residual_ratio,
        "actual_setup_forward_calls": state.setup_forward_calls,
        "actual_setup_adjoint_calls": state.setup_adjoint_calls,
        "basis_uses_evaluated_case_truth": basis.uses_evaluated_case_truth,
        "radius_applies_to_entire_correction": True,
    }


def _evaluate(
    by_kind: Mapping[str, list[n17.BasisState]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kinds = [str(value) for value in config["candidate_basis_kinds"]]
    counts = {len(by_kind[kind]) for kind in kinds}
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        raise RuntimeError("candidate state counts drifted")
    case_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    budget = config["budget"]
    oracle = config["oracle_screen"]
    for index in range(next(iter(counts))):
        states = {kind: by_kind[kind][index] for kind in kinds}
        anchor = states[kinds[0]]
        case_rows.append(n17._low_reference_row(anchor, config))
        case_rows.append(
            n17._run_refinement(
                anchor,
                candidate_id="component_damping",
                correction_normalized=anchor.deployable.damping_normalized,
                config=config,
                refine_iterations=int(budget["component_damping_refine_iterations"]),
                setup_forward_calls=0,
                setup_adjoint_calls=0,
                setup_seconds=0.0,
                evaluator_only=False,
            )
        )
        for kind, state in states.items():
            schedule = config["candidate_schedules"][kind]
            target = state.evaluator.mismatch_normalized
            measurement = measurement_projection_oracle(
                state.basis, target, coefficient_radius=state.coefficient_radius
            )
            operator = state.deployable.operator
            _, adjoint_map = n15a._operator_maps(operator)
            support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
            before = operator.call_report()
            adjoint = adjoint_projection_oracle(
                state.basis,
                target,
                adjoint=adjoint_map,
                support=support,
                coefficient_radius=state.coefficient_radius,
                l2=float(oracle["adjoint_l2"]),
            )
            if n17._call_delta(before, operator.call_report()) != {
                "forward_calls": 0,
                "adjoint_calls": adjoint.evaluator_adjoint_calls,
            }:
                raise RuntimeError("hybrid adjoint oracle call ledger drifted")
            basis_rows.append(_basis_row(state, measurement, adjoint))
            for role, projection in (
                ("measurement", measurement),
                ("adjoint", adjoint),
            ):
                candidate_id = f"{kind}_{role}_oracle"
                correction = state.basis.synthesize(projection.coefficients)
                case_rows.append(
                    n17._run_refinement(
                        state,
                        candidate_id=candidate_id,
                        correction_normalized=correction,
                        config=config,
                        refine_iterations=int(schedule["refine_iterations"]),
                        setup_forward_calls=state.setup_forward_calls,
                        setup_adjoint_calls=state.setup_adjoint_calls,
                        setup_seconds=state.basis_setup_seconds,
                        evaluator_only=True,
                        metadata={
                            "coefficient_norm": projection.coefficient_norm,
                            "coefficient_radius": state.coefficient_radius,
                            "coefficient_clipped": projection.clipped,
                            "projection_target_residual_ratio": projection.residual_ratio,
                            "radius_applies_to_entire_correction": True,
                            "basis_rank": state.basis.rank,
                        },
                    )
                )
                diagnostic_rows.append(
                    n17._target_diagnostics(
                        state,
                        candidate_id=candidate_id,
                        correction_normalized=correction,
                    )
                )

        exact = anchor.evaluator.mismatch_normalized
        case_rows.append(
            n17._run_refinement(
                anchor,
                candidate_id="exact_mismatch_oracle",
                correction_normalized=exact,
                config=config,
                refine_iterations=int(budget["candidate_refine_iterations"]),
                setup_forward_calls=anchor.setup_forward_calls,
                setup_adjoint_calls=anchor.setup_adjoint_calls,
                setup_seconds=anchor.basis_setup_seconds,
                evaluator_only=True,
            )
        )
        diagnostic_rows.append(
            n17._target_diagnostics(
                anchor,
                candidate_id="exact_mismatch_oracle",
                correction_normalized=exact,
            )
        )
        teacher = HighOrderTeacherMaps(anchor.deployable.operator)
        started = time.perf_counter()
        high = teacher.correction(
            anchor.deployable.warm_field,
            low_projection=anchor.deployable.warm_projection,
        ) / anchor.deployable.signal_scale
        teacher_seconds = time.perf_counter() - started
        if teacher.call_report() != {"forward_calls": 1, "adjoint_calls": 0}:
            raise RuntimeError("high-order teacher control ledger drifted")
        beta = float(oracle["high_order_teacher_beta"])
        teacher_correction = anchor.deployable.damping_normalized + beta * (
            high - anchor.deployable.damping_normalized
        )
        case_rows.append(
            n17._run_refinement(
                anchor,
                candidate_id="high_order_teacher_b0p75",
                correction_normalized=teacher_correction,
                config=config,
                refine_iterations=int(budget["candidate_refine_iterations"]),
                setup_forward_calls=anchor.setup_forward_calls,
                setup_adjoint_calls=anchor.setup_adjoint_calls,
                setup_seconds=anchor.basis_setup_seconds,
                evaluator_only=True,
                high_order_forward_calls=1,
                metadata={"high_order_teacher_setup_seconds": teacher_seconds},
            )
        )
        diagnostic_rows.append(
            n17._target_diagnostics(
                anchor,
                candidate_id="high_order_teacher_b0p75",
                correction_normalized=teacher_correction,
            )
        )
    return case_rows, diagnostic_rows, basis_rows


def _gate(
    aggregate: Mapping[str, Any],
    *,
    extra_headroom_retention: float,
    frozen_n17_field_gain: float,
    basis_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["design_selection_gate"]
    extra_retention = float(extra_headroom_retention)
    family = aggregate["family_mean_field_gain_over_component_damping"]
    relevant_basis = [
        row
        for row in basis_rows
        if f"{row['basis_kind']}_measurement_oracle" == aggregate["candidate_id"]
    ]
    reconstruction_checks = {
        "mean_field_gain_over_low": float(aggregate["mean_field_gain_over_low_cgls24"])
        >= float(gates["mean_field_gain_over_low_cgls24_minimum"]),
        "mean_h1_gain_over_low": float(aggregate["mean_h1_gain_over_low_cgls24"])
        >= float(gates["mean_h1_gain_over_low_cgls24_minimum"]),
        "mean_field_gain_over_damping": float(
            aggregate["mean_field_gain_over_component_damping"]
        )
        >= float(gates["mean_field_gain_over_component_damping_minimum"]),
        "gain_over_frozen_n1_7": float(aggregate["mean_field_gain_over_low_cgls24"])
        - frozen_n17_field_gain
        >= float(gates["mean_field_gain_over_frozen_n1_7_minimum"]),
        "worst_case_gain_over_low": float(
            aggregate["worst_case_field_gain_over_low_cgls24"]
        )
        >= float(gates["worst_case_field_gain_over_low_cgls24_minimum"]),
        "worst_geometry_gain_over_low": float(
            aggregate["worst_geometry_field_gain_over_low_cgls24"]
        )
        >= float(gates["worst_geometry_field_gain_over_low_cgls24_minimum"]),
        "harm_rate_vs_low": float(aggregate["case_harm_over_one_percent_rate_vs_low"])
        <= float(gates["case_field_harm_over_one_percent_rate_vs_low_maximum"]),
        "harm_rate_vs_damping": float(
            aggregate["case_harm_over_one_percent_rate_vs_component_damping"]
        )
        <= float(
            gates["case_field_harm_over_one_percent_rate_vs_component_damping_maximum"]
        ),
        "each_family_gain_over_damping": all(float(value) > 0.0 for value in family.values()),
        "exact_gain_retention": float(aggregate["mean_exact_oracle_field_gain_retention"])
        >= float(gates["mean_exact_oracle_field_gain_retention_minimum"]),
        "extra_headroom_retention": extra_retention
        >= float(gates["extra_headroom_retention_over_component_damping_minimum"]),
        "low_forward_budget": int(aggregate["low_forward_calls"])
        == int(gates["required_low_forward_calls"]),
        "low_adjoint_budget": int(aggregate["low_adjoint_calls"])
        == int(gates["required_low_adjoint_calls"]),
        "zero_high_order_forward": int(aggregate["high_order_forward_calls"])
        == int(gates["required_high_order_forward_calls"]),
        "zero_high_order_adjoint": int(aggregate["high_order_adjoint_calls"])
        == int(gates["required_high_order_adjoint_calls"]),
        "basis_rank": min(int(row["basis_rank"]) for row in relevant_basis)
        >= int(config["basis"]["minimum_accepted_rank"]),
        "basis_orthonormality": max(
            float(row["orthonormality_defect"]) for row in relevant_basis
        )
        <= float(config["basis"]["maximum_orthonormality_defect"]),
    }
    adjoint_gain = float(
        aggregate["evaluator_mean_adjoint_residual_gain_over_component_damping"]
    )
    physics_fidelity = adjoint_gain >= float(
        gates["forward_correction_role_adjoint_gain_minimum"]
    )
    reconstruction_passed = all(reconstruction_checks.values())
    role = _representation_role(
        reconstruction_passed=reconstruction_passed,
        adjoint_gain=adjoint_gain,
        gates=gates,
    )
    return {
        "candidate_id": aggregate["candidate_id"],
        "reconstruction_checks": reconstruction_checks,
        "reconstruction_passed_count": sum(reconstruction_checks.values()),
        "reconstruction_check_count": len(reconstruction_checks),
        "reconstruction_passed": reconstruction_passed,
        "physics_fidelity_adjoint_gain": adjoint_gain,
        "physics_fidelity_passed": physics_fidelity,
        "solver_aware_fidelity_passed": adjoint_gain
        >= float(gates["solver_aware_role_adjoint_gain_minimum"]),
        "extra_headroom_retention_over_component_damping": extra_retention,
        "role": role,
    }


def _representation_role(
    *, reconstruction_passed: bool, adjoint_gain: float, gates: Mapping[str, Any]
) -> str:
    gain = float(adjoint_gain)
    if not np.isfinite(gain) or not reconstruction_passed:
        return "REPRESENTATION_NO_GO"
    if gain < float(gates["solver_aware_role_adjoint_gain_minimum"]):
        return "REPRESENTATION_NO_GO"
    if gain >= float(gates["forward_correction_role_adjoint_gain_minimum"]):
        return "FORWARD_CORRECTION_REPRESENTATION_ELIGIBLE"
    return "SOLVER_AWARE_REPRESENTATION_ELIGIBLE"


def _extra_headroom_retention(
    case_rows: list[dict[str, Any]], *, candidate_id: str
) -> float:
    """Aggregate incremental error reduction beyond component damping.

    This deliberately uses a ratio of summed error differences.  Averaging
    per-case ratios would let cases with nearly zero exact headroom dominate.
    """

    candidate = n16._index_rows(case_rows, candidate_id)
    damping = n16._index_rows(case_rows, "component_damping")
    exact = n16._index_rows(case_rows, "exact_mismatch_oracle")
    if not candidate or set(candidate) != set(damping) or set(candidate) != set(exact):
        raise RuntimeError("extra-headroom case sets drifted")
    numerator = sum(
        float(damping[key]["field_relative_l2"])
        - float(candidate[key]["field_relative_l2"])
        for key in candidate
    )
    denominator = sum(
        float(damping[key]["field_relative_l2"])
        - float(exact[key]["field_relative_l2"])
        for key in candidate
    )
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 1e-30:
        raise RuntimeError("exact oracle provides no positive aggregate extra headroom")
    return float(numerator / denominator)


def _select(gates: list[dict[str, Any]], aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in gates
        if row["reconstruction_passed"] and row["role"] != "REPRESENTATION_NO_GO"
    ]
    if not eligible:
        return {
            "authorized": False,
            "selected_candidate_id": None,
            "status": "NO_N1_8_CONFIRMATION_AUTHORIZATION",
        }
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(aggregates[row["candidate_id"]]["mean_field_gain_over_low_cgls24"]),
            -float(row["extra_headroom_retention_over_component_damping"]),
            -float(row["physics_fidelity_adjoint_gain"]),
            int(aggregates[row["candidate_id"]]["basis_rank"]),
            str(row["candidate_id"]),
        ),
    )
    selected = ranked[0]
    return {
        "authorized": True,
        "selected_candidate_id": selected["candidate_id"],
        "selected_role": selected["role"],
        "status": "N1_8_NEW_SPLIT_PREREGISTRATION_AUTHORIZED",
    }


def _case_signature(rows: list[Mapping[str, Any]], *, partition: str) -> set[tuple[str, ...]]:
    return {
        (
            str(row["partition"]),
            str(row["base_seed"]),
            str(row["family"]),
            str(row["case_id"]),
            str(row["geometry_digest"]),
        )
        for row in rows
        if str(row["partition"]) == partition
    }


def _assert_same_development_cases(
    current_manifest: list[Mapping[str, Any]],
    frozen_n17_manifest: list[Mapping[str, Any]],
    *,
    allow_current_subset: bool = False,
) -> None:
    current = _case_signature(current_manifest, partition="development")
    frozen = _case_signature(frozen_n17_manifest, partition="development")
    matches = current.issubset(frozen) if allow_current_subset else current == frozen
    if not current or not matches:
        raise RuntimeError("N1.7/N1.8 development case or geometry digest drifted")


def _plot(aggregates: list[dict[str, Any]], gates: list[dict[str, Any]], output: Path) -> None:
    candidates = [
        row
        for row in aggregates
        if str(row["candidate_id"]).endswith("_measurement_oracle")
    ]
    labels = [str(row["candidate_id"]).replace("_measurement_oracle", "") for row in candidates]
    fields = [100.0 * float(row["mean_field_gain_over_low_cgls24"]) for row in candidates]
    retentions = [100.0 * float(row["mean_exact_oracle_field_gain_retention"]) for row in candidates]
    adjoints = [
        100.0 * float(row["evaluator_mean_adjoint_residual_gain_over_component_damping"])
        for row in candidates
    ]
    colors = ["#2f7f75", "#537b9f", "#a45b45", "#8b6f3d", "#745b8f"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, values, title, threshold in (
        (axes[0], fields, "Field gain vs CGLS-24", 5.0),
        (axes[1], retentions, "Exact-headroom retention", 70.0),
        (axes[2], adjoints, "Support-adjoint gain vs damping", 50.0),
    ):
        axis.bar(np.arange(len(labels)), values, color=colors[: len(labels)])
        axis.axhline(threshold, color="#8b3d32", linestyle="--", linewidth=1.4)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
        axis.set_ylabel("percent")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("N1.8 opened-data hybrid representation design screen", fontsize=14)
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
    design_freeze_path = ROOT / config["design_freeze_document"]
    if not design_freeze_path.is_file():
        raise FileNotFoundError(f"missing N1.8 design freeze: {design_freeze_path}")
    n15_path = ROOT / config["source_n1_5_a_config"]
    source_path = ROOT / config["source_t0_config"]
    n15_config = n16._read_json(n15_path)
    source = n16._read_json(source_path)
    prepared, manifest = n16._prepare_cases(
        config, n15_config, source, seed_limit=args.seed_limit
    )
    fit_modes = _fit_centered_modes(prepared, config)
    by_kind = _candidate_states(prepared, fit_modes, config)
    case_rows, diagnostic_rows, basis_rows = _evaluate(by_kind, config)
    aggregates = n17._aggregate_partition(
        case_rows, diagnostic_rows, partition="development"
    )
    aggregate_map = {str(row["candidate_id"]): row for row in aggregates}
    n17_summary = n16._read_json(
        ROOT / config["source_n1_7_frozen_result"] / "summary.json"
    )
    n17_manifest_path = ROOT / config["source_n1_7_frozen_result"] / "case_manifest.csv"
    with n17_manifest_path.open(encoding="utf-8", newline="") as handle:
        n17_manifest = list(csv.DictReader(handle))
    _assert_same_development_cases(
        manifest,
        n17_manifest,
        allow_current_subset=args.seed_limit is not None,
    )
    frozen_field = float(
        n17_summary["primary_development_aggregate"]["mean_field_gain_over_low_cgls24"]
    )
    gates = []
    for kind in config["candidate_basis_kinds"]:
        candidate_id = f"{kind}_measurement_oracle"
        extra_retention = _extra_headroom_retention(
            case_rows, candidate_id=candidate_id
        )
        gates.append(
            _gate(
                aggregate_map[candidate_id],
                extra_headroom_retention=extra_retention,
                frozen_n17_field_gain=frozen_field,
                basis_rows=basis_rows,
                config=config,
            )
        )
    selection = _select(gates, aggregate_map)
    summary = {
        "schema": REPORT_SCHEMA,
        "status": selection["status"],
        "evidence_level": config["evidence_level"],
        "runtime_seconds": time.perf_counter() - started,
        "seed_limit": args.seed_limit,
        "opened_geometry_cluster_count": len(
            {row["base_seed"] for row in basis_rows}
        ),
        "opened_case_count": len(
            {(row["base_seed"], row["family"]) for row in basis_rows}
        ),
        "candidate_basis_kinds": config["candidate_basis_kinds"],
        "fit_mode_rank": int(fit_modes.shape[0]),
        "fit_mode_mean_is_excluded": True,
        "radius_applies_to_entire_correction": True,
        "finite_k_truth_search_was_run": False,
        "learner_was_trained": False,
        "opens_new_geometry": False,
        "may_claim_algorithm_gain": False,
        "n1_7_development_case_identity_verified": True,
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
        "schema": "jacru-n1-8-hybrid-design-provenance-1.0",
        "git_commit": n16._git_commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": n16._sha256(config_path),
        "design_freeze_path": str(design_freeze_path.relative_to(ROOT)),
        "design_freeze_sha256": n16._sha256(design_freeze_path),
        "source_n1_5_a_config_sha256": n16._sha256(n15_path),
        "source_t0_config_sha256": n16._sha256(source_path),
        "source_n1_7_summary_sha256": n16._sha256(
            ROOT / config["source_n1_7_frozen_result"] / "summary.json"
        ),
        "source_n1_7_manifest_path": str(n17_manifest_path.relative_to(ROOT)),
        "source_n1_7_manifest_sha256": n16._sha256(n17_manifest_path),
        "runner_path": str(Path(__file__).relative_to(ROOT)),
        "runner_sha256": n16._sha256(Path(__file__)),
        "model_module_path": "demo_t16_operator/jacru_n1_8_camera_ray_hybrid.py",
        "model_module_sha256": n16._sha256(
            ROOT / "demo_t16_operator/jacru_n1_8_camera_ray_hybrid.py"
        ),
        "development_was_already_opened": True,
        "new_geometry_opened": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# N1.8 hybrid representation design screen\n\n"
        f"Status: **{selection['status']}**.\n\n"
        "This package reuses already-opened synthetic development only. It may "
        "select one hypothesis for preregistration on new geometry, but it cannot "
        "establish algorithm gain, confirmation, OOD transfer, or real-BOST evidence.\n",
        encoding="utf-8",
    )
    n16._write_checksums(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
