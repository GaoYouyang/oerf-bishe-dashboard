#!/usr/bin/env python3
"""Run the N1.7 post-open geometry-Krylov representation oracle gate."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.interface_baselines import cgls_baseline  # noqa: E402
from demo_t16_operator.jacru_n1_5_high_order_correction import (  # noqa: E402
    HighOrderTeacherMaps,
    warm_start_cgls,
)
from demo_t16_operator.jacru_n1_7_krylov_correction import (  # noqa: E402
    GeometryKrylovBasis,
    ProjectionOracle,
    adjoint_projection_oracle,
    build_geometry_krylov_basis,
    measurement_projection_oracle,
    project_to_l2_ball,
)
from demo_t16_operator.psu_b0_streaming_operator import (  # noqa: E402
    zero_outer_boundary_support,
)
from site_tools import run_jacru_n1_5_approximation_error_headroom as n15a  # noqa: E402
from site_tools import run_jacru_n1_5_reconstruction_aware_postopen as n15b  # noqa: E402
from site_tools import run_jacru_n1_6_adjoint_low_rank as n16  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/jacru_n1_7_geometry_krylov_postopen_v1.json"
)
DEFAULT_OUTPUT = ROOT / "demo_t16_operator/results/jacru_n1_7_geometry_krylov_scratch"
REPORT_SCHEMA = "jacru-n1-7-geometry-krylov-report-1.0"


@dataclass(frozen=True)
class DeployableCaseView:
    """Truth-free state allowed to enter basis construction and refinement."""

    measured_observation: torch.Tensor
    signal_scale: float
    warm_field: torch.Tensor
    warm_projection: torch.Tensor
    damping_normalized: torch.Tensor
    shared_warm_seconds: float
    operator: Any


@dataclass(frozen=True)
class BasisState:
    deployable: DeployableCaseView
    evaluator: n15a.CaseRecord
    basis: GeometryKrylovBasis
    warm_residual_normalized: torch.Tensor
    coefficient_radius: float
    basis_setup_seconds: float
    setup_forward_calls: int
    setup_adjoint_calls: int

    @property
    def key(self) -> tuple[str, int, str]:
        return self.evaluator.partition, self.evaluator.base_seed, self.evaluator.family


@dataclass(frozen=True)
class FieldOracleResult:
    coefficients: torch.Tensor
    field_relative_l2: float
    requested_evaluations: int
    unique_evaluations: int
    evaluator_forward_calls: int
    evaluator_adjoint_calls: int
    optimizer_runs: int
    successful_optimizer_runs: int
    coefficient_norm: float
    runtime_seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _call_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        "forward_calls": int(after["forward_calls"]) - int(before["forward_calls"]),
        "adjoint_calls": int(after["adjoint_calls"]) - int(before["adjoint_calls"]),
    }


def _validate_config(
    config: Mapping[str, Any],
    n15a_config: Mapping[str, Any],
    *,
    seed_limit: int | None,
) -> None:
    if config.get("schema_version") != "jacru-n1-7-geometry-krylov-postopen-1.0":
        raise ValueError("unexpected N1.7 config schema")
    if config.get("status") != "POSTOPEN_REPRESENTATION_ORACLE_ONLY":
        raise ValueError("N1.7 may only run as a post-open representation oracle")
    if config.get("may_construct_or_evaluate_ood") is not False:
        raise ValueError("N1.7 must not construct or evaluate OOD")
    contract = config["deployment_contract"]
    if contract.get("learner_is_present") is not False:
        raise ValueError("the N1.7 representation gate must not contain a learner")
    if int(contract["high_order_forward_calls"]) != 0 or int(
        contract["high_order_adjoint_calls"]
    ) != 0:
        raise ValueError("deployable N1.7 must use zero high-order calls")
    if contract.get("independent_learned_adjoint") is not False:
        raise ValueError("N1.7 must use the current low-order adjoint")
    required_forbidden = {
        "truth_volume",
        "fresh_exact_mismatch",
        "phantom_family_label",
        "confirmation_target",
        "high_order_forward_output",
    }
    if set(contract["forbidden_inputs"]) != required_forbidden:
        raise ValueError("N1.7 forbidden-input contract drifted")
    if contract["basis_order"] != [
        "damping",
        "warm_residual",
        "A_P_At_damping",
        "A_P_At_warm_residual",
    ]:
        raise ValueError("N1.7 basis order or support projection drifted")

    budget = config["budget"]
    warm = int(budget["warm_cgls_iterations"])
    projection = int(budget["warm_projection_forward_calls"])
    probe_f = int(budget["krylov_probe_forward_calls"])
    probe_at = int(budget["krylov_probe_adjoint_calls"])
    refine = int(budget["candidate_refine_iterations"])
    expected = (
        int(budget["deployable_total_low_forward_calls"]),
        int(budget["deployable_total_low_adjoint_calls"]),
    )
    if (warm + projection + probe_f + refine, warm + probe_at + refine) != expected:
        raise ValueError("candidate physical-call budget is not matched")
    damping_refine = int(budget["component_damping_refine_iterations"])
    if (warm + projection + damping_refine, warm + damping_refine) != expected:
        raise ValueError("component-damping physical-call budget is not matched")
    low_iterations = int(budget["low_reference_cgls_iterations"])
    low_projection = int(budget["low_reference_final_projection_forward_calls"])
    if (low_iterations + low_projection, low_iterations) != expected:
        raise ValueError("low-reference physical-call budget is not matched")
    if (probe_f, probe_at) != (2, 2):
        raise ValueError("N1.7 requires two sequential low-order operator probes")

    basis = config["basis"]
    if basis.get("normal_operator") != "A_P_At":
        raise ValueError("basis must use the solver-consistent A P A^T operator")
    trust = config["coefficient_trust_region"]
    if trust.get("uses_exact_target") is not False:
        raise ValueError("coefficient radius must not access the exact target")
    if any(
        float(trust[key]) <= 0.0
        for key in (
            "damping_norm_cap_multiplier",
            "warm_residual_norm_multiplier",
            "damping_norm_floor_multiplier",
        )
    ):
        raise ValueError("coefficient trust-region multipliers must be positive")
    if n15a_config.get("status") != "DEVELOPMENT_ONLY_OPENED_NOT_CONFIRMATORY":
        raise ValueError("source N1.5-A contract drifted")
    claim = config["claim_boundary"]
    for key in (
        "may_claim_learned_algorithm_gain",
        "may_claim_confirmed_algorithm_gain",
        "is_real_bost_evidence",
        "opens_ood_fresh_or_final",
    ):
        if claim.get(key) is not False:
            raise ValueError(f"claim boundary drifted: {key}")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")


def _coefficient_radius(
    damping: torch.Tensor,
    residual: torch.Tensor,
    config: Mapping[str, Any],
) -> float:
    trust = config["coefficient_trust_region"]
    damping_norm = float(torch.linalg.vector_norm(damping))
    residual_norm = float(torch.linalg.vector_norm(residual))
    cap = float(trust["damping_norm_cap_multiplier"]) * damping_norm
    expanded = float(trust["warm_residual_norm_multiplier"]) * residual_norm
    floor = float(trust["damping_norm_floor_multiplier"]) * damping_norm
    radius = min(cap, max(expanded, floor))
    if not np.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("visible coefficient radius is not positive and finite")
    return radius


def _prepare_basis_states(
    config: Mapping[str, Any],
    n15a_config: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    seed_limit: int | None,
) -> tuple[list[BasisState], list[dict[str, Any]]]:
    prepared, manifest = n16._prepare_cases(
        config, n15a_config, source, seed_limit=seed_limit
    )
    states: list[BasisState] = []
    for state in prepared:
        view = DeployableCaseView(
            measured_observation=state.measured_observation,
            signal_scale=state.signal_scale,
            warm_field=state.warm_field,
            warm_projection=state.warm_projection,
            damping_normalized=state.damping_normalized,
            shared_warm_seconds=state.shared_warm_seconds,
            operator=state.record.case.inference.operator,
        )
        operator = view.operator
        forward, adjoint = n15a._operator_maps(operator)
        support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
        warm_residual = (
            view.measured_observation - view.warm_projection
        ) / view.signal_scale
        before = operator.call_report()
        started = time.perf_counter()
        basis = build_geometry_krylov_basis(
            damping=view.damping_normalized,
            warm_residual=warm_residual,
            forward=forward,
            adjoint=adjoint,
            support=support,
            dependence_tolerance=float(config["basis"]["dependence_tolerance"]),
        )
        setup_seconds = time.perf_counter() - started
        delta = _call_delta(before, operator.call_report())
        expected = {
            "forward_calls": int(config["budget"]["krylov_probe_forward_calls"]),
            "adjoint_calls": int(config["budget"]["krylov_probe_adjoint_calls"]),
        }
        if delta != expected or delta != {
            "forward_calls": basis.setup_forward_calls,
            "adjoint_calls": basis.setup_adjoint_calls,
        }:
            raise RuntimeError("actual Krylov setup call ledger drifted")
        states.append(
            BasisState(
                deployable=view,
                evaluator=state.record,
                basis=basis,
                warm_residual_normalized=warm_residual,
                coefficient_radius=_coefficient_radius(
                    view.damping_normalized, warm_residual, config
                ),
                basis_setup_seconds=setup_seconds,
                setup_forward_calls=delta["forward_calls"],
                setup_adjoint_calls=delta["adjoint_calls"],
            )
        )
    for row in manifest:
        row["n1_7_basis_uses_truth"] = False
        row["n1_7_basis_uses_high_order_output"] = False
        row["n1_7_normal_operator"] = "A_P_At"
        row["n1_7_learner_present"] = False
    return states, manifest


def _field_metrics(field: torch.Tensor, state: BasisState) -> dict[str, float]:
    return n15b._field_metrics(field, state.evaluator)


def _run_refinement(
    state: BasisState,
    *,
    candidate_id: str,
    correction_normalized: torch.Tensor,
    config: Mapping[str, Any],
    refine_iterations: int,
    setup_forward_calls: int,
    setup_adjoint_calls: int,
    setup_seconds: float,
    evaluator_only: bool,
    high_order_forward_calls: int = 0,
    high_order_adjoint_calls: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    view = state.deployable
    record = state.evaluator
    operator = view.operator
    forward, adjoint = n15a._operator_maps(operator)
    support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
    correction = torch.as_tensor(correction_normalized, dtype=torch.float64)
    if correction.shape != view.measured_observation.shape:
        raise ValueError("correction shape drifted from the observation")
    before = operator.call_report()
    started = time.perf_counter()
    result = warm_start_cgls(
        view.measured_observation - correction * view.signal_scale,
        forward=forward,
        adjoint=adjoint,
        support=support,
        initial_field=view.warm_field,
        initial_projection=view.warm_projection,
        iterations=refine_iterations,
    )
    refine_seconds = time.perf_counter() - started
    delta = _call_delta(before, operator.call_report())
    if delta != {
        "forward_calls": refine_iterations,
        "adjoint_calls": refine_iterations,
    } or (result.forward_calls, result.adjoint_calls) != (
        refine_iterations,
        refine_iterations,
    ):
        raise RuntimeError("refinement operator ledger drifted")
    budget = config["budget"]
    warm_iterations = int(budget["warm_cgls_iterations"])
    warm_projection_calls = int(budget["warm_projection_forward_calls"])
    total_forward = (
        warm_iterations
        + warm_projection_calls
        + int(setup_forward_calls)
        + refine_iterations
    )
    total_adjoint = warm_iterations + int(setup_adjoint_calls) + refine_iterations
    expected = (
        int(budget["deployable_total_low_forward_calls"]),
        int(budget["deployable_total_low_adjoint_calls"]),
    )
    if (total_forward, total_adjoint) != expected:
        raise RuntimeError("summed deployment-stage call ledger is not matched")
    metrics = _field_metrics(result.field, state)
    row: dict[str, Any] = {
        "partition": record.partition,
        "base_seed": record.base_seed,
        "family": record.family,
        "case_id": record.case.inference.case_id,
        "geometry_digest": record.case.inference.geometry.digest,
        "candidate_id": candidate_id,
        "field_relative_l2": metrics["field_relative_l2"],
        "h1_seminorm_relative_l2": metrics["h1_seminorm_relative_error"],
        "field_mean_bias": metrics["field_mean_bias"],
        "data_residual_relative_l2": float(torch.linalg.vector_norm(result.residual))
        / max(float(torch.linalg.vector_norm(view.measured_observation)), 1e-30),
        "fallback": False,
        "fallback_reason": None,
        "takeover": True,
        "evaluator_only": evaluator_only,
        "basis_rank": state.basis.rank if setup_forward_calls else None,
        "warm_forward_calls": warm_iterations,
        "warm_adjoint_calls": warm_iterations,
        "warm_projection_forward_calls": warm_projection_calls,
        "setup_forward_calls": int(setup_forward_calls),
        "setup_adjoint_calls": int(setup_adjoint_calls),
        "refine_forward_calls": delta["forward_calls"],
        "refine_adjoint_calls": delta["adjoint_calls"],
        "low_forward_calls": total_forward,
        "low_adjoint_calls": total_adjoint,
        "high_order_forward_calls": int(high_order_forward_calls),
        "high_order_adjoint_calls": int(high_order_adjoint_calls),
        "shared_warm_seconds": view.shared_warm_seconds,
        "deployment_setup_seconds": setup_seconds,
        "candidate_refine_seconds": refine_seconds,
        "end_to_end_seconds": view.shared_warm_seconds + setup_seconds + refine_seconds,
    }
    if metadata:
        row.update(metadata)
    return row


def _low_reference_row(state: BasisState, config: Mapping[str, Any]) -> dict[str, Any]:
    view = state.deployable
    record = state.evaluator
    operator = view.operator
    forward, adjoint = n15a._operator_maps(operator)
    support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
    iterations = int(config["budget"]["low_reference_cgls_iterations"])
    operator.reset_call_counts()
    started = time.perf_counter()
    result = cgls_baseline(
        view.measured_observation,
        forward=forward,
        adjoint=adjoint,
        support=support,
        spacing_xyz=operator.spacing_xyz,
        iterations=iterations,
    )
    projection = operator(result.field[None, None])[0]
    elapsed = time.perf_counter() - started
    actual = operator.call_report()
    expected = {
        "forward_calls": int(config["budget"]["deployable_total_low_forward_calls"]),
        "adjoint_calls": int(config["budget"]["deployable_total_low_adjoint_calls"]),
    }
    if actual != expected:
        raise RuntimeError("matched low-reference call ledger drifted")
    metrics = _field_metrics(result.field, state)
    return {
        "partition": record.partition,
        "base_seed": record.base_seed,
        "family": record.family,
        "case_id": record.case.inference.case_id,
        "geometry_digest": record.case.inference.geometry.digest,
        "candidate_id": "low_cgls24_matched",
        "field_relative_l2": metrics["field_relative_l2"],
        "h1_seminorm_relative_l2": metrics["h1_seminorm_relative_error"],
        "field_mean_bias": metrics["field_mean_bias"],
        "data_residual_relative_l2": float(
            torch.linalg.vector_norm(view.measured_observation - projection)
        )
        / max(float(torch.linalg.vector_norm(view.measured_observation)), 1e-30),
        "fallback": False,
        "fallback_reason": None,
        "takeover": True,
        "evaluator_only": False,
        "basis_rank": None,
        "warm_forward_calls": 0,
        "warm_adjoint_calls": 0,
        "warm_projection_forward_calls": 0,
        "setup_forward_calls": 0,
        "setup_adjoint_calls": 0,
        "refine_forward_calls": actual["forward_calls"],
        "refine_adjoint_calls": actual["adjoint_calls"],
        "low_forward_calls": actual["forward_calls"],
        "low_adjoint_calls": actual["adjoint_calls"],
        "high_order_forward_calls": 0,
        "high_order_adjoint_calls": 0,
        "shared_warm_seconds": 0.0,
        "deployment_setup_seconds": 0.0,
        "candidate_refine_seconds": elapsed,
        "end_to_end_seconds": elapsed,
    }


def _correction_from_projection(
    state: BasisState, oracle: ProjectionOracle
) -> torch.Tensor:
    return state.deployable.damping_normalized + state.basis.synthesize(
        oracle.coefficients
    )


def _target_diagnostics(
    state: BasisState,
    *,
    candidate_id: str,
    correction_normalized: torch.Tensor,
) -> dict[str, Any]:
    view = state.deployable
    record = state.evaluator
    exact = record.mismatch_normalized
    damping = view.damping_normalized
    denominator = max(float(torch.linalg.vector_norm(exact - damping)), 1e-30)
    measurement_ratio = float(torch.linalg.vector_norm(exact - correction_normalized)) / denominator
    operator = view.operator
    _, adjoint = n15a._operator_maps(operator)
    support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
    before = operator.call_report()
    candidate_adjoint = adjoint(exact - correction_normalized) * support
    damping_adjoint = adjoint(exact - damping) * support
    delta = _call_delta(before, operator.call_report())
    if delta != {"forward_calls": 0, "adjoint_calls": 2}:
        raise RuntimeError("target diagnostic call ledger drifted")
    adjoint_ratio = float(torch.linalg.vector_norm(candidate_adjoint)) / max(
        float(torch.linalg.vector_norm(damping_adjoint)), 1e-30
    )
    return {
        "partition": record.partition,
        "base_seed": record.base_seed,
        "family": record.family,
        "case_id": record.case.inference.case_id,
        "geometry_digest": record.case.inference.geometry.digest,
        "candidate_id": candidate_id,
        "measurement_residual_ratio_to_component_damping": measurement_ratio,
        "measurement_residual_gain_over_component_damping": 1.0 - measurement_ratio,
        "adjoint_residual_ratio_to_component_damping": adjoint_ratio,
        "adjoint_residual_gain_over_component_damping": 1.0 - adjoint_ratio,
        "fresh_exact_mismatch_access": True,
        "evaluator_only": True,
        "evaluator_forward_calls": 0,
        "evaluator_adjoint_calls": 2,
    }


def _finite_k_field_oracle(
    state: BasisState,
    *,
    starts: list[torch.Tensor],
    config: Mapping[str, Any],
) -> FieldOracleResult:
    view = state.deployable
    operator = view.operator
    forward, adjoint = n15a._operator_maps(operator)
    support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
    refine = int(config["budget"]["candidate_refine_iterations"])
    oracle = config["oracle_screen"]
    radius = state.coefficient_radius
    cache: dict[bytes, tuple[float, torch.Tensor]] = {}
    requested = 0

    def objective(raw: np.ndarray) -> float:
        nonlocal requested
        requested += 1
        coefficients = project_to_l2_ball(
            torch.as_tensor(raw, dtype=torch.float64), radius=radius
        )
        key = coefficients.detach().cpu().numpy().tobytes()
        if key in cache:
            return cache[key][0]
        correction = view.damping_normalized + state.basis.synthesize(coefficients)
        result = warm_start_cgls(
            view.measured_observation - correction * view.signal_scale,
            forward=forward,
            adjoint=adjoint,
            support=support,
            initial_field=view.warm_field,
            initial_projection=view.warm_projection,
            iterations=refine,
        )
        score = _field_metrics(result.field, state)["field_relative_l2"]
        cache[key] = (score, coefficients.clone())
        return score

    before = operator.call_report()
    started = time.perf_counter()
    successful = 0
    bounds = [(-radius, radius)] * state.basis.rank
    for start in starts:
        initial = project_to_l2_ball(start, radius=radius).detach().cpu().numpy()
        objective(initial)
        result = minimize(
            objective,
            initial,
            method="Powell",
            bounds=bounds,
            options={
                "maxfev": int(oracle["finite_k_max_function_evaluations_per_start"]),
                "xtol": float(oracle["finite_k_xtol"]),
                "ftol": float(oracle["finite_k_ftol"]),
                "disp": False,
            },
        )
        objective(np.asarray(result.x, dtype=np.float64))
        successful += int(bool(result.success))
    elapsed = time.perf_counter() - started
    delta = _call_delta(before, operator.call_report())
    unique = len(cache)
    expected = {"forward_calls": unique * refine, "adjoint_calls": unique * refine}
    if delta != expected:
        raise RuntimeError("finite-K evaluator search call ledger drifted")
    best_score, best_coefficients = min(
        cache.values(), key=lambda item: (item[0], tuple(float(value) for value in item[1]))
    )
    return FieldOracleResult(
        coefficients=best_coefficients,
        field_relative_l2=float(best_score),
        requested_evaluations=requested,
        unique_evaluations=unique,
        evaluator_forward_calls=delta["forward_calls"],
        evaluator_adjoint_calls=delta["adjoint_calls"],
        optimizer_runs=len(starts),
        successful_optimizer_runs=successful,
        coefficient_norm=float(torch.linalg.vector_norm(best_coefficients)),
        runtime_seconds=elapsed,
    )


def _basis_row(
    state: BasisState,
    measurement: ProjectionOracle,
    adjoint: ProjectionOracle,
) -> dict[str, Any]:
    record = state.evaluator
    return {
        "partition": record.partition,
        "base_seed": record.base_seed,
        "family": record.family,
        "case_id": record.case.inference.case_id,
        "geometry_digest": record.case.inference.geometry.digest,
        "basis_names": "|".join(state.basis.names),
        "basis_rank": state.basis.rank,
        "dropped_names": "|".join(state.basis.dropped_names),
        "raw_norm_damping": state.basis.raw_norms[0],
        "raw_norm_warm_residual": state.basis.raw_norms[1],
        "raw_norm_normal_damping": state.basis.raw_norms[2],
        "raw_norm_normal_warm_residual": state.basis.raw_norms[3],
        "orthonormality_defect": state.basis.orthonormality_defect,
        "coefficient_radius": state.coefficient_radius,
        "measurement_coefficient_norm": measurement.coefficient_norm,
        "measurement_coefficient_clipped": measurement.clipped,
        "measurement_target_residual_ratio": measurement.residual_ratio,
        "adjoint_coefficient_norm": adjoint.coefficient_norm,
        "adjoint_coefficient_clipped": adjoint.clipped,
        "adjoint_target_residual_ratio": adjoint.residual_ratio,
        "actual_setup_forward_calls": state.setup_forward_calls,
        "actual_setup_adjoint_calls": state.setup_adjoint_calls,
        "basis_uses_truth": False,
    }


def _evaluate_cases(
    states: list[BasisState], config: Mapping[str, Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    case_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    partitions = set(config["evaluated_partitions"])
    budget = config["budget"]
    oracle_config = config["oracle_screen"]
    frozen_model = n16._read_json(
        ROOT / config["source_n1_6_result"] / "selected_model.json"
    )
    global_mean = torch.as_tensor(frozen_model["basis_mean"], dtype=torch.float64)
    global_vectors = torch.as_tensor(
        frozen_model["basis_vectors"], dtype=torch.float64
    )
    if int(frozen_model["basis_rank"]) != 4 or global_vectors.shape[0] != 4:
        raise RuntimeError("frozen N1.6 global PCA rank-4 control drifted")
    for state in states:
        view = state.deployable
        record = state.evaluator
        if record.partition not in partitions:
            continue
        case_rows.append(_low_reference_row(state, config))
        component_row = _run_refinement(
            state,
            candidate_id="component_damping",
            correction_normalized=view.damping_normalized,
            config=config,
            refine_iterations=int(budget["component_damping_refine_iterations"]),
            setup_forward_calls=0,
            setup_adjoint_calls=0,
            setup_seconds=0.0,
            evaluator_only=False,
        )
        case_rows.append(component_row)
        anchor_row = _run_refinement(
            state,
            candidate_id="krylov_schedule_damping_anchor",
            correction_normalized=view.damping_normalized,
            config=config,
            refine_iterations=int(budget["candidate_refine_iterations"]),
            setup_forward_calls=state.setup_forward_calls,
            setup_adjoint_calls=state.setup_adjoint_calls,
            setup_seconds=state.basis_setup_seconds,
            evaluator_only=False,
        )
        case_rows.append(anchor_row)

        target_residual = record.mismatch_normalized - view.damping_normalized
        measurement = measurement_projection_oracle(
            state.basis,
            target_residual,
            coefficient_radius=state.coefficient_radius,
        )
        operator = view.operator
        _, adjoint_map = n15a._operator_maps(operator)
        support = zero_outer_boundary_support(operator.grid_shape, dtype=torch.float64)
        before = operator.call_report()
        adjoint = adjoint_projection_oracle(
            state.basis,
            target_residual,
            adjoint=adjoint_map,
            support=support,
            coefficient_radius=state.coefficient_radius,
            l2=float(oracle_config["adjoint_l2"]),
        )
        adjoint_delta = _call_delta(before, operator.call_report())
        if adjoint_delta != {
            "forward_calls": 0,
            "adjoint_calls": adjoint.evaluator_adjoint_calls,
        }:
            raise RuntimeError("adjoint projection evaluator ledger drifted")
        basis_rows.append(_basis_row(state, measurement, adjoint))

        projected_corrections = {
            "measurement_projection_oracle": (
                _correction_from_projection(state, measurement),
                measurement,
            ),
            "adjoint_projection_oracle": (
                _correction_from_projection(state, adjoint),
                adjoint,
            ),
        }
        for candidate_id, (correction, projection) in projected_corrections.items():
            case_rows.append(
                _run_refinement(
                    state,
                    candidate_id=candidate_id,
                    correction_normalized=correction,
                    config=config,
                    refine_iterations=int(budget["candidate_refine_iterations"]),
                    setup_forward_calls=state.setup_forward_calls,
                    setup_adjoint_calls=state.setup_adjoint_calls,
                    setup_seconds=state.basis_setup_seconds,
                    evaluator_only=True,
                    metadata={
                        "coefficient_norm": projection.coefficient_norm,
                        "coefficient_radius": state.coefficient_radius,
                        "coefficient_clipped": projection.clipped,
                        "projection_target_residual_ratio": projection.residual_ratio,
                        "coefficient_evaluator_forward_calls": projection.evaluator_forward_calls,
                        "coefficient_evaluator_adjoint_calls": projection.evaluator_adjoint_calls,
                    },
                )
            )
            diagnostic_rows.append(
                _target_diagnostics(
                    state,
                    candidate_id=candidate_id,
                    correction_normalized=correction,
                )
            )

        if oracle_config["include_frozen_global_pca_rank4_probe_tax_control"]:
            target_flat = target_residual.reshape(-1)
            if global_mean.numel() != target_flat.numel() or global_vectors.shape[1] != target_flat.numel():
                raise RuntimeError("frozen global PCA observation shape drifted")
            global_unconstrained = global_vectors @ (target_flat - global_mean)
            global_coefficients = project_to_l2_ball(
                global_unconstrained, radius=state.coefficient_radius
            )
            global_residual = global_mean + global_coefficients @ global_vectors
            global_correction = view.damping_normalized + global_residual.reshape_as(
                view.damping_normalized
            )
            global_ratio = float(
                torch.linalg.vector_norm(target_flat - global_residual)
            ) / max(float(torch.linalg.vector_norm(target_flat)), 1e-30)
            case_rows.append(
                _run_refinement(
                    state,
                    candidate_id="frozen_global_pca_rank4_probe_tax_oracle",
                    correction_normalized=global_correction,
                    config=config,
                    refine_iterations=int(budget["candidate_refine_iterations"]),
                    setup_forward_calls=state.setup_forward_calls,
                    setup_adjoint_calls=state.setup_adjoint_calls,
                    setup_seconds=state.basis_setup_seconds,
                    evaluator_only=True,
                    metadata={
                        "basis_rank": 4,
                        "coefficient_norm": float(torch.linalg.vector_norm(global_coefficients)),
                        "coefficient_radius": state.coefficient_radius,
                        "coefficient_clipped": float(torch.linalg.vector_norm(global_unconstrained))
                        > state.coefficient_radius,
                        "projection_target_residual_ratio": global_ratio,
                        "frozen_source_candidate_id": frozen_model["candidate_id"],
                    },
                )
            )
            diagnostic_rows.append(
                _target_diagnostics(
                    state,
                    candidate_id="frozen_global_pca_rank4_probe_tax_oracle",
                    correction_normalized=global_correction,
                )
            )

        starts = [
            torch.zeros(state.basis.rank, dtype=torch.float64),
            measurement.coefficients,
            adjoint.coefficients,
        ]
        field_oracle = _finite_k_field_oracle(state, starts=starts, config=config)
        field_correction = view.damping_normalized + state.basis.synthesize(
            field_oracle.coefficients
        )
        case_rows.append(
            _run_refinement(
                state,
                candidate_id="truth_conditioned_finite_k_oracle_search",
                correction_normalized=field_correction,
                config=config,
                refine_iterations=int(budget["candidate_refine_iterations"]),
                setup_forward_calls=state.setup_forward_calls,
                setup_adjoint_calls=state.setup_adjoint_calls,
                setup_seconds=state.basis_setup_seconds,
                evaluator_only=True,
                metadata={
                    "coefficient_norm": field_oracle.coefficient_norm,
                    "coefficient_radius": state.coefficient_radius,
                    "evaluator_search_forward_calls": field_oracle.evaluator_forward_calls,
                    "evaluator_search_adjoint_calls": field_oracle.evaluator_adjoint_calls,
                    "evaluator_search_unique_evaluations": field_oracle.unique_evaluations,
                },
            )
        )
        diagnostic_rows.append(
            _target_diagnostics(
                state,
                candidate_id="truth_conditioned_finite_k_oracle_search",
                correction_normalized=field_correction,
            )
        )
        exact = record.mismatch_normalized
        if oracle_config["include_exact_mismatch_refine12_control"]:
            case_rows.append(
                _run_refinement(
                    state,
                    candidate_id="exact_mismatch_refine12_oracle",
                    correction_normalized=exact,
                    config=config,
                    refine_iterations=int(budget["component_damping_refine_iterations"]),
                    setup_forward_calls=0,
                    setup_adjoint_calls=0,
                    setup_seconds=0.0,
                    evaluator_only=True,
                )
            )
            diagnostic_rows.append(
                _target_diagnostics(
                    state,
                    candidate_id="exact_mismatch_refine12_oracle",
                    correction_normalized=exact,
                )
            )
        search_rows.append(
            {
                "partition": record.partition,
                "base_seed": record.base_seed,
                "family": record.family,
                "case_id": record.case.inference.case_id,
                "geometry_digest": record.case.inference.geometry.digest,
                "candidate_id": "truth_conditioned_finite_k_oracle_search",
                "field_relative_l2": field_oracle.field_relative_l2,
                "requested_evaluations": field_oracle.requested_evaluations,
                "unique_evaluations": field_oracle.unique_evaluations,
                "evaluator_forward_calls": field_oracle.evaluator_forward_calls,
                "evaluator_adjoint_calls": field_oracle.evaluator_adjoint_calls,
                "optimizer_runs": field_oracle.optimizer_runs,
                "successful_optimizer_runs": field_oracle.successful_optimizer_runs,
                "coefficient_norm": field_oracle.coefficient_norm,
                "coefficient_radius": state.coefficient_radius,
                "runtime_seconds": field_oracle.runtime_seconds,
                "guaranteed_global_optimum": False,
            }
        )

        case_rows.append(
            _run_refinement(
                state,
                candidate_id="exact_mismatch_oracle",
                correction_normalized=exact,
                config=config,
                refine_iterations=int(budget["candidate_refine_iterations"]),
                setup_forward_calls=state.setup_forward_calls,
                setup_adjoint_calls=state.setup_adjoint_calls,
                setup_seconds=state.basis_setup_seconds,
                evaluator_only=True,
            )
        )
        diagnostic_rows.append(
            _target_diagnostics(
                state,
                candidate_id="exact_mismatch_oracle",
                correction_normalized=exact,
            )
        )

        teacher = HighOrderTeacherMaps(operator)
        teacher_started = time.perf_counter()
        high_correction = teacher.correction(
            view.warm_field, low_projection=view.warm_projection
        ) / view.signal_scale
        teacher_seconds = time.perf_counter() - teacher_started
        if teacher.call_report() != {"forward_calls": 1, "adjoint_calls": 0}:
            raise RuntimeError("high-order teacher control ledger drifted")
        beta = float(oracle_config["high_order_teacher_beta"])
        teacher_correction = view.damping_normalized + beta * (
            high_correction - view.damping_normalized
        )
        case_rows.append(
            _run_refinement(
                state,
                candidate_id="high_order_teacher_b0p75",
                correction_normalized=teacher_correction,
                config=config,
                refine_iterations=int(budget["candidate_refine_iterations"]),
                setup_forward_calls=state.setup_forward_calls,
                setup_adjoint_calls=state.setup_adjoint_calls,
                setup_seconds=state.basis_setup_seconds,
                evaluator_only=True,
                high_order_forward_calls=1,
                metadata={"high_order_teacher_setup_seconds": teacher_seconds},
            )
        )
        diagnostic_rows.append(
            _target_diagnostics(
                state,
                candidate_id="high_order_teacher_b0p75",
                correction_normalized=teacher_correction,
            )
        )
    return case_rows, diagnostic_rows, basis_rows, search_rows


def _aggregate_partition(
    case_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    *,
    partition: str,
) -> list[dict[str, Any]]:
    rows = [row for row in case_rows if row["partition"] == partition]
    diagnostics = [row for row in diagnostic_rows if row["partition"] == partition]
    low = n16._index_rows(rows, "low_cgls24_matched")
    damping = n16._index_rows(rows, "component_damping")
    teacher = n16._index_rows(rows, "high_order_teacher_b0p75")
    candidate_ids = sorted({str(row["candidate_id"]) for row in rows})
    aggregates = []
    for candidate_id in candidate_ids:
        selected_diagnostics = [
            row for row in diagnostics if row["candidate_id"] == candidate_id
        ]
        aggregates.append(
            n16._aggregate_candidate(
                rows,
                candidate_id=candidate_id,
                low_rows=low,
                damping_rows=damping,
                teacher_rows=teacher,
                adjoint_diagnostic_rows=selected_diagnostics or None,
            )
        )
    exact = next(row for row in aggregates if row["candidate_id"] == "exact_mismatch_oracle")
    exact_gain = float(exact["mean_field_gain_over_low_cgls24"])
    for row in aggregates:
        gain = float(row["mean_field_gain_over_low_cgls24"])
        row["mean_exact_oracle_field_gain_retention"] = (
            gain / exact_gain if exact_gain > 1e-30 else None
        )
        selected = n16._index_rows(rows, str(row["candidate_id"]))
        family_gains: dict[str, float] = {}
        for family in sorted({family for _, family in selected}):
            values = []
            for key, candidate in selected.items():
                if key[1] != family:
                    continue
                values.append(
                    1.0
                    - float(candidate["field_relative_l2"])
                    / max(float(damping[key]["field_relative_l2"]), 1e-30)
                )
            family_gains[family] = float(np.mean(values))
        row["family_mean_field_gain_over_component_damping"] = family_gains
    return aggregates


def _gate(
    aggregate: Mapping[str, Any],
    basis_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["representation_gate"]
    checks = {
        "mean_field_gain_over_low": float(aggregate["mean_field_gain_over_low_cgls24"])
        >= float(gates["mean_field_gain_over_low_cgls24_minimum"]),
        "mean_h1_gain_over_low": float(aggregate["mean_h1_gain_over_low_cgls24"])
        >= float(gates["mean_h1_gain_over_low_cgls24_minimum"]),
        "mean_field_gain_over_damping": float(
            aggregate["mean_field_gain_over_component_damping"]
        )
        >= float(gates["mean_field_gain_over_component_damping_minimum"]),
        "mean_field_gain_over_teacher": float(
            aggregate["mean_field_gain_over_high_order_teacher_b0p75"]
        )
        >= float(gates["mean_field_gain_over_high_order_teacher_b0p75_minimum"]),
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
        "adjoint_residual_gain": float(
            aggregate["evaluator_mean_adjoint_residual_gain_over_component_damping"]
        )
        >= float(gates["mean_adjoint_residual_gain_over_component_damping_minimum"]),
        "exact_oracle_gain_retention": float(
            aggregate["mean_exact_oracle_field_gain_retention"]
        )
        >= float(gates["mean_exact_oracle_field_gain_retention_minimum"]),
        "each_family_gain_over_damping": all(
            float(value) > 0.0
            for value in aggregate[
                "family_mean_field_gain_over_component_damping"
            ].values()
        ),
        "low_forward_budget": int(aggregate["low_forward_calls"])
        == int(gates["required_low_forward_calls"]),
        "low_adjoint_budget": int(aggregate["low_adjoint_calls"])
        == int(gates["required_low_adjoint_calls"]),
        "zero_high_order_forward": int(aggregate["high_order_forward_calls"])
        == int(gates["required_high_order_forward_calls"]),
        "zero_high_order_adjoint": int(aggregate["high_order_adjoint_calls"])
        == int(gates["required_high_order_adjoint_calls"]),
        "minimum_basis_rank": min(int(row["basis_rank"]) for row in basis_rows)
        >= int(config["basis"]["minimum_accepted_rank"]),
        "basis_orthonormality": max(
            float(row["orthonormality_defect"]) for row in basis_rows
        )
        <= float(config["basis"]["maximum_orthonormality_defect"]),
    }
    return {
        "candidate_id": aggregate["candidate_id"],
        "checks": checks,
        "passed_count": sum(bool(value) for value in checks.values()),
        "check_count": len(checks),
        "passed": all(checks.values()),
    }


def _plot(
    aggregates: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    development = [row for row in aggregates if row["partition"] == "development"]
    order = [
        "component_damping",
        "krylov_schedule_damping_anchor",
        "measurement_projection_oracle",
        "adjoint_projection_oracle",
        "frozen_global_pca_rank4_probe_tax_oracle",
        "truth_conditioned_finite_k_oracle_search",
        "high_order_teacher_b0p75",
        "exact_mismatch_oracle",
    ]
    indexed = {row["candidate_id"]: row for row in development}
    selected = [indexed[name] for name in order]
    labels = [
        "damping",
        "probe cost, c=0",
        "measurement oracle",
        "adjoint oracle",
        "global PCA rank-4",
        "finite-K search",
        "high-order .75",
        "exact mismatch",
    ]
    colors = [
        "#477c72",
        "#7d8891",
        "#2f6f9f",
        "#6d5c91",
        "#7a6b4d",
        "#a45b45",
        "#b28a3e",
        "#37495b",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axes[0].barh(
        labels,
        [100.0 * float(row["mean_field_gain_over_low_cgls24"]) for row in selected],
        color=colors,
    )
    axes[0].axvline(5.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("mean field gain vs CGLS-24 (%)")
    axes[0].set_title("Opened development representation ceiling")

    rows = [
        row
        for row in case_rows
        if row["partition"] == "development"
        and row["candidate_id"] in {
            "measurement_projection_oracle",
            "truth_conditioned_finite_k_oracle_search",
        }
    ]
    low = n16._index_rows(case_rows, "low_cgls24_matched")
    seeds = sorted({int(row["base_seed"]) for row in rows})
    x = np.arange(len(seeds), dtype=float)
    for offset, candidate_id, label, color in (
        (-0.16, "measurement_projection_oracle", "measurement", "#2f6f9f"),
        (0.16, "truth_conditioned_finite_k_oracle_search", "finite-K", "#a45b45"),
    ):
        values = []
        for seed in seeds:
            cluster = [
                row
                for row in rows
                if int(row["base_seed"]) == seed and row["candidate_id"] == candidate_id
            ]
            gains = [
                1.0
                - float(row["field_relative_l2"])
                / float(low[(seed, str(row["family"]))]["field_relative_l2"])
                for row in cluster
            ]
            values.append(100.0 * float(np.mean(gains)))
        axes[1].bar(x + offset, values, width=0.32, label=label, color=color)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x, [str(seed) for seed in seeds])
    axes[1].set_xlabel("geometry seed")
    axes[1].set_ylabel("paired-family field gain (%)")
    axes[1].set_title("Geometry-cluster tails")
    axes[1].legend(frameon=False)

    dev_basis = [row for row in basis_rows if row["partition"] == "development"]
    axes[2].scatter(
        [float(row["measurement_target_residual_ratio"]) for row in dev_basis],
        [float(row["adjoint_target_residual_ratio"]) for row in dev_basis],
        c=[int(row["basis_rank"]) for row in dev_basis],
        cmap="viridis",
        edgecolors="black",
        linewidths=0.4,
    )
    axes[2].axvline(1.0, color="#777777", linewidth=0.8)
    axes[2].axhline(1.0, color="#777777", linewidth=0.8)
    axes[2].set_xlabel("measurement target residual ratio")
    axes[2].set_ylabel("adjoint target residual ratio")
    axes[2].set_title("What the per-geometry span retains")
    fig.suptitle("N1.7 geometry-conditioned Krylov representation gate", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "diagnostic.png", dpi=180)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.replace_output:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = n16._read_json(config_path)
    n15a_config_path = ROOT / config["source_n1_5_a_config"]
    n15a_config = n16._read_json(n15a_config_path)
    source_path = ROOT / config["source_t0_config"]
    source = n16._read_json(source_path)
    confirmation_manifest_path = (
        ROOT / config["source_n1_5_confirmation"] / "case_manifest.csv"
    )
    confirmation_manifest = n16._read_csv(confirmation_manifest_path)
    preregistration_path = ROOT / config["preregistration_document"]
    _validate_config(config, n15a_config, seed_limit=args.seed_limit)
    states, manifest = _prepare_basis_states(
        config, n15a_config, source, seed_limit=args.seed_limit
    )
    split_audit = n16._validate_split_integrity(
        manifest,
        confirmation_manifest,
        families=[str(value) for value in n15a_config["families"]],
    )
    case_rows, diagnostic_rows, basis_rows, search_rows = _evaluate_cases(states, config)
    aggregates = []
    for partition in config["evaluated_partitions"]:
        aggregates.extend(
            _aggregate_partition(
                case_rows, diagnostic_rows, partition=str(partition)
            )
        )
    development = {
        str(row["candidate_id"]): row
        for row in aggregates
        if row["partition"] == "development"
    }
    development_basis = [row for row in basis_rows if row["partition"] == "development"]
    primary_id = str(config["oracle_screen"]["primary_representation_candidate"])
    primary_gate = _gate(development[primary_id], development_basis, config)
    field_id = "truth_conditioned_finite_k_oracle_search"
    field_gate = _gate(development[field_id], development_basis, config)
    if primary_gate["passed"]:
        status = "REPRESENTATION_ELIGIBLE_LEARNER_NOT_YET_TESTED"
    elif field_gate["passed"]:
        status = "FIELD_ORACLE_ONLY_REFRAME_OR_STOP"
    else:
        status = "REPRESENTATION_NO_GO_STOP_BEFORE_LEARNER"

    evaluator_partitions = {
        partition: {
            "forward_calls": sum(
                int(row["evaluator_forward_calls"])
                for row in search_rows
                if row["partition"] == partition
            ),
            "adjoint_calls": sum(
                int(row["evaluator_adjoint_calls"])
                for row in search_rows
                if row["partition"] == partition
            ),
        }
        for partition in config["evaluated_partitions"]
    }
    evaluator_package = {
        key: sum(int(values[key]) for values in evaluator_partitions.values())
        for key in ("forward_calls", "adjoint_calls")
    }
    development_evaluator = evaluator_partitions["development"]
    calibration_evaluator = evaluator_partitions["calibration"]
    summary = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "evidence_level": config["evidence_level"],
        "runtime_seconds": time.perf_counter() - started,
        "seed_limit": args.seed_limit,
        "independent_unit": "base_seed_geometry_cluster",
        "paired_families_are_not_independent_rigs": True,
        "development_geometry_cluster_count": len(
            {row["base_seed"] for row in development_basis}
        ),
        "development_case_count": len(development_basis),
        "split_integrity_audit": split_audit,
        "primary_representation_candidate": primary_id,
        "primary_development_aggregate": development[primary_id],
        "finite_k_development_aggregate": development[field_id],
        "exact_mismatch_development_aggregate": development["exact_mismatch_oracle"],
        "primary_representation_gate": primary_gate,
        "finite_k_diagnostic_gate": field_gate,
        "development_aggregates": [
            row for row in aggregates if row["partition"] == "development"
        ],
        "minimum_development_basis_rank": min(
            int(row["basis_rank"]) for row in development_basis
        ),
        "maximum_development_orthonormality_defect": max(
            float(row["orthonormality_defect"]) for row in development_basis
        ),
        "finite_k_evaluator_development_forward_calls": development_evaluator[
            "forward_calls"
        ],
        "finite_k_evaluator_development_adjoint_calls": development_evaluator[
            "adjoint_calls"
        ],
        "finite_k_evaluator_calibration_forward_calls": calibration_evaluator[
            "forward_calls"
        ],
        "finite_k_evaluator_calibration_adjoint_calls": calibration_evaluator[
            "adjoint_calls"
        ],
        "finite_k_evaluator_package_forward_calls": evaluator_package[
            "forward_calls"
        ],
        "finite_k_evaluator_package_adjoint_calls": evaluator_package[
            "adjoint_calls"
        ],
        # Historical N1.7 summaries used "total" for development-only search.
        "finite_k_evaluator_total_forward_calls": development_evaluator[
            "forward_calls"
        ],
        "finite_k_evaluator_total_adjoint_calls": development_evaluator[
            "adjoint_calls"
        ],
        "finite_k_evaluator_total_scope": "development_only_legacy_field_name",
        "finite_k_is_guaranteed_global_optimum": False,
        "learner_was_trained": False,
        "may_claim_learned_algorithm_gain": False,
        "may_claim_confirmed_algorithm_gain": False,
        "opens_ood_fresh_or_final": False,
        "real_bost_claim": False,
        "deployment_contract": config["deployment_contract"],
        "budget": config["budget"],
        "claim_boundary": config["claim_boundary"],
        "physics_scope": (
            "continuous analytic-gradient renderer versus voxel finite-difference/"
            "trilinear representation mismatch only"
        ),
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
    n16._write_csv(output / "finite_k_search_diagnostics.csv", search_rows)
    n16._write_csv(output / "aggregate_metrics.csv", aggregates)
    _plot(aggregates, case_rows, basis_rows, output)
    provenance = {
        "schema": "jacru-n1-7-provenance-1.0",
        "git_commit": n16._git_commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": n16._sha256(config_path),
        "source_n1_5_a_config_sha256": n16._sha256(n15a_config_path),
        "source_n1_5_confirmation_manifest_sha256": n16._sha256(
            confirmation_manifest_path
        ),
        "source_n1_6_summary_sha256": n16._sha256(
            ROOT / config["source_n1_6_result"] / "summary.json"
        ),
        "preregistration_document_sha256": n16._sha256(preregistration_path),
        "source_t0_config_sha256": n16._sha256(source_path),
        "runner_sha256": n16._sha256(Path(__file__)),
        "model_module_sha256": n16._sha256(
            ROOT / "demo_t16_operator/jacru_n1_7_krylov_correction.py"
        ),
        "development_was_already_opened": True,
        "confirmation_or_ood_opened": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# N1.7 geometry-conditioned Krylov representation gate

Status: **{status}**.

This is an opened synthetic evaluator-only representation screen. It contains
no learned coefficient model and cannot establish algorithm gain, real-BOST
performance, OOD transfer, or novelty.

- Primary representation oracle: `{primary_id}`.
- Production-equivalent path: 25 low F / 24 low A^T / 0 high-order calls.
- Finite-K search calls are reported separately and are not deployment cost.
- No OOD, fresh, final, or real-BOST evidence was opened.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    n16._write_checksums(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
