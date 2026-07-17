#!/usr/bin/env python3
"""Screen a true robust-data PDHG base before any N1.3 neural correction.

The development screen uses only the already-opened synthetic development
split. It compares Huber measurement fidelity with no whitening, isotropic
flow-off whitening, and structured flow-off whitening. A deterministic sparse
flow-on displacement-outlier stress is evaluated beside the nominal packet.
OOD construction and evaluation are structurally disabled in this runner.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
import uuid

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.interface_baselines import (  # noqa: E402
    edge_preserving_pdhg_baseline,
    robust_data_pdhg_baseline,
)
from demo_t16_operator.jacru_n1_2_session_conformal import (  # noqa: E402
    SelectorPayload,
    stable_digest,
    stable_seed,
    verify_session_packet,
)
from site_tools import run_jacru_m2_1_data_consistency_diagnostic as m21  # noqa: E402
from site_tools import run_jacru_m2_learned_residual_gate as m2  # noqa: E402
from site_tools import (  # noqa: E402
    run_jacru_n1_2_session_conformal_dual_reference as n12,
)


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "jacru_n1_3_robust_data_whitening_development_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_3_robust_data_whitening_development_scratch"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(materialized) / len(materialized)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_manifest(config_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = [
        config_path,
        ROOT / str(config["source_t0_config"]),
        ROOT / str(config["source_n1_2_config"]),
        ROOT / "demo_t16_operator/interface_baselines.py",
        ROOT / "demo_t16_operator/jacru_n1_2_session_conformal.py",
        ROOT / "demo_t16_operator/jacru_n1_2_dual_reference.py",
        ROOT / "demo_t16_operator/jacru_n1_flowoff_covariance.py",
        ROOT / "demo_t16_operator/jacru_synthetic_fixture.py",
        ROOT / "site_tools/run_jacru_m2_1_data_consistency_diagnostic.py",
        ROOT / "site_tools/run_jacru_m2_learned_residual_gate.py",
        ROOT / "site_tools/run_jacru_n1_2_session_conformal_dual_reference.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _validate_config(config: Mapping[str, Any], seed_limit: int | None) -> None:
    if config.get("status") != "DEVELOPMENT_ONLY_NOT_PREREGISTERED_NOT_FORMAL":
        raise RuntimeError("N1.3 runner accepts only the explicit development config")
    if config.get("evaluated_split") != "development":
        raise RuntimeError("this runner is development-only")
    if config.get("may_construct_or_evaluate_ood") is not False:
        raise RuntimeError("OOD construction must remain disabled")
    if seed_limit is not None and seed_limit < 1:
        raise ValueError("seed-limit must be positive")
    budget = config["solve_budget"]
    iterations = int(budget["iterations"])
    if iterations != int(budget["forward_calls"]) or iterations != int(
        budget["adjoint_calls"]
    ):
        raise ValueError("solve budget must use one physical pair per iteration")


def _token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _candidate_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = config["candidate_grid"]
    output: list[dict[str, Any]] = []
    for mean_policy in grid["mean_policies"]:
        for policy in grid["whitening_policies"]:
            for delta in grid["data_huber_delta_sigma_multipliers"]:
                for weight in grid["standardized_edge_regularization_weights"]:
                    output.append(
                        {
                            "candidate_id": (
                                f"robust_data__mean_{mean_policy}__{policy}__"
                                f"d{_token(delta)}__l{_token(weight)}"
                            ),
                            "solver_kind": "huber_measurement_pdhg",
                            "mean_policy": str(mean_policy),
                            "whitening_policy": str(policy),
                            "data_huber_delta_sigma_multiplier": float(delta),
                            "standardized_edge_regularization_weight": float(weight),
                        }
                    )
            if grid["include_quadratic_controls_all_whitening_policies"]:
                for weight in grid["standardized_edge_regularization_weights"]:
                    output.append(
                        {
                            "candidate_id": (
                                f"quadratic_data__mean_{mean_policy}__{policy}__"
                                f"l{_token(weight)}"
                            ),
                            "solver_kind": "quadratic_measurement_pdhg_control",
                            "mean_policy": str(mean_policy),
                            "whitening_policy": str(policy),
                            "data_huber_delta_sigma_multiplier": None,
                            "standardized_edge_regularization_weight": float(weight),
                        }
                    )
    identifiers = [str(row["candidate_id"]) for row in output]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("candidate IDs must be unique")
    return output


def _development_source_config(
    source: Mapping[str, Any], seed_limit: int | None
) -> dict[str, Any]:
    value = copy.deepcopy(source)
    value["splits"]["train"]["base_seeds"] = []
    value["splits"]["ood"]["base_seeds"] = []
    if seed_limit is not None:
        value["splits"]["development"]["base_seeds"] = value["splits"][
            "development"
        ]["base_seeds"][:seed_limit]
    if not value["splits"]["development"]["base_seeds"]:
        raise ValueError("development split must contain at least one seed")
    return value


def _selector_maps(
    packets: Mapping[str, Any], n12_config: Mapping[str, Any]
) -> tuple[dict[tuple[str, str], SelectorPayload], list[dict[str, Any]]]:
    selectors, calibration_rows, _ = n12._calibrate_candidates(
        packets=packets,
        config=n12_config,
    )
    selected_ids = {
        "isotropic_flowoff": "isotropic_joint_two_sided_95",
        "structured_flowoff": "structured_joint_two_sided_95",
        "diagonal_flowoff": "structured_joint_two_sided_95",
        "unwhitened": "structured_joint_two_sided_95",
    }
    output: dict[tuple[str, str], SelectorPayload] = {}
    for session_id, packet in packets.items():
        verify_session_packet(packet)
        for policy, candidate_id in selected_ids.items():
            output[(session_id, policy)] = selectors[(session_id, candidate_id)]
    return output, calibration_rows


def _sparse_outlier_delta(
    observation_uv: torch.Tensor,
    *,
    camera_index: torch.Tensor,
    covariance: torch.Tensor,
    fraction_per_camera: float,
    minimum_components_per_camera: int,
    standardized_amplitude: float,
    seed: int,
) -> torch.Tensor:
    observation = torch.as_tensor(observation_uv).detach().cpu().to(torch.float64)
    cameras = torch.as_tensor(camera_index).detach().cpu().to(torch.int64)
    covariance = torch.as_tensor(covariance).detach().cpu().to(torch.float64)
    if observation.ndim != 2 or observation.shape[1] != 2:
        raise ValueError("observation_uv must have shape [ray,2]")
    if cameras.shape != (observation.shape[0],):
        raise ValueError("camera_index does not match observation rays")
    if covariance.shape != (observation.numel(), observation.numel()):
        raise ValueError("covariance does not match flattened observation")
    fraction = float(fraction_per_camera)
    amplitude = float(standardized_amplitude)
    minimum = int(minimum_components_per_camera)
    if not 0.0 < fraction <= 1.0 or amplitude <= 0.0 or minimum < 1:
        raise ValueError("outlier policy must use positive fraction, amplitude, and count")
    component_camera = cameras[:, None].expand(-1, 2).reshape(-1)
    standard_deviation = torch.sqrt(torch.diag(covariance).clamp_min(1e-30))
    generator = torch.Generator().manual_seed(int(seed))
    flattened = torch.zeros(observation.numel(), dtype=torch.float64)
    for camera in range(int(torch.max(cameras)) + 1):
        indices = torch.nonzero(component_camera == camera, as_tuple=False).reshape(-1)
        count = max(minimum, int(math.ceil(fraction * indices.numel())))
        count = min(count, indices.numel())
        chosen = indices[torch.randperm(indices.numel(), generator=generator)[:count]]
        signs = 2.0 * torch.randint(
            0, 2, (count,), generator=generator, dtype=torch.int64
        ).to(torch.float64) - 1.0
        flattened[chosen] = amplitude * standard_deviation[chosen] * signs
    return flattened.reshape_as(observation)


def _expanded_stress_records(
    records: list[m2.PreparedRecord],
    *,
    case_to_session: Mapping[str, str],
    selectors: Mapping[tuple[str, str], SelectorPayload],
    stress_config: Mapping[str, Any],
) -> tuple[list[m2.PreparedRecord], dict[str, str], dict[str, str], list[dict[str, Any]]]:
    output: list[m2.PreparedRecord] = []
    expanded_sessions: dict[str, str] = {}
    stress_by_case: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []
    for record in records:
        case_id = record.case.inference.case_id
        session_id = str(case_to_session[case_id])
        output.append(record)
        expanded_sessions[case_id] = session_id
        stress_by_case[case_id] = "nominal"
        manifest.append(
            {
                "case_id": case_id,
                "source_case_id": case_id,
                "session_id": session_id,
                "stress": "nominal",
                "outlier_component_count": 0,
                "outlier_uses_truth": False,
            }
        )
        if not stress_config["enabled"]:
            continue
        selector = selectors[(session_id, "structured_flowoff")]
        observation = record.case.inference.observations_uv[0]
        delta = _sparse_outlier_delta(
            observation,
            camera_index=selector.camera_index,
            covariance=selector.proximal_covariance,
            fraction_per_camera=float(stress_config["fraction_per_camera"]),
            minimum_components_per_camera=int(
                stress_config["minimum_components_per_camera"]
            ),
            standardized_amplitude=float(stress_config["standardized_amplitude"]),
            seed=stable_seed(
                session_id,
                case_id,
                "flowon-sparse-outlier",
                int(stress_config["seed"]),
            ),
        )
        stressed_observation = observation.detach().cpu().to(torch.float64) + delta
        stressed_id = f"{case_id}:stress=sparse-flowon-outlier"
        inference = replace(
            record.case.inference,
            case_id=stressed_id,
            observations_uv=stressed_observation[None],
            observation_digest=stable_digest(
                stressed_observation,
                metadata={
                    "source_case_id": case_id,
                    "stress": "sparse-flowon-outlier",
                    "session_id": session_id,
                },
            ),
        )
        evaluation = replace(
            record.case.evaluation,
            case_id=stressed_id,
            additive_noise_uv=(
                record.case.evaluation.additive_noise_uv.detach().cpu().to(torch.float64)
                + delta[None]
            ),
        )
        stressed_record = replace(
            record,
            case=replace(record.case, inference=inference, evaluation=evaluation),
        )
        output.append(stressed_record)
        expanded_sessions[stressed_id] = session_id
        stress_by_case[stressed_id] = "sparse_flowon_outlier"
        manifest.append(
            {
                "case_id": stressed_id,
                "source_case_id": case_id,
                "session_id": session_id,
                "stress": "sparse_flowon_outlier",
                "outlier_component_count": int(torch.count_nonzero(delta)),
                "outlier_uses_truth": False,
            }
        )
    return output, expanded_sessions, stress_by_case, manifest


def _whitening_factor(selector: SelectorPayload) -> torch.Tensor:
    covariance = selector.proximal_covariance.detach().cpu().to(torch.float64)
    factor, info = torch.linalg.cholesky_ex(covariance)
    if int(torch.max(info)) != 0:
        raise ValueError("flow-off covariance is not positive definite")
    return factor


def _whitening_factor_for_policy(
    selector: SelectorPayload,
    policy: str,
) -> torch.Tensor | None:
    if policy == "unwhitened":
        return None
    if policy == "diagonal_flowoff":
        diagonal = torch.diag(selector.proximal_covariance).clamp_min(1e-30)
        return torch.diag(torch.sqrt(diagonal))
    if policy in {"isotropic_flowoff", "structured_flowoff"}:
        return _whitening_factor(selector)
    raise ValueError(f"unsupported whitening policy: {policy}")


def _whiten_vector(value: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    shape = value.shape
    flattened = value.reshape(-1, 1).to(torch.float64)
    return torch.linalg.solve_triangular(
        factor, flattened, upper=False
    ).reshape(shape)


def _whiten_transpose_vector(value: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    shape = value.shape
    flattened = value.reshape(-1, 1).to(torch.float64)
    return torch.linalg.solve_triangular(
        factor.mT, flattened, upper=True
    ).reshape(shape)


def _candidate_operator(
    *,
    forward: Any,
    adjoint: Any,
    factor: torch.Tensor | None,
) -> tuple[Any, Any]:
    if factor is None:
        return forward, adjoint

    def whitened_forward(field: torch.Tensor) -> torch.Tensor:
        return _whiten_vector(forward(field), factor)

    def whitened_adjoint(measurement: torch.Tensor) -> torch.Tensor:
        return adjoint(_whiten_transpose_vector(measurement, factor))

    return whitened_forward, whitened_adjoint


def _gain(reference: float, candidate: float) -> float:
    return (float(reference) - float(candidate)) / max(float(reference), 1e-30)


def _evaluate_candidates(
    *,
    records: list[m2.PreparedRecord],
    case_to_session: Mapping[str, str],
    stress_by_case: Mapping[str, str],
    selectors: Mapping[tuple[str, str], SelectorPayload],
    matrices: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    references: Mapping[tuple[str, str], Mapping[str, Any]],
    candidates: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    whitening_cache: dict[
        tuple[str, str, str], tuple[torch.Tensor | None, float, float]
    ] = {}
    budget = config["solve_budget"]
    grid = config["candidate_grid"]
    iterations = int(budget["iterations"])
    for record in records:
        case = record.case
        case_id = case.inference.case_id
        session_id = str(case_to_session[case_id])
        stress = str(stress_by_case[case_id])
        operator = case.inference.operator
        observation = case.inference.observations_uv[0].detach().cpu().to(torch.float64)
        support = operator.support.detach().cpu().to(torch.float64)
        physical_forward, physical_adjoint = m2._operator_maps(operator)
        matrix, _ = matrices[case.inference.geometry.digest]
        registered_huber = references[(case_id, "huber_pdhg_matched")]
        registered_cgls = references[(case_id, "cgls_matched")]
        for candidate in candidates:
            policy = str(candidate["whitening_policy"])
            selector = selectors[(session_id, policy)]
            mean_policy = str(candidate["mean_policy"])
            if mean_policy == "estimated":
                centered_observation = observation - selector.mean_uv
            elif mean_policy == "zero":
                centered_observation = observation
            else:
                raise ValueError(f"unsupported mean policy: {mean_policy}")
            standard_deviation = torch.sqrt(
                torch.diag(selector.proximal_covariance).clamp_min(1e-30)
            )
            raw_unit_scale = float(torch.median(standard_deviation))
            cache_key = (session_id, policy, case.inference.geometry.digest)
            if cache_key not in whitening_cache:
                factor = _whitening_factor_for_policy(selector, policy)
                transformed_matrix = (
                    matrix
                    if factor is None
                    else torch.linalg.solve_triangular(factor, matrix, upper=False)
                )
                largest = float(torch.linalg.svdvals(transformed_matrix).max())
                norm_bound = float(budget["norm_safety_factor"]) * largest**2
                unit_scale = raw_unit_scale if factor is None else 1.0
                whitening_cache[cache_key] = (factor, norm_bound, unit_scale)
            factor, norm_bound, unit_scale = whitening_cache[cache_key]
            target = (
                centered_observation
                if factor is None
                else _whiten_vector(centered_observation, factor)
            )
            forward, adjoint = _candidate_operator(
                forward=physical_forward,
                adjoint=physical_adjoint,
                factor=factor,
            )
            edge_weight = (
                float(candidate["standardized_edge_regularization_weight"])
                * unit_scale
            )
            operator.reset_call_counts()
            started = time.perf_counter()
            if candidate["solver_kind"] == "huber_measurement_pdhg":
                delta = (
                    float(candidate["data_huber_delta_sigma_multiplier"])
                    * unit_scale
                )
                result = robust_data_pdhg_baseline(
                    target,
                    forward=forward,
                    adjoint=adjoint,
                    support=support,
                    spacing_xyz=operator.spacing_xyz,
                    iterations=iterations,
                    regularization_weight=edge_weight,
                    data_norm_squared_bound=norm_bound,
                    data_huber_delta=delta,
                    edge_penalty=str(grid["edge_penalty"]),
                    edge_huber_delta=float(grid["edge_huber_delta"]),
                    ridge_weight=float(grid["ridge_weight"]),
                    step_safety=float(budget["step_safety"]),
                )
            elif candidate["solver_kind"] == "quadratic_measurement_pdhg_control":
                delta = None
                result = edge_preserving_pdhg_baseline(
                    target,
                    forward=forward,
                    adjoint=adjoint,
                    support=support,
                    spacing_xyz=operator.spacing_xyz,
                    iterations=iterations,
                    regularization_weight=edge_weight,
                    data_norm_squared_bound=norm_bound,
                    penalty=str(grid["edge_penalty"]),
                    huber_delta=float(grid["edge_huber_delta"]),
                    step_safety=float(budget["step_safety"]),
                )
            else:
                raise ValueError(f"unsupported solver kind: {candidate['solver_kind']}")
            solve_seconds = time.perf_counter() - started
            physical_calls = operator.call_report()
            expected_calls = {
                "forward_calls": iterations,
                "adjoint_calls": iterations,
            }
            if physical_calls != expected_calls:
                raise RuntimeError(
                    f"candidate physical-call contract drifted: {physical_calls}"
                )
            score = m2._score_prediction(
                record=record,
                method=str(candidate["candidate_id"]),
                model_seed=-1,
                prediction=result.field,
                gate=None,
                correction_rms=None,
                optimization_forward_calls=result.forward_calls,
                optimization_adjoint_calls=result.adjoint_calls,
                grouped_adjoint_calls=0,
                neural_inference_seconds=0.0,
            )
            row = {
                **score,
                **candidate,
                "session_id": session_id,
                "stress": stress,
                "selector_digest": selector.digest,
                "uses_truth_or_clean_projection": False,
                "data_huber_delta_effective": delta,
                "edge_regularization_weight_effective": edge_weight,
                "measurement_unit_scale": unit_scale,
                "whitening_condition_number": (
                    1.0
                    if factor is None
                    else float(torch.linalg.cond(factor @ factor.mT))
                ),
                "operator_norm_squared_bound": norm_bound,
                "dense_norm_setup_in_budget": False,
                "candidate_budget_matched": True,
                "solve_seconds": solve_seconds,
                "terminal_objective": result.history[-1][
                    "extrapolated_total_objective"
                ],
                "terminal_primal_update_norm": result.history[-1][
                    "primal_update_norm"
                ],
                "terminal_data_dual_saturation_fraction": result.history[-1].get(
                    "data_dual_saturation_fraction"
                ),
                "field_gain_to_registered_huber": _gain(
                    registered_huber["field_relative_l2"], score["field_relative_l2"]
                ),
                "h1_gain_to_registered_huber": _gain(
                    registered_huber["h1_seminorm_relative_error"],
                    score["h1_seminorm_relative_error"],
                ),
                "clean_reprojection_ratio_to_cgls": float(
                    score["clean_reprojection_relative_l2"]
                )
                / max(float(registered_cgls["clean_reprojection_relative_l2"]), 1e-30),
                "measured_reprojection_ratio_to_cgls": float(
                    score["measured_reprojection_relative_l2"]
                )
                / max(
                    float(registered_cgls["measured_reprojection_relative_l2"]),
                    1e-30,
                ),
            }
            rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]], harm_threshold: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["candidate_id"]), str(row["stress"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (candidate_id, stress), values in sorted(grouped.items()):
        family_means = {
            family: _mean(
                row["field_gain_to_registered_huber"]
                for row in values
                if row["family"] == family
            )
            for family in sorted({str(row["family"]) for row in values})
        }
        output.append(
            {
                "candidate_id": candidate_id,
                "solver_kind": values[0]["solver_kind"],
                "mean_policy": values[0]["mean_policy"],
                "whitening_policy": values[0]["whitening_policy"],
                "stress": stress,
                "row_count": len(values),
                "field_gain_to_huber_mean": _mean(
                    row["field_gain_to_registered_huber"] for row in values
                ),
                "h1_gain_to_huber_mean": _mean(
                    row["h1_gain_to_registered_huber"] for row in values
                ),
                "field_harm_rate_vs_huber": _mean(
                    float(row["field_gain_to_registered_huber"]) < -harm_threshold
                    for row in values
                ),
                "worst_field_gain_to_huber": min(
                    float(row["field_gain_to_registered_huber"]) for row in values
                ),
                "clean_reprojection_ratio_to_cgls_mean": _mean(
                    row["clean_reprojection_ratio_to_cgls"] for row in values
                ),
                "clean_reprojection_ratio_to_cgls_maximum": max(
                    float(row["clean_reprojection_ratio_to_cgls"]) for row in values
                ),
                "minimum_family_mean_field_gain_to_huber": min(family_means.values()),
                "family_mean_field_gains_to_huber_json": _canonical_json(family_means),
                "exact_call_budget_all_rows": all(
                    int(row["optimization_forward_calls"]) == 24
                    and int(row["optimization_adjoint_calls"]) == 24
                    for row in values
                ),
            }
        )
    return output


def _factorial_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["candidate_id"]), str(row["case_id"])): row
        for row in rows
    }
    grouped: dict[tuple[str, str, float, float, str], list[dict[str, float]]] = {}
    for robust in rows:
        if robust["solver_kind"] != "huber_measurement_pdhg":
            continue
        mean_policy = str(robust["mean_policy"])
        whitening_policy = str(robust["whitening_policy"])
        weight = float(robust["standardized_edge_regularization_weight"])
        delta = float(robust["data_huber_delta_sigma_multiplier"])
        quadratic_id = (
            f"quadratic_data__mean_{mean_policy}__{whitening_policy}__"
            f"l{_token(weight)}"
        )
        quadratic = lookup[(quadratic_id, str(robust["case_id"]))]
        grouped.setdefault(
            (mean_policy, whitening_policy, delta, weight, str(robust["stress"])),
            [],
        ).append(
            {
                "field_gain_huber_data_to_quadratic_data": _gain(
                    quadratic["field_relative_l2"], robust["field_relative_l2"]
                ),
                "h1_gain_huber_data_to_quadratic_data": _gain(
                    quadratic["h1_seminorm_relative_error"],
                    robust["h1_seminorm_relative_error"],
                ),
                "clean_reprojection_ratio_huber_data_to_quadratic_data": float(
                    robust["clean_reprojection_relative_l2"]
                )
                / max(float(quadratic["clean_reprojection_relative_l2"]), 1e-30),
            }
        )
    output: list[dict[str, Any]] = []
    for (mean_policy, whitening_policy, delta, weight, stress), values in sorted(
        grouped.items()
    ):
        output.append(
            {
                "mean_policy": mean_policy,
                "whitening_policy": whitening_policy,
                "data_huber_delta_sigma_multiplier": delta,
                "standardized_edge_regularization_weight": weight,
                "stress": stress,
                "paired_case_count": len(values),
                "field_gain_huber_data_to_quadratic_data_mean": _mean(
                    row["field_gain_huber_data_to_quadratic_data"] for row in values
                ),
                "field_gain_huber_data_to_quadratic_data_worst": min(
                    row["field_gain_huber_data_to_quadratic_data"] for row in values
                ),
                "h1_gain_huber_data_to_quadratic_data_mean": _mean(
                    row["h1_gain_huber_data_to_quadratic_data"] for row in values
                ),
                "clean_reprojection_ratio_huber_data_to_quadratic_data_mean": _mean(
                    row["clean_reprojection_ratio_huber_data_to_quadratic_data"]
                    for row in values
                ),
            }
        )
    return output


def _decisions(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    *,
    gates: Mapping[str, Any],
    full_screen_complete: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if any(str(row.get("split")) != "development" for row in rows):
        raise ValueError("N1.3 development decisions reject non-development rows")
    lookup = {(row["candidate_id"], row["stress"]): row for row in aggregates}
    decisions: list[dict[str, Any]] = []
    candidate_ids = sorted({str(row["candidate_id"]) for row in aggregates})
    for candidate_id in candidate_ids:
        nominal = lookup[(candidate_id, "nominal")]
        outlier = lookup[(candidate_id, "sparse_flowon_outlier")]
        known = [
            row
            for row in rows
            if row["candidate_id"] == candidate_id
            and int(row["base_seed"]) == int(gates["known_interface_seed"])
        ]
        known_gain = (
            min(float(row["field_gain_to_registered_huber"]) for row in known)
            if known
            else None
        )
        checks = {
            "full_development_seed_set": full_screen_complete,
            "nominal_field_mean": float(nominal["field_gain_to_huber_mean"])
            >= float(gates["nominal_field_gain_to_huber_mean_minimum"]),
            "nominal_h1_mean": float(nominal["h1_gain_to_huber_mean"])
            >= float(gates["nominal_h1_gain_to_huber_mean_minimum"]),
            "nominal_harm_rate": float(nominal["field_harm_rate_vs_huber"])
            <= float(gates["nominal_field_harm_rate_maximum"]),
            "nominal_worst": float(nominal["worst_field_gain_to_huber"])
            >= float(gates["nominal_worst_field_gain_to_huber_minimum"]),
            "outlier_field_mean": float(outlier["field_gain_to_huber_mean"])
            >= float(gates["outlier_field_gain_to_huber_mean_minimum"]),
            "outlier_h1_mean": float(outlier["h1_gain_to_huber_mean"])
            >= float(gates["outlier_h1_gain_to_huber_mean_minimum"]),
            "outlier_harm_rate": float(outlier["field_harm_rate_vs_huber"])
            <= float(gates["outlier_field_harm_rate_maximum"]),
            "outlier_worst": float(outlier["worst_field_gain_to_huber"])
            >= float(gates["outlier_worst_field_gain_to_huber_minimum"]),
            "clean_mean_both_stresses": max(
                float(nominal["clean_reprojection_ratio_to_cgls_mean"]),
                float(outlier["clean_reprojection_ratio_to_cgls_mean"]),
            ) <= float(gates["clean_reprojection_ratio_to_cgls_mean_maximum"]),
            "clean_worst_both_stresses": max(
                float(nominal["clean_reprojection_ratio_to_cgls_maximum"]),
                float(outlier["clean_reprojection_ratio_to_cgls_maximum"]),
            ) <= float(gates["clean_reprojection_ratio_to_cgls_worst_maximum"]),
            "nominal_family_tail": float(
                nominal["minimum_family_mean_field_gain_to_huber"]
            ) >= float(gates["minimum_family_mean_field_gain_nominal"]),
            "outlier_family_tail": float(
                outlier["minimum_family_mean_field_gain_to_huber"]
            ) >= float(gates["minimum_family_mean_field_gain_outlier"]),
            "known_interface_seed_present_and_safe": known_gain is not None
            and known_gain >= float(gates["known_interface_seed_field_gain_minimum"]),
            "exact_call_budget": bool(nominal["exact_call_budget_all_rows"])
            and bool(outlier["exact_call_budget_all_rows"]),
        }
        decisions.append(
            {
                "candidate_id": candidate_id,
                "checks": checks,
                "known_interface_seed_worst_gain": known_gain,
                "nominal": nominal,
                "outlier": outlier,
                "passed": all(checks.values()),
            }
        )
    eligible = [row for row in decisions if row["passed"]]
    selection = None
    if eligible:
        selection = max(
            eligible,
            key=lambda row: (
                min(
                    float(row["nominal"]["field_gain_to_huber_mean"]),
                    float(row["outlier"]["field_gain_to_huber_mean"]),
                ),
                min(
                    float(row["nominal"]["h1_gain_to_huber_mean"]),
                    float(row["outlier"]["h1_gain_to_huber_mean"]),
                ),
            ),
        )
    return decisions, selection


def _plot(
    aggregates: list[dict[str, Any]],
    output: Path,
    *,
    title: str = "N1.3 robust measurement-fidelity development screen",
) -> None:
    lookup = {(row["candidate_id"], row["stress"]): row for row in aggregates}
    candidate_ids = sorted({str(row["candidate_id"]) for row in aggregates})
    candidate_ids = sorted(
        candidate_ids,
        key=lambda candidate_id: float(
            lookup[(candidate_id, "sparse_flowon_outlier")][
                "field_gain_to_huber_mean"
            ]
        ),
        reverse=True,
    )
    labels = [value.replace("robust_data__", "R ").replace("quadratic_data__", "Q ") for value in candidate_ids]
    y = np.arange(len(candidate_ids))
    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, 0.28 * len(labels))), constrained_layout=True)
    nominal = [lookup[(key, "nominal")]["field_gain_to_huber_mean"] for key in candidate_ids]
    outlier = [lookup[(key, "sparse_flowon_outlier")]["field_gain_to_huber_mean"] for key in candidate_ids]
    axes[0].barh(y - 0.18, nominal, height=0.34, label="nominal", color="#2a6f97")
    axes[0].barh(y + 0.18, outlier, height=0.34, label="sparse outlier", color="#bc4749")
    axes[0].axvline(0.0, color="#202020", linewidth=1)
    axes[0].set_yticks(y, labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("field gain vs fixed Huber-24")
    axes[0].legend()
    clean = [
        max(
            lookup[(key, "nominal")]["clean_reprojection_ratio_to_cgls_mean"],
            lookup[(key, "sparse_flowon_outlier")][
                "clean_reprojection_ratio_to_cgls_mean"
            ],
        )
        for key in candidate_ids
    ]
    axes[1].barh(y, clean, color="#6a994e")
    axes[1].axvline(1.0, color="#202020", linewidth=1)
    axes[1].set_yticks([])
    axes[1].set_xlabel("worst-stress mean clean ratio vs CGLS-24")
    worst = [
        min(
            lookup[(key, "nominal")]["worst_field_gain_to_huber"],
            lookup[(key, "sparse_flowon_outlier")]["worst_field_gain_to_huber"],
        )
        for key in candidate_ids
    ]
    axes[2].barh(y, worst, color="#7b2cbf")
    axes[2].axvline(0.0, color="#202020", linewidth=1)
    axes[2].set_yticks([])
    axes[2].set_xlabel("worst case field gain vs Huber-24")
    fig.suptitle(title, fontsize=15)
    fig.savefig(output / "diagnostic.png", dpi=170)
    fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def _write_checksums(output: Path) -> None:
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    _validate_config(config, args.seed_limit)
    git_commit_at_start = _git_commit()
    source_hashes_at_start = _source_manifest(config_path, config)
    source_config = _development_source_config(
        _read_json(ROOT / config["source_t0_config"]),
        args.seed_limit,
    )
    n12_config = _read_json(ROOT / config["source_n1_2_config"])
    output = args.output_dir.resolve()
    if output.exists():
        if not args.replace_output:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    started = time.perf_counter()

    try:
        records, packets, case_to_session, session_rows = n12._prepare_session_records(
            source_config,
            n12_config,
        )
        if any(record.split != "development" for record in records):
            raise RuntimeError("non-development record escaped the construction firewall")
        selectors, calibration_rows = _selector_maps(packets, n12_config)
        expanded, expanded_sessions, stress_by_case, stress_rows = _expanded_stress_records(
            records,
            case_to_session=case_to_session,
            selectors=selectors,
            stress_config=config["flowon_sparse_outlier_stress"],
        )
        matrices, dense_rows = n12._prepare_dense_evaluator(expanded, n12_config)
        norm_cache = n12._norm_cache(expanded, source_config)
        reference_rows = m21._matched_baseline_rows(
            records=expanded,
            source_config=source_config,
            diagnostic_config={"step_safety_factor": 0.9},
            norm_cache=norm_cache,
            steps=[int(config["registered_references"]["projection_step"])],
        )
        reference_lookup = {
            (str(row["case_id"]), str(row["baseline_kind"])): row
            for row in reference_rows
        }
        candidates = _candidate_specs(config)
        metric_rows = _evaluate_candidates(
            records=expanded,
            case_to_session=expanded_sessions,
            stress_by_case=stress_by_case,
            selectors=selectors,
            matrices=matrices,
            references=reference_lookup,
            candidates=candidates,
            config=config,
        )
        aggregates = _aggregate(
            metric_rows,
            float(config["development_gates"]["field_harm_threshold_fraction"]),
        )
        factorial_contrasts = _factorial_contrasts(metric_rows)
        full_screen_complete = args.seed_limit is None
        decisions, selection = _decisions(
            metric_rows,
            aggregates,
            gates=config["development_gates"],
            full_screen_complete=full_screen_complete,
        )
        if not full_screen_complete:
            status = "N1_3_ROBUST_DATA_WHITENING_PILOT_ONLY"
        elif selection is None:
            status = "N1_3_ROBUST_DATA_WHITENING_DEVELOPMENT_NO_GO"
        else:
            status = "N1_3_ROBUST_DATA_WHITENING_DEVELOPMENT_SIGNAL_TO_FREEZE"
        summary = {
            "schema_version": config["report_schema_version"],
            "status": status,
            "evidence_level": config["evidence_level"],
            "development_only": True,
            "ood_constructed_or_evaluated": False,
            "full_screen_complete": full_screen_complete,
            "exact_cli": [sys.executable, *sys.argv],
            "runtime_seconds": time.perf_counter() - started,
            "git_commit_at_start": git_commit_at_start,
            "source_hashes_at_start": source_hashes_at_start,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "session_count": len(session_rows),
            "nominal_case_count": len(records),
            "stress_case_count": len(expanded),
            "candidate_count": len(candidates),
            "metric_row_count": len(metric_rows),
            "aggregate_row_count": len(aggregates),
            "factorial_contrast_row_count": len(factorial_contrasts),
            "decision_count": len(decisions),
            "selection": selection,
            "passed_candidate_count": sum(row["passed"] for row in decisions),
            "decisions": decisions,
            "authorization": {
                "claim_algorithm_superiority": False,
                "claim_real_bost_generalization": False,
                "open_ood": selection is not None and full_screen_complete,
                "train_geometry_guarded_neural_correction": False,
                "continue_robust_base_protocol_development": True,
            },
            "claim_boundary": config["claim_boundary"],
        }
        _write_csv(temporary / "session_rows.csv", session_rows)
        _write_csv(temporary / "calibration_rows.csv", calibration_rows)
        _write_csv(temporary / "stress_manifest_rows.csv", stress_rows)
        _write_csv(temporary / "dense_setup_rows.csv", dense_rows)
        _write_csv(temporary / "reference_rows.csv", reference_rows)
        _write_csv(temporary / "metric_rows.csv", metric_rows)
        _write_csv(temporary / "aggregate_rows.csv", aggregates)
        _write_csv(temporary / "factorial_contrast_rows.csv", factorial_contrasts)
        _write_csv(
            temporary / "decision_rows.csv",
            [
                {
                    "candidate_id": row["candidate_id"],
                    "passed": row["passed"],
                    "known_interface_seed_worst_gain": row[
                        "known_interface_seed_worst_gain"
                    ],
                    "failed_checks_json": _canonical_json(
                        [name for name, passed in row["checks"].items() if not passed]
                    ),
                }
                for row in decisions
            ],
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            "# JACRU N1.3 robust-data whitening development screen\n\n"
            f"- Status: `{status}`\n"
            "- The solver uses Huber measurement fidelity; the older Huber-PDHG "
            "reference uses quadratic measurement fidelity and Huber spatial regularization.\n"
            "- A complete zero/estimated mean x unwhitened/diagonal/isotropic/structured "
            "whitening x quadratic/Huber data-loss factorial is reported.\n"
            "- Sparse flow-on outliers are deterministic synthetic stress, not measured BOST.\n"
            "- OOD was neither constructed nor evaluated by this runner.\n"
            "- Dense whitened-norm setup is outside the matched solve budget and is not deployable.\n"
            "- No algorithm, real-data, generalization, efficiency, or publication claim is authorized.\n",
            encoding="utf-8",
        )
        _plot(aggregates, temporary)
        _write_checksums(temporary)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": status,
                "candidate_count": len(candidates),
                "metric_rows": len(metric_rows),
                "passed_candidates": sum(row["passed"] for row in decisions),
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
