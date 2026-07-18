#!/usr/bin/env python3
"""Run the N2 operator-consistent bend-homotopy mechanism bridge.

This development-only runner compares the previous continuous affine
variational predictor, an analytic operator-consistent bend-homotopy tangent,
one/two-sweep Picard baselines, and the complete 128/256/512-step curved ray.
The 256/512 routes are evaluator-only.  No reserved phantom family is opened,
and no result here authorizes a real-BOST, reconstruction, novelty, or
generalization claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

try:
    from .discrete_rk4_jvp_predictor import predict_discrete_rk4_jvp_residual
    from .field_dependent_ray import relative_l2
    from .operator_consistent_homotopy_predictor import (
        predict_operator_consistent_homotopy_residual,
    )
    from .picard_curved_ray_baseline import trace_picard_curved_rays
    from .run_n2_pvgr_n0_trifidelity_development import _high_route
    from .run_n2_pvgr_n1_variational_development import _build_case_context
    from .shared_straight_state import build_straight_path_state
    from .trajectory_variational_predictor import (
        predict_trajectory_variational_residual,
    )
except ImportError:
    from discrete_rk4_jvp_predictor import predict_discrete_rk4_jvp_residual
    from field_dependent_ray import relative_l2
    from operator_consistent_homotopy_predictor import (
        predict_operator_consistent_homotopy_residual,
    )
    from picard_curved_ray_baseline import trace_picard_curved_rays
    from run_n2_pvgr_n0_trifidelity_development import _high_route
    from run_n2_pvgr_n1_variational_development import _build_case_context
    from shared_straight_state import build_straight_path_state
    from trajectory_variational_predictor import (
        predict_trajectory_variational_residual,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "n2_pvgr_n2_operator_consistent_bridge_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator"
    / "results"
    / "n2_pvgr_n2_operator_consistent_bridge_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _validate_contracts(config: dict[str, Any], source: dict[str, Any]) -> None:
    if config.get("status") != "development_only_no_audit_authorization":
        raise ValueError("config must remain development_only_no_audit_authorization")
    if source.get("dtype") != "float64" or source.get("device") != "cpu":
        raise ValueError("source contract must remain CPU float64")
    reserved = tuple(config["reserved_audit_families_not_opened"])
    if reserved != tuple(source["reserved_audit_families_not_opened"]):
        raise ValueError("reserved-family contract drifted from the source config")
    opened = {str(case["phantom_family"]) for case in source["development_cases"]}
    if opened.intersection(reserved):
        raise ValueError("a reserved audit family was opened")
    steps = (
        int(config["execution_step_count"]),
        int(config["reference_step_count"]),
        int(config["reference_sentinel_step_count"]),
    )
    if not (steps[0] < steps[1] < steps[2]):
        raise ValueError("execution/reference/sentinel steps must increase strictly")
    if steps != (128, 256, 512):
        raise ValueError("this frozen bridge requires the 128/256/512 contract")
    if tuple(int(value) for value in config["picard_sweep_counts"]) != (1, 2):
        raise ValueError("this frozen bridge requires Picard-1 and Picard-2")


def _safe_variance_ratio(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.var(reference.detach(), dim=0, unbiased=True).sum()
    numerator = torch.var(candidate.detach(), dim=0, unbiased=True).sum()
    if float(denominator) <= 1e-30:
        return 0.0 if float(numerator) <= 1e-30 else float("inf")
    return float(numerator / denominator)


def _safe_spearman(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    left = candidate.detach().cpu().numpy().reshape(-1)
    right = reference.detach().cpu().numpy().reshape(-1)
    if np.ptp(left) <= 1e-30 or np.ptp(right) <= 1e-30:
        return 1.0 if np.allclose(left, right) else -1.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else -1.0


def _norm_ratio(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(candidate.detach())
        / torch.linalg.vector_norm(reference.detach()).clamp_min(1e-30)
    )


def _q95_ray_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    errors = torch.linalg.vector_norm(candidate.detach() - reference.detach(), dim=-1)
    return float(torch.quantile(errors, 0.95))


def _straight_output(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: Any,
    source: dict[str, Any],
    *,
    scale: float,
    step_count: int,
) -> torch.Tensor:
    certificate = source["certificate"]
    return build_straight_path_state(
        values,
        states,
        rig,
        difference_step=float(source["difference_step"]),
        refractivity_scale=scale,
        step_count=step_count,
        frustum_half_width_u=float(certificate["frustum_half_width_u"]),
        frustum_half_width_v=float(certificate["frustum_half_width_v"]),
    ).projected_outputs.detach()


def _method_metrics(
    output: torch.Tensor,
    risk: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    medium128: torch.Tensor,
    high128: torch.Tensor,
    high256: torch.Tensor,
) -> dict[str, float]:
    exact_matched_residual = high128 - medium128
    predicted_residual = output - medium128
    remaining_residual = high128 - output
    exact_risk = torch.linalg.vector_norm(exact_matched_residual, dim=-1)
    high_reference_error = relative_l2(high128, high256)
    candidate_reference_error = relative_l2(output, high256)
    high_q95 = _q95_ray_error(high128, high256)
    candidate_q95 = _q95_ray_error(output, high256)
    high_rms = float(torch.sqrt(torch.mean(high256.detach().square())).clamp_min(1e-30))
    return {
        "matched_residual_prediction_relative_l2": relative_l2(
            predicted_residual,
            exact_matched_residual,
        ),
        "corrected_residual_variance_ratio": _safe_variance_ratio(
            remaining_residual,
            exact_matched_residual,
        ),
        "per_ray_risk_spearman": _safe_spearman(risk, exact_risk),
        "valid_ray_fraction": float(torch.mean(valid_mask.to(torch.float64))),
        "candidate_to_high_reference_relative_l2": candidate_reference_error,
        "high_execution_to_high_reference_relative_l2": high_reference_error,
        "candidate_reference_error_to_high_execution_reference_error_ratio": (
            candidate_reference_error / max(high_reference_error, 1e-30)
        ),
        "candidate_q95_reference_error": candidate_q95,
        "high_execution_q95_reference_error": high_q95,
        "candidate_q95_reference_error_to_high_execution_q95_ratio": (
            candidate_q95 / max(high_q95, 1e-30)
        ),
        "candidate_q95_reference_error_normalized_by_high256_rms": (
            candidate_q95 / high_rms
        ),
    }


def _primary_gates(
    metrics: dict[str, float],
    screens: dict[str, float],
) -> dict[str, bool]:
    return {
        "matched_residual_gate_met": metrics[
            "matched_residual_prediction_relative_l2"
        ]
        <= float(screens["maximum_matched_residual_prediction_relative_l2"]),
        "residual_variance_gate_met": metrics[
            "corrected_residual_variance_ratio"
        ]
        <= float(screens["maximum_corrected_residual_variance_ratio"]),
        "risk_spearman_gate_met": metrics["per_ray_risk_spearman"]
        >= float(screens["minimum_per_ray_risk_spearman"]),
        "valid_fraction_gate_met": metrics["valid_ray_fraction"]
        >= float(screens["minimum_valid_ray_fraction"]),
        "absolute_reference_gate_met": metrics[
            "candidate_to_high_reference_relative_l2"
        ]
        <= float(screens["maximum_candidate_to_high_reference_relative_l2"]),
        "reference_no_harm_gate_met": metrics[
            "candidate_reference_error_to_high_execution_reference_error_ratio"
        ]
        <= float(
            screens[
                "maximum_candidate_reference_error_to_high_execution_reference_error_ratio"
            ]
        ),
        "q95_reference_no_harm_gate_met": metrics[
            "candidate_q95_reference_error_to_high_execution_q95_ratio"
        ]
        <= float(
            screens[
                "maximum_candidate_q95_reference_error_to_high_execution_q95_ratio"
            ]
        ),
    }


def _case_scale_bundle(
    case: dict[str, Any],
    source: dict[str, Any],
    config: dict[str, Any],
    *,
    stress: float,
) -> dict[str, Any]:
    values, states, rig = _build_case_context(case, source)
    scale = float(source["base_refractivity_scale"]) * float(stress)
    delta = float(source["difference_step"])
    execution_steps = int(config["execution_step_count"])
    reference_steps = int(config["reference_step_count"])
    sentinel_steps = int(config["reference_sentinel_step_count"])
    screens = config["development_screens"]

    medium128 = _straight_output(
        values,
        states,
        rig,
        source,
        scale=scale,
        step_count=execution_steps,
    )
    continuous = predict_trajectory_variational_residual(
        values,
        states,
        rig,
        refractivity_scale=scale,
        step_count=execution_steps,
        domain_margin=delta,
    )
    continuous_all_valid = bool(torch.all(continuous.valid_mask))
    continuous_output = (
        medium128 + continuous.residual_prediction_uv
        if continuous_all_valid
        else medium128
    )
    candidate = predict_operator_consistent_homotopy_residual(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=execution_steps,
        domain_margin=delta,
    )
    candidate_all_valid = bool(torch.all(candidate.valid_mask))
    candidate_output = candidate.candidate_output_uv if candidate_all_valid else medium128

    picard_results = {
        sweep: trace_picard_curved_rays(
            values,
            states,
            rig,
            difference_step=delta,
            refractivity_scale=scale,
            step_count=execution_steps,
            sweep_count=sweep,
            domain_margin=delta,
            refractive_index_floor=0.500001,
        )
        for sweep in (1, 2)
    }
    teacher = predict_discrete_rk4_jvp_residual(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=execution_steps,
        domain_margin=delta,
    )

    high128, _ = _high_route(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=execution_steps,
        create_graph=False,
    )
    high256, _ = _high_route(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=reference_steps,
        create_graph=False,
    )
    high512, _ = _high_route(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=sentinel_steps,
        create_graph=False,
    )
    medium256 = _straight_output(
        values,
        states,
        rig,
        source,
        scale=scale,
        step_count=reference_steps,
    )
    medium512 = _straight_output(
        values,
        states,
        rig,
        source,
        scale=scale,
        step_count=sentinel_steps,
    )

    method_outputs = {
        "continuous_affine_n1": (
            continuous_output,
            continuous.risk_norm,
            continuous.valid_mask,
            continuous_all_valid,
        ),
        "operator_consistent_homotopy": (
            candidate_output,
            candidate.risk_norm,
            candidate.valid_mask,
            candidate_all_valid,
        ),
        "picard_1": (
            picard_results[1].detector_plane_deflection,
            torch.linalg.vector_norm(
                picard_results[1].detector_plane_deflection - medium128,
                dim=-1,
            ),
            picard_results[1].valid_mask,
            True,
        ),
        "picard_2": (
            picard_results[2].detector_plane_deflection,
            torch.linalg.vector_norm(
                picard_results[2].detector_plane_deflection - medium128,
                dim=-1,
            ),
            picard_results[2].valid_mask,
            True,
        ),
    }
    method_rows = []
    primary_gates: dict[str, bool] | None = None
    for method_id, (output, risk, valid_mask, correction_applied) in method_outputs.items():
        metrics = _method_metrics(
            output,
            risk,
            valid_mask,
            medium128=medium128,
            high128=high128,
            high256=high256,
        )
        gates = (
            _primary_gates(metrics, screens)
            if method_id == "operator_consistent_homotopy"
            else {}
        )
        if method_id == "operator_consistent_homotopy":
            base_relative_l2 = relative_l2(candidate.base_output_uv, medium128)
            gates["base_output_consistency_gate_met"] = base_relative_l2 <= float(
                screens["maximum_base_output_relative_l2"]
            )
            metrics["base_output_relative_l2"] = base_relative_l2
            primary_gates = gates
        method_rows.append(
            {
                "case_id": str(case["id"]),
                "phantom_family": str(case["phantom_family"]),
                "phantom_seed": int(case["phantom_seed"]),
                "dimensionless_stress_multiplier": float(stress),
                "refractivity_scale": scale,
                "method_id": method_id,
                "ray_count": len(states),
                "step_count": execution_steps,
                "correction_applied": bool(correction_applied),
                "metrics": metrics,
                "gates": gates,
                "all_primary_gates_pass": bool(gates) and all(gates.values()),
            }
        )
    if primary_gates is None:
        raise RuntimeError("primary candidate row was not constructed")

    teacher_metrics = {
        "output_relative_l2": relative_l2(
            candidate.residual_prediction_uv,
            teacher.residual_prediction_uv,
        ),
        "position_tangent_relative_l2": relative_l2(
            candidate.delta_positions,
            teacher.delta_positions,
        ),
        "direction_tangent_relative_l2": relative_l2(
            candidate.delta_directions,
            teacher.delta_directions,
        ),
        "teacher_valid_ray_fraction": float(
            torch.mean(teacher.valid_mask.to(torch.float64))
        ),
    }
    teacher_gates = {
        "output_gate_met": teacher_metrics["output_relative_l2"]
        <= float(screens["maximum_teacher_output_relative_l2"]),
        "position_gate_met": teacher_metrics["position_tangent_relative_l2"]
        <= float(screens["maximum_teacher_position_tangent_relative_l2"]),
        "direction_gate_met": teacher_metrics["direction_tangent_relative_l2"]
        <= float(screens["maximum_teacher_direction_tangent_relative_l2"]),
        "teacher_valid_gate_met": teacher_metrics["teacher_valid_ray_fraction"]
        >= float(screens["minimum_valid_ray_fraction"]),
    }
    sentinel_metrics = {
        "high256_to_high512_output_relative_l2": relative_l2(high256, high512),
        "matched_residual_256_to_512_relative_l2": relative_l2(
            high256 - medium256,
            high512 - medium512,
        ),
    }
    sentinel_gates = {
        "output_convergence_gate_met": sentinel_metrics[
            "high256_to_high512_output_relative_l2"
        ]
        <= float(screens["maximum_high256_to_high512_output_relative_l2"]),
        "matched_residual_convergence_gate_met": sentinel_metrics[
            "matched_residual_256_to_512_relative_l2"
        ]
        <= float(screens["maximum_matched_residual_256_to_512_relative_l2"]),
    }
    return {
        "case": case,
        "values": values,
        "states": states,
        "rig": rig,
        "scale": scale,
        "method_rows": method_rows,
        "primary_all_pass": all(primary_gates.values()),
        "teacher_row": {
            "case_id": str(case["id"]),
            "dimensionless_stress_multiplier": float(stress),
            "metrics": teacher_metrics,
            "gates": teacher_gates,
            "all_gates_pass": all(teacher_gates.values()),
        },
        "sentinel_row": {
            "case_id": str(case["id"]),
            "dimensionless_stress_multiplier": float(stress),
            "metrics": sentinel_metrics,
            "gates": sentinel_gates,
            "all_gates_pass": all(sentinel_gates.values()),
        },
        "query_accounting": {
            "operator_consistent_homotopy": candidate.query_accounting,
            "teacher_discrete_jvp": teacher.query_accounting,
            "picard_1": picard_results[1].query_accounting.as_dict(),
            "picard_2": picard_results[2].query_accounting.as_dict(),
            "high128": {
                "logical_scalar_grid_point_queries": 35
                * len(states)
                * execution_steps,
                "interpolation_dispatches": 35 * execution_steps,
                "exact_high_evaluations": 1,
            },
        },
    }


def _timed_samples(
    closures: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup_repeats: int,
    measured_repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    checksum = 0.0
    for closure in closures.values():
        for _ in range(int(warmup_repeats)):
            checksum += float(closure().detach().sum())
    labels = [
        label
        for _ in range(int(measured_repeats))
        for label in closures
    ]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(labels)
    samples = {label: [] for label in closures}
    for label in labels:
        started = time.perf_counter_ns()
        checksum += float(closures[label]().detach().sum())
        samples[label].append((time.perf_counter_ns() - started) * 1e-9)
    if not np.isfinite(checksum):
        raise RuntimeError("timing closure produced a non-finite checksum")
    return samples


def _timing_bundle(
    case: dict[str, Any],
    source: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    values, states, rig = _build_case_context(case, source)
    scale = float(source["base_refractivity_scale"]) * float(
        config["timing_stress_multiplier"]
    )
    delta = float(source["difference_step"])
    steps = int(config["execution_step_count"])

    def continuous_closure() -> torch.Tensor:
        medium = _straight_output(
            values,
            states,
            rig,
            source,
            scale=scale,
            step_count=steps,
        )
        prediction = predict_trajectory_variational_residual(
            values,
            states,
            rig,
            refractivity_scale=scale,
            step_count=steps,
            domain_margin=delta,
        )
        if not bool(torch.all(prediction.valid_mask)):
            return medium
        return medium + prediction.residual_prediction_uv

    def candidate_closure() -> torch.Tensor:
        result = predict_operator_consistent_homotopy_residual(
            values,
            states,
            rig,
            difference_step=delta,
            refractivity_scale=scale,
            step_count=steps,
            domain_margin=delta,
        )
        return result.candidate_output_uv if bool(torch.all(result.valid_mask)) else result.base_output_uv

    def picard_closure(sweep_count: int) -> Callable[[], torch.Tensor]:
        def closure() -> torch.Tensor:
            return trace_picard_curved_rays(
                values,
                states,
                rig,
                difference_step=delta,
                refractivity_scale=scale,
                step_count=steps,
                sweep_count=sweep_count,
                domain_margin=delta,
                refractive_index_floor=0.500001,
            ).detector_plane_deflection

        return closure

    def high_closure() -> torch.Tensor:
        output, _ = _high_route(
            values,
            states,
            rig,
            difference_step=delta,
            refractivity_scale=scale,
            step_count=steps,
            create_graph=False,
        )
        return output

    closures = {
        "continuous_affine_n1": continuous_closure,
        "operator_consistent_homotopy": candidate_closure,
        "picard_1": picard_closure(1),
        "picard_2": picard_closure(2),
        "high128": high_closure,
    }
    timing = config["timing"]
    samples = _timed_samples(
        closures,
        warmup_repeats=int(timing["warmup_repeats"]),
        measured_repeats=int(timing["measured_repeats"]),
        seed=int(timing["interleave_seed"])
        + int(hashlib.sha256(str(case["id"]).encode()).hexdigest()[:8], 16),
    )
    high_p10 = float(np.quantile(samples["high128"], 0.10))
    rows = []
    for method_id, values_seconds in samples.items():
        row = {
            "case_id": str(case["id"]),
            "dimensionless_stress_multiplier": float(
                config["timing_stress_multiplier"]
            ),
            "method_id": method_id,
            "sample_count": len(values_seconds),
            "p10_seconds": float(np.quantile(values_seconds, 0.10)),
            "p50_seconds": float(np.quantile(values_seconds, 0.50)),
            "p90_seconds": float(np.quantile(values_seconds, 0.90)),
            "candidate_p90_to_high128_p10_wall_time_ratio": (
                float(np.quantile(values_seconds, 0.90)) / max(high_p10, 1e-30)
            ),
        }
        if method_id == "operator_consistent_homotopy":
            row["timing_gate_met"] = row[
                "candidate_p90_to_high128_p10_wall_time_ratio"
            ] <= float(
                config["development_screens"][
                    "maximum_candidate_p90_to_high128_p10_wall_time_ratio"
                ]
            )
        rows.append(row)
    return rows


def _flatten_method_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key not in {"metrics", "gates"}}
    flat.update(row["metrics"])
    flat.update(row["gates"])
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(
    path: Path,
    method_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    sentinel_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    screens: dict[str, float],
) -> None:
    methods = [
        "continuous_affine_n1",
        "operator_consistent_homotopy",
        "picard_1",
        "picard_2",
    ]
    labels = ["N1 affine", "OCBH", "Picard-1", "Picard-2"]
    colors = ["#9a5b4a", "#286f69", "#426a9c", "#7d6098"]
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    def method_values(metric: str, method: str) -> list[float]:
        return [
            float(row["metrics"][metric])
            for row in method_rows
            if row["method_id"] == method
        ]

    x = np.arange(len(methods))
    matched = [max(method_values("matched_residual_prediction_relative_l2", method)) for method in methods]
    axes[0, 0].bar(x, matched, color=colors)
    axes[0, 0].axhline(
        float(screens["maximum_matched_residual_prediction_relative_l2"]),
        color="#b44b3f",
        linestyle="--",
        linewidth=1.5,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 0].set_title("worst matched residual error")

    no_harm = [
        max(
            method_values(
                "candidate_reference_error_to_high_execution_reference_error_ratio",
                method,
            )
        )
        for method in methods
    ]
    axes[0, 1].bar(x, no_harm, color=colors)
    axes[0, 1].axhline(
        float(
            screens[
                "maximum_candidate_reference_error_to_high_execution_reference_error_ratio"
            ]
        ),
        color="#b44b3f",
        linestyle="--",
        linewidth=1.5,
    )
    axes[0, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 1].set_title("worst H256 no-harm ratio")

    tails = [
        max(
            method_values(
                "candidate_q95_reference_error_to_high_execution_q95_ratio",
                method,
            )
        )
        for method in methods
    ]
    axes[0, 2].bar(x, tails, color=colors)
    axes[0, 2].axhline(
        float(
            screens[
                "maximum_candidate_q95_reference_error_to_high_execution_q95_ratio"
            ]
        ),
        color="#b44b3f",
        linestyle="--",
        linewidth=1.5,
    )
    axes[0, 2].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 2].set_title("worst per-ray Q95 no-harm ratio")

    teacher_metrics = [
        "output_relative_l2",
        "position_tangent_relative_l2",
        "direction_tangent_relative_l2",
    ]
    teacher_labels = ["output", "position", "direction"]
    teacher_max = [
        max(float(row["metrics"][metric]) for row in teacher_rows)
        for metric in teacher_metrics
    ]
    axes[1, 0].bar(np.arange(3), teacher_max, color="#286f69")
    axes[1, 0].axhline(
        float(screens["maximum_teacher_output_relative_l2"]),
        color="#b44b3f",
        linestyle="--",
        linewidth=1.5,
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(np.arange(3), teacher_labels)
    axes[1, 0].set_title("analytic OCBH vs forward-mode teacher")

    output_sentinel = [
        float(row["metrics"]["high256_to_high512_output_relative_l2"])
        for row in sentinel_rows
    ]
    residual_sentinel = [
        float(row["metrics"]["matched_residual_256_to_512_relative_l2"])
        for row in sentinel_rows
    ]
    axes[1, 1].scatter(output_sentinel, residual_sentinel, color="#426a9c", s=44)
    axes[1, 1].axvline(
        float(screens["maximum_high256_to_high512_output_relative_l2"]),
        color="#b44b3f",
        linestyle="--",
    )
    axes[1, 1].axhline(
        float(screens["maximum_matched_residual_256_to_512_relative_l2"]),
        color="#b44b3f",
        linestyle="--",
    )
    axes[1, 1].set_xlabel("H256 vs H512 output relative-L2")
    axes[1, 1].set_ylabel("matched residual relative-L2")
    axes[1, 1].set_title("reference sentinel")

    timing_methods = methods + ["high128"]
    timing_labels = labels + ["H128"]
    timing_colors = colors + ["#50565c"]
    timing_max = [
        max(
            float(row["candidate_p90_to_high128_p10_wall_time_ratio"])
            for row in timing_rows
            if row["method_id"] == method
        )
        for method in timing_methods
    ]
    axes[1, 2].bar(np.arange(len(timing_methods)), timing_max, color=timing_colors)
    axes[1, 2].axhline(
        float(screens["maximum_candidate_p90_to_high128_p10_wall_time_ratio"]),
        color="#b44b3f",
        linestyle="--",
        linewidth=1.5,
    )
    axes[1, 2].set_xticks(
        np.arange(len(timing_methods)),
        timing_labels,
        rotation=18,
        ha="right",
    )
    axes[1, 2].set_title("worst p90 / H128 p10 wall ratio")
    figure.suptitle(
        "N2-PVGR-N2 operator-consistent mechanism bridge: development only",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    source_path = _resolve_from_root(str(config["source_config"]))
    previous_path = _resolve_from_root(str(config["previous_result"]))
    source = _read_json(source_path)
    previous = _read_json(previous_path)
    _validate_contracts(config, source)
    if previous.get("machine_decision") != "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION":
        raise ValueError("previous N1 result is not the frozen 7/9 NO-AUTH decision")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    method_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    sentinel_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    primary_pass_count = 0
    for case in source["development_cases"]:
        for stress in source["dimensionless_stress_scale_multipliers"]:
            bundle = _case_scale_bundle(
                case,
                source,
                config,
                stress=float(stress),
            )
            method_rows.extend(bundle["method_rows"])
            teacher_rows.append(bundle["teacher_row"])
            sentinel_rows.append(bundle["sentinel_row"])
            query_rows.append(
                {
                    "case_id": str(case["id"]),
                    "dimensionless_stress_multiplier": float(stress),
                    **bundle["query_accounting"],
                }
            )
            primary_pass_count += int(bundle["primary_all_pass"])

    timing_rows = []
    for case in source["development_cases"]:
        timing_rows.extend(_timing_bundle(case, source, config))

    primary_rows = [
        row for row in method_rows if row["method_id"] == "operator_consistent_homotopy"
    ]
    primary_required = len(primary_rows)
    teacher_pass_count = sum(int(row["all_gates_pass"]) for row in teacher_rows)
    sentinel_pass_count = sum(int(row["all_gates_pass"]) for row in sentinel_rows)
    candidate_timing_rows = [
        row
        for row in timing_rows
        if row["method_id"] == "operator_consistent_homotopy"
    ]
    timing_pass_count = sum(int(row["timing_gate_met"]) for row in candidate_timing_rows)
    screens = config["development_screens"]
    point_query_pass_count = 0
    point_query_ratios = []
    for query_row in query_rows:
        candidate_queries = int(
            query_row["operator_consistent_homotopy"][
                "logical_scalar_grid_point_queries"
            ]
        )
        high_queries = int(query_row["high128"]["logical_scalar_grid_point_queries"])
        ratio = candidate_queries / high_queries
        point_query_ratios.append(ratio)
        point_query_pass_count += int(
            ratio
            <= float(
                screens[
                    "maximum_candidate_to_high128_logical_point_query_ratio"
                ]
            )
        )

    all_bridge_screens_pass = (
        primary_pass_count == primary_required
        and teacher_pass_count == len(teacher_rows)
        and sentinel_pass_count == len(sentinel_rows)
        and timing_pass_count == len(candidate_timing_rows)
        and point_query_pass_count == len(query_rows)
    )
    machine_decision = (
        "MECHANISM_BRIDGE_SIGNAL_ONLY_96_CELL_RECONSTRUCTION_AND_REAL_DATA_GATES_CLOSED"
        if all_bridge_screens_pass
        else "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION"
    )
    figure_name = "n2_pvgr_n2_operator_consistent_bridge.png"
    _write_figure(
        output_dir / figure_name,
        method_rows,
        teacher_rows,
        sentinel_rows,
        timing_rows,
        screens,
    )

    result = {
        "schema": str(config["schema"]),
        "candidate_id": str(config["candidate_id"]),
        "machine_decision": machine_decision,
        "development_bridge_authorization": False,
        "reserved_audit_authorization": False,
        "real_data_authorization": False,
        "paper_claim_authorization": False,
        "primary_screen_pass_count": primary_pass_count,
        "primary_screen_required_count": primary_required,
        "teacher_screen_pass_count": teacher_pass_count,
        "teacher_screen_required_count": len(teacher_rows),
        "reference_sentinel_pass_count": sentinel_pass_count,
        "reference_sentinel_required_count": len(sentinel_rows),
        "timing_screen_pass_count": timing_pass_count,
        "timing_screen_required_count": len(candidate_timing_rows),
        "point_query_screen_pass_count": point_query_pass_count,
        "point_query_screen_required_count": len(query_rows),
        "maximum_candidate_to_high128_logical_point_query_ratio": max(
            point_query_ratios
        ),
        "execution_step_count": int(config["execution_step_count"]),
        "reference_step_count": int(config["reference_step_count"]),
        "reference_sentinel_step_count": int(
            config["reference_sentinel_step_count"]
        ),
        "development_screens": screens,
        "method_rows": method_rows,
        "teacher_rows": teacher_rows,
        "reference_sentinel_rows": sentinel_rows,
        "timing_rows": timing_rows,
        "query_accounting_rows": query_rows,
        "timing_environment": {
            "platform": "local Mac CPU",
            "dtype": "torch.float64",
            "device": "cpu",
            "candidate_and_high_share_process": True,
            "peak_rss_measured": False,
            "field_jvp_vjp_timed": False,
            "host_scalar_synchronizations_instrumented": False,
            "wall_time_claim_boundary": (
                "current Python/PyTorch implementation only; not an algorithmic "
                "complexity or hardware-general speed claim"
            ),
        },
        "scientific_positioning": {
            "old_n1_role": (
                "continuous affine/Newton-like straight-path correction; not the "
                "exact discrete bend-homotopy derivative"
            ),
            "operator_consistent_role": (
                "analytic exact equivalent of the local discrete bend JVP at "
                "epsilon=0, using the same central-difference operator"
            ),
            "picard_role": (
                "strong successive-approximation baseline with final-path output "
                "re-evaluation; not a novelty claim"
            ),
            "reference_role": "H256 evaluator with H512 development sentinel",
        },
        "claim_boundary": (
            "The bridge uses only three original development rigs and two opened "
            "synthetic phantom families. Passing it proves neither a 96-cell "
            "factorial result, differentiable reconstruction, comparison with "
            "DeepONet/FNO, real BOST validity, generalization, nor novelty."
        ),
        "mandatory_next_gates": [
            "96-cell grouped-by-field factorial development screen",
            "cell/topology/caustic and finite-difference remainder certificate",
            "field JVP/VJP dot tests and differentiable reconstruction benchmark",
            "peak RSS and host synchronization instrumentation",
            "cone-ray aperture baseline",
            "reserved-family audit only after every development gate is frozen",
            "real OERF geometry, noise, and independent physical endpoint",
            "TDBOST distortion-module overlap review with He Yuanzhe",
        ],
        "reserved_audit_families_not_opened": list(
            config["reserved_audit_families_not_opened"]
        ),
        "figures": [figure_name],
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "metrics.csv", [_flatten_method_row(row) for row in method_rows])
    _write_csv(
        output_dir / "teacher_metrics.csv",
        [
            {
                "case_id": row["case_id"],
                "dimensionless_stress_multiplier": row[
                    "dimensionless_stress_multiplier"
                ],
                **row["metrics"],
                **row["gates"],
                "all_gates_pass": row["all_gates_pass"],
            }
            for row in teacher_rows
        ],
    )
    _write_csv(
        output_dir / "reference_sentinel.csv",
        [
            {
                "case_id": row["case_id"],
                "dimensionless_stress_multiplier": row[
                    "dimensionless_stress_multiplier"
                ],
                **row["metrics"],
                **row["gates"],
                "all_gates_pass": row["all_gates_pass"],
            }
            for row in sentinel_rows
        ],
    )
    _write_csv(output_dir / "timing.csv", timing_rows)
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# N2-PVGR-N2 operator-consistent mechanism bridge",
                "",
                f"- Machine decision: `{machine_decision}`",
                f"- Primary OCBH rows: `{primary_pass_count}/{primary_required}`",
                f"- Analytic/forward-JVP teacher checks: `{teacher_pass_count}/{len(teacher_rows)}`",
                f"- H256/H512 reference sentinels: `{sentinel_pass_count}/{len(sentinel_rows)}`",
                f"- OCBH timing rigs: `{timing_pass_count}/{len(candidate_timing_rows)}`",
                f"- Logical point-query rows: `{point_query_pass_count}/{len(query_rows)}`",
                "- Picard-1/2 are mandatory strong baselines, not claimed innovations.",
                "- No reserved family, real data, reconstruction, DeepONet/FNO, or paper claim is authorized.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest_inputs = {
        "config": config_path,
        "config_snapshot": output_dir / "config_snapshot.json",
        "source_config": source_path,
        "previous_result": previous_path,
        "runner": Path(__file__),
        "operator_consistent_predictor": ROOT
        / "demo_t16_operator/operator_consistent_homotopy_predictor.py",
        "discrete_jvp_teacher": ROOT
        / "demo_t16_operator/discrete_rk4_jvp_predictor.py",
        "picard_baseline": ROOT / "demo_t16_operator/picard_curved_ray_baseline.py",
        "result": result_path,
        "metrics": output_dir / "metrics.csv",
        "teacher_metrics": output_dir / "teacher_metrics.csv",
        "reference_sentinel": output_dir / "reference_sentinel.csv",
        "timing": output_dir / "timing.csv",
        "figure": output_dir / figure_name,
        "summary": output_dir / "summary.md",
        "report": ROOT
        / "docs/n2_pvgr_n2_operator_consistent_bridge_2026-07-18.md",
        "test_discrete_jvp_teacher": ROOT
        / "demo_t16_operator/test_discrete_rk4_jvp_predictor.py",
        "test_operator_consistent_predictor": ROOT
        / "demo_t16_operator/test_operator_consistent_homotopy_predictor.py",
        "test_picard_baseline": ROOT
        / "demo_t16_operator/test_picard_curved_ray_baseline.py",
        "test_runner": ROOT
        / "demo_t16_operator/test_run_n2_pvgr_n2_operator_consistent_bridge.py",
    }
    manifest = {
        "schema": "n2-pvgr-n2-operator-consistent-manifest-1.1",
        "files": {
            key: {
                "path": _manifest_path(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in manifest_inputs.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(json.dumps({
        "machine_decision": result["machine_decision"],
        "primary": [
            result["primary_screen_pass_count"],
            result["primary_screen_required_count"],
        ],
        "teacher": [
            result["teacher_screen_pass_count"],
            result["teacher_screen_required_count"],
        ],
        "sentinel": [
            result["reference_sentinel_pass_count"],
            result["reference_sentinel_required_count"],
        ],
        "timing": [
            result["timing_screen_pass_count"],
            result["timing_screen_required_count"],
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
