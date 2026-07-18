#!/usr/bin/env python3
"""Development audit for a first-order BOST trajectory defect correction.

The candidate predicts the curved-minus-straight measurement residual from a
frozen straight path.  This runner checks matched-discretization accuracy,
variance reduction, per-ray risk ranking, reference-aware error, and the full
candidate closure wall time.  It opens development phantoms only and cannot
authorize real BOST, reconstruction, generalization, novelty, or paper claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

try:
    from .analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from .automatic_discrete_multifidelity import trace_sample_variance
    from .field_dependent_ray import relative_l2, sample_pupil_sobol
    from .run_n2_pvgr_n0_trifidelity_development import (
        _high_route,
        _rig_from_case,
        stable_seed,
    )
    from .shared_straight_state import build_straight_path_state
    from .trajectory_variational_predictor import (
        predict_trajectory_variational_residual,
    )
except ImportError:
    from analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from automatic_discrete_multifidelity import trace_sample_variance
    from field_dependent_ray import relative_l2, sample_pupil_sobol
    from run_n2_pvgr_n0_trifidelity_development import (
        _high_route,
        _rig_from_case,
        stable_seed,
    )
    from shared_straight_state import build_straight_path_state
    from trajectory_variational_predictor import (
        predict_trajectory_variational_residual,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "n2_pvgr_n1_variational_development_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator"
    / "results"
    / "n2_pvgr_n1_variational_development_v1"
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


def _validate_contracts(
    config: dict[str, Any],
    source: dict[str, Any],
    convergence: dict[str, Any],
) -> tuple[int, int, tuple[str, ...]]:
    execution_step = int(config["matched_execution_step_count"])
    reference_step = int(config["reference_step_count"])
    if execution_step < 2 or reference_step <= execution_step:
        raise ValueError("execution/reference step counts are inconsistent")
    if execution_step != int(convergence["accepted_execution_step_count"]):
        raise ValueError("execution step differs from the residual convergence audit")
    if reference_step != int(convergence["reference_step_count"]):
        raise ValueError("reference step differs from the residual convergence audit")
    if convergence["machine_decision"] != (
        "RESIDUAL_TARGET_128_ACCEPTED_DEVELOPMENT_ONLY"
    ):
        raise RuntimeError("the residual convergence audit has not accepted execution")

    reserved = tuple(str(value) for value in config["reserved_audit_families_not_opened"])
    source_reserved = tuple(
        str(value) for value in source["reserved_audit_families_not_opened"]
    )
    convergence_reserved = tuple(
        str(value) for value in convergence["reserved_audit_families_not_opened"]
    )
    if set(reserved) != set(source_reserved) or set(reserved) != set(
        convergence_reserved
    ):
        raise ValueError("reserved-family contracts do not match")
    development = {str(case["phantom_family"]) for case in source["development_cases"]}
    if development & set(reserved):
        raise RuntimeError("a reserved audit family appears in development cases")

    timing_stress = float(config["timing_stress_multiplier"])
    available_stress = {
        float(value) for value in source["dimensionless_stress_scale_multipliers"]
    }
    if timing_stress not in available_stress:
        raise ValueError("timing stress is absent from the source development contract")
    return execution_step, reference_step, reserved


def _safe_relative_variance(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    denominator = trace_sample_variance(reference)
    numerator = trace_sample_variance(candidate)
    if denominator <= 1e-30:
        return 0.0 if numerator <= 1e-30 else float("inf")
    return float(numerator / denominator)


def _norm_ratio(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(candidate)
        / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    )


def _safe_spearman(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate_array = candidate.detach().cpu().numpy()
    reference_array = reference.detach().cpu().numpy()
    if np.ptp(candidate_array) <= 1e-30 or np.ptp(reference_array) <= 1e-30:
        return 1.0 if np.allclose(candidate_array, reference_array) else -1.0
    statistic = float(spearmanr(candidate_array, reference_array).statistic)
    return statistic if np.isfinite(statistic) else -1.0


def _build_case_context(
    case: dict[str, Any],
    source: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    spec = make_analytic_phantom(
        family=str(case["phantom_family"]),
        seed=int(case["phantom_seed"]),
    )
    values = analytic_phantom_grid(
        spec,
        grid_shape=tuple(int(value) for value in source["grid_shape_zyx"]),
        dtype=torch.float64,
        device="cpu",
    ).field
    states = sample_pupil_sobol(
        int(source["population_count"]),
        seed=stable_seed(
            int(source["seed_roles"]["population_state_base"]),
            case["id"],
        ),
    )
    return values, states, _rig_from_case(case)


def _candidate_output(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: Any,
    source: dict[str, Any],
    *,
    scale: float,
    step_count: int,
) -> tuple[torch.Tensor, Any, Any]:
    certificate = source["certificate"]
    medium_state = build_straight_path_state(
        values,
        states,
        rig,
        difference_step=float(source["difference_step"]),
        refractivity_scale=scale,
        step_count=step_count,
        frustum_half_width_u=float(certificate["frustum_half_width_u"]),
        frustum_half_width_v=float(certificate["frustum_half_width_v"]),
    )
    prediction = predict_trajectory_variational_residual(
        values,
        states,
        rig,
        refractivity_scale=scale,
        step_count=step_count,
        domain_margin=float(source["difference_step"]),
    )
    corrected = medium_state.projected_outputs + prediction.residual_prediction_uv
    return corrected.detach(), medium_state, prediction


def _case_scale_row(
    case: dict[str, Any],
    source: dict[str, Any],
    screens: dict[str, float],
    *,
    values: torch.Tensor,
    states: torch.Tensor,
    rig: Any,
    stress: float,
    execution_step: int,
    reference_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scale = float(source["base_refractivity_scale"]) * float(stress)
    candidate, medium_state, prediction = _candidate_output(
        values,
        states,
        rig,
        source,
        scale=scale,
        step_count=execution_step,
    )
    medium = medium_state.projected_outputs.detach()
    high_execution, _ = _high_route(
        values,
        states,
        rig,
        difference_step=float(source["difference_step"]),
        refractivity_scale=scale,
        step_count=execution_step,
        create_graph=False,
    )
    high_reference, _ = _high_route(
        values,
        states,
        rig,
        difference_step=float(source["difference_step"]),
        refractivity_scale=scale,
        step_count=reference_step,
        create_graph=False,
    )
    exact_matched_residual = high_execution - medium
    predicted_residual = prediction.residual_prediction_uv
    corrected_matched_residual = exact_matched_residual - predicted_residual
    exact_reference_residual = high_reference - medium
    corrected_reference_residual = high_reference - candidate
    exact_per_ray_risk = torch.linalg.vector_norm(exact_matched_residual, dim=-1)
    valid_fraction = float(torch.mean(prediction.valid_mask.to(torch.float64)))
    candidate_to_reference = relative_l2(candidate, high_reference)
    high_execution_to_reference = relative_l2(high_execution, high_reference)

    metrics = {
        "matched_residual_prediction_relative_l2": relative_l2(
            predicted_residual,
            exact_matched_residual,
        ),
        "corrected_residual_variance_ratio": _safe_relative_variance(
            corrected_matched_residual,
            exact_matched_residual,
        ),
        "per_ray_risk_spearman": _safe_spearman(
            prediction.risk_norm,
            exact_per_ray_risk,
        ),
        "valid_ray_fraction": valid_fraction,
        "candidate_to_reference_residual_relative_l2": _norm_ratio(
            corrected_reference_residual,
            exact_reference_residual,
        ),
        "high_execution_to_reference_residual_relative_l2": _norm_ratio(
            high_reference - high_execution,
            exact_reference_residual,
        ),
        "candidate_to_high_reference_relative_l2": candidate_to_reference,
        "high_execution_to_high_reference_relative_l2": high_execution_to_reference,
        "candidate_reference_error_to_high_execution_reference_error_ratio": (
            candidate_to_reference / max(high_execution_to_reference, 1e-30)
        ),
        "medium_to_high_reference_relative_l2": relative_l2(
            medium,
            high_reference,
        ),
        "reference_residual_norm_to_high_output_norm": float(
            torch.linalg.vector_norm(exact_reference_residual)
            / torch.linalg.vector_norm(high_reference).clamp_min(1e-30)
        ),
    }
    gates = {
        "matched_residual_prediction_gate_met": metrics[
            "matched_residual_prediction_relative_l2"
        ]
        <= float(screens["maximum_matched_residual_prediction_relative_l2"]),
        "corrected_residual_variance_gate_met": metrics[
            "corrected_residual_variance_ratio"
        ]
        <= float(screens["maximum_corrected_residual_variance_ratio"]),
        "per_ray_risk_spearman_gate_met": metrics["per_ray_risk_spearman"]
        >= float(screens["minimum_per_ray_risk_spearman"]),
        "valid_ray_fraction_gate_met": metrics["valid_ray_fraction"]
        >= float(screens["minimum_valid_ray_fraction"]),
        "candidate_high_reference_absolute_gate_met": metrics[
            "candidate_to_high_reference_relative_l2"
        ]
        <= float(screens["maximum_candidate_to_high_reference_relative_l2"]),
        "candidate_reference_no_harm_gate_met": metrics[
            "candidate_reference_error_to_high_execution_reference_error_ratio"
        ]
        <= float(
            screens[
                "maximum_candidate_reference_error_to_high_execution_reference_error_ratio"
            ]
        ),
    }
    row = {
        "case_id": str(case["id"]),
        "phantom_family": str(case["phantom_family"]),
        "phantom_seed": int(case["phantom_seed"]),
        "dimensionless_stress_multiplier": float(stress),
        "refractivity_scale": scale,
        "ray_count": len(states),
        "matched_execution_step_count": execution_step,
        "reference_step_count": reference_step,
        "gradient_contract": (
            "central-difference M/H with automatic-coordinate Hessian predictor"
        ),
        "metrics": metrics,
        "gates": gates,
        "all_non_timing_screens_pass": all(gates.values()),
    }
    scatter = {
        "case_id": str(case["id"]),
        "stress": float(stress),
        "predicted_risk": prediction.risk_norm.detach().cpu().numpy(),
        "exact_risk": exact_per_ray_risk.detach().cpu().numpy(),
    }
    return row, scatter


def _timed_samples(
    candidate_closure: Callable[[], torch.Tensor],
    high_closure: Callable[[], torch.Tensor],
    *,
    warmup_repeats: int,
    measured_repeats: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    if warmup_repeats < 0 or measured_repeats < 5:
        raise ValueError("timing requires nonnegative warmup and at least five repeats")
    for _ in range(warmup_repeats):
        candidate_closure()
        high_closure()
    labels = ["candidate", "high"] * measured_repeats
    random.Random(seed).shuffle(labels)
    samples = {"candidate": [], "high": []}
    for label in labels:
        closure = candidate_closure if label == "candidate" else high_closure
        started = time.perf_counter()
        output = closure()
        elapsed = time.perf_counter() - started
        if torch.any(~torch.isfinite(output)):
            raise RuntimeError(f"{label} timing closure returned a non-finite output")
        samples[label].append(float(elapsed))
    return samples["candidate"], samples["high"]


def _timing_row(
    case: dict[str, Any],
    source: dict[str, Any],
    config: dict[str, Any],
    screens: dict[str, float],
    *,
    values: torch.Tensor,
    states: torch.Tensor,
    rig: Any,
    execution_step: int,
) -> dict[str, Any]:
    stress = float(config["timing_stress_multiplier"])
    scale = float(source["base_refractivity_scale"]) * stress

    def candidate_closure() -> torch.Tensor:
        output, _, _ = _candidate_output(
            values,
            states,
            rig,
            source,
            scale=scale,
            step_count=execution_step,
        )
        return output

    def high_closure() -> torch.Tensor:
        output, _ = _high_route(
            values,
            states,
            rig,
            difference_step=float(source["difference_step"]),
            refractivity_scale=scale,
            step_count=execution_step,
            create_graph=False,
        )
        return output

    timing = config["timing"]
    candidate_times, high_times = _timed_samples(
        candidate_closure,
        high_closure,
        warmup_repeats=int(timing["warmup_repeats"]),
        measured_repeats=int(timing["measured_repeats"]),
        seed=stable_seed(int(timing["interleave_seed"]), case["id"]),
    )
    candidate_quantiles = {
        "p10_seconds": float(np.quantile(candidate_times, 0.1)),
        "p50_seconds": float(np.quantile(candidate_times, 0.5)),
        "p90_seconds": float(np.quantile(candidate_times, 0.9)),
    }
    high_quantiles = {
        "p10_seconds": float(np.quantile(high_times, 0.1)),
        "p50_seconds": float(np.quantile(high_times, 0.5)),
        "p90_seconds": float(np.quantile(high_times, 0.9)),
    }
    ratio = candidate_quantiles["p90_seconds"] / high_quantiles["p10_seconds"]
    threshold = float(
        screens["maximum_candidate_p90_to_full_high_p10_wall_time_ratio"]
    )
    return {
        "case_id": str(case["id"]),
        "dimensionless_stress_multiplier": stress,
        "warmup_repeats": int(timing["warmup_repeats"]),
        "measured_repeats_per_route": int(timing["measured_repeats"]),
        "randomly_interleaved": True,
        "candidate_closure": (
            "shared central-difference medium state plus automatic-Hessian "
            "variational prediction and correction"
        ),
        "full_high_closure": "central-difference curved RK4 plus path integral",
        "candidate": candidate_quantiles,
        "full_high": high_quantiles,
        "candidate_p90_to_full_high_p10_wall_time_ratio": ratio,
        "wall_time_gate_met": ratio <= threshold,
    }


def _query_accounting(ray_count: int, step_count: int) -> dict[str, Any]:
    medium_points = 7 * ray_count * step_count
    predictor_points = ray_count * (2 * step_count + 1)
    high_points = 35 * ray_count * step_count
    return {
        "warning": (
            "point counts are not wall-time equivalents; automatic gradient/Hessian "
            "VJPs and sequential dispatch depth are reported separately"
        ),
        "shared_medium": {
            "logical_scalar_grid_point_queries": medium_points,
            "vectorized_interpolation_forward_batches": 1,
        },
        "variational_predictor": {
            "path_point_count": predictor_points,
            "vectorized_interpolation_forward_batches": 1,
            "coordinate_vjp_batches": 4,
            "coordinate_vjp_point_visits": 4 * predictor_points,
            "explicit_hessian_components_per_point": 9,
        },
        "full_high": {
            "logical_scalar_grid_point_queries": high_points,
            "sequential_rhs_depth": 4 * step_count + 1,
            "interpolation_dispatches": 7 * (4 * step_count + 1),
        },
    }


def _write_figure(
    rows: list[dict[str, Any]],
    scatter_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    path: Path,
    *,
    screens: dict[str, float],
) -> None:
    colors = {
        "smooth_narrow_aperture": "#176b67",
        "wrinkled_wide_aperture": "#a34e3f",
        "smooth_wide_aperture": "#405a8a",
    }
    markers = {1.0: "o", 3.0: "s", 10.0: "^"}
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    positive_values: list[float] = []
    for item in scatter_rows:
        predicted = np.maximum(item["predicted_risk"], 1e-16)
        exact = np.maximum(item["exact_risk"], 1e-16)
        positive_values.extend(predicted.tolist())
        positive_values.extend(exact.tolist())
        axes[0, 0].scatter(
            exact,
            predicted,
            s=14,
            alpha=0.55,
            marker=markers[item["stress"]],
            color=colors.get(item["case_id"], "#4c5e65"),
        )
    lower = max(min(positive_values), 1e-16)
    upper = max(positive_values)
    axes[0, 0].plot([lower, upper], [lower, upper], "--", color="#20282c")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("exact matched |H-M| per ray")
    axes[0, 0].set_ylabel("variational predicted risk")
    axes[0, 0].set_title("directional risk tracks the exact residual")

    labels = [
        f"{row['case_id'].replace('_', ' ')}\n{row['dimensionless_stress_multiplier']:g}x"
        for row in rows
    ]
    x = np.arange(len(rows))
    matched_error = [
        row["metrics"]["matched_residual_prediction_relative_l2"] for row in rows
    ]
    axes[0, 1].bar(
        x,
        matched_error,
        color=[colors.get(row["case_id"], "#4c5e65") for row in rows],
    )
    axes[0, 1].axhline(
        float(screens["maximum_matched_residual_prediction_relative_l2"]),
        color="#20282c",
        linestyle="--",
    )
    axes[0, 1].set_xticks(x, labels, rotation=38, ha="right", fontsize=7)
    axes[0, 1].set_ylabel("relative-L2 normalized by matched H-M")
    axes[0, 1].set_title("matched 128-step prediction error")

    variance = [row["metrics"]["corrected_residual_variance_ratio"] for row in rows]
    axes[0, 2].bar(
        x,
        variance,
        color=[colors.get(row["case_id"], "#4c5e65") for row in rows],
    )
    axes[0, 2].axhline(
        float(screens["maximum_corrected_residual_variance_ratio"]),
        color="#20282c",
        linestyle="--",
    )
    axes[0, 2].set_xticks(x, labels, rotation=38, ha="right", fontsize=7)
    axes[0, 2].set_ylabel("Var(H - M - prediction) / Var(H - M)")
    axes[0, 2].set_title("development control-residual variance")

    reference_no_harm = [
        row["metrics"][
            "candidate_reference_error_to_high_execution_reference_error_ratio"
        ]
        for row in rows
    ]
    axes[1, 0].bar(
        x,
        reference_no_harm,
        color=[colors.get(row["case_id"], "#4c5e65") for row in rows],
    )
    axes[1, 0].axhline(
        float(
            screens[
                "maximum_candidate_reference_error_to_high_execution_reference_error_ratio"
            ]
        ),
        color="#20282c",
        linestyle="--",
    )
    axes[1, 0].set_xticks(x, labels, rotation=38, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("candidate reference error / high-128 reference error")
    axes[1, 0].set_title("mixed-discretization no-harm gate")

    width = 0.38
    candidate_reference = [
        row["metrics"]["candidate_to_high_reference_relative_l2"] for row in rows
    ]
    high_reference = [
        row["metrics"]["high_execution_to_high_reference_relative_l2"] for row in rows
    ]
    axes[1, 1].bar(
        x - width / 2,
        candidate_reference,
        width,
        label="M128 + prediction",
    )
    axes[1, 1].bar(
        x + width / 2,
        high_reference,
        width,
        label="full H128",
    )
    axes[1, 1].axhline(
        float(screens["maximum_candidate_to_high_reference_relative_l2"]),
        color="#20282c",
        linestyle="--",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(x, labels, rotation=38, ha="right", fontsize=7)
    axes[1, 1].set_ylabel("relative-L2 to full H256")
    axes[1, 1].set_title("absolute reference accuracy")
    axes[1, 1].legend(fontsize=8)

    timing_x = np.arange(len(timing_rows))
    timing_ratio = [
        item["candidate_p90_to_full_high_p10_wall_time_ratio"]
        for item in timing_rows
    ]
    axes[1, 2].bar(
        timing_x,
        timing_ratio,
        color=[colors.get(item["case_id"], "#4c5e65") for item in timing_rows],
    )
    axes[1, 2].axhline(
        float(screens["maximum_candidate_p90_to_full_high_p10_wall_time_ratio"]),
        color="#20282c",
        linestyle="--",
    )
    axes[1, 2].set_xticks(
        timing_x,
        [item["case_id"].replace("_", "\n") for item in timing_rows],
        fontsize=8,
    )
    axes[1, 2].set_ylabel("candidate p90 / full-high p10 wall time")
    axes[1, 2].set_title("full candidate closure wall time")

    figure.suptitle(
        "N2-PVGR-N1 variational defect correction: development evidence only",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    source_path = _resolve_from_root(str(config["source_config"]))
    convergence_path = _resolve_from_root(str(config["residual_convergence_audit"]))
    source = _read_json(source_path)
    convergence = _read_json(convergence_path)
    execution_step, reference_step, reserved = _validate_contracts(
        config,
        source,
        convergence,
    )
    screens = {key: float(value) for key, value in config["development_screens"].items()}

    rows: list[dict[str, Any]] = []
    scatter_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for case in source["development_cases"]:
        values, states, rig = _build_case_context(case, source)
        for stress in source["dimensionless_stress_scale_multipliers"]:
            row, scatter = _case_scale_row(
                case,
                source,
                screens,
                values=values,
                states=states,
                rig=rig,
                stress=float(stress),
                execution_step=execution_step,
                reference_step=reference_step,
            )
            rows.append(row)
            scatter_rows.append(scatter)
        timing_rows.append(
            _timing_row(
                case,
                source,
                config,
                screens,
                values=values,
                states=states,
                rig=rig,
                execution_step=execution_step,
            )
        )

    timing_by_case = {item["case_id"]: item for item in timing_rows}
    for row in rows:
        timing_gate = timing_by_case[row["case_id"]]["wall_time_gate_met"]
        row["timing_gate_met"] = timing_gate
        row["all_development_screens_pass"] = (
            row["all_non_timing_screens_pass"] and timing_gate
        )
    pass_count = sum(row["all_development_screens_pass"] for row in rows)
    all_pass = pass_count == len(rows)
    machine_decision = str(
        config[
            "hard_conclusion_if_all_development_screens_pass"
            if all_pass
            else "hard_conclusion_otherwise"
        ]
    )
    ray_count = int(source["population_count"])
    result = {
        "schema": str(config["schema"]),
        "candidate_id": str(config["candidate_id"]),
        "machine_decision": machine_decision,
        "claim_boundary": (
            "synthetic development measurement-operator evidence only; no reserved "
            "family, real BOST, reconstruction, generalization, novelty, or paper "
            "authorization"
        ),
        "scientific_positioning": (
            "first-order frozen-path trajectory defect correction for a BOST "
            "measurement operator; not first Born, first differentiable ray tracing, "
            "or a new multifidelity estimator"
        ),
        "matched_execution_step_count": execution_step,
        "reference_step_count": reference_step,
        "case_scale_count": len(rows),
        "development_screen_pass_count": pass_count,
        "reserved_audit_families_not_opened": list(reserved),
        "development_screens": screens,
        "query_accounting": _query_accounting(ray_count, execution_step),
        "timing_environment": {
            "device": "cpu",
            "dtype": "float64",
            "torch_num_threads": torch.get_num_threads(),
            "timing_scope": "forward measurement operator only; no JVP or VJP timing",
        },
        "timing_rows": timing_rows,
        "rows": rows,
        "figures": ["n2_pvgr_n1_variational_development.png"],
        "mandatory_next_gates": [
            "discrete RK4-JVP and normalization-consistent baseline",
            "one- and two-step Picard historical baselines",
            "cone-ray finite-aperture separation",
            "sealed nondevelopment families",
            "field-reconstruction and JVP/VJP end-to-end cost",
            "mentor confirmation of TDBOST distortion-correction overlap",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    snapshot_path = output_dir / "config_snapshot.json"
    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.md"
    figure_path = output_dir / result["figures"][0]
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "case_id",
        "stress_multiplier",
        "matched_residual_prediction_relative_l2",
        "corrected_residual_variance_ratio",
        "per_ray_risk_spearman",
        "valid_ray_fraction",
        "candidate_to_reference_residual_relative_l2",
        "candidate_to_high_reference_relative_l2",
        "high_execution_to_high_reference_relative_l2",
        "candidate_reference_error_to_high_execution_reference_error_ratio",
        "candidate_p90_to_full_high_p10_wall_time_ratio",
        "all_development_screens_pass",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            timing_ratio = timing_by_case[row["case_id"]][
                "candidate_p90_to_full_high_p10_wall_time_ratio"
            ]
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "stress_multiplier": row["dimensionless_stress_multiplier"],
                    "matched_residual_prediction_relative_l2": metrics[
                        "matched_residual_prediction_relative_l2"
                    ],
                    "corrected_residual_variance_ratio": metrics[
                        "corrected_residual_variance_ratio"
                    ],
                    "per_ray_risk_spearman": metrics["per_ray_risk_spearman"],
                    "valid_ray_fraction": metrics["valid_ray_fraction"],
                    "candidate_to_reference_residual_relative_l2": metrics[
                        "candidate_to_reference_residual_relative_l2"
                    ],
                    "candidate_to_high_reference_relative_l2": metrics[
                        "candidate_to_high_reference_relative_l2"
                    ],
                    "high_execution_to_high_reference_relative_l2": metrics[
                        "high_execution_to_high_reference_relative_l2"
                    ],
                    "candidate_reference_error_to_high_execution_reference_error_ratio": metrics[
                        "candidate_reference_error_to_high_execution_reference_error_ratio"
                    ],
                    "candidate_p90_to_full_high_p10_wall_time_ratio": timing_ratio,
                    "all_development_screens_pass": row[
                        "all_development_screens_pass"
                    ],
                }
            )
    _write_figure(
        rows,
        scatter_rows,
        timing_rows,
        figure_path,
        screens=screens,
    )
    summary_path.write_text(
        "\n".join(
            (
                "# N2-PVGR-N1 variational development audit",
                "",
                f"- Machine decision: `{machine_decision}`.",
                f"- Development screens: {pass_count}/{len(rows)} case x stress rows.",
                f"- Matched execution/reference steps: {execution_step}/{reference_step}.",
                "- Candidate timing includes medium state, Hessian predictor, and correction.",
                "- Reserved families remain closed.",
                "- This is not real-data, reconstruction, generalization, novelty, or paper evidence.",
                "",
            )
        ),
        encoding="utf-8",
    )
    generated = (result_path, snapshot_path, csv_path, summary_path, figure_path)
    manifest = {
        "schema": str(config["schema"]),
        "source_sha256": {
            "runner": _sha256(Path(__file__)),
            "config": _sha256(config_path),
            "source_config": _sha256(source_path),
            "residual_convergence_audit": _sha256(convergence_path),
            "shared_straight_state": _sha256(
                ROOT / "demo_t16_operator/shared_straight_state.py"
            ),
            "trajectory_variational_predictor": _sha256(
                ROOT / "demo_t16_operator/trajectory_variational_predictor.py"
            ),
            "field_dependent_ray": _sha256(
                ROOT / "demo_t16_operator/field_dependent_ray.py"
            ),
        },
        "files": {path.name: _sha256(path) for path in generated},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "machine_decision": result["machine_decision"],
                "screen": (
                    f"{result['development_screen_pass_count']}/"
                    f"{result['case_scale_count']}"
                ),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
