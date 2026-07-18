#!/usr/bin/env python3
"""Run the N2-ADRC-N1 development-only two-level BOST mechanism pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

try:
    from .analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from .automatic_discrete_multifidelity import (
        AUTOMATIC_DISCRETE_MF_SCHEMA,
        SyntheticRayRig,
        evaluate_automatic_discrete_pair,
        evaluate_automatic_projected,
        evaluate_discrete_projected,
        joint_state_geometry,
        optimal_two_level_allocation,
        sample_joint_pupil_path_sobol,
        two_level_efficiency,
        two_level_mean,
    )
except ImportError:
    from analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from automatic_discrete_multifidelity import (
        AUTOMATIC_DISCRETE_MF_SCHEMA,
        SyntheticRayRig,
        evaluate_automatic_discrete_pair,
        evaluate_automatic_projected,
        evaluate_discrete_projected,
        joint_state_geometry,
        optimal_two_level_allocation,
        sample_joint_pupil_path_sobol,
        two_level_efficiency,
        two_level_mean,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "n2_adrc_n1_development_pilot_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "n2_adrc_n1_development_pilot_v1"
RESULT_SCHEMA = "n2-adrc-n1-development-result-1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(int(base)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1e-30)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _rig_from_case(case: dict[str, Any]) -> SyntheticRayRig:
    values = case["rig"]
    return SyntheticRayRig(
        rig_id=str(case["id"]),
        view_angle_degrees=float(values["view_angle_degrees"]),
        detector_u=float(values["detector_u"]),
        detector_z=float(values["detector_z"]),
        aperture_radius=float(values["aperture_radius"]),
        path_half_length=float(values["path_half_length"]),
        cone_u=float(values["cone_u"]),
        cone_z=float(values["cone_z"]),
        bend=float(values["bend"]),
    )


def _benchmark(
    function: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    *,
    sample_count: int,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, float]:
    checksum = 0.0
    for _ in range(int(warmup_repeats)):
        output = function()
        tensors = output if isinstance(output, tuple) else (output,)
        checksum += sum(float(value.detach().sum()) for value in tensors)
    elapsed = []
    for _ in range(int(measured_repeats)):
        started = time.perf_counter_ns()
        output = function()
        tensors = output if isinstance(output, tuple) else (output,)
        checksum += sum(float(value.detach().sum()) for value in tensors)
        elapsed.append(time.perf_counter_ns() - started)
    if not np.isfinite(checksum):
        raise RuntimeError("timing route produced a non-finite checksum")
    per_sample = np.asarray(elapsed, dtype=np.float64) / float(sample_count)
    return {
        "p10_nanoseconds_per_sample": float(np.quantile(per_sample, 0.1)),
        "median_nanoseconds_per_sample": float(np.median(per_sample)),
        "p90_nanoseconds_per_sample": float(np.quantile(per_sample, 0.9)),
        "batch_median_seconds": float(np.median(elapsed) * 1e-9),
    }


def _component_correlations(high: np.ndarray, low: np.ndarray) -> list[float]:
    correlations = []
    for component in range(high.shape[1]):
        high_column = high[:, component]
        low_column = low[:, component]
        if np.std(high_column, ddof=1) <= 0.0 or np.std(low_column, ddof=1) <= 0.0:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(high_column, low_column)[0, 1]))
    return correlations


def _smooth_direction(values: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(values.shape, generator=generator, dtype=values.dtype)
    smoothed = functional.avg_pool3d(
        noise[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    scale = 0.2 * float(torch.linalg.vector_norm(values)) / max(
        float(torch.linalg.vector_norm(smoothed)),
        1e-30,
    )
    return smoothed * scale


def _draw_estimates(
    low: np.ndarray,
    high: np.ndarray,
    *,
    low_count: int,
    residual_count: int,
    high_count: int,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    population = len(high)
    low_indices = rng.integers(population, size=(replicates, low_count))
    residual_indices = rng.integers(population, size=(replicates, residual_count))
    high_indices = rng.integers(population, size=(replicates, high_count))
    two_level = low[low_indices].mean(axis=1) + (
        high[residual_indices] - low[residual_indices]
    ).mean(axis=1)
    high_only = high[high_indices].mean(axis=1)
    return {
        "two_level": two_level,
        "high_only": high_only,
        "low_indices": low_indices,
        "residual_indices": residual_indices,
        "high_indices": high_indices,
    }


def _bootstrap_mse_gain(
    high_squared_error: np.ndarray,
    two_level_squared_error: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(int(seed))
    count = len(high_squared_error)
    ratios = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        selected = rng.integers(count, size=count)
        ratios[index] = float(
            np.mean(high_squared_error[selected])
            / max(float(np.mean(two_level_squared_error[selected])), 1e-30)
        )
    return (
        float(np.mean(high_squared_error) / max(float(np.mean(two_level_squared_error)), 1e-30)),
        float(np.quantile(ratios, 0.025)),
        float(np.quantile(ratios, 0.975)),
    )


def _resampled_two_level(
    values: np.ndarray,
    paired_high: np.ndarray,
    paired_low: np.ndarray,
    low_indices: np.ndarray,
    residual_indices: np.ndarray,
) -> np.ndarray:
    return values[low_indices].mean(axis=1) + (
        paired_high[residual_indices] - paired_low[residual_indices]
    ).mean(axis=1)


def _nonlinear_gradient_check(
    *,
    low: np.ndarray,
    high: np.ndarray,
    low_direction: np.ndarray,
    high_direction: np.ndarray,
    low_count: int,
    residual_count: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    first = _draw_estimates(
        low,
        high,
        low_count=low_count,
        residual_count=residual_count,
        high_count=2,
        replicates=replicates,
        seed=stable_seed(seed, "replica_a"),
    )
    second = _draw_estimates(
        low,
        high,
        low_count=low_count,
        residual_count=residual_count,
        high_count=2,
        replicates=replicates,
        seed=stable_seed(seed, "replica_b"),
    )
    mu = np.mean(high, axis=0)
    jacobian_direction = np.mean(high_direction, axis=0)
    observation = 0.85 * mu
    exact_gradient = float(np.dot(jacobian_direction, mu - observation))
    mu_a = first["two_level"]
    mu_b = second["two_level"]
    jac_a = _resampled_two_level(
        low_direction,
        high_direction,
        low_direction,
        first["low_indices"],
        first["residual_indices"],
    )
    jac_b = _resampled_two_level(
        low_direction,
        high_direction,
        low_direction,
        second["low_indices"],
        second["residual_indices"],
    )
    plugin = np.sum(jac_a * (mu_a - observation), axis=1)
    double_sample = 0.5 * (
        np.sum(jac_a * (mu_b - observation), axis=1)
        + np.sum(jac_b * (mu_a - observation), axis=1)
    )
    scale = max(abs(exact_gradient), float(np.std(plugin, ddof=1)), 1e-30)
    return {
        "exact_directional_loss_gradient": exact_gradient,
        "plugin_mean_directional_loss_gradient": float(np.mean(plugin)),
        "double_sample_mean_directional_loss_gradient": float(np.mean(double_sample)),
        "plugin_absolute_bias": float(np.mean(plugin) - exact_gradient),
        "double_sample_absolute_bias": float(np.mean(double_sample) - exact_gradient),
        "plugin_bias_over_scale": float((np.mean(plugin) - exact_gradient) / scale),
        "double_sample_bias_over_scale": float(
            (np.mean(double_sample) - exact_gradient) / scale
        ),
        "plugin_standard_error": float(np.std(plugin, ddof=1) / math.sqrt(replicates)),
        "double_sample_standard_error": float(
            np.std(double_sample, ddof=1) / math.sqrt(replicates)
        ),
        "contract": "double_sample_is_unbiased_in_expectation_not_exact_per_run",
    }


def _fixed_state_derivative_check(
    *,
    values: torch.Tensor,
    direction: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    difference_step: float,
    low_indices: np.ndarray,
    residual_indices: np.ndarray,
) -> dict[str, float]:
    low_state = states[torch.as_tensor(low_indices, dtype=torch.long)]
    residual_state = states[torch.as_tensor(residual_indices, dtype=torch.long)]

    def estimate(grid: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        low_only = evaluate_automatic_projected(
            grid,
            low_state,
            rig,
            create_graph=create_graph,
        )
        paired_low, paired_high = evaluate_automatic_discrete_pair(
            grid,
            residual_state,
            rig,
            difference_step=difference_step,
            create_graph=create_graph,
        )
        return two_level_mean(low_only, paired_high, paired_low)

    differentiable_values = values.detach().clone().requires_grad_(True)
    output = estimate(differentiable_values, create_graph=True)
    jvp_components = []
    for component in range(len(output)):
        gradient = torch.autograd.grad(
            output[component],
            differentiable_values,
            retain_graph=True,
        )[0]
        jvp_components.append(torch.sum(gradient * direction))
    automatic_jvp = torch.stack(jvp_components)
    epsilon = 1e-5
    plus = estimate(values + epsilon * direction, create_graph=False)
    minus = estimate(values - epsilon * direction, create_graph=False)
    finite_difference_jvp = (plus - minus) / (2.0 * epsilon)
    jvp_error = relative_l2(
        automatic_jvp.detach().numpy(),
        finite_difference_jvp.detach().numpy(),
    )
    cotangent = torch.as_tensor([0.6, -0.8], dtype=values.dtype)
    vjp = torch.autograd.grad(
        torch.sum(output * cotangent),
        differentiable_values,
    )[0]
    lhs = float(torch.sum(vjp * direction))
    rhs = float(torch.sum(cotangent * automatic_jvp))
    return {
        "fixed_state_jvp_relative_error": jvp_error,
        "fixed_state_vjp_dot_relative_error": abs(lhs - rhs)
        / max(abs(lhs), abs(rhs), 1e-30),
        "finite_difference_epsilon": epsilon,
        "trajectory_sensitivity_audited": False,
        "trajectory_contract": "prescribed_geometry_only",
        "state_index_sha256": array_sha256(low_indices, residual_indices),
    }


def _stencil_diagnostics(
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    grid_shape: tuple[int, int, int],
) -> dict[str, float]:
    points, _, _ = joint_state_geometry(states, rig, high_geometry=True)
    sizes = torch.as_tensor(
        [grid_shape[2] - 1, grid_shape[1] - 1, grid_shape[0] - 1],
        dtype=points.dtype,
    )
    crossing = torch.zeros(len(points), dtype=torch.bool)
    for axis in range(3):
        offset = torch.zeros_like(points)
        offset[:, axis] = float(difference_step)
        left = torch.floor(0.5 * (points - offset + 1.0) * sizes)
        right = torch.floor(0.5 * (points + offset + 1.0) * sizes)
        crossing = crossing | torch.any(left != right, dim=1)
    margin = 1.0 - torch.max(torch.abs(points), dim=1).values
    return {
        "central_stencil_cell_crossing_fraction": float(crossing.to(torch.float64).mean()),
        "minimum_domain_margin": float(torch.min(margin)),
        "support_crossing_audited": False,
        "mask_or_frustum_crossing_audited": False,
    }


def _case_result(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case["id"])
    grid_shape = tuple(int(value) for value in config["grid_shape_zyx"])
    difference_step = float(config["difference_step"])
    seeds = config["seed_roles"]
    spec = make_analytic_phantom(
        family=str(case["phantom_family"]),
        seed=int(case["phantom_seed"]),
    )
    values = analytic_phantom_grid(
        spec,
        grid_shape=grid_shape,
        dtype=torch.float64,
        device="cpu",
    ).field
    rig = _rig_from_case(case)
    calibration_states = sample_joint_pupil_path_sobol(
        int(config["calibration_population_count"]),
        seed=stable_seed(seeds["calibration_state_base"], case_id),
    )
    reference_states = sample_joint_pupil_path_sobol(
        int(config["reference_population_count"]),
        seed=stable_seed(seeds["reference_state_base"], case_id),
    )
    low_calibration, high_calibration = evaluate_automatic_discrete_pair(
        values,
        calibration_states,
        rig,
        difference_step=difference_step,
        create_graph=False,
    )
    low_reference, high_reference = evaluate_automatic_discrete_pair(
        values,
        reference_states,
        rig,
        difference_step=difference_step,
        create_graph=False,
    )
    timing_config = config["timing"]
    timing_low = _benchmark(
        lambda: evaluate_automatic_projected(
            values,
            calibration_states,
            rig,
            create_graph=False,
        ),
        sample_count=len(calibration_states),
        warmup_repeats=int(timing_config["warmup_repeats"]),
        measured_repeats=int(timing_config["measured_repeats"]),
    )
    timing_high = _benchmark(
        lambda: evaluate_discrete_projected(
            values,
            calibration_states,
            rig,
            difference_step=difference_step,
        ),
        sample_count=len(calibration_states),
        warmup_repeats=int(timing_config["warmup_repeats"]),
        measured_repeats=int(timing_config["measured_repeats"]),
    )
    timing_pair = _benchmark(
        lambda: evaluate_automatic_discrete_pair(
            values,
            calibration_states,
            rig,
            difference_step=difference_step,
            create_graph=False,
        ),
        sample_count=len(calibration_states),
        warmup_repeats=int(timing_config["warmup_repeats"]),
        measured_repeats=int(timing_config["measured_repeats"]),
    )
    efficiency = two_level_efficiency(
        high_calibration,
        low_calibration,
        high_cost=timing_high["median_nanoseconds_per_sample"],
        low_cost=timing_low["median_nanoseconds_per_sample"],
        residual_cost=timing_pair["median_nanoseconds_per_sample"],
    )
    conservative_timing_efficiency = two_level_efficiency(
        high_calibration,
        low_calibration,
        high_cost=timing_high["p10_nanoseconds_per_sample"],
        low_cost=timing_low["p90_nanoseconds_per_sample"],
        residual_cost=timing_pair["p90_nanoseconds_per_sample"],
    )
    total_cost = (
        int(config["matched_cost_high_equivalent_samples"])
        * timing_high["median_nanoseconds_per_sample"]
    )
    low_count, residual_count, consumed_cost = optimal_two_level_allocation(
        total_cost=total_cost,
        low_variance=efficiency.low_trace_variance,
        residual_variance=efficiency.residual_trace_variance,
        low_cost=efficiency.low_cost,
        residual_cost=efficiency.residual_cost,
    )
    high_count = max(2, int(math.floor(total_cost / efficiency.high_cost)))
    low_numpy = low_reference.detach().numpy()
    high_numpy = high_reference.detach().numpy()
    truth = np.mean(high_numpy, axis=0)
    estimates = _draw_estimates(
        low_numpy,
        high_numpy,
        low_count=low_count,
        residual_count=residual_count,
        high_count=high_count,
        replicates=int(config["estimator_replicates"]),
        seed=stable_seed(seeds["resampling_base"], case_id),
    )
    high_squared_error = np.sum((estimates["high_only"] - truth) ** 2, axis=1)
    two_level_squared_error = np.sum((estimates["two_level"] - truth) ** 2, axis=1)
    gain, gain_lower, gain_upper = _bootstrap_mse_gain(
        high_squared_error,
        two_level_squared_error,
        replicates=int(config["bootstrap_replicates"]),
        seed=stable_seed(seeds["bootstrap_base"], case_id),
    )
    direction = _smooth_direction(
        values,
        seed=stable_seed(seeds["direction_base"], case_id),
    )
    low_direction, high_direction = evaluate_automatic_discrete_pair(
        direction,
        reference_states,
        rig,
        difference_step=difference_step,
        create_graph=False,
    )
    nonlinear = _nonlinear_gradient_check(
        low=low_numpy,
        high=high_numpy,
        low_direction=low_direction.detach().numpy(),
        high_direction=high_direction.detach().numpy(),
        low_count=low_count,
        residual_count=residual_count,
        replicates=int(config["estimator_replicates"]),
        seed=stable_seed(seeds["resampling_base"], case_id, "nonlinear"),
    )
    derivative_config = config["fixed_state_derivative_check"]
    derivative_rng = np.random.default_rng(
        stable_seed(seeds["direction_base"], case_id, "fixed_state_indices")
    )
    derivative_low_indices = derivative_rng.integers(
        len(reference_states),
        size=int(derivative_config["low_count"]),
        dtype=np.int64,
    )
    derivative_residual_indices = derivative_rng.integers(
        len(reference_states),
        size=int(derivative_config["residual_count"]),
        dtype=np.int64,
    )
    derivative = _fixed_state_derivative_check(
        values=values,
        direction=direction,
        states=reference_states,
        rig=rig,
        difference_step=difference_step,
        low_indices=derivative_low_indices,
        residual_indices=derivative_residual_indices,
    )
    diagnostics = _stencil_diagnostics(
        reference_states,
        rig,
        difference_step=difference_step,
        grid_shape=grid_shape,
    )
    finite_reference_sensitivity = relative_l2(
        np.mean(high_numpy[: len(high_numpy) // 2], axis=0),
        truth,
    )
    screens = config["development_promotion_screens"]
    case_screen = bool(
        efficiency.predicted_efficiency_gain
        >= float(screens["minimum_predicted_measured_cost_gain"])
        and conservative_timing_efficiency.predicted_efficiency_gain
        >= float(screens["minimum_conservative_timing_efficiency_gain"])
        and gain >= float(screens["minimum_empirical_mse_gain"])
        and gain_lower >= float(screens["minimum_empirical_gain_bootstrap_lower_95"])
        and finite_reference_sensitivity
        <= float(screens["maximum_reference_half_to_full_relative_l2"])
        and derivative["fixed_state_jvp_relative_error"]
        <= float(screens["maximum_fixed_state_jvp_relative_error"])
        and derivative["fixed_state_vjp_dot_relative_error"]
        <= float(screens["maximum_fixed_state_vjp_dot_relative_error"])
    )
    correlations = _component_correlations(high_numpy, low_numpy)
    return {
        "case_id": case_id,
        "phantom_family": str(case["phantom_family"]),
        "phantom_seed": int(case["phantom_seed"]),
        "rig": case["rig"],
        "target_mean": truth.tolist(),
        "low_mean": np.mean(low_numpy, axis=0).tolist(),
        "low_to_high_mean_relative_l2": relative_l2(np.mean(low_numpy, axis=0), truth),
        "reference_half_to_full_relative_l2": finite_reference_sensitivity,
        "high_trace_variance": efficiency.high_trace_variance,
        "low_trace_variance": efficiency.low_trace_variance,
        "residual_trace_variance": efficiency.residual_trace_variance,
        "residual_to_high_variance_ratio": efficiency.residual_trace_variance
        / max(efficiency.high_trace_variance, 1e-30),
        "component_correlations": correlations,
        "minimum_component_correlation": float(np.nanmin(correlations)),
        "timing": {
            "low_automatic": timing_low,
            "high_discrete": timing_high,
            "paired_residual": timing_pair,
            "machine_specific": True,
            "primitive_ledger": {
                "low": {"field_queries": 1, "coordinate_vjp": 1},
                "high": {"field_queries": 6, "coordinate_vjp": 0},
                "paired": {"field_queries": 7, "coordinate_vjp": 1},
            },
        },
        "predicted_measured_cost_efficiency_gain": efficiency.predicted_efficiency_gain,
        "predicted_conservative_timing_efficiency_gain": (
            conservative_timing_efficiency.predicted_efficiency_gain
        ),
        "matched_cost": {
            "budget_nanoseconds_proxy": total_cost,
            "low_count": low_count,
            "residual_count": residual_count,
            "high_only_count": high_count,
            "two_level_consumed_nanoseconds_proxy": consumed_cost,
            "high_only_mse": float(np.mean(high_squared_error)),
            "two_level_mse": float(np.mean(two_level_squared_error)),
            "empirical_mse_gain": gain,
            "empirical_mse_gain_bootstrap_95": [gain_lower, gain_upper],
            "bootstrap_unit": "complete_estimator_replicate",
        },
        "forward_sampling": {
            "two_level_empirical_mean_relative_l2": relative_l2(
                np.mean(estimates["two_level"], axis=0),
                truth,
            ),
            "identity": "E[low]+E[high-low]=E[high]",
            "unbiased_only_for_declared_finite_population": True,
        },
        "nonlinear_loss_gradient": nonlinear,
        "fixed_state_derivative": derivative,
        "stencil_diagnostics": diagnostics,
        "development_case_screen_met": case_screen,
        "limitations": [
            "finite_population_target_not_continuous_integral",
            "gridded_morphology_proxy_not_experimental_data",
            "prescribed_bend_has_no_field_dependent_trajectory_sensitivity",
            "timing_is_cpu_batch_and_machine_specific",
            "no_reconstruction_or_operator_learning_model_trained",
        ],
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("status") != "development_only_no_audit_authorization":
        raise ValueError("pilot config must remain development-only")
    if config.get("hard_conclusion") != "DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION":
        raise ValueError("hard conclusion cannot be relaxed in a development run")
    if str(config.get("device")) != "cpu" or str(config.get("dtype")) != "float64":
        raise ValueError("v1 pilot is frozen to CPU float64")
    cases = [_case_result(config, case) for case in config["development_cases"]]
    minimum = int(
        config["development_promotion_screens"][
            "minimum_cases_passing_efficiency_screen"
        ]
    )
    passed = sum(bool(case["development_case_screen_met"]) for case in cases)
    return {
        "schema": RESULT_SCHEMA,
        "primitive_schema": AUTOMATIC_DISCRETE_MF_SCHEMA,
        "candidate_id": str(config["candidate_id"]),
        "machine_decision": str(config["hard_conclusion"]),
        "promotion_screen_met": bool(passed >= minimum),
        "promotion_screen_meaning": "may_design_unseen_audit_only",
        "case_screen_count": passed,
        "case_count": len(cases),
        "declared_target": str(config["target_contract"]),
        "scope": str(config["scope_contract"]),
        "reserved_audit_families_not_opened": list(
            config["reserved_audit_families_not_opened"]
        ),
        "cases": cases,
        "global_limitations": [
            "development_results_cannot_authorize_a_paper_claim",
            "reserved_audit_families_were_not_evaluated",
            "no_interface_mask_frustum_or_support_crossing_audit",
            "no_field_dependent_curved_ray_vjp",
            "no_real_bost_data_or_external_physics_endpoint",
        ],
    }


def _metric_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in result["cases"]:
        matched = case["matched_cost"]
        derivative = case["fixed_state_derivative"]
        nonlinear = case["nonlinear_loss_gradient"]
        timing = case["timing"]
        rows.append(
            {
                "case_id": case["case_id"],
                "phantom_family": case["phantom_family"],
                "bend": case["rig"]["bend"],
                "low_to_high_mean_relative_l2": case[
                    "low_to_high_mean_relative_l2"
                ],
                "reference_half_to_full_relative_l2": case[
                    "reference_half_to_full_relative_l2"
                ],
                "residual_to_high_variance_ratio": case[
                    "residual_to_high_variance_ratio"
                ],
                "minimum_component_correlation": case[
                    "minimum_component_correlation"
                ],
                "low_ns_per_sample": timing["low_automatic"][
                    "median_nanoseconds_per_sample"
                ],
                "high_ns_per_sample": timing["high_discrete"][
                    "median_nanoseconds_per_sample"
                ],
                "paired_ns_per_sample": timing["paired_residual"][
                    "median_nanoseconds_per_sample"
                ],
                "predicted_measured_cost_efficiency_gain": case[
                    "predicted_measured_cost_efficiency_gain"
                ],
                "predicted_conservative_timing_efficiency_gain": case[
                    "predicted_conservative_timing_efficiency_gain"
                ],
                "empirical_mse_gain": matched["empirical_mse_gain"],
                "empirical_mse_gain_bootstrap_lower_95": matched[
                    "empirical_mse_gain_bootstrap_95"
                ][0],
                "empirical_mse_gain_bootstrap_upper_95": matched[
                    "empirical_mse_gain_bootstrap_95"
                ][1],
                "fixed_state_jvp_relative_error": derivative[
                    "fixed_state_jvp_relative_error"
                ],
                "fixed_state_vjp_dot_relative_error": derivative[
                    "fixed_state_vjp_dot_relative_error"
                ],
                "plugin_gradient_bias_over_scale": nonlinear[
                    "plugin_bias_over_scale"
                ],
                "double_sample_gradient_bias_over_scale": nonlinear[
                    "double_sample_bias_over_scale"
                ],
                "development_case_screen_met": case[
                    "development_case_screen_met"
                ],
                "machine_decision": result["machine_decision"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, result: dict[str, Any]) -> None:
    labels = [case["case_id"].replace("_", "\n") for case in result["cases"]]
    positions = np.arange(len(labels))
    residual = [case["residual_to_high_variance_ratio"] for case in result["cases"]]
    predicted = [
        case["predicted_measured_cost_efficiency_gain"] for case in result["cases"]
    ]
    empirical = [case["matched_cost"]["empirical_mse_gain"] for case in result["cases"]]
    plugin = [
        abs(case["nonlinear_loss_gradient"]["plugin_bias_over_scale"])
        for case in result["cases"]
    ]
    double = [
        abs(case["nonlinear_loss_gradient"]["double_sample_bias_over_scale"])
        for case in result["cases"]
    ]
    low_time = [
        case["timing"]["low_automatic"]["median_nanoseconds_per_sample"] * 1e-3
        for case in result["cases"]
    ]
    high_time = [
        case["timing"]["high_discrete"]["median_nanoseconds_per_sample"] * 1e-3
        for case in result["cases"]
    ]
    pair_time = [
        case["timing"]["paired_residual"]["median_nanoseconds_per_sample"] * 1e-3
        for case in result["cases"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    axes[0, 0].bar(positions, residual, color="#0f766e")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Residual variance / high variance")
    axes[0, 0].set_xticks(positions, labels)
    width = 0.36
    axes[0, 1].bar(positions - width / 2, predicted, width, label="predicted", color="#2563eb")
    axes[0, 1].bar(positions + width / 2, empirical, width, label="resampled", color="#ca8a04")
    axes[0, 1].axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
    axes[0, 1].set_title("Matched-cost MSE gain over high-only")
    axes[0, 1].set_xticks(positions, labels)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].bar(positions - width / 2, np.maximum(plugin, 1e-12), width, label="plug-in", color="#b42318")
    axes[1, 0].bar(positions + width / 2, np.maximum(double, 1e-12), width, label="double sample", color="#7c3aed")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Directional squared-loss gradient |bias| / scale")
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].legend(frameon=False)
    width_time = 0.25
    axes[1, 1].bar(positions - width_time, low_time, width_time, label="low", color="#16a34a")
    axes[1, 1].bar(positions, high_time, width_time, label="high", color="#dc2626")
    axes[1, 1].bar(positions + width_time, pair_time, width_time, label="pair", color="#4f46e5")
    axes[1, 1].set_title("CPU batch timing proxy (microseconds/sample)")
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "N2-ADRC-N1 development-only mechanism screen\n"
        "Finite Sobol population; no audit or paper authorization",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# N2-ADRC-N1 development pilot",
        "",
        f"**Machine decision:** `{result['machine_decision']}`",
        "",
        f"Promotion screen met: **{result['promotion_screen_met']}** "
        "(this permits design of an unseen audit only).",
        "",
        "| Case | residual/high variance | predicted gain | conservative timing | empirical MSE gain (95% bootstrap) | JVP rel. err. | screen |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in result["cases"]:
        matched = case["matched_cost"]
        interval = matched["empirical_mse_gain_bootstrap_95"]
        lines.append(
            f"| {case['case_id']} | {case['residual_to_high_variance_ratio']:.4g} | "
            f"{case['predicted_measured_cost_efficiency_gain']:.3f}x | "
            f"{case['predicted_conservative_timing_efficiency_gain']:.3f}x | "
            f"{matched['empirical_mse_gain']:.3f}x "
            f"[{interval[0]:.3f}, {interval[1]:.3f}] | "
            f"{case['fixed_state_derivative']['fixed_state_jvp_relative_error']:.3g} | "
            f"{case['development_case_screen_met']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The target is a frozen finite-population high-renderer mean. The fields are gridded analytic morphology proxies. The bend is prescribed and has no field-dependent trajectory derivative. Timings are CPU batch proxies. No neural field, reconstruction, experimental data, or held-out audit is present.",
            "",
            "The plug-in and double-sample gradient columns are Monte Carlo diagnostics. The double-sample construction is unbiased in expectation; a smaller observed bias in one finite run is not itself a proof.",
            "",
            "Reserved audit families were not opened: "
            + ", ".join(result["reserved_audit_families_not_opened"])
            + ".",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    *,
    config: dict[str, Any],
    result: dict[str, Any],
    config_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "metrics.csv", _metric_rows(result))
    _plot(output_dir / "n2_adrc_n1_development_pilot.png", result)
    _write_summary(output_dir / "summary.md", result)
    try:
        public_config_source = str(
            config_path.resolve().relative_to(ROOT.parent.resolve())
        )
    except ValueError:
        public_config_source = config_path.name
    manifest = {
        "schema": "n2-adrc-n1-development-manifest-1.0",
        "config_source": public_config_source,
        "config_source_sha256": sha256(config_path),
        "source_files": {
            str(path.relative_to(ROOT.parent)): sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "automatic_discrete_multifidelity.py",
                ROOT / "analytic_bost_phantoms.py",
            )
        },
        "files": {},
    }
    for file in sorted(output_dir.iterdir()):
        if file.name != "manifest.json" and file.is_file():
            manifest["files"][file.name] = sha256(file)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    result = run_experiment(config)
    write_outputs(
        args.output_dir,
        config=config,
        result=result,
        config_path=args.config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
