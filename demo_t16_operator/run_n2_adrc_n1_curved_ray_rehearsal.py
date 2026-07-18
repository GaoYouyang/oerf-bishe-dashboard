#!/usr/bin/env python3
"""Run the pre-audit curved-ray rehearsal without opening reserved families."""

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
import torch
import torch.nn.functional as functional

try:
    from .analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from .automatic_discrete_multifidelity import (
        SyntheticRayRig,
        two_level_efficiency,
    )
    from .field_dependent_ray import (
        FIELD_DEPENDENT_RAY_SCHEMA,
        RayDomainError,
        exit_direction_deflection,
        path_integrated_deflection,
        path_topology_diagnostics,
        ray_momentum_balance,
        relative_l2,
        sample_pupil_sobol,
        straight_ray_deflection,
        trace_field_dependent_rays,
    )
except ImportError:
    from analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from automatic_discrete_multifidelity import SyntheticRayRig, two_level_efficiency
    from field_dependent_ray import (
        FIELD_DEPENDENT_RAY_SCHEMA,
        RayDomainError,
        exit_direction_deflection,
        path_integrated_deflection,
        path_topology_diagnostics,
        ray_momentum_balance,
        relative_l2,
        sample_pupil_sobol,
        straight_ray_deflection,
        trace_field_dependent_rays,
    )


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs" / "n2_adrc_n1_curved_ray_rehearsal_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "n2_adrc_n1_curved_ray_rehearsal_v1"
RESULT_SCHEMA = "n2-adrc-n1-curved-ray-rehearsal-result-1.0"


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


def _rig_from_case(case: dict[str, Any]) -> SyntheticRayRig:
    values = case["rig"]
    if float(values.get("bend", 0.0)) != 0.0:
        raise ValueError("curved-ray rehearsal forbids prescribed bend")
    return SyntheticRayRig(
        rig_id=str(case["id"]),
        view_angle_degrees=float(values["view_angle_degrees"]),
        detector_u=float(values["detector_u"]),
        detector_z=float(values["detector_z"]),
        aperture_radius=float(values["aperture_radius"]),
        path_half_length=float(values["path_half_length"]),
        cone_u=float(values["cone_u"]),
        cone_z=float(values["cone_z"]),
        bend=0.0,
    )


def _smooth_direction(values: torch.Tensor, *, seed: int, relative_norm: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(values.shape, generator=generator, dtype=values.dtype)
    smoothed = functional.avg_pool3d(
        noise[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    scale = float(relative_norm) * float(torch.linalg.vector_norm(values)) / max(
        float(torch.linalg.vector_norm(smoothed)),
        1e-30,
    )
    return smoothed * scale


def _benchmark(
    function: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, float]:
    checksum = 0.0
    for _ in range(int(warmup_repeats)):
        output = function()
        tensors = output if isinstance(output, tuple) else (output,)
        checksum += sum(float(item.detach().sum()) for item in tensors)
    elapsed: list[int] = []
    for _ in range(int(measured_repeats)):
        started = time.perf_counter_ns()
        output = function()
        tensors = output if isinstance(output, tuple) else (output,)
        checksum += sum(float(item.detach().sum()) for item in tensors)
        elapsed.append(time.perf_counter_ns() - started)
    if not np.isfinite(checksum):
        raise RuntimeError("timing route produced a non-finite checksum")
    samples = np.asarray(elapsed, dtype=np.float64)
    return {
        "p10_seconds": float(np.quantile(samples, 0.1) * 1e-9),
        "median_seconds": float(np.median(samples) * 1e-9),
        "p90_seconds": float(np.quantile(samples, 0.9) * 1e-9),
    }


def _high_route(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    create_graph: bool,
) -> tuple[torch.Tensor, Any]:
    trace = trace_field_dependent_rays(
        values,
        states,
        rig,
        gradient_mode="central",
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
        create_graph=create_graph,
    )
    output = path_integrated_deflection(
        values,
        trace,
        gradient_mode="central",
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        create_graph=create_graph,
        detach_path=False,
    )
    return output, trace


def _trajectory_derivative_audit(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    epsilon: float,
    direction: torch.Tensor,
    support_threshold_fraction: float,
    frustum_half_width_u: float,
    frustum_half_width_v: float,
) -> dict[str, Any]:
    def forward(grid: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, Any]:
        output, trace = _high_route(
            grid,
            states,
            rig,
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=create_graph,
        )
        return torch.mean(output, dim=0), trace

    variable = values.detach().clone().requires_grad_(True)
    full, trace = forward(variable, create_graph=True)
    frozen = torch.mean(
        path_integrated_deflection(
            variable,
            trace,
            gradient_mode="central",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            create_graph=True,
            detach_path=True,
        ),
        dim=0,
    )
    full_jvp = torch.stack(
        [
            torch.sum(
                torch.autograd.grad(full[index], variable, retain_graph=True)[0]
                * direction
            )
            for index in range(len(full))
        ]
    )
    frozen_jvp = torch.stack(
        [
            torch.sum(
                torch.autograd.grad(frozen[index], variable, retain_graph=True)[0]
                * direction
            )
            for index in range(len(frozen))
        ]
    )
    plus, plus_trace = forward(values + epsilon * direction, create_graph=False)
    minus, minus_trace = forward(values - epsilon * direction, create_graph=False)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    cotangent = torch.as_tensor([0.6, -0.8], dtype=values.dtype)
    vjp = torch.autograd.grad(torch.sum(full * cotangent), variable)[0]
    vjp_lhs = float(torch.sum(vjp * direction))
    vjp_rhs = float(torch.sum(cotangent * full_jvp))

    peak = float(torch.max(torch.abs(values)))
    topology_args = {
        "support_threshold": float(support_threshold_fraction) * peak,
        "frustum_half_width_u": float(frustum_half_width_u),
        "frustum_half_width_v": float(frustum_half_width_v),
    }
    nominal_signature = path_topology_diagnostics(values, trace, **topology_args)
    plus_signature = path_topology_diagnostics(values + epsilon * direction, plus_trace, **topology_args)
    minus_signature = path_topology_diagnostics(values - epsilon * direction, minus_trace, **topology_args)
    signature_stable = (
        nominal_signature.support_crossings_per_ray
        == plus_signature.support_crossings_per_ray
        == minus_signature.support_crossings_per_ray
        and nominal_signature.frustum_violations_per_ray
        == plus_signature.frustum_violations_per_ray
        == minus_signature.frustum_violations_per_ray
    )
    trajectory_delta = torch.linalg.vector_norm(full_jvp - frozen_jvp)
    full_scale = torch.linalg.vector_norm(full_jvp).clamp_min(1e-30)
    return {
        "full_output_vs_frozen_output_relative_l2": relative_l2(full, frozen),
        "full_trajectory_jvp_relative_error": relative_l2(full_jvp, finite_difference),
        "full_trajectory_vjp_dot_relative_error": abs(vjp_lhs - vjp_rhs)
        / max(abs(vjp_lhs), abs(vjp_rhs), 1e-30),
        "trajectory_jvp_fraction": float(trajectory_delta / full_scale),
        "full_jvp": [float(value) for value in full_jvp.detach()],
        "frozen_path_jvp": [float(value) for value in frozen_jvp.detach()],
        "finite_difference_jvp": [float(value) for value in finite_difference.detach()],
        "topology_signature_stable_under_fd_perturbation": bool(signature_stable),
        "support_crossings_per_ray": list(nominal_signature.support_crossings_per_ray),
        "frustum_violations_per_ray": list(nominal_signature.frustum_violations_per_ray),
        "minimum_frustum_margin": nominal_signature.minimum_frustum_margin,
    }


def _case_result(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case["id"])
    rig = _rig_from_case(case)
    spec = make_analytic_phantom(
        family=str(case["phantom_family"]),
        seed=int(case["phantom_seed"]),
    )
    values = analytic_phantom_grid(
        spec,
        grid_shape=tuple(int(value) for value in config["grid_shape_zyx"]),
        dtype=torch.float64,
        device="cpu",
    ).field
    delta = float(config["difference_step"])
    refractivity_scale = float(config["refractivity_scale"])
    step_counts = [int(value) for value in config["reference_step_counts"]]
    states = sample_pupil_sobol(
        int(config["population_count"]),
        seed=stable_seed(config["seed_roles"]["population_state_base"], case_id),
    )
    low = straight_ray_deflection(
        values,
        states,
        rig,
        gradient_mode="automatic",
        difference_step=delta,
        refractivity_scale=refractivity_scale,
        step_count=step_counts[0],
        create_graph=False,
    )
    high_half, half_trace = _high_route(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=refractivity_scale,
        step_count=step_counts[0],
        create_graph=False,
    )
    high_full, full_trace = _high_route(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=refractivity_scale,
        step_count=step_counts[1],
        create_graph=False,
    )
    exit_full = exit_direction_deflection(full_trace)
    high_reference_error = relative_l2(high_half, high_full)
    exit_integral_error = relative_l2(high_full, exit_full)
    endpoint_momentum, integrated_index_gradient = ray_momentum_balance(
        values,
        full_trace,
        gradient_mode="central",
        difference_step=delta,
        refractivity_scale=refractivity_scale,
        create_graph=False,
    )
    momentum_balance_error = relative_l2(
        integrated_index_gradient,
        endpoint_momentum,
    )

    timing = config["timing"]
    low_timing = _benchmark(
        lambda: straight_ray_deflection(
            values,
            states,
            rig,
            gradient_mode="automatic",
            difference_step=delta,
            refractivity_scale=refractivity_scale,
            step_count=step_counts[0],
            create_graph=False,
        ),
        warmup_repeats=int(timing["warmup_repeats"]),
        measured_repeats=int(timing["measured_repeats"]),
    )
    high_timing = _benchmark(
        lambda: _high_route(
            values,
            states,
            rig,
            difference_step=delta,
            refractivity_scale=refractivity_scale,
            step_count=step_counts[0],
            create_graph=False,
        )[0],
        warmup_repeats=int(timing["warmup_repeats"]),
        measured_repeats=int(timing["measured_repeats"]),
    )
    pair_timing = _benchmark(
        lambda: (
            straight_ray_deflection(
                values,
                states,
                rig,
                gradient_mode="automatic",
                difference_step=delta,
                refractivity_scale=refractivity_scale,
                step_count=step_counts[0],
                create_graph=False,
            ),
            _high_route(
                values,
                states,
                rig,
                difference_step=delta,
                refractivity_scale=refractivity_scale,
                step_count=step_counts[0],
                create_graph=False,
            )[0],
        ),
        warmup_repeats=int(timing["warmup_repeats"]),
        measured_repeats=int(timing["measured_repeats"]),
    )
    efficiency = two_level_efficiency(
        high_half,
        low,
        high_cost=high_timing["median_seconds"],
        low_cost=low_timing["median_seconds"],
        residual_cost=pair_timing["median_seconds"],
    )
    conservative_efficiency = two_level_efficiency(
        high_half,
        low,
        high_cost=high_timing["p10_seconds"],
        low_cost=low_timing["p90_seconds"],
        residual_cost=pair_timing["p90_seconds"],
    )

    derivative_config = config["derivative_check"]
    derivative_states = sample_pupil_sobol(
        int(derivative_config["ray_count"]),
        seed=stable_seed(config["seed_roles"]["derivative_state_base"], case_id),
    )
    direction = _smooth_direction(
        values,
        seed=stable_seed(config["seed_roles"]["direction_base"], case_id),
        relative_norm=float(derivative_config["direction_relative_norm"]),
    )
    derivative = _trajectory_derivative_audit(
        values,
        derivative_states,
        rig,
        difference_step=delta,
        refractivity_scale=refractivity_scale,
        step_count=int(derivative_config["step_count"]),
        epsilon=float(derivative_config["finite_difference_epsilon"]),
        direction=direction,
        support_threshold_fraction=float(
            config["topology"]["support_threshold_fraction_of_grid_peak"]
        ),
        frustum_half_width_u=float(config["topology"]["frustum_half_width_u"]),
        frustum_half_width_v=float(config["topology"]["frustum_half_width_v"]),
    )
    stress_envelope: list[dict[str, Any]] = []
    for multiplier_value in config["dimensionless_stress_scale_multipliers"]:
        multiplier = float(multiplier_value)
        stress_scale = refractivity_scale * multiplier
        try:
            if multiplier == 1.0:
                stress_low = low
                stress_high = high_half
                stress_efficiency = efficiency
                stress_derivative = derivative
            else:
                stress_low = straight_ray_deflection(
                    values,
                    states,
                    rig,
                    gradient_mode="automatic",
                    difference_step=delta,
                    refractivity_scale=stress_scale,
                    step_count=step_counts[0],
                    create_graph=False,
                )
                stress_high, _ = _high_route(
                    values,
                    states,
                    rig,
                    difference_step=delta,
                    refractivity_scale=stress_scale,
                    step_count=step_counts[0],
                    create_graph=False,
                )
                stress_efficiency = two_level_efficiency(
                    stress_high,
                    stress_low,
                    high_cost=high_timing["p10_seconds"],
                    low_cost=low_timing["p90_seconds"],
                    residual_cost=pair_timing["p90_seconds"],
                )
                stress_derivative = _trajectory_derivative_audit(
                    values,
                    derivative_states,
                    rig,
                    difference_step=delta,
                    refractivity_scale=stress_scale,
                    step_count=int(derivative_config["step_count"]),
                    epsilon=float(derivative_config["finite_difference_epsilon"]),
                    direction=direction,
                    support_threshold_fraction=float(
                        config["topology"][
                            "support_threshold_fraction_of_grid_peak"
                        ]
                    ),
                    frustum_half_width_u=float(
                        config["topology"]["frustum_half_width_u"]
                    ),
                    frustum_half_width_v=float(
                        config["topology"]["frustum_half_width_v"]
                    ),
                )
            frustum_count = sum(
                bool(value)
                for value in stress_derivative["frustum_violations_per_ray"]
            )
            low_high_error = relative_l2(stress_low, stress_high)
            stress_envelope.append(
                {
                    "scale_multiplier": multiplier,
                    "refractivity_scale": stress_scale,
                    "low_to_high_relative_l2": low_high_error,
                    "residual_to_high_variance_ratio": (
                        stress_efficiency.residual_trace_variance
                        / max(stress_efficiency.high_trace_variance, 1e-30)
                    ),
                    "conservative_timing_efficiency_ceiling": (
                        stress_efficiency.predicted_efficiency_gain
                    ),
                    "trajectory_jvp_fraction": stress_derivative[
                        "trajectory_jvp_fraction"
                    ],
                    "topology_signature_stable": stress_derivative[
                        "topology_signature_stable_under_fd_perturbation"
                    ],
                    "frustum_violation_count": frustum_count,
                    "minimum_frustum_margin": stress_derivative[
                        "minimum_frustum_margin"
                    ],
                    "diagnostic_break": bool(
                        low_high_error > 0.01
                        or stress_derivative["trajectory_jvp_fraction"] > 0.01
                        or not stress_derivative[
                            "topology_signature_stable_under_fd_perturbation"
                        ]
                        or frustum_count > 0
                    ),
                    "status": "computed",
                }
            )
        except RayDomainError as error:
            stress_envelope.append(
                {
                    "scale_multiplier": multiplier,
                    "refractivity_scale": stress_scale,
                    "diagnostic_break": True,
                    "status": "fail_closed_ray_domain",
                    "error": str(error),
                }
            )
    breakpoints = [
        item["scale_multiplier"]
        for item in stress_envelope
        if item["diagnostic_break"]
    ]
    screens = config["rehearsal_screens"]
    screen_checks = {
        "reference_convergence": high_reference_error
        <= float(screens["maximum_high_half_to_full_relative_l2"]),
        "exit_integral_consistency": exit_integral_error
        <= float(screens["maximum_exit_vs_integral_relative_l2"]),
        "momentum_balance": momentum_balance_error
        <= float(screens["maximum_momentum_balance_relative_l2"]),
        "full_trajectory_jvp": derivative["full_trajectory_jvp_relative_error"]
        <= float(screens["maximum_full_trajectory_jvp_relative_error"]),
        "full_trajectory_vjp_dot": derivative[
            "full_trajectory_vjp_dot_relative_error"
        ]
        <= float(screens["maximum_full_trajectory_vjp_dot_relative_error"]),
        "stencil_margin": min(
            half_trace.minimum_stencil_margin,
            full_trace.minimum_stencil_margin,
        )
        >= float(screens["minimum_stencil_margin"]),
        "topology_signature": bool(
            derivative["topology_signature_stable_under_fd_perturbation"]
        ),
    }
    return {
        "case_id": case_id,
        "phantom_family": str(case["phantom_family"]),
        "phantom_seed": int(case["phantom_seed"]),
        "rig": case["rig"],
        "field_range": [float(torch.min(values)), float(torch.max(values))],
        "high_half_to_full_relative_l2": high_reference_error,
        "exit_vs_integral_relative_l2": exit_integral_error,
        "momentum_balance_relative_l2": momentum_balance_error,
        "low_to_high_relative_l2": relative_l2(low, high_half),
        "residual_to_high_variance_ratio": efficiency.residual_trace_variance
        / max(efficiency.high_trace_variance, 1e-30),
        "predicted_measured_timing_efficiency_gain": efficiency.predicted_efficiency_gain,
        "predicted_conservative_timing_efficiency_gain": conservative_efficiency.predicted_efficiency_gain,
        "timing": {
            "low_straight_automatic": low_timing,
            "high_curved_central": high_timing,
            "paired_route": pair_timing,
            "machine_specific": True,
        },
        "primitive_ledger_per_ray": {
            "low_field_values": 2 * step_counts[0],
            "low_coordinate_gradient_calls": 1,
            "high_field_values": 35 * step_counts[0],
            "high_rk4_stages": 4 * step_counts[0],
            "high_path_integral_midpoints": step_counts[0],
        },
        "trajectory_derivative": derivative,
        "dimensionless_stress_envelope": stress_envelope,
        "first_diagnostic_break_multiplier": min(breakpoints) if breakpoints else None,
        "geometry_diagnostics": {
            "half_minimum_domain_margin": half_trace.minimum_domain_margin,
            "half_minimum_stencil_margin": half_trace.minimum_stencil_margin,
            "full_minimum_domain_margin": full_trace.minimum_domain_margin,
            "full_minimum_stencil_margin": full_trace.minimum_stencil_margin,
            "maximum_direction_norm_error": max(
                half_trace.maximum_direction_norm_error,
                full_trace.maximum_direction_norm_error,
            ),
        },
        "rehearsal_screen_checks": screen_checks,
        "rehearsal_screen_met": bool(all(screen_checks.values())),
    }


def _write_figure(results: list[dict[str, Any]], path: Path) -> None:
    labels = [item["case_id"].replace("_", "\n") for item in results]
    x = np.arange(len(results))
    residual = [item["residual_to_high_variance_ratio"] for item in results]
    gain = [item["predicted_conservative_timing_efficiency_gain"] for item in results]
    convergence = [item["high_half_to_full_relative_l2"] for item in results]
    trajectory = [item["trajectory_derivative"]["trajectory_jvp_fraction"] for item in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(x, residual, color="#28766c")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Residual variance / curved-ray high variance")
    axes[0, 1].bar(x, gain, color="#b98527")
    axes[0, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Conservative measured-time efficiency ceiling")
    axes[1, 0].bar(x, convergence, color="#3f69a8")
    axes[1, 0].axhline(0.01, color="#a34f43", linestyle="--", linewidth=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("RK4 half-to-full step sensitivity")
    axes[1, 1].bar(x, trajectory, color="#a34f43")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Trajectory contribution / full directional JVP")
    for axis in axes.flat:
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "N2-ADRC-N1 curved-ray rehearsal only\nreserved audit families remain unopened",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_envelope_figure(results: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    colors = ("#28766c", "#a34f43", "#3f69a8")
    for color, item in zip(colors, results):
        envelope = item["dimensionless_stress_envelope"]
        multiplier = np.asarray(
            [row["scale_multiplier"] for row in envelope],
            dtype=np.float64,
        )
        low_high = np.asarray(
            [row.get("low_to_high_relative_l2", np.nan) for row in envelope],
            dtype=np.float64,
        )
        trajectory = np.asarray(
            [row.get("trajectory_jvp_fraction", np.nan) for row in envelope],
            dtype=np.float64,
        )
        frustum = np.asarray(
            [row.get("minimum_frustum_margin", np.nan) for row in envelope],
            dtype=np.float64,
        )
        label = item["case_id"].replace("_", " ")
        axes[0].plot(multiplier, low_high, marker="o", color=color, label=label)
        axes[1].plot(multiplier, trajectory, marker="o", color=color, label=label)
        axes[2].plot(multiplier, frustum, marker="o", color=color, label=label)
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
        axis.set_xlabel("dimensionless refractivity multiplier")
    axes[0].set_yscale("log")
    axes[0].axhline(0.01, color="#222222", linestyle="--", linewidth=1)
    axes[0].set_title("straight / curved output mismatch")
    axes[0].set_ylabel("relative L2")
    axes[1].set_yscale("log")
    axes[1].axhline(0.01, color="#222222", linestyle="--", linewidth=1)
    axes[1].set_title("trajectory share of directional JVP")
    axes[2].axhline(0.0, color="#222222", linestyle="--", linewidth=1)
    axes[2].set_title("synthetic frustum margin")
    axes[2].set_ylabel("normalized coordinate")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Curved-ray validity envelope: numerical stress levels, not physical conditions",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if config["reserved_audit_families_not_opened"] != [
        "oblique_compression_sheet",
        "shock_expansion_pair",
    ]:
        raise ValueError("reserved-family contract changed")
    opened = {
        str(case["phantom_family"])
        for case in config["development_rehearsal_cases"]
    }
    if opened.intersection(config["reserved_audit_families_not_opened"]):
        raise RuntimeError("curved-ray rehearsal attempted to open a reserved family")
    results = [_case_result(case, config) for case in config["development_rehearsal_cases"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_name = "n2_adrc_n1_curved_ray_rehearsal.png"
    envelope_figure_name = "n2_adrc_n1_refractivity_envelope.png"
    _write_figure(results, output_dir / figure_name)
    _write_envelope_figure(results, output_dir / envelope_figure_name)
    result = {
        "schema": RESULT_SCHEMA,
        "candidate_id": str(config["candidate_id"]),
        "ray_schema": FIELD_DEPENDENT_RAY_SCHEMA,
        "machine_decision": str(config["hard_conclusion"]),
        "case_count": len(results),
        "case_screen_count": sum(bool(item["rehearsal_screen_met"]) for item in results),
        "cases": results,
        "reserved_audit_families_not_opened": config[
            "reserved_audit_families_not_opened"
        ],
        "claim_boundary": [
            "normalized_grid_proxy_not_physical_density",
            "no_Gladstone_Dale_wavelength_or_composition_calibration",
            "no_camera_calibration_or_background_image_formation",
            "no_reserved_family_opening",
            "no_real_BOST_data",
            "no_reconstruction_or_generalization_claim",
            "timing_is_machine_specific",
        ],
        "promotion_meaning": "may_freeze_blind_audit_protocol_only",
        "figure": figure_name,
        "figures": [figure_name, envelope_figure_name],
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "screen_met",
                "reference_relative_l2",
                "exit_integral_relative_l2",
                "momentum_balance_relative_l2",
                "residual_variance_ratio",
                "conservative_timing_gain",
                "trajectory_jvp_fraction",
                "trajectory_jvp_relative_error",
                "topology_signature_stable",
            ],
        )
        writer.writeheader()
        for item in results:
            derivative = item["trajectory_derivative"]
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "screen_met": item["rehearsal_screen_met"],
                    "reference_relative_l2": item["high_half_to_full_relative_l2"],
                    "exit_integral_relative_l2": item["exit_vs_integral_relative_l2"],
                    "momentum_balance_relative_l2": item[
                        "momentum_balance_relative_l2"
                    ],
                    "residual_variance_ratio": item[
                        "residual_to_high_variance_ratio"
                    ],
                    "conservative_timing_gain": item[
                        "predicted_conservative_timing_efficiency_gain"
                    ],
                    "trajectory_jvp_fraction": derivative[
                        "trajectory_jvp_fraction"
                    ],
                    "trajectory_jvp_relative_error": derivative[
                        "full_trajectory_jvp_relative_error"
                    ],
                    "topology_signature_stable": derivative[
                        "topology_signature_stable_under_fd_perturbation"
                    ],
                }
            )
    summary_lines = [
        "# N2-ADRC-N1 curved-ray rehearsal",
        "",
        f"Machine decision: `{result['machine_decision']}`",
        "",
        (
            f"Numerical validity screens passed: {result['case_screen_count']} / "
            f"{result['case_count']}. This does not authorize an audit or paper claim."
        ),
        "",
        "| Case | 64->128 reference | Momentum balance | Base trajectory JVP share | First stress break |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        summary_lines.append(
            "| {case} | {reference:.4%} | {momentum:.4%} | {trajectory:.4%} | {breakpoint}x |".format(
                case=item["case_id"],
                reference=item["high_half_to_full_relative_l2"],
                momentum=item["momentum_balance_relative_l2"],
                trajectory=item["trajectory_derivative"]["trajectory_jvp_fraction"],
                breakpoint=item["first_diagnostic_break_multiplier"],
            )
        )
    summary_lines.extend(
        [
            "",
            "The scale ladder is dimensionless and cannot be mapped to a real flame without the missing physical and camera contract.",
            "Reserved morphology families were not evaluated by this runner.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )
    generated_files = (
        "result.json",
        "config_snapshot.json",
        "metrics.csv",
        "summary.md",
        figure_name,
        envelope_figure_name,
    )
    manifest = {
        "schema": "n2-adrc-n1-curved-ray-rehearsal-manifest-1.0",
        "config": config_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "ray_module_sha256": sha256(ROOT / "field_dependent_ray.py"),
        "analytic_phantom_sha256": sha256(ROOT / "analytic_bost_phantoms.py"),
        "reserved_audit_families_opened": False,
        "files": {
            filename: sha256(output_dir / filename) for filename in generated_files
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    result = run(args.config.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "machine_decision": result["machine_decision"],
                "case_screen_count": result["case_screen_count"],
                "case_count": result["case_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
