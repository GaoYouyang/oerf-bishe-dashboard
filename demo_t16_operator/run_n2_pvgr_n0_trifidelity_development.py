#!/usr/bin/env python3
"""Run the N2-PVGR-N0 fail-closed trifidelity development rehearsal.

This runner uses only the declared non-reserved development families.  It
separates three mechanisms:

* L0: straight ray with automatic coordinate gradients;
* M: straight ray with central-difference gradients;
* H: field-dependent curved ray with central-difference gradients.

The randomized estimator uses M as the all-ray baseline and evaluates H on a
Bernoulli subset with an inverse-probability residual correction.  Complete H
replays are computed only to audit the development estimator; a separate sparse
execution proves that the online path can avoid unselected H evaluations.
Nothing in this file authorizes a real-BOST, reconstruction, generalization, or
novelty claim.
"""

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

try:
    from .analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from .automatic_discrete_multifidelity import (
        SyntheticRayRig,
        trace_sample_variance,
    )
    from .field_dependent_ray import (
        path_integrated_deflection,
        path_topology_diagnostics,
        relative_l2,
        sample_pupil_sobol,
        straight_ray_deflection,
        trace_field_dependent_rays,
    )
    from .ray_safety_certificate import low_path_safety_certificate
    from .topology_certified_routing import (
        allocate_inclusion_probabilities,
        conditional_trace_variance,
        horvitz_thompson_mean,
        horvitz_thompson_sparse_mean,
    )
except ImportError:
    from analytic_bost_phantoms import analytic_phantom_grid, make_analytic_phantom
    from automatic_discrete_multifidelity import SyntheticRayRig, trace_sample_variance
    from field_dependent_ray import (
        path_integrated_deflection,
        path_topology_diagnostics,
        relative_l2,
        sample_pupil_sobol,
        straight_ray_deflection,
        trace_field_dependent_rays,
    )
    from ray_safety_certificate import low_path_safety_certificate
    from topology_certified_routing import (
        allocate_inclusion_probabilities,
        conditional_trace_variance,
        horvitz_thompson_mean,
        horvitz_thompson_sparse_mean,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator"
    / "configs"
    / "n2_pvgr_n0_trifidelity_development_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator"
    / "results"
    / "n2_pvgr_n0_trifidelity_development_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


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
        bend=float(values.get("bend", 0.0)),
    )


def _smooth_direction(
    values: torch.Tensor,
    *,
    seed: int,
    relative_norm: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(values.shape, generator=generator, dtype=values.dtype)
    smoothed = torch.nn.functional.avg_pool3d(
        raw[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    target = float(relative_norm) * torch.linalg.vector_norm(values.detach())
    return smoothed * target / torch.linalg.vector_norm(smoothed).clamp_min(1e-30)


def _benchmark(
    function: Callable[[], torch.Tensor],
    *,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, float]:
    checksum = 0.0
    for _ in range(int(warmup_repeats)):
        checksum += float(function().detach().sum())
    elapsed: list[int] = []
    for _ in range(int(measured_repeats)):
        started = time.perf_counter_ns()
        checksum += float(function().detach().sum())
        elapsed.append(time.perf_counter_ns() - started)
    if not np.isfinite(checksum):
        raise RuntimeError("timing route produced a non-finite checksum")
    samples = np.asarray(elapsed, dtype=np.float64) * 1e-9
    return {
        "p10_seconds": float(np.quantile(samples, 0.1)),
        "median_seconds": float(np.median(samples)),
        "p90_seconds": float(np.quantile(samples, 0.9)),
    }


def _straight_route(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    gradient_mode: str,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    create_graph: bool,
) -> torch.Tensor:
    return straight_ray_deflection(
        values,
        states,
        rig,
        gradient_mode=gradient_mode,
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
        create_graph=create_graph,
    )


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


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_rank = torch.argsort(torch.argsort(left.detach())).to(torch.float64)
    right_rank = torch.argsort(torch.argsort(right.detach())).to(torch.float64)
    left_rank -= torch.mean(left_rank)
    right_rank -= torch.mean(right_rank)
    denominator = torch.linalg.vector_norm(left_rank) * torch.linalg.vector_norm(
        right_rank
    )
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.sum(left_rank * right_rank) / denominator)


def _route_probabilities(
    risk_proxy: torch.Tensor,
    residual: torch.Tensor,
    unsafe_mask: torch.Tensor,
    *,
    pi_floor: float,
    safe_average_probability: float,
) -> tuple[dict[str, torch.Tensor], float]:
    sample_count = len(risk_proxy)
    safe_count = int(torch.sum(~unsafe_mask))
    unsafe_count = sample_count - safe_count
    if safe_count == 0:
        probabilities = torch.ones_like(risk_proxy)
        return {
            "uniform": probabilities,
            "proxy": probabilities,
            "oracle": probabilities,
        }, 1.0
    safe_probability = float(safe_average_probability)
    if safe_probability < float(pi_floor) or safe_probability > 1.0:
        raise ValueError("safe_average_probability must lie in [pi_floor,1]")
    average_budget = (
        unsafe_count + safe_probability * safe_count
    ) / sample_count
    uniform = torch.full_like(risk_proxy, safe_probability)
    uniform[unsafe_mask] = 1.0
    proxy = allocate_inclusion_probabilities(
        risk_proxy,
        average_high_fidelity_budget=average_budget,
        pi_floor=pi_floor,
        unsafe_mask=unsafe_mask,
    )
    oracle = allocate_inclusion_probabilities(
        torch.linalg.vector_norm(residual.detach(), dim=-1),
        average_high_fidelity_budget=average_budget,
        pi_floor=pi_floor,
        unsafe_mask=unsafe_mask,
    )
    return {"uniform": uniform, "proxy": proxy, "oracle": oracle}, average_budget


def _replay_metrics(
    low: torch.Tensor,
    high: torch.Tensor,
    probabilities: torch.Tensor,
    unsafe_mask: torch.Tensor,
    *,
    pi_floor: float,
    replicate_count: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    uniforms = torch.rand(
        (int(replicate_count), len(low)),
        generator=generator,
        dtype=low.dtype,
    )
    estimates = horvitz_thompson_mean(
        low,
        high,
        probabilities,
        uniforms,
        pi_min=pi_floor,
        unsafe_mask=unsafe_mask,
    )
    target = torch.mean(high, dim=0)
    squared_error = torch.sum((estimates - target).square(), dim=-1)
    exact_variance = float(
        conditional_trace_variance(
            low,
            high,
            probabilities,
            pi_min=pi_floor,
            unsafe_mask=unsafe_mask,
        )
    )
    empirical_mse = float(torch.mean(squared_error))
    if exact_variance == 0.0:
        roundoff_tolerance = 128.0 * torch.finfo(low.dtype).eps * max(
            float(torch.sum(target.square())),
            1e-30,
        )
        roundoff_only = empirical_mse <= roundoff_tolerance
        variance_relative_error = 0.0 if roundoff_only else 1e300
        bias_standard_errors = 0.0 if roundoff_only else 1e300
    else:
        variance_relative_error = abs(empirical_mse - exact_variance) / exact_variance
        mean_bias = torch.linalg.vector_norm(torch.mean(estimates, dim=0) - target)
        bias_standard_errors = float(
            mean_bias / math.sqrt(exact_variance / int(replicate_count))
        )
    return {
        "exact_conditional_trace_variance": exact_variance,
        "empirical_mse": empirical_mse,
        "empirical_to_exact_relative_error": variance_relative_error,
        "empirical_bias_in_trace_standard_errors": bias_standard_errors,
        "q95_squared_error": float(torch.quantile(squared_error, 0.95)),
        "expected_high_route_count": float(torch.sum(probabilities)),
        "expected_high_route_fraction": float(torch.mean(probabilities)),
        "minimum_probability": float(torch.min(probabilities)),
        "maximum_probability": float(torch.max(probabilities)),
    }


def _sparse_execution_audit(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    medium: torch.Tensor,
    full_high: torch.Tensor,
    probabilities: torch.Tensor,
    unsafe_mask: torch.Tensor,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    pi_floor: float,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    uniforms = torch.rand(len(states), generator=generator, dtype=states.dtype)
    selected = torch.nonzero(uniforms < probabilities, as_tuple=False).flatten()
    if selected.numel() == 0:
        selected_high = torch.empty((0, full_high.shape[1]), dtype=full_high.dtype)
    else:
        selected_high, _ = _high_route(
            values,
            states.index_select(0, selected),
            rig,
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=False,
        )
    sparse = horvitz_thompson_sparse_mean(
        medium,
        selected_high,
        selected,
        probabilities,
        pi_min=pi_floor,
        unsafe_mask=unsafe_mask,
    )
    replay = horvitz_thompson_mean(
        medium,
        full_high,
        probabilities,
        uniforms,
        pi_min=pi_floor,
        unsafe_mask=unsafe_mask,
    )
    sparse_error = relative_l2(sparse, replay)
    selected_reference_error = (
        0.0
        if selected.numel() == 0
        else relative_l2(selected_high, full_high.index_select(0, selected))
    )
    return (
        {
            "selected_high_route_count": int(selected.numel()),
            "selected_high_route_fraction": float(selected.numel() / len(states)),
            "sparse_to_replay_relative_l2": sparse_error,
            "selected_high_to_full_batch_relative_l2": selected_reference_error,
        },
        selected,
    )


def _timing_and_cost_audit(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    probabilities: torch.Tensor,
    selected: torch.Tensor,
    certificate_arguments: dict[str, Any],
    *,
    difference_step: float,
    refractivity_scale: float,
    step_counts: dict[str, int],
    pi_floor: float,
    unsafe_mask: torch.Tensor,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, Any]:
    low_steps = int(step_counts["low_automatic"])
    medium_steps = int(step_counts["medium_straight_central"])
    high_steps = int(step_counts["high_curved_central"])

    low_timing = _benchmark(
        lambda: _straight_route(
            values,
            states,
            rig,
            gradient_mode="automatic",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=low_steps,
            create_graph=False,
        ),
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
    )
    medium_timing = _benchmark(
        lambda: _straight_route(
            values,
            states,
            rig,
            gradient_mode="central",
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=medium_steps,
            create_graph=False,
        ),
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
    )
    high_timing = _benchmark(
        lambda: _high_route(
            values,
            states,
            rig,
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=high_steps,
            create_graph=False,
        )[0],
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
    )
    certificate_timing = _benchmark(
        lambda: low_path_safety_certificate(
            values,
            states,
            rig,
            **certificate_arguments,
        ).residual_risk_proxy,
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
    )

    if selected.numel() == 0:
        def sparse_route() -> torch.Tensor:
            medium = _straight_route(
                values,
                states,
                rig,
                gradient_mode="central",
                difference_step=difference_step,
                refractivity_scale=refractivity_scale,
                step_count=medium_steps,
                create_graph=False,
            )
            return torch.mean(medium, dim=0)
    else:
        selected_states = states.index_select(0, selected)

        def sparse_route() -> torch.Tensor:
            medium = _straight_route(
                values,
                states,
                rig,
                gradient_mode="central",
                difference_step=difference_step,
                refractivity_scale=refractivity_scale,
                step_count=medium_steps,
                create_graph=False,
            )
            selected_high, _ = _high_route(
                values,
                selected_states,
                rig,
                difference_step=difference_step,
                refractivity_scale=refractivity_scale,
                step_count=high_steps,
                create_graph=False,
            )
            return horvitz_thompson_sparse_mean(
                medium,
                selected_high,
                selected,
                probabilities,
                pi_min=pi_floor,
                unsafe_mask=unsafe_mask,
            )

    sparse_timing = _benchmark(
        sparse_route,
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
    )
    interval_count = int(certificate_arguments["support_interval_count"])
    population_count = len(states)
    certificate_field_queries_per_ray = 3 * interval_count + 1
    certificate_coordinate_vjps_per_ray = interval_count
    low_field_queries_per_ray = 2 * low_steps
    low_coordinate_vjps_per_ray = low_steps
    medium_field_queries_per_ray = 7 * medium_steps
    high_field_queries_per_ray = 35 * high_steps
    expected_field_queries = (
        population_count
        * (certificate_field_queries_per_ray + medium_field_queries_per_ray)
        + float(torch.sum(probabilities)) * high_field_queries_per_ray
    )
    full_high_field_queries = population_count * high_field_queries_per_ray
    return {
        "wall_time_seconds_machine_specific": {
            "low_automatic_all": low_timing,
            "medium_straight_central_all": medium_timing,
            "high_curved_central_all": high_timing,
            "certificate_all": certificate_timing,
            "sparse_medium_plus_selected_high": sparse_timing,
            "observed_end_to_end_ratio_to_full_high": (
                certificate_timing["median_seconds"]
                + sparse_timing["median_seconds"]
            )
            / high_timing["median_seconds"],
        },
        "primitive_contract": {
            "low_field_queries_per_ray": low_field_queries_per_ray,
            "low_coordinate_vjps_per_ray": low_coordinate_vjps_per_ray,
            "medium_field_queries_per_ray": medium_field_queries_per_ray,
            "high_field_queries_per_ray": high_field_queries_per_ray,
            "certificate_field_queries_per_ray": certificate_field_queries_per_ray,
            "certificate_coordinate_vjps_per_ray": (
                certificate_coordinate_vjps_per_ray
            ),
            "expected_routed_field_queries": expected_field_queries,
            "full_high_field_queries": full_high_field_queries,
            "expected_routed_to_full_high_field_query_ratio": (
                expected_field_queries / full_high_field_queries
            ),
        },
    }


def _derivative_contract(
    values: torch.Tensor,
    rig: SyntheticRayRig,
    config: dict[str, Any],
    case_id: str,
    *,
    refractivity_scale: float,
    support_threshold: float,
) -> dict[str, Any]:
    derivative = config["derivative_contract"]
    routing = config["routing"]
    certificate_config = config["certificate"]
    ray_count = int(derivative["ray_count"])
    step_count = int(derivative["step_count"])
    pi_floor = float(routing["pi_floor"])
    states = sample_pupil_sobol(
        ray_count,
        seed=stable_seed(
            config["seed_roles"]["derivative_state_base"],
            case_id,
        ),
    )
    certificate = low_path_safety_certificate(
        values,
        states,
        rig,
        refractivity_scale=refractivity_scale,
        difference_step=float(config["difference_step"]),
        support_threshold=support_threshold,
        frustum_half_width_u=float(certificate_config["frustum_half_width_u"]),
        frustum_half_width_v=float(certificate_config["frustum_half_width_v"]),
        support_interval_count=max(8, min(32, int(certificate_config["support_interval_count"]))),
        numerical_path_buffer=float(certificate_config["numerical_path_buffer"]),
    )
    unsafe = ~certificate.domain_frustum_safe_mask
    medium = _straight_route(
        values,
        states,
        rig,
        gradient_mode="central",
        difference_step=float(config["difference_step"]),
        refractivity_scale=refractivity_scale,
        step_count=step_count,
        create_graph=False,
    )
    high, _ = _high_route(
        values,
        states,
        rig,
        difference_step=float(config["difference_step"]),
        refractivity_scale=refractivity_scale,
        step_count=step_count,
        create_graph=False,
    )
    probabilities, _ = _route_probabilities(
        certificate.residual_risk_proxy,
        high - medium,
        unsafe,
        pi_floor=pi_floor,
        safe_average_probability=float(routing["safe_ray_average_probability"]),
    )
    proxy_probability = probabilities["proxy"]
    generator = torch.Generator(device="cpu").manual_seed(
        stable_seed(config["seed_roles"]["derivative_route_base"], case_id)
    )
    fixed_uniforms = torch.rand(ray_count, generator=generator, dtype=values.dtype)
    selected = torch.nonzero(
        fixed_uniforms < proxy_probability,
        as_tuple=False,
    ).flatten()
    if selected.numel() == 0:
        selected = torch.tensor(
            [int(torch.argmax(proxy_probability))],
            dtype=torch.long,
        )
    direction = _smooth_direction(
        values,
        seed=stable_seed(
            config["seed_roles"]["derivative_direction_base"],
            case_id,
        ),
        relative_norm=float(derivative["direction_relative_norm"]),
    )

    def routed_function(field: torch.Tensor) -> torch.Tensor:
        medium_output = _straight_route(
            field,
            states,
            rig,
            gradient_mode="central",
            difference_step=float(config["difference_step"]),
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=True,
        )
        selected_high, _ = _high_route(
            field,
            states.index_select(0, selected),
            rig,
            difference_step=float(config["difference_step"]),
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=True,
        )
        return horvitz_thompson_sparse_mean(
            medium_output,
            selected_high,
            selected,
            proxy_probability,
            pi_min=pi_floor,
            unsafe_mask=unsafe,
        )

    base_output, routed_jvp = torch.autograd.functional.jvp(
        routed_function,
        values,
        direction,
        create_graph=False,
        strict=True,
    )
    epsilon = float(derivative["finite_difference_epsilon"])
    finite_difference = (
        routed_function(values + epsilon * direction)
        - routed_function(values - epsilon * direction)
    ) / (2.0 * epsilon)
    jvp_fd_error = relative_l2(routed_jvp, finite_difference)

    differentiable_values = values.detach().clone().requires_grad_(True)
    differentiable_output = routed_function(differentiable_values)
    cotangent = torch.as_tensor(
        [0.7, -0.4],
        dtype=values.dtype,
        device=values.device,
    )
    vjp = torch.autograd.grad(
        torch.sum(differentiable_output * cotangent),
        differentiable_values,
    )[0]
    left = torch.sum(vjp * direction)
    right = torch.sum(cotangent * routed_jvp)
    vjp_dot_error = float(
        torch.abs(left - right)
        / torch.maximum(
            torch.maximum(torch.abs(left), torch.abs(right)),
            torch.as_tensor(1e-30, dtype=values.dtype),
        )
    )

    _, medium_jvp = torch.autograd.functional.jvp(
        lambda field: _straight_route(
            field,
            states,
            rig,
            gradient_mode="central",
            difference_step=float(config["difference_step"]),
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=True,
        ),
        values,
        direction,
        create_graph=False,
        strict=True,
    )
    _, high_jvp = torch.autograd.functional.jvp(
        lambda field: _high_route(
            field,
            states,
            rig,
            difference_step=float(config["difference_step"]),
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            create_graph=True,
        )[0],
        values,
        direction,
        create_graph=False,
        strict=True,
    )
    replicate_count = int(derivative["monte_carlo_replicates"])
    uniform_generator = torch.Generator(device="cpu").manual_seed(
        stable_seed(config["seed_roles"]["derivative_route_base"], case_id, "mean")
    )
    derivative_uniforms = torch.rand(
        (replicate_count, ray_count),
        generator=uniform_generator,
        dtype=values.dtype,
    )
    routed_jvp_replicates = horvitz_thompson_mean(
        medium_jvp,
        high_jvp,
        proxy_probability,
        derivative_uniforms,
        pi_min=pi_floor,
        unsafe_mask=unsafe,
    )
    exact_high_jvp = torch.mean(high_jvp, dim=0)
    derivative_variance = float(
        conditional_trace_variance(
            medium_jvp,
            high_jvp,
            proxy_probability,
            pi_min=pi_floor,
            unsafe_mask=unsafe,
        )
    )
    derivative_bias = torch.linalg.vector_norm(
        torch.mean(routed_jvp_replicates, dim=0) - exact_high_jvp
    )
    derivative_bias_standard_errors = (
        0.0
        if derivative_variance == 0.0
        else float(
            derivative_bias
            / math.sqrt(derivative_variance / replicate_count)
        )
    )

    loss_replicates = int(config["routing"]["quadratic_loss_replicates"])
    loss_generator_a = torch.Generator(device="cpu").manual_seed(
        stable_seed(config["seed_roles"]["derivative_route_base"], case_id, "loss-a")
    )
    loss_generator_b = torch.Generator(device="cpu").manual_seed(
        stable_seed(config["seed_roles"]["derivative_route_base"], case_id, "loss-b")
    )
    uniforms_a = torch.rand(
        (loss_replicates, ray_count),
        generator=loss_generator_a,
        dtype=values.dtype,
    )
    uniforms_b = torch.rand(
        (loss_replicates, ray_count),
        generator=loss_generator_b,
        dtype=values.dtype,
    )
    estimate_a = horvitz_thompson_mean(
        medium,
        high,
        proxy_probability,
        uniforms_a,
        pi_min=pi_floor,
        unsafe_mask=unsafe,
    )
    estimate_b = horvitz_thompson_mean(
        medium,
        high,
        proxy_probability,
        uniforms_b,
        pi_min=pi_floor,
        unsafe_mask=unsafe,
    )
    jvp_a = horvitz_thompson_mean(
        medium_jvp,
        high_jvp,
        proxy_probability,
        uniforms_a,
        pi_min=pi_floor,
        unsafe_mask=unsafe,
    )
    jvp_b = horvitz_thompson_mean(
        medium_jvp,
        high_jvp,
        proxy_probability,
        uniforms_b,
        pi_min=pi_floor,
        unsafe_mask=unsafe,
    )
    high_mean = torch.mean(high, dim=0)
    high_jvp_mean = torch.mean(high_jvp, dim=0)
    jvp_norm = torch.linalg.vector_norm(high_jvp_mean)
    if float(jvp_norm) > 1e-30:
        target_direction = high_jvp_mean / jvp_norm
    else:
        target_direction = torch.as_tensor(
            [1.0, -1.0],
            dtype=values.dtype,
        ) / math.sqrt(2.0)
    offset_scale = 0.25 * max(float(torch.linalg.vector_norm(high_mean)), 1e-8)
    target = high_mean + offset_scale * target_direction
    true_half_loss = 0.5 * torch.sum((high_mean - target).square())
    cross_loss = 0.5 * torch.sum(
        (estimate_a - target) * (estimate_b - target),
        dim=-1,
    )
    true_directional_derivative = torch.sum(
        (high_mean - target) * high_jvp_mean
    )
    cross_directional_derivative = 0.5 * (
        torch.sum(jvp_a * (estimate_b - target), dim=-1)
        + torch.sum(jvp_b * (estimate_a - target), dim=-1)
    )
    same_replica_loss = 0.5 * torch.sum((estimate_a - target).square(), dim=-1)
    forward_variance = float(
        conditional_trace_variance(
            medium,
            high,
            proxy_probability,
            pi_min=pi_floor,
            unsafe_mask=unsafe,
        )
    )

    def scalar_relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
        return float(
            torch.abs(candidate - reference)
            / torch.maximum(
                torch.abs(reference),
                torch.as_tensor(1e-30, dtype=values.dtype),
            )
        )

    return {
        "fixed_route_selected_count": int(selected.numel()),
        "fixed_route_state_role": "derivative_contract_only_not_estimator_replicate",
        "fixed_route_output": [float(value) for value in base_output],
        "jvp_finite_difference_relative_error": jvp_fd_error,
        "vjp_dot_relative_error": vjp_dot_error,
        "routed_jvp_mean_bias_in_trace_standard_errors": (
            derivative_bias_standard_errors
        ),
        "routed_jvp_exact_conditional_trace_variance": derivative_variance,
        "cross_loss_true": float(true_half_loss),
        "cross_loss_empirical_mean": float(torch.mean(cross_loss)),
        "cross_loss_relative_error": scalar_relative_error(
            torch.mean(cross_loss),
            true_half_loss,
        ),
        "cross_loss_directional_derivative_true": float(
            true_directional_derivative
        ),
        "cross_loss_directional_derivative_empirical_mean": float(
            torch.mean(cross_directional_derivative)
        ),
        "cross_loss_directional_derivative_relative_error": scalar_relative_error(
            torch.mean(cross_directional_derivative),
            true_directional_derivative,
        ),
        "same_replica_loss_empirical_mean": float(torch.mean(same_replica_loss)),
        "same_replica_predicted_positive_bias": 0.5 * forward_variance,
        "probabilities_detached": not proxy_probability.requires_grad,
    }


def _case_scale_result(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    case: dict[str, Any],
    config: dict[str, Any],
    *,
    multiplier: float,
) -> dict[str, Any]:
    case_id = str(case["id"])
    delta = float(config["difference_step"])
    scale = float(config["base_refractivity_scale"]) * float(multiplier)
    step_counts = {key: int(value) for key, value in config["route_step_counts"].items()}
    low = _straight_route(
        values,
        states,
        rig,
        gradient_mode="automatic",
        difference_step=delta,
        refractivity_scale=scale,
        step_count=step_counts["low_automatic"],
        create_graph=False,
    )
    medium = _straight_route(
        values,
        states,
        rig,
        gradient_mode="central",
        difference_step=delta,
        refractivity_scale=scale,
        step_count=step_counts["medium_straight_central"],
        create_graph=False,
    )
    high, high_trace = _high_route(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=step_counts["high_curved_central"],
        create_graph=False,
    )
    high_reference, reference_trace = _high_route(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=step_counts["high_reference"],
        create_graph=False,
    )
    high_reference_error = relative_l2(high, high_reference)
    certificate_config = config["certificate"]
    support_threshold = float(
        certificate_config["support_threshold_fraction_of_grid_peak"]
    ) * float(torch.max(torch.abs(values)))
    certificate_arguments = {
        "refractivity_scale": scale,
        "difference_step": delta,
        "support_threshold": support_threshold,
        "frustum_half_width_u": float(certificate_config["frustum_half_width_u"]),
        "frustum_half_width_v": float(certificate_config["frustum_half_width_v"]),
        "support_interval_count": int(certificate_config["support_interval_count"]),
        "numerical_path_buffer": float(certificate_config["numerical_path_buffer"]),
    }
    certificate = low_path_safety_certificate(
        values,
        states,
        rig,
        **certificate_arguments,
    )
    active_safe = certificate.domain_frustum_safe_mask
    active_unsafe = ~active_safe
    strict_safe = certificate.safe_mask
    topology = path_topology_diagnostics(
        values,
        reference_trace,
        support_threshold=support_threshold,
        frustum_half_width_u=float(certificate_config["frustum_half_width_u"]),
        frustum_half_width_v=float(certificate_config["frustum_half_width_v"]),
    )
    geometry_false_safe = sum(
        bool(active_safe[index]) and topology.frustum_violations_per_ray[index]
        for index in range(len(states))
    )
    strict_support_false_safe = sum(
        bool(strict_safe[index])
        and certificate.straight_support_crossings_per_ray[index]
        != topology.support_crossings_per_ray[index]
        for index in range(len(states))
    )
    routing = config["routing"]
    probabilities, expected_high_fraction = _route_probabilities(
        certificate.residual_risk_proxy,
        high - medium,
        active_unsafe,
        pi_floor=float(routing["pi_floor"]),
        safe_average_probability=float(routing["safe_ray_average_probability"]),
    )
    replay = {
        name: _replay_metrics(
            medium,
            high,
            probability,
            active_unsafe,
            pi_floor=float(routing["pi_floor"]),
            replicate_count=int(routing["monte_carlo_replicates"]),
            seed=stable_seed(
                config["seed_roles"]["routing_uniform_base"],
                case_id,
                multiplier,
                name,
            ),
        )
        for name, probability in probabilities.items()
    }
    high_only_probability = torch.full_like(
        probabilities["proxy"],
        float(torch.mean(probabilities["proxy"])),
    )
    replay["high_only_equal_high_call_fraction"] = _replay_metrics(
        torch.zeros_like(high),
        high,
        high_only_probability,
        torch.zeros_like(active_unsafe),
        pi_floor=float(torch.min(high_only_probability)),
        replicate_count=int(routing["monte_carlo_replicates"]),
        seed=stable_seed(
            config["seed_roles"]["routing_uniform_base"],
            case_id,
            multiplier,
            "high-only",
        ),
    )
    sparse, selected = _sparse_execution_audit(
        values,
        states,
        rig,
        medium,
        high,
        probabilities["proxy"],
        active_unsafe,
        difference_step=delta,
        refractivity_scale=scale,
        step_count=step_counts["high_curved_central"],
        pi_floor=float(routing["pi_floor"]),
        seed=stable_seed(
            config["seed_roles"]["sparse_execution_base"],
            case_id,
            multiplier,
        ),
    )
    total_residual_variance = trace_sample_variance(high - low)
    trajectory_residual_variance = trace_sample_variance(high - medium)
    gradient_residual_variance = trace_sample_variance(medium - low)
    trajectory_ratio = trajectory_residual_variance / max(
        total_residual_variance,
        1e-30,
    )
    uniform_variance = replay["uniform"]["exact_conditional_trace_variance"]
    proxy_variance = replay["proxy"]["exact_conditional_trace_variance"]
    oracle_variance = replay["oracle"]["exact_conditional_trace_variance"]
    proxy_ratio = 1.0 if uniform_variance == 0.0 else proxy_variance / uniform_variance
    oracle_ratio = 1.0 if uniform_variance == 0.0 else oracle_variance / uniform_variance
    target = torch.mean(high, dim=0)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-30)
    screens = config["development_screens"]
    numerical_contract_met = bool(
        high_reference_error
        <= float(screens["maximum_high_reference_relative_l2"])
        and sparse["sparse_to_replay_relative_l2"]
        <= float(screens["maximum_sparse_replay_relative_l2"])
        and geometry_false_safe
        <= int(screens["maximum_observed_geometry_false_safe_count"])
        and all(
            item["empirical_to_exact_relative_error"]
            <= float(screens["maximum_empirical_to_exact_variance_relative_error"])
            for item in replay.values()
        )
    )
    proxy_screen_met = bool(
        int(torch.sum(active_safe)) > 0
        and proxy_ratio
        <= float(screens["maximum_proxy_to_uniform_exact_variance_ratio"])
    )
    oracle_screen_met = bool(
        int(torch.sum(active_safe)) > 0
        and oracle_ratio
        <= float(screens["maximum_oracle_to_uniform_exact_variance_ratio"])
    )
    mechanism_screen_met = bool(
        trajectory_ratio
        <= float(
            screens[
                "maximum_trajectory_residual_to_total_residual_variance_ratio"
            ]
        )
    )
    result: dict[str, Any] = {
        "case_id": case_id,
        "phantom_family": str(case["phantom_family"]),
        "phantom_seed": int(case["phantom_seed"]),
        "dimensionless_stress_multiplier": float(multiplier),
        "refractivity_scale": scale,
        "route_roles": {
            "low": "straight_automatic_gradient",
            "medium": "straight_central_difference",
            "high": "field_dependent_curved_central_difference",
        },
        "reference_relative_l2": high_reference_error,
        "mechanism_decomposition": {
            "total_high_minus_low_trace_variance": total_residual_variance,
            "gradient_medium_minus_low_trace_variance": gradient_residual_variance,
            "trajectory_high_minus_medium_trace_variance": (
                trajectory_residual_variance
            ),
            "trajectory_to_total_residual_variance_ratio": trajectory_ratio,
            "automatic_only_mean_relative_bias": float(
                torch.linalg.vector_norm(torch.mean(low, dim=0) - target) / target_norm
            ),
            "medium_only_mean_relative_bias": float(
                torch.linalg.vector_norm(torch.mean(medium, dim=0) - target)
                / target_norm
            ),
        },
        "certificate": {
            "active_contract": "domain_and_synthetic_frustum_only",
            "support_contract": "report_only_renderer_has_no_mask_branch",
            "active_safe_count": int(torch.sum(active_safe)),
            "active_unsafe_count": int(torch.sum(active_unsafe)),
            "strict_future_mask_safe_count": int(torch.sum(strict_safe)),
            "observed_geometry_false_safe_count": int(geometry_false_safe),
            "observed_strict_support_false_safe_count": int(
                strict_support_false_safe
            ),
            "continuous_path_deviation_bound_maximum": (
                certificate.continuous_path_deviation_bound
            ),
            "minimum_certified_frustum_margin": float(
                torch.min(certificate.certified_frustum_margin)
            ),
            "minimum_certified_domain_margin": float(
                torch.min(certificate.certified_domain_margin)
            ),
            "outward_rounded_interval_arithmetic": bool(
                certificate_config["outward_rounded_interval_arithmetic"]
            ),
        },
        "routing": {
            "expected_high_route_fraction": expected_high_fraction,
            "residual_risk_to_true_residual_rank_correlation": _rank_correlation(
                certificate.residual_risk_proxy,
                torch.linalg.vector_norm(high - medium, dim=-1),
            ),
            "proxy_to_uniform_exact_variance_ratio": proxy_ratio,
            "oracle_to_uniform_exact_variance_ratio": oracle_ratio,
            "replay_metrics": replay,
            "sparse_execution": sparse,
        },
        "screens": {
            "numerical_contract_met": numerical_contract_met,
            "mechanism_headroom_met": mechanism_screen_met,
            "proxy_router_screen_met": proxy_screen_met,
            "oracle_router_headroom_met": oracle_screen_met,
        },
    }
    if float(multiplier) == 1.0:
        timing = config["timing"]
        result["timing_and_cost"] = _timing_and_cost_audit(
            values,
            states,
            rig,
            probabilities["proxy"],
            selected,
            certificate_arguments,
            difference_step=delta,
            refractivity_scale=scale,
            step_counts=step_counts,
            pi_floor=float(routing["pi_floor"]),
            unsafe_mask=active_unsafe,
            warmup_repeats=int(timing["warmup_repeats"]),
            measured_repeats=int(timing["measured_repeats"]),
        )
        result["screens"]["implementation_cost_screen_met"] = bool(
            result["timing_and_cost"]["primitive_contract"][
                "expected_routed_to_full_high_field_query_ratio"
            ]
            <= float(screens["maximum_expected_field_query_cost_ratio"])
            and result["timing_and_cost"]["wall_time_seconds_machine_specific"][
                "observed_end_to_end_ratio_to_full_high"
            ]
            <= float(screens["maximum_observed_end_to_end_wall_time_ratio"])
        )
        derivative_contract = _derivative_contract(
            values,
            rig,
            config,
            case_id,
            refractivity_scale=scale,
            support_threshold=support_threshold,
        )
        derivative_screen_met = bool(
            derivative_contract["jvp_finite_difference_relative_error"]
            <= float(screens["maximum_jvp_finite_difference_relative_error"])
            and derivative_contract["vjp_dot_relative_error"]
            <= float(screens["maximum_vjp_dot_relative_error"])
            and derivative_contract[
                "routed_jvp_mean_bias_in_trace_standard_errors"
            ]
            <= float(
                screens[
                    "maximum_derivative_mean_relative_error_in_standard_errors"
                ]
            )
            and derivative_contract["cross_loss_relative_error"]
            <= float(screens["maximum_cross_loss_relative_error"])
            and derivative_contract[
                "cross_loss_directional_derivative_relative_error"
            ]
            <= float(
                screens[
                    "maximum_cross_loss_directional_derivative_relative_error"
                ]
            )
        )
        derivative_contract["screen_met"] = derivative_screen_met
        result["derivative_contract"] = derivative_contract
    return result


def _case_results(case: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
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
    states = sample_pupil_sobol(
        int(config["population_count"]),
        seed=stable_seed(
            config["seed_roles"]["population_state_base"],
            case["id"],
        ),
    )
    rig = _rig_from_case(case)
    return [
        _case_scale_result(
            values,
            states,
            rig,
            case,
            config,
            multiplier=float(multiplier),
        )
        for multiplier in config["dimensionless_stress_scale_multipliers"]
    ]


def _write_main_figure(rows: list[dict[str, Any]], path: Path) -> None:
    case_ids = sorted({row["case_id"] for row in rows})
    colors = dict(zip(case_ids, ("#176b67", "#a34e3f", "#405a8a"), strict=False))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for case_id in case_ids:
        case_rows = sorted(
            (row for row in rows if row["case_id"] == case_id),
            key=lambda row: row["dimensionless_stress_multiplier"],
        )
        x = [row["dimensionless_stress_multiplier"] for row in case_rows]
        axes[0, 0].plot(
            x,
            [
                row["mechanism_decomposition"][
                    "trajectory_to_total_residual_variance_ratio"
                ]
                for row in case_rows
            ],
            marker="o",
            color=colors[case_id],
            label=case_id.replace("_", " "),
        )
        axes[0, 1].plot(
            x,
            [row["routing"]["proxy_to_uniform_exact_variance_ratio"] for row in case_rows],
            marker="o",
            color=colors[case_id],
            label=f"proxy: {case_id.replace('_', ' ')}",
        )
        axes[0, 1].plot(
            x,
            [row["routing"]["oracle_to_uniform_exact_variance_ratio"] for row in case_rows],
            marker="x",
            linestyle="--",
            color=colors[case_id],
            label=f"oracle: {case_id.replace('_', ' ')}",
        )
        population = max(
            row["certificate"]["active_safe_count"]
            + row["certificate"]["active_unsafe_count"]
            for row in case_rows
        )
        axes[1, 0].plot(
            x,
            [row["certificate"]["active_safe_count"] / population for row in case_rows],
            marker="o",
            color=colors[case_id],
            label=f"geometry: {case_id.replace('_', ' ')}",
        )
        axes[1, 0].plot(
            x,
            [
                row["certificate"]["strict_future_mask_safe_count"] / population
                for row in case_rows
            ],
            marker="x",
            linestyle="--",
            color=colors[case_id],
            label=f"strict mask: {case_id.replace('_', ' ')}",
        )

    base_rows = [row for row in rows if row["dimensionless_stress_multiplier"] == 1.0]
    labels = [row["case_id"].replace("_", "\n") for row in base_rows]
    x_positions = np.arange(len(base_rows))
    axes[1, 1].bar(
        x_positions - 0.18,
        [
            row["timing_and_cost"]["primitive_contract"][
                "expected_routed_to_full_high_field_query_ratio"
            ]
            for row in base_rows
        ],
        width=0.36,
        color="#176b67",
        label="field-query contract",
    )
    axes[1, 1].bar(
        x_positions + 0.18,
        [
            row["timing_and_cost"]["wall_time_seconds_machine_specific"][
                "observed_end_to_end_ratio_to_full_high"
            ]
            for row in base_rows
        ],
        width=0.36,
        color="#c79537",
        label="Mac wall time",
    )
    axes[1, 1].set_xticks(x_positions, labels)

    axes[0, 0].set_title("trajectory residual variance / original mixed residual")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(0.1, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("dimensionless stress multiplier")
    axes[0, 0].set_ylabel("variance ratio")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("routing variance relative to constant-probability HT")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(0.9, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("dimensionless stress multiplier")
    axes[0, 1].set_ylabel("exact conditional variance ratio")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].set_title("certificate headroom: active geometry vs future mask")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_xlabel("dimensionless stress multiplier")
    axes[1, 0].set_ylabel("certified ray fraction")
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 1].set_title("base-scale routed cost / full high cost")
    axes[1, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("ratio; below one is required")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(
        "N2-PVGR-N0 development: mechanism headroom is not a paper claim",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_derivative_figure(rows: list[dict[str, Any]], path: Path) -> None:
    base_rows = [row for row in rows if "derivative_contract" in row]
    labels = [row["case_id"].replace("_", "\n") for row in base_rows]
    metrics = (
        ("jvp_finite_difference_relative_error", "JVP vs FD"),
        ("vjp_dot_relative_error", "VJP dot"),
        ("cross_loss_relative_error", "cross loss"),
        (
            "cross_loss_directional_derivative_relative_error",
            "cross-loss derivative",
        ),
    )
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.18
    colors = ("#176b67", "#405a8a", "#c79537", "#a34e3f")
    for index, ((key, label), color) in enumerate(zip(metrics, colors, strict=False)):
        axis.bar(
            x + (index - 1.5) * width,
            [max(row["derivative_contract"][key], 1e-16) for row in base_rows],
            width=width,
            color=color,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_ylabel("relative error")
    axis.set_title(
        "Frozen-route derivative and independent-replica loss contracts (development)"
    )
    axis.legend(ncol=4, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    reserved = set(str(value) for value in config["reserved_audit_families_not_opened"])
    development_families = {
        str(case["phantom_family"]) for case in config["development_cases"]
    }
    overlap = sorted(reserved & development_families)
    if overlap:
        raise RuntimeError(
            "reserved audit families must remain unopened: " + ", ".join(overlap)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in config["development_cases"]:
        rows.extend(_case_results(case, config))

    base_rows = [row for row in rows if row["dimensionless_stress_multiplier"] == 1.0]
    numerical_count = sum(row["screens"]["numerical_contract_met"] for row in rows)
    mechanism_count = sum(row["screens"]["mechanism_headroom_met"] for row in rows)
    proxy_count = sum(row["screens"]["proxy_router_screen_met"] for row in rows)
    oracle_count = sum(row["screens"]["oracle_router_headroom_met"] for row in rows)
    derivative_count = sum(
        row.get("derivative_contract", {}).get("screen_met", False)
        for row in base_rows
    )
    implementation_cost_count = sum(
        row["screens"].get("implementation_cost_screen_met", False)
        for row in base_rows
    )
    if numerical_count != len(rows):
        diagnosis = "NUMERICAL_OR_STATISTICAL_CONTRACT_FAILED"
    elif proxy_count == 0 and oracle_count > 0:
        diagnosis = "ORACLE_HEADROOM_CURRENT_PROXY_AND_IMPLEMENTATION_NO_GO"
    elif mechanism_count == len(rows) and proxy_count < len(rows):
        diagnosis = "TRIFIDELITY_MECHANISM_HEADROOM_PROXY_ROUTER_NOT_ESTABLISHED"
    elif mechanism_count == len(rows) and proxy_count == len(rows):
        diagnosis = "PROXY_ROUTER_DEVELOPMENT_SCREEN_MET_NOT_AUDIT"
    else:
        diagnosis = "TRIFIDELITY_MECHANISM_HEADROOM_NOT_UNIFORM"
    result: dict[str, Any] = {
        "schema": str(config["schema"]),
        "candidate_id": str(config["candidate_id"]),
        "machine_decision": str(config["hard_conclusion"]),
        "development_diagnosis": diagnosis,
        "case_scale_count": len(rows),
        "numerical_contract_count": int(numerical_count),
        "mechanism_headroom_count": int(mechanism_count),
        "proxy_router_screen_count": int(proxy_count),
        "oracle_router_headroom_count": int(oracle_count),
        "derivative_contract_count": int(derivative_count),
        "implementation_cost_screen_count": int(implementation_cost_count),
        "reserved_audit_families_not_opened": sorted(reserved),
        "claim_boundary": [
            "finite-population synthetic development only",
            "complete high replays are audit evidence, not sparse execution cost",
            "support certificate is report-only because this renderer has no mask branch",
            "float64 derivative bounds are not outward-rounded interval proofs",
            "wall times are machine-specific and cannot establish algorithmic speedup",
            "no reconstruction, real BOST, physical units, generalization, or novelty claim",
        ],
        "rows": rows,
        "figures": [
            "n2_pvgr_n0_trifidelity_development.png",
            "n2_pvgr_n0_derivative_contract.png",
        ],
    }
    result_path = output_dir / "result.json"
    config_snapshot_path = output_dir / "config_snapshot.json"
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.md"
    main_figure_path = output_dir / result["figures"][0]
    derivative_figure_path = output_dir / result["figures"][1]
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_snapshot_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "stress_multiplier",
            "reference_relative_l2",
            "trajectory_to_total_variance_ratio",
            "active_safe_count",
            "strict_mask_safe_count",
            "expected_high_fraction",
            "risk_residual_rank_correlation",
            "proxy_to_uniform_variance_ratio",
            "oracle_to_uniform_variance_ratio",
            "sparse_replay_relative_l2",
            "numerical_contract_met",
            "mechanism_headroom_met",
            "proxy_router_screen_met",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "stress_multiplier": row["dimensionless_stress_multiplier"],
                    "reference_relative_l2": row["reference_relative_l2"],
                    "trajectory_to_total_variance_ratio": row[
                        "mechanism_decomposition"
                    ]["trajectory_to_total_residual_variance_ratio"],
                    "active_safe_count": row["certificate"]["active_safe_count"],
                    "strict_mask_safe_count": row["certificate"][
                        "strict_future_mask_safe_count"
                    ],
                    "expected_high_fraction": row["routing"][
                        "expected_high_route_fraction"
                    ],
                    "risk_residual_rank_correlation": row["routing"][
                        "residual_risk_to_true_residual_rank_correlation"
                    ],
                    "proxy_to_uniform_variance_ratio": row["routing"][
                        "proxy_to_uniform_exact_variance_ratio"
                    ],
                    "oracle_to_uniform_variance_ratio": row["routing"][
                        "oracle_to_uniform_exact_variance_ratio"
                    ],
                    "sparse_replay_relative_l2": row["routing"]["sparse_execution"][
                        "sparse_to_replay_relative_l2"
                    ],
                    "numerical_contract_met": row["screens"][
                        "numerical_contract_met"
                    ],
                    "mechanism_headroom_met": row["screens"][
                        "mechanism_headroom_met"
                    ],
                    "proxy_router_screen_met": row["screens"][
                        "proxy_router_screen_met"
                    ],
                }
            )
    _write_main_figure(rows, main_figure_path)
    _write_derivative_figure(rows, derivative_figure_path)
    summary_lines = [
        "# N2-PVGR-N0 trifidelity development result",
        "",
        f"- Hard conclusion: `{result['machine_decision']}`.",
        f"- Development diagnosis: `{result['development_diagnosis']}`.",
        f"- Numerical/statistical contracts: {numerical_count}/{len(rows)}.",
        f"- Trifidelity mechanism headroom: {mechanism_count}/{len(rows)}.",
        f"- Proxy router screen: {proxy_count}/{len(rows)}.",
        f"- Oracle routing headroom: {oracle_count}/{len(rows)}.",
        f"- Base-scale derivative contracts: {derivative_count}/{len(base_rows)}.",
        f"- Base-scale implementation cost screens: {implementation_cost_count}/{len(base_rows)}.",
        "",
        "The medium route removes gradient-formulation mismatch before the randomized",
        "curved-path correction. Complete high replays validate the finite-population",
        "statistics; the sparse audit separately proves unselected high rays are not",
        "required online. These are synthetic mechanism results only.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    manifest_files = [
        result_path,
        config_snapshot_path,
        metrics_path,
        summary_path,
        main_figure_path,
        derivative_figure_path,
    ]
    manifest = {
        "schema": str(config["schema"]),
        "source_sha256": {
            "runner": sha256(Path(__file__)),
            "config": sha256(config_path),
            "field_dependent_ray": sha256(
                ROOT / "demo_t16_operator" / "field_dependent_ray.py"
            ),
            "ray_safety_certificate": sha256(
                ROOT / "demo_t16_operator" / "ray_safety_certificate.py"
            ),
            "topology_certified_routing": sha256(
                ROOT / "demo_t16_operator" / "topology_certified_routing.py"
            ),
            "analytic_phantoms": sha256(
                ROOT / "demo_t16_operator" / "analytic_bost_phantoms.py"
            ),
        },
        "files": {path.name: sha256(path) for path in manifest_files},
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
                "development_diagnosis": result["development_diagnosis"],
                "case_scale_count": result["case_scale_count"],
                "proxy_router_screen_count": result["proxy_router_screen_count"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
