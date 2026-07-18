"""Cancellation-aware residual quadrature for the N2-PVGR N5 audit.

The frozen N4 observable is the difference between a curved-path midpoint
integral and its straight-path counterpart.  Linearity of the midpoint rule
means that subtracting matched integrands before summation must represent the
same discrete observable as subtracting the two completed sums.  This module
implements that paired ordering without changing the ray equation, gradient
stencil, nodes, detector basis, or logical field-query budget.

It is a numerical-reference mechanism, not a reconstruction algorithm.  In
particular, improved summation agreement cannot authorize a neural operator or
replace step-refinement and experimental noise-floor checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import torch

try:
    from .automatic_discrete_multifidelity import SyntheticRayRig, smoothstep_grid_field
    from .field_dependent_ray import (
        RayDomainError,
        RayTraceResult,
        initial_pupil_rays,
        trace_field_dependent_rays,
    )
except ImportError:
    from automatic_discrete_multifidelity import SyntheticRayRig, smoothstep_grid_field
    from field_dependent_ray import (
        RayDomainError,
        RayTraceResult,
        initial_pupil_rays,
        trace_field_dependent_rays,
    )


CANCELLATION_AWARE_RESIDUAL_SCHEMA = "n2-pvgr-cancellation-aware-residual-1.0"
_DOMAIN_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PairedResidualQueryAccounting:
    ray_count: int
    step_count: int
    curved_trace_point_queries: int
    paired_integrand_point_queries: int
    total_field_point_queries: int
    paired_interpolation_calls: int

    def __post_init__(self) -> None:
        midpoint_count = self.ray_count * self.step_count
        if (
            self.curved_trace_point_queries != 28 * midpoint_count
            or self.paired_integrand_point_queries != 14 * midpoint_count
            or self.total_field_point_queries != 42 * midpoint_count
            or self.paired_interpolation_calls != 1
        ):
            raise ValueError("paired residual query accounting drifted")


@dataclass(frozen=True, slots=True)
class PairedResidualEvaluation:
    """Matched discrete outputs under four accumulation orderings."""

    trace: RayTraceResult
    curved_output_naive: torch.Tensor
    straight_output_naive: torch.Tensor
    raw_separate_subtraction: torch.Tensor
    paired_naive: torch.Tensor
    paired_pairwise: torch.Tensor
    paired_neumaier: torch.Tensor
    separate_neumaier_subtraction: torch.Tensor
    minimum_paired_domain_margin: float
    minimum_paired_stencil_margin: float
    maximum_paired_direction_norm_error: float
    query_accounting: PairedResidualQueryAccounting

    def __post_init__(self) -> None:
        outputs = (
            self.curved_output_naive,
            self.straight_output_naive,
            self.raw_separate_subtraction,
            self.paired_naive,
            self.paired_pairwise,
            self.paired_neumaier,
            self.separate_neumaier_subtraction,
        )
        expected = (self.query_accounting.ray_count, 2)
        if any(tuple(value.shape) != expected for value in outputs):
            raise ValueError("paired residual outputs have an invalid shape")
        if any(value.dtype != torch.float64 for value in outputs):
            raise TypeError("paired residual outputs must use float64")
        if any(not bool(torch.all(torch.isfinite(value))) for value in outputs):
            raise RayDomainError("paired residual output became non-finite")


def _validated_step_count(step_count: int) -> int:
    try:
        steps = operator.index(step_count)
    except TypeError as error:
        raise TypeError("step_count must be an integer") from error
    if isinstance(step_count, bool) or steps < 2:
        raise ValueError("step_count must be an integer of at least two")
    return int(steps)


def _validate_inputs(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    values = torch.as_tensor(values_zyx)
    states = torch.as_tensor(pupil_states)
    if values.ndim != 3 or any(int(size) < 3 for size in values.shape):
        raise ValueError("values_zyx must have shape [z,y,x] with size at least three")
    if values.dtype != torch.float64:
        raise TypeError("values_zyx must use float64")
    if states.ndim != 2 or states.shape[1] != 2 or len(states) < 1:
        raise ValueError("pupil_states must have shape [ray,2]")
    if states.dtype != torch.float64:
        raise TypeError("pupil_states must use float64")
    if values.device != states.device:
        raise ValueError("values and pupil states must share a device")
    if not bool(torch.all(torch.isfinite(values))) or not bool(
        torch.all(torch.isfinite(states))
    ):
        raise ValueError("values and pupil states must be finite")
    if bool(torch.any((states < 0.0) | (states > 1.0))):
        raise ValueError("pupil_states must lie in [0,1]^2")
    delta = float(difference_step)
    scale = float(refractivity_scale)
    if not math.isfinite(delta) or not 0.0 < delta < 0.25:
        raise ValueError("difference_step must lie in (0,0.25)")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("refractivity_scale must be finite and positive")
    return values, states, delta, scale, _validated_step_count(step_count)


def pairwise_sum(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Balanced binary-tree sum with an explicit, reproducible ordering."""

    tensor = torch.as_tensor(values)
    if tensor.shape[dim] < 1:
        raise ValueError("pairwise_sum requires a nonempty dimension")
    work = torch.movedim(tensor, dim, 0)
    while len(work) > 1:
        pair_count = len(work) // 2
        paired = work[: 2 * pair_count : 2] + work[1 : 2 * pair_count : 2]
        work = torch.cat((paired, work[-1:]), dim=0) if len(work) % 2 else paired
    return work[0]


def neumaier_sum(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Neumaier-compensated sum along one dimension."""

    tensor = torch.as_tensor(values)
    if tensor.shape[dim] < 1:
        raise ValueError("neumaier_sum requires a nonempty dimension")
    work = torch.movedim(tensor, dim, 0)
    total = torch.zeros_like(work[0])
    compensation = torch.zeros_like(work[0])
    for value in work:
        updated = total + value
        correction = torch.where(
            torch.abs(total) >= torch.abs(value),
            (total - updated) + value,
            (value - updated) + total,
        )
        compensation = compensation + correction
        total = updated
    return total + compensation


def _paired_field_and_gradient(
    values_zyx: torch.Tensor,
    positions: torch.Tensor,
    *,
    difference_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Query values and central gradients for curved/straight nodes together."""

    flat = positions.reshape(-1, 3)
    identity = torch.eye(3, dtype=flat.dtype, device=flat.device)
    offsets = torch.cat(
        (
            torch.zeros((1, 3), dtype=flat.dtype, device=flat.device),
            difference_step * identity,
            -difference_step * identity,
        ),
        dim=0,
    )
    queries = flat[:, None, :] + offsets[None, :, :]
    if bool(torch.any(torch.abs(queries) > 1.0 + _DOMAIN_TOLERANCE)):
        raise RayDomainError("paired central-difference query left the grid domain")
    samples = smoothstep_grid_field(
        values_zyx,
        queries.reshape(-1, 3),
    ).reshape(len(flat), 7)
    scalar = samples[:, 0]
    gradient = (samples[:, 1:4] - samples[:, 4:7]) / (2.0 * difference_step)
    return scalar, gradient


def _paired_integrands(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    trace: RayTraceResult,
    *,
    difference_step: float,
    refractivity_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    curved_positions = 0.5 * (trace.positions[:, :-1] + trace.positions[:, 1:])
    curved_directions = 0.5 * (trace.directions[:, :-1] + trace.directions[:, 1:])
    start, straight_direction, projection_u, projection_v = initial_pupil_rays(
        pupil_states,
        rig,
    )
    start = start.to(dtype=torch.float64, device=values_zyx.device)
    straight_direction = straight_direction.to(dtype=torch.float64, device=values_zyx.device)
    projection_u = projection_u.to(dtype=torch.float64, device=values_zyx.device)
    projection_v = projection_v.to(dtype=torch.float64, device=values_zyx.device)
    steps = curved_positions.shape[1]
    midpoint_distance = (
        torch.arange(steps, dtype=torch.float64, device=values_zyx.device) + 0.5
    ) * float(trace.step_size)
    straight_positions = (
        start[:, None, :]
        + midpoint_distance[None, :, None] * straight_direction[:, None, :]
    )
    straight_directions = straight_direction[:, None, :].expand(-1, steps, -1)

    positions = torch.stack((curved_positions, straight_positions), dim=0)
    directions = torch.stack((curved_directions, straight_directions), dim=0)
    scalar, gradient = _paired_field_and_gradient(
        values_zyx,
        positions,
        difference_step=difference_step,
    )
    scalar = scalar.reshape(2, len(pupil_states), steps)
    gradient = gradient.reshape(2, len(pupil_states), steps, 3)
    unit_direction = directions / torch.linalg.vector_norm(
        directions,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-30)
    refractive_index = 1.0 + refractivity_scale * scalar
    if not bool(torch.all(torch.isfinite(refractive_index))) or bool(
        torch.any(refractive_index <= 0.5)
    ):
        raise RayDomainError("paired refractive index became invalid")
    gradient_n = refractivity_scale * gradient
    longitudinal = torch.sum(gradient_n * unit_direction, dim=-1, keepdim=True)
    curvature = (gradient_n - longitudinal * unit_direction) / refractive_index[..., None]
    basis_u = projection_u[None, :, None, :]
    basis_v = projection_v[None, :, None, :]
    projected = torch.stack(
        (
            torch.sum(curvature * basis_u, dim=-1),
            torch.sum(curvature * basis_v, dim=-1),
        ),
        dim=-1,
    )
    domain_margin = 1.0 - torch.amax(torch.abs(positions), dim=-1)
    stencil_margin = domain_margin - difference_step
    direction_error = torch.max(
        torch.abs(torch.linalg.vector_norm(unit_direction, dim=-1) - 1.0)
    )
    if bool(torch.any(stencil_margin < -_DOMAIN_TOLERANCE)):
        raise RayDomainError("paired midpoint stencil left the grid domain")
    return (
        projected[0],
        projected[1],
        float(torch.min(domain_margin)),
        float(torch.min(stencil_margin)),
        float(direction_error),
    )


def evaluate_paired_residual(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
) -> PairedResidualEvaluation:
    """Evaluate one N5 paired residual without changing the N4 observable."""

    values, states, delta, scale, steps = _validate_inputs(
        values_zyx,
        pupil_states,
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
    )
    trace = trace_field_dependent_rays(
        values,
        states,
        rig,
        gradient_mode="central",
        difference_step=delta,
        refractivity_scale=scale,
        step_count=steps,
        create_graph=False,
    )
    curved, straight, domain_margin, stencil_margin, direction_error = _paired_integrands(
        values,
        states,
        rig,
        trace,
        difference_step=delta,
        refractivity_scale=scale,
    )
    step_size = float(trace.step_size)
    curved_naive = torch.sum(curved, dim=1) * step_size
    straight_naive = torch.sum(straight, dim=1) * step_size
    residual_integrand = curved - straight
    raw = curved_naive - straight_naive
    paired_naive = torch.sum(residual_integrand, dim=1) * step_size
    paired_pairwise = pairwise_sum(residual_integrand, dim=1) * step_size
    paired_neumaier = neumaier_sum(residual_integrand, dim=1) * step_size
    separate_neumaier = (
        neumaier_sum(curved, dim=1) - neumaier_sum(straight, dim=1)
    ) * step_size
    ray_count = len(states)
    accounting = PairedResidualQueryAccounting(
        ray_count=ray_count,
        step_count=steps,
        curved_trace_point_queries=28 * ray_count * steps,
        paired_integrand_point_queries=14 * ray_count * steps,
        total_field_point_queries=42 * ray_count * steps,
        paired_interpolation_calls=1,
    )
    return PairedResidualEvaluation(
        trace=trace,
        curved_output_naive=curved_naive,
        straight_output_naive=straight_naive,
        raw_separate_subtraction=raw,
        paired_naive=paired_naive,
        paired_pairwise=paired_pairwise,
        paired_neumaier=paired_neumaier,
        separate_neumaier_subtraction=separate_neumaier,
        minimum_paired_domain_margin=domain_margin,
        minimum_paired_stencil_margin=stencil_margin,
        maximum_paired_direction_norm_error=direction_error,
        query_accounting=accounting,
    )


__all__ = [
    "CANCELLATION_AWARE_RESIDUAL_SCHEMA",
    "PairedResidualEvaluation",
    "PairedResidualQueryAccounting",
    "evaluate_paired_residual",
    "neumaier_sum",
    "pairwise_sum",
]
