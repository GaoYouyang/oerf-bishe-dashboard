"""Differentiable field-dependent ray tracing for BOST mechanism audits.

The geometric-optics state follows

    d(n d) / ds = grad(n),

where ``d`` is the unit ray direction.  Expanding the derivative gives the
arc-length system

    dp / ds = d,
    dd / ds = (grad(n) - d * dot(d, grad(n))) / n.

This module is intentionally a small, clean-room audit kernel.  It does not
model a calibrated camera, an experimental BOS image, or a neural field.  Its
purpose is to expose the trajectory derivative that a prescribed straight or
fixed-bend path cannot test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    from .automatic_discrete_multifidelity import (
        SyntheticRayRig,
        central_difference_spatial_gradient,
        joint_state_geometry,
        smoothstep_grid_field,
    )
except ImportError:
    from automatic_discrete_multifidelity import (
        SyntheticRayRig,
        central_difference_spatial_gradient,
        joint_state_geometry,
        smoothstep_grid_field,
    )


FIELD_DEPENDENT_RAY_SCHEMA = "field-dependent-refractive-ray-1.0"


class RayDomainError(RuntimeError):
    """Raised when a trace or derivative stencil leaves its frozen domain."""


@dataclass(frozen=True)
class RayTraceResult:
    """Differentiable ray states plus detached geometric diagnostics."""

    positions: torch.Tensor
    directions: torch.Tensor
    projection_u: torch.Tensor
    projection_v: torch.Tensor
    step_size: float
    gradient_mode: str
    minimum_domain_margin: float
    minimum_stencil_margin: float
    maximum_direction_norm_error: float


@dataclass(frozen=True)
class PathTopologyDiagnostics:
    """Discrete support and synthetic-frustum signature for one trace."""

    support_crossings_per_ray: tuple[int, ...]
    frustum_violations_per_ray: tuple[bool, ...]
    minimum_frustum_margin: float
    minimum_domain_margin: float


def sample_pupil_sobol(
    count: int,
    *,
    seed: int,
    scramble: bool = True,
) -> torch.Tensor:
    """Draw finite pupil states in ``[0,1]^2``."""

    if int(count) < 1:
        raise ValueError("count must be positive")
    engine = torch.quasirandom.SobolEngine(
        dimension=2,
        scramble=bool(scramble),
        seed=int(seed),
    )
    return engine.draw(int(count)).to(torch.float64)


def _validated_pupil_states(states: torch.Tensor) -> torch.Tensor:
    unit = torch.as_tensor(states)
    if unit.ndim != 2 or unit.shape[1] != 2 or len(unit) < 1:
        raise ValueError("pupil states must have shape [ray,2]")
    if not unit.is_floating_point():
        unit = unit.to(torch.float64)
    if torch.any(~torch.isfinite(unit)) or torch.any(unit < 0.0) or torch.any(unit > 1.0):
        raise ValueError("pupil states must be finite and lie in [0,1]^2")
    return unit


def initial_pupil_rays(
    states: torch.Tensor,
    rig: SyntheticRayRig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return entrance points, directions, and detector-plane basis vectors."""

    unit = _validated_pupil_states(states)
    zeros = torch.zeros((len(unit), 1), dtype=unit.dtype, device=unit.device)
    ones = torch.ones_like(zeros)
    start, projection_u, projection_v = joint_state_geometry(
        torch.cat((unit, zeros), dim=1),
        rig,
        high_geometry=False,
    )
    end, _, _ = joint_state_geometry(
        torch.cat((unit, ones), dim=1),
        rig,
        high_geometry=False,
    )
    direction = end - start
    direction = direction / torch.linalg.vector_norm(
        direction,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-30)
    return start, direction, projection_u, projection_v


def coupled_automatic_spatial_gradient(
    values_zyx: torch.Tensor,
    points_xyz: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Coordinate gradient that preserves an existing trajectory graph."""

    points = torch.as_tensor(points_xyz)
    if not points.is_floating_point():
        points = points.to(torch.float64)
    if not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)
    field = smoothstep_grid_field(values_zyx, points)
    return torch.autograd.grad(
        field,
        points,
        grad_outputs=torch.ones_like(field),
        create_graph=bool(create_graph),
        retain_graph=bool(create_graph),
    )[0]


def _validate_trace_parameters(
    *,
    gradient_mode: str,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
) -> tuple[str, float, float, int]:
    mode = str(gradient_mode)
    if mode not in {"automatic", "central"}:
        raise ValueError("gradient_mode must be 'automatic' or 'central'")
    delta = float(difference_step)
    if not np.isfinite(delta) or delta <= 0.0 or delta >= 0.25:
        raise ValueError("difference_step must lie in (0,0.25)")
    scale = float(refractivity_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("refractivity_scale must be finite and positive")
    steps = int(step_count)
    if steps < 2:
        raise ValueError("step_count must be at least two")
    return mode, delta, scale, steps


def _ray_rhs(
    values_zyx: torch.Tensor,
    position: torch.Tensor,
    direction: torch.Tensor,
    *,
    gradient_mode: str,
    difference_step: float,
    refractivity_scale: float,
    create_graph: bool,
    stage_label: str,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    unit_direction = direction / torch.linalg.vector_norm(
        direction,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-30)
    stencil_margin = difference_step if gradient_mode == "central" else 0.0
    domain_margin = 1.0 - float(torch.max(torch.abs(position.detach())))
    available_stencil_margin = domain_margin - stencil_margin
    if available_stencil_margin < -1e-12:
        raise RayDomainError(
            f"{stage_label} left the trace/stencil domain: "
            f"domain_margin={domain_margin:.6g}, required={stencil_margin:.6g}"
        )
    scalar = smoothstep_grid_field(values_zyx, position)
    refractive_index = 1.0 + float(refractivity_scale) * scalar
    if torch.any(~torch.isfinite(refractive_index)) or torch.any(refractive_index <= 0.5):
        raise RayDomainError("refractive index became non-finite or non-positive")
    if gradient_mode == "automatic":
        gradient = coupled_automatic_spatial_gradient(
            values_zyx,
            position,
            create_graph=bool(create_graph),
        )
    else:
        gradient = central_difference_spatial_gradient(
            values_zyx,
            position,
            step=float(difference_step),
        )
    gradient_n = float(refractivity_scale) * gradient
    longitudinal = torch.sum(gradient_n * unit_direction, dim=-1, keepdim=True)
    curvature = (gradient_n - longitudinal * unit_direction) / refractive_index[:, None]
    return unit_direction, curvature, domain_margin, available_stencil_margin


def trace_field_dependent_rays(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    gradient_mode: str,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    create_graph: bool,
) -> RayTraceResult:
    """Integrate the refractive ray equation with explicit RK4."""

    mode, delta, scale, steps = _validate_trace_parameters(
        gradient_mode=gradient_mode,
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
    )
    values = torch.as_tensor(values_zyx)
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError("values_zyx must be a floating [z,y,x] grid")
    position, direction, projection_u, projection_v = initial_pupil_rays(
        pupil_states,
        rig,
    )
    position = position.to(dtype=values.dtype, device=values.device)
    direction = direction.to(dtype=values.dtype, device=values.device)
    projection_u = projection_u.to(dtype=values.dtype, device=values.device)
    projection_v = projection_v.to(dtype=values.dtype, device=values.device)
    step_size = 2.0 * float(rig.path_half_length) / float(steps)
    positions = [position]
    directions = [direction]
    domain_margins: list[float] = []
    stencil_margins: list[float] = []

    def rhs(
        p: torch.Tensor,
        d: torch.Tensor,
        *,
        step_index: int,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dp, dd, domain_margin, stencil_margin = _ray_rhs(
            values,
            p,
            d,
            gradient_mode=mode,
            difference_step=delta,
            refractivity_scale=scale,
            create_graph=bool(create_graph),
            stage_label=f"step={step_index},stage={stage}",
        )
        domain_margins.append(domain_margin)
        stencil_margins.append(stencil_margin)
        return dp, dd

    for index in range(steps):
        k1p, k1d = rhs(position, direction, step_index=index, stage="k1")
        k2p, k2d = rhs(
            position + 0.5 * step_size * k1p,
            direction + 0.5 * step_size * k1d,
            step_index=index,
            stage="k2",
        )
        k3p, k3d = rhs(
            position + 0.5 * step_size * k2p,
            direction + 0.5 * step_size * k2d,
            step_index=index,
            stage="k3",
        )
        k4p, k4d = rhs(
            position + step_size * k3p,
            direction + step_size * k3d,
            step_index=index,
            stage="k4",
        )
        position = position + (step_size / 6.0) * (
            k1p + 2.0 * k2p + 2.0 * k3p + k4p
        )
        direction = direction + (step_size / 6.0) * (
            k1d + 2.0 * k2d + 2.0 * k3d + k4d
        )
        direction = direction / torch.linalg.vector_norm(
            direction,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-30)
        positions.append(position)
        directions.append(direction)

    stacked_positions = torch.stack(positions, dim=1)
    stacked_directions = torch.stack(directions, dim=1)
    final_domain_margin = 1.0 - float(torch.max(torch.abs(stacked_positions.detach())))
    domain_margins.append(final_domain_margin)
    stencil_margins.append(final_domain_margin - (delta if mode == "central" else 0.0))
    norm_error = torch.max(
        torch.abs(torch.linalg.vector_norm(stacked_directions.detach(), dim=-1) - 1.0)
    )
    return RayTraceResult(
        positions=stacked_positions,
        directions=stacked_directions,
        projection_u=projection_u,
        projection_v=projection_v,
        step_size=step_size,
        gradient_mode=mode,
        minimum_domain_margin=float(min(domain_margins)),
        minimum_stencil_margin=float(min(stencil_margins)),
        maximum_direction_norm_error=float(norm_error),
    )


def exit_direction_deflection(trace: RayTraceResult) -> torch.Tensor:
    """Project the traced exit-minus-entrance direction onto detector axes."""

    change = trace.directions[:, -1] - trace.directions[:, 0]
    return torch.stack(
        (
            torch.sum(change * trace.projection_u, dim=-1),
            torch.sum(change * trace.projection_v, dim=-1),
        ),
        dim=-1,
    )


def path_integrated_deflection(
    values_zyx: torch.Tensor,
    trace: RayTraceResult,
    *,
    gradient_mode: str,
    difference_step: float,
    refractivity_scale: float,
    create_graph: bool,
    detach_path: bool,
) -> torch.Tensor:
    """Integrate transverse curvature on the traced or frozen path.

    ``detach_path=True`` keeps the nominal ray coordinates and directions but
    removes their dependency on the field.  The resulting VJP is the direct
    integrand derivative only; its difference from the full VJP isolates the
    field-to-trajectory contribution at the frozen state.
    """

    position = 0.5 * (trace.positions[:, :-1] + trace.positions[:, 1:])
    direction = 0.5 * (trace.directions[:, :-1] + trace.directions[:, 1:])
    if detach_path:
        position = position.detach()
        direction = direction.detach()
    ray_count, interval_count, _ = position.shape
    _, curvature, _, _ = _ray_rhs(
        values_zyx,
        position.reshape(-1, 3),
        direction.reshape(-1, 3),
        gradient_mode=str(gradient_mode),
        difference_step=float(difference_step),
        refractivity_scale=float(refractivity_scale),
        create_graph=bool(create_graph),
        stage_label="path_integral_midpoints",
    )
    integrated = curvature.reshape(ray_count, interval_count, 3).sum(dim=1)
    integrated = integrated * float(trace.step_size)
    return torch.stack(
        (
            torch.sum(integrated * trace.projection_u, dim=-1),
            torch.sum(integrated * trace.projection_v, dim=-1),
        ),
        dim=-1,
    )


def ray_momentum_balance(
    values_zyx: torch.Tensor,
    trace: RayTraceResult,
    *,
    gradient_mode: str,
    difference_step: float,
    refractivity_scale: float,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return endpoint momentum change and midpoint ``integral grad(n) ds``."""

    mode, delta, scale, _ = _validate_trace_parameters(
        gradient_mode=gradient_mode,
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=trace.positions.shape[1] - 1,
    )
    values = torch.as_tensor(values_zyx)
    endpoints = torch.stack((trace.positions[:, 0], trace.positions[:, -1]), dim=1)
    endpoint_field = smoothstep_grid_field(values, endpoints.reshape(-1, 3)).reshape(
        len(endpoints),
        2,
    )
    endpoint_index = 1.0 + scale * endpoint_field
    endpoint_change = (
        endpoint_index[:, 1, None] * trace.directions[:, -1]
        - endpoint_index[:, 0, None] * trace.directions[:, 0]
    )

    midpoint = 0.5 * (trace.positions[:, :-1] + trace.positions[:, 1:])
    ray_count, interval_count, _ = midpoint.shape
    flat = midpoint.reshape(-1, 3)
    if mode == "automatic":
        gradient = coupled_automatic_spatial_gradient(
            values,
            flat,
            create_graph=bool(create_graph),
        )
    else:
        gradient = central_difference_spatial_gradient(values, flat, step=delta)
    integrated_gradient = (
        scale
        * gradient.reshape(ray_count, interval_count, 3).sum(dim=1)
        * float(trace.step_size)
    )
    return endpoint_change, integrated_gradient


def straight_ray_deflection(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    gradient_mode: str,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    create_graph: bool,
) -> torch.Tensor:
    """Born-style transverse-gradient integral on the frozen straight path."""

    mode, delta, scale, steps = _validate_trace_parameters(
        gradient_mode=gradient_mode,
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
    )
    start, direction, projection_u, projection_v = initial_pupil_rays(
        pupil_states,
        rig,
    )
    values = torch.as_tensor(values_zyx)
    start = start.to(dtype=values.dtype, device=values.device)
    direction = direction.to(dtype=values.dtype, device=values.device)
    projection_u = projection_u.to(dtype=values.dtype, device=values.device)
    projection_v = projection_v.to(dtype=values.dtype, device=values.device)
    step_size = 2.0 * float(rig.path_half_length) / float(steps)
    midpoint_distance = (
        torch.arange(steps, dtype=values.dtype, device=values.device) + 0.5
    ) * step_size
    points = start[:, None, :] + midpoint_distance[None, :, None] * direction[:, None, :]
    _, curvature, _, _ = _ray_rhs(
        values,
        points.reshape(-1, 3),
        direction[:, None, :].expand(-1, steps, -1).reshape(-1, 3),
        gradient_mode=mode,
        difference_step=delta,
        refractivity_scale=scale,
        create_graph=bool(create_graph),
        stage_label="straight_path_midpoints",
    )
    integrated = curvature.reshape(len(start), steps, 3).sum(dim=1) * step_size
    return torch.stack(
        (
            torch.sum(integrated * projection_u, dim=-1),
            torch.sum(integrated * projection_v, dim=-1),
        ),
        dim=-1,
    )


def path_topology_diagnostics(
    values_zyx: torch.Tensor,
    trace: RayTraceResult,
    *,
    support_threshold: float,
    frustum_half_width_u: float,
    frustum_half_width_v: float,
) -> PathTopologyDiagnostics:
    """Return a detached topology signature for derivative fail-closed gates."""

    threshold = float(support_threshold)
    half_u = float(frustum_half_width_u)
    half_v = float(frustum_half_width_v)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("support_threshold must be finite and nonnegative")
    if any(not np.isfinite(value) or value <= 0.0 for value in (half_u, half_v)):
        raise ValueError("frustum half widths must be finite and positive")
    positions = trace.positions.detach()
    field = smoothstep_grid_field(
        values_zyx.detach(),
        positions.reshape(-1, 3),
    ).reshape(positions.shape[:2])
    inside_support = torch.abs(field) >= threshold
    crossings = torch.sum(inside_support[:, 1:] != inside_support[:, :-1], dim=1)

    progress = torch.arange(
        positions.shape[1],
        dtype=positions.dtype,
        device=positions.device,
    ) * float(trace.step_size)
    straight = positions[:, :1, :] + progress[None, :, None] * trace.directions[:, :1, :].detach()
    deviation = positions - straight
    offset_u = torch.sum(deviation * trace.projection_u[:, None, :].detach(), dim=-1)
    offset_v = torch.sum(deviation * trace.projection_v[:, None, :].detach(), dim=-1)
    margin_u = half_u - torch.abs(offset_u)
    margin_v = half_v - torch.abs(offset_v)
    minimum_per_ray = torch.minimum(
        torch.min(margin_u, dim=1).values,
        torch.min(margin_v, dim=1).values,
    )
    violations = minimum_per_ray < 0.0
    return PathTopologyDiagnostics(
        support_crossings_per_ray=tuple(int(value) for value in crossings.cpu()),
        frustum_violations_per_ray=tuple(bool(value) for value in violations.cpu()),
        minimum_frustum_margin=float(torch.min(minimum_per_ray)),
        minimum_domain_margin=float(trace.minimum_domain_margin),
    )


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Return a detached relative L2 diagnostic."""

    numerator = torch.linalg.vector_norm(candidate.detach() - reference.detach())
    denominator = torch.linalg.vector_norm(reference.detach()).clamp_min(1e-30)
    return float(numerator / denominator)
