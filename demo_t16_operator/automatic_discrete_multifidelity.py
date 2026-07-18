"""Automatic/discrete-gradient multi-fidelity primitives for BOST audits.

This module implements a deliberately small mechanism surrogate for one
specific cost conflict in neural refractive-index reconstruction: an automatic
coordinate gradient needs one field query plus a coordinate VJP, whereas a
central-difference gradient needs several field queries.  The code does not
reproduce NeRIF, the 2026 neural-refractive-index-primitive implementation, or
an experimental camera.  It provides a finite-population test bed in which a
low-fidelity automatic-gradient renderer and a high-fidelity discrete-gradient
renderer can be paired without hiding the residual correction.

The two-level identity is statistical, not a physical correction::

    E[high] = E[low] + E[high - low]

It is unbiased only for the declared sampling population and only when every
term is estimated with the correct marginal measure.  An unbiased forward
estimate also does not by itself make a nonlinear loss or its gradient
unbiased.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


AUTOMATIC_DISCRETE_MF_SCHEMA = "automatic-discrete-bost-multifidelity-1.0"


@dataclass(frozen=True)
class SyntheticRayRig:
    """Prescribed weak-ray geometry for a mechanism audit only."""

    rig_id: str
    view_angle_degrees: float
    detector_u: float
    detector_z: float
    aperture_radius: float
    path_half_length: float = 0.72
    cone_u: float = 0.06
    cone_z: float = 0.04
    bend: float = 0.0


@dataclass(frozen=True)
class TwoLevelEfficiency:
    high_trace_variance: float
    low_trace_variance: float
    residual_trace_variance: float
    high_cost: float
    low_cost: float
    residual_cost: float
    high_work_variance: float
    optimal_two_level_work_variance: float
    predicted_efficiency_gain: float


def _validate_grid_values(values: torch.Tensor) -> torch.Tensor:
    grid = torch.as_tensor(values)
    if grid.ndim != 3 or any(int(size) < 3 for size in grid.shape):
        raise ValueError("grid values must have shape [z,y,x] with size at least 3")
    if not grid.is_floating_point():
        grid = grid.to(torch.float64)
    if torch.any(~torch.isfinite(grid)):
        raise ValueError("grid values must be finite")
    return grid


def _validate_points(points_xyz: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
    points = torch.as_tensor(points_xyz)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError("points_xyz must have shape [sample,3]")
    if not points.is_floating_point():
        points = points.to(torch.float64)
    if torch.any(~torch.isfinite(points)):
        raise ValueError("points_xyz must be finite")
    bound = 1.0 - float(margin)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("margin must be finite and smaller than one")
    if torch.any(torch.abs(points) > bound + 1e-12):
        raise ValueError("points_xyz must lie inside the declared grid domain")
    return points


def smoothstep_grid_field(
    values_zyx: torch.Tensor,
    points_xyz: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a smoothstep-interpolated scalar grid on ``[-1,1]^3``.

    The integer cell choice is piecewise constant.  Within a cell, cubic
    smoothstep weights give nonzero second derivatives.  This is an original
    compact surrogate, not a copy of an external hash-grid implementation.
    """

    values = _validate_grid_values(values_zyx)
    points = _validate_points(points_xyz).to(dtype=values.dtype, device=values.device)
    nz, ny, nx = (int(size) for size in values.shape)
    sizes_xyz = torch.as_tensor(
        [nx - 1, ny - 1, nz - 1],
        dtype=points.dtype,
        device=points.device,
    )
    scaled = 0.5 * (points + 1.0) * sizes_xyz
    lower = torch.floor(scaled).to(torch.long)
    maximum_lower = torch.as_tensor(
        [nx - 2, ny - 2, nz - 2],
        dtype=torch.long,
        device=points.device,
    )
    lower = torch.minimum(torch.maximum(lower, torch.zeros_like(lower)), maximum_lower)
    fraction = scaled - lower.to(scaled.dtype)
    weight = fraction.square() * (3.0 - 2.0 * fraction)

    output = torch.zeros(len(points), dtype=values.dtype, device=values.device)
    for dx in (0, 1):
        wx = weight[:, 0] if dx else 1.0 - weight[:, 0]
        ix = lower[:, 0] + dx
        for dy in (0, 1):
            wy = weight[:, 1] if dy else 1.0 - weight[:, 1]
            iy = lower[:, 1] + dy
            for dz in (0, 1):
                wz = weight[:, 2] if dz else 1.0 - weight[:, 2]
                iz = lower[:, 2] + dz
                output = output + wx * wy * wz * values[iz, iy, ix]
    return output


def automatic_spatial_gradient(
    values_zyx: torch.Tensor,
    points_xyz: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Differentiate the scalar primitive with respect to coordinates."""

    points = _validate_points(points_xyz).detach().clone().requires_grad_(True)
    values = _validate_grid_values(values_zyx)
    points = points.to(dtype=values.dtype, device=values.device)
    field = smoothstep_grid_field(values, points)
    gradient = torch.autograd.grad(
        field,
        points,
        grad_outputs=torch.ones_like(field),
        create_graph=bool(create_graph),
        retain_graph=bool(create_graph),
    )[0]
    return gradient


def central_difference_spatial_gradient(
    values_zyx: torch.Tensor,
    points_xyz: torch.Tensor,
    *,
    step: float,
) -> torch.Tensor:
    """Evaluate a six-query central-difference gradient."""

    delta = float(step)
    if not np.isfinite(delta) or delta <= 0.0 or delta >= 0.25:
        raise ValueError("step must be finite and lie in (0,0.25)")
    values = _validate_grid_values(values_zyx)
    points = _validate_points(points_xyz, margin=delta).to(
        dtype=values.dtype,
        device=values.device,
    )
    columns = []
    for axis in range(3):
        offset = torch.zeros_like(points)
        offset[:, axis] = delta
        right = smoothstep_grid_field(values, points + offset)
        left = smoothstep_grid_field(values, points - offset)
        columns.append((right - left) / (2.0 * delta))
    return torch.stack(columns, dim=-1)


def sample_joint_pupil_path_sobol(
    count: int,
    *,
    seed: int,
    scramble: bool = True,
) -> torch.Tensor:
    """Sample ``[pupil_u,pupil_v,path]`` states in the unit cube."""

    if int(count) < 1:
        raise ValueError("count must be positive")
    engine = torch.quasirandom.SobolEngine(
        dimension=3,
        scramble=bool(scramble),
        seed=int(seed),
    )
    return engine.draw(int(count)).to(torch.float64)


def joint_state_geometry(
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    high_geometry: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map pupil/path states to prescribed ray points and detector axes."""

    unit = torch.as_tensor(states)
    if unit.ndim != 2 or unit.shape[1] != 3 or len(unit) < 1:
        raise ValueError("states must have shape [sample,3]")
    if not unit.is_floating_point():
        unit = unit.to(torch.float64)
    if torch.any(~torch.isfinite(unit)) or torch.any(unit < 0.0) or torch.any(unit > 1.0):
        raise ValueError("states must be finite and lie in [0,1]^3")
    scalars = (
        rig.view_angle_degrees,
        rig.detector_u,
        rig.detector_z,
        rig.aperture_radius,
        rig.path_half_length,
        rig.cone_u,
        rig.cone_z,
        rig.bend,
    )
    if any(not np.isfinite(float(value)) for value in scalars):
        raise ValueError("rig parameters must be finite")
    if rig.aperture_radius < 0.0 or rig.path_half_length <= 0.0:
        raise ValueError("rig aperture and path length are invalid")

    theta = math.radians(float(rig.view_angle_degrees))
    line = torch.as_tensor(
        [math.cos(theta), math.sin(theta), 0.0],
        dtype=unit.dtype,
        device=unit.device,
    )
    transverse = torch.as_tensor(
        [-math.sin(theta), math.cos(theta), 0.0],
        dtype=unit.dtype,
        device=unit.device,
    )
    vertical = torch.as_tensor([0.0, 0.0, 1.0], dtype=unit.dtype, device=unit.device)
    radius = torch.sqrt(unit[:, 0])
    angle = 2.0 * math.pi * unit[:, 1]
    disk_u = radius * torch.cos(angle)
    disk_z = radius * torch.sin(angle)
    subray_u = float(rig.detector_u) + float(rig.aperture_radius) * disk_u
    subray_z = float(rig.detector_z) + float(rig.aperture_radius) * disk_z
    origins = subray_u[:, None] * transverse + subray_z[:, None] * vertical
    directions = (
        line[None, :]
        + float(rig.cone_u) * subray_u[:, None] * transverse
        + float(rig.cone_z) * subray_z[:, None] * vertical
    )
    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(1e-30)
    distance = (2.0 * unit[:, 2] - 1.0) * float(rig.path_half_length)
    points = origins + distance[:, None] * directions
    if high_geometry and rig.bend != 0.0:
        normalized = distance / float(rig.path_half_length)
        curve = (
            float(rig.bend)
            * (1.0 - normalized.square())
            * (0.35 + torch.abs(subray_u))
        )
        points = points + curve[:, None] * transverse
    if torch.any(torch.abs(points) > 1.0):
        raise ValueError("rig maps states outside the normalized field domain")
    projection_u = transverse.expand(len(unit), -1)
    projection_v = vertical.expand(len(unit), -1)
    return points, projection_u, projection_v


def evaluate_automatic_discrete_pair(
    values_zyx: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return low automatic and high discrete projected integrands.

    Both fidelities receive the same pupil/path state.  The low renderer uses a
    straight prescribed path, while the high renderer may include the declared
    bend.  Consequently the residual contains both gradient-formulation and
    geometry discrepancy.  This is intentional and must be reported as such.
    """

    low = evaluate_automatic_projected(
        values_zyx,
        states,
        rig,
        create_graph=bool(create_graph),
    )
    high = evaluate_discrete_projected(
        values_zyx,
        states,
        rig,
        difference_step=float(difference_step),
    )
    return low, high


def evaluate_automatic_projected(
    values_zyx: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Evaluate the low-fidelity automatic-gradient projected integrand."""

    points, projection_u, projection_v = joint_state_geometry(
        states,
        rig,
        high_geometry=False,
    )
    gradient = automatic_spatial_gradient(
        values_zyx,
        points,
        create_graph=bool(create_graph),
    )
    return torch.stack(
        (
            torch.sum(gradient * projection_u, dim=-1),
            torch.sum(gradient * projection_v, dim=-1),
        ),
        dim=-1,
    )


def evaluate_discrete_projected(
    values_zyx: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
) -> torch.Tensor:
    """Evaluate the high-fidelity discrete-gradient projected integrand."""

    points, projection_u, projection_v = joint_state_geometry(
        states,
        rig,
        high_geometry=True,
    )
    gradient = central_difference_spatial_gradient(
        values_zyx,
        points,
        step=float(difference_step),
    )
    return torch.stack(
        (
            torch.sum(gradient * projection_u, dim=-1),
            torch.sum(gradient * projection_v, dim=-1),
        ),
        dim=-1,
    )


def two_level_mean(
    low_only_values: torch.Tensor,
    paired_high_values: torch.Tensor,
    paired_low_values: torch.Tensor,
) -> torch.Tensor:
    """Return ``mean(low) + mean(high-low)`` without hiding correction."""

    low = torch.as_tensor(low_only_values)
    high = torch.as_tensor(paired_high_values)
    paired_low = torch.as_tensor(paired_low_values)
    if low.ndim < 1 or len(low) < 1:
        raise ValueError("low_only_values must contain at least one sample")
    if high.shape != paired_low.shape or high.ndim < 1 or len(high) < 1:
        raise ValueError("paired high/low values must have equal non-empty shape")
    if low.shape[1:] != high.shape[1:]:
        raise ValueError("all values must share their trailing shape")
    if low.dtype != high.dtype or high.dtype != paired_low.dtype:
        raise ValueError("all values must share their dtype")
    return torch.mean(low, dim=0) + torch.mean(high - paired_low, dim=0)


def trace_sample_variance(values: torch.Tensor) -> float:
    """Return the trace of the unbiased sample covariance."""

    samples = torch.as_tensor(values)
    if samples.ndim < 1 or len(samples) < 2:
        raise ValueError("variance requires at least two samples")
    flat = samples.reshape(len(samples), -1).to(torch.float64)
    return float(torch.sum(torch.var(flat, dim=0, unbiased=True)))


def two_level_efficiency(
    high_values: torch.Tensor,
    low_values: torch.Tensor,
    *,
    high_cost: float,
    low_cost: float,
    residual_cost: float,
) -> TwoLevelEfficiency:
    """Compute the continuous-allocation work-normalized variance ceiling."""

    high = torch.as_tensor(high_values)
    low = torch.as_tensor(low_values)
    if high.shape != low.shape:
        raise ValueError("high and low samples must have the same shape")
    costs = tuple(float(value) for value in (high_cost, low_cost, residual_cost))
    if any(not np.isfinite(value) or value <= 0.0 for value in costs):
        raise ValueError("costs must be finite and strictly positive")
    high_variance = trace_sample_variance(high)
    low_variance = trace_sample_variance(low)
    residual_variance = trace_sample_variance(high - low)
    high_work = high_variance * costs[0]
    optimal_work = (
        math.sqrt(low_variance * costs[1])
        + math.sqrt(residual_variance * costs[2])
    ) ** 2
    gain = high_work / max(optimal_work, 1e-30)
    return TwoLevelEfficiency(
        high_trace_variance=high_variance,
        low_trace_variance=low_variance,
        residual_trace_variance=residual_variance,
        high_cost=costs[0],
        low_cost=costs[1],
        residual_cost=costs[2],
        high_work_variance=high_work,
        optimal_two_level_work_variance=optimal_work,
        predicted_efficiency_gain=gain,
    )


def optimal_two_level_allocation(
    *,
    total_cost: float,
    low_variance: float,
    residual_variance: float,
    low_cost: float,
    residual_cost: float,
    minimum_count: int = 2,
) -> tuple[int, int, float]:
    """Return the exact integer allocation under the declared scalar budget.

    For each count of the smaller feasible range, the other count is set to
    the largest affordable value.  Since ``V / n`` is strictly decreasing in
    ``n``, this enumerates every potentially optimal boundary point without a
    continuous-allocation rounding assumption.
    """

    budget = float(total_cost)
    values = tuple(
        float(value)
        for value in (low_variance, residual_variance, low_cost, residual_cost)
    )
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("total_cost must be finite and positive")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("variances and costs must be finite and positive")
    minimum = int(minimum_count)
    if minimum < 1:
        raise ValueError("minimum_count must be positive")
    floor_cost = minimum * (values[2] + values[3])
    if budget < floor_cost:
        raise ValueError("total_cost cannot fund the minimum allocation")
    maximum_low = int(math.floor((budget - minimum * values[3]) / values[2]))
    maximum_residual = int(math.floor((budget - minimum * values[2]) / values[3]))
    best: tuple[float, float, int, int] | None = None

    def consider(low_count: int, residual_count: int) -> None:
        nonlocal best
        consumed = low_count * values[2] + residual_count * values[3]
        if consumed > budget + 1e-9 * max(budget, 1.0):
            return
        objective = values[0] / low_count + values[1] / residual_count
        candidate = (objective, -consumed, low_count, residual_count)
        if best is None or candidate < best:
            best = candidate

    if maximum_low - minimum <= maximum_residual - minimum:
        for low_count in range(minimum, maximum_low + 1):
            residual_count = int(
                math.floor((budget - low_count * values[2]) / values[3])
            )
            if residual_count >= minimum:
                consider(low_count, residual_count)
    else:
        for residual_count in range(minimum, maximum_residual + 1):
            low_count = int(
                math.floor((budget - residual_count * values[3]) / values[2])
            )
            if low_count >= minimum:
                consider(low_count, residual_count)
    if best is None:
        raise RuntimeError("failed to construct a feasible integer allocation")
    _, negative_consumed, low_count, residual_count = best
    return low_count, residual_count, float(-negative_consumed)
