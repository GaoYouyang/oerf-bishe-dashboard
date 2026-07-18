"""Reusable vectorized state for the straight central-difference ray route.

The N0 tri-fidelity rehearsal evaluated the straight medium route and its
low-path diagnostics independently.  This module exposes the common midpoint
state without changing the declared physics: a frozen straight path, a
smoothstep scalar grid, central-difference coordinate gradients, and the
Born-style transverse-curvature integral used by ``straight_ray_deflection``.

The builder is deliberately fail closed.  It accepts float64 tensors only,
rejects an invalid stencil or refractive index, and records point-query cost
separately from vectorized interpolation-call cost.  Cell identifiers and
geometric margins are diagnostics; they are never used to silently clamp an
invalid physical query.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import operator
from typing import Any

import torch

try:
    from .automatic_discrete_multifidelity import (
        SyntheticRayRig,
        smoothstep_grid_field,
    )
    from .field_dependent_ray import RayDomainError, initial_pupil_rays
except ImportError:
    from automatic_discrete_multifidelity import (
        SyntheticRayRig,
        smoothstep_grid_field,
    )
    from field_dependent_ray import RayDomainError, initial_pupil_rays


SHARED_STRAIGHT_STATE_SCHEMA = "shared-straight-path-state-1.0"
_DOMAIN_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class StraightPathQueryAccounting:
    """Exact accounting for construction of one shared straight-path state.

    A field *point query* is one scalar-grid evaluation at one coordinate.  A
    vectorized interpolation call may evaluate many such points.  Keeping both
    quantities prevents batching from being reported as a physical-query
    reduction.
    """

    ray_count: int
    step_count: int
    midpoint_count: int
    scalar_value_point_queries: int
    central_difference_point_queries: int
    total_field_point_queries: int
    vectorized_interpolation_calls: int
    projected_output_additional_point_queries: int
    cell_and_margin_additional_point_queries: int

    @property
    def field_queries_per_midpoint(self) -> int:
        return self.total_field_point_queries // self.midpoint_count

    @property
    def total_query_count(self) -> int:
        """Alias used by downstream cost ledgers."""

        return self.total_field_point_queries

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["field_queries_per_midpoint"] = self.field_queries_per_midpoint
        result["total_query_count"] = self.total_query_count
        result["query_unit"] = "scalar_grid_evaluation_at_one_coordinate"
        return result


@dataclass(frozen=True, slots=True)
class StraightPathState:
    """Shared differentiable tensors for one straight central-difference route.

    ``positions`` are interval midpoints with shape ``[ray, step, xyz]``.
    ``cell_ids`` use ``xyz`` ordering even though the grid storage is ``zyx``.
    ``frustum_margins[..., 0/1]`` are the remaining synthetic ``u/v`` widths
    relative to the ray's own frozen straight reference.  They therefore equal
    the declared half widths before a later certificate subtracts a curved-path
    deviation bound.
    """

    positions: torch.Tensor
    start_positions: torch.Tensor
    directions: torch.Tensor
    projection_u: torch.Tensor
    projection_v: torch.Tensor
    scalar_values: torch.Tensor
    central_difference_gradients: torch.Tensor
    refractive_indices: torch.Tensor
    curvatures: torch.Tensor
    projected_integrands: torch.Tensor
    projected_outputs: torch.Tensor
    cell_ids: torch.Tensor
    domain_margins: torch.Tensor
    stencil_domain_margins: torch.Tensor
    frustum_margins: torch.Tensor
    step_size: float
    difference_step: float
    refractivity_scale: float
    grid_shape_zyx: tuple[int, int, int]
    query_accounting: StraightPathQueryAccounting

    def __post_init__(self) -> None:
        ray_count, step_count = self._validate_shapes()
        self._validate_numeric_tensors()
        self._validate_discrete_metadata(ray_count, step_count)

    def _validate_shapes(self) -> tuple[int, int]:
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape [ray,step,3]")
        ray_count, step_count, _ = self.positions.shape
        expected = {
            "start_positions": (ray_count, 3),
            "directions": (ray_count, 3),
            "projection_u": (ray_count, 3),
            "projection_v": (ray_count, 3),
            "scalar_values": (ray_count, step_count),
            "central_difference_gradients": (ray_count, step_count, 3),
            "refractive_indices": (ray_count, step_count),
            "curvatures": (ray_count, step_count, 3),
            "projected_integrands": (ray_count, step_count, 2),
            "projected_outputs": (ray_count, 2),
            "cell_ids": (ray_count, step_count, 3),
            "domain_margins": (ray_count, step_count),
            "stencil_domain_margins": (ray_count, step_count),
            "frustum_margins": (ray_count, step_count, 2),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != tuple(shape):
                raise ValueError(f"{name} must have shape {shape}")
        return int(ray_count), int(step_count)

    def _validate_numeric_tensors(self) -> None:
        floating_names = (
            "positions",
            "start_positions",
            "directions",
            "projection_u",
            "projection_v",
            "scalar_values",
            "central_difference_gradients",
            "refractive_indices",
            "curvatures",
            "projected_integrands",
            "projected_outputs",
            "domain_margins",
            "stencil_domain_margins",
            "frustum_margins",
        )
        device = self.positions.device
        for name in floating_names:
            tensor = getattr(self, name)
            if tensor.dtype != torch.float64:
                raise TypeError(f"{name} must use torch.float64")
            if tensor.device != device:
                raise ValueError("all StraightPathState tensors must share a device")
            if not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(f"{name} must be finite")
        if not bool(torch.all(self.refractive_indices > 0.5)):
            raise RayDomainError("refractive index lower gate failed")
        if bool(torch.any(self.stencil_domain_margins < -_DOMAIN_TOLERANCE)):
            raise RayDomainError("central-difference stencil leaves the grid domain")
        if bool(torch.any(self.frustum_margins < 0.0)):
            raise RayDomainError("straight path leaves the declared synthetic frustum")

    def _validate_discrete_metadata(self, ray_count: int, step_count: int) -> None:
        if self.cell_ids.dtype != torch.long or self.cell_ids.device != self.positions.device:
            raise TypeError("cell_ids must be torch.long on the state device")
        if len(self.grid_shape_zyx) != 3 or any(size < 3 for size in self.grid_shape_zyx):
            raise ValueError("grid_shape_zyx must contain three sizes of at least three")
        nz, ny, nx = self.grid_shape_zyx
        maximum_xyz = torch.tensor(
            [nx - 2, ny - 2, nz - 2],
            dtype=torch.long,
            device=self.cell_ids.device,
        )
        if bool(torch.any(self.cell_ids < 0)) or bool(
            torch.any(self.cell_ids > maximum_xyz)
        ):
            raise ValueError("cell_ids leave the declared grid-cell range")
        accounting = self.query_accounting
        if accounting.ray_count != ray_count or accounting.step_count != step_count:
            raise ValueError("query accounting dimensions do not match the state")
        midpoint_count = ray_count * step_count
        if (
            accounting.midpoint_count != midpoint_count
            or accounting.scalar_value_point_queries != midpoint_count
            or accounting.central_difference_point_queries != 6 * midpoint_count
            or accounting.total_field_point_queries != 7 * midpoint_count
            or accounting.vectorized_interpolation_calls != 1
            or accounting.projected_output_additional_point_queries != 0
            or accounting.cell_and_margin_additional_point_queries != 0
        ):
            raise ValueError("query accounting is inconsistent with state construction")
        scalars = (self.step_size, self.difference_step, self.refractivity_scale)
        if any(not math.isfinite(value) or value <= 0.0 for value in scalars):
            raise ValueError("state step and scale metadata must be finite and positive")

    @property
    def minimum_domain_margin_per_ray(self) -> torch.Tensor:
        return torch.amin(self.domain_margins, dim=1)

    @property
    def minimum_stencil_margin_per_ray(self) -> torch.Tensor:
        return torch.amin(self.stencil_domain_margins, dim=1)

    @property
    def minimum_frustum_margin_per_ray(self) -> torch.Tensor:
        return torch.amin(self.frustum_margins, dim=(1, 2))

    @classmethod
    def build(
        cls,
        values_zyx: torch.Tensor,
        pupil_states: torch.Tensor,
        rig: SyntheticRayRig,
        *,
        difference_step: float,
        refractivity_scale: float,
        step_count: int,
        frustum_half_width_u: float,
        frustum_half_width_v: float,
    ) -> StraightPathState:
        return build_straight_path_state(
            values_zyx,
            pupil_states,
            rig,
            difference_step=difference_step,
            refractivity_scale=refractivity_scale,
            step_count=step_count,
            frustum_half_width_u=frustum_half_width_u,
            frustum_half_width_v=frustum_half_width_v,
        )


def _validated_float64_grid(values_zyx: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values_zyx)
    if values.ndim != 3 or any(int(size) < 3 for size in values.shape):
        raise ValueError("values_zyx must have shape [z,y,x] with size at least three")
    if values.dtype != torch.float64:
        raise TypeError("values_zyx must use torch.float64")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("values_zyx must be finite")
    return values


def _validated_float64_pupil_states(pupil_states: torch.Tensor) -> torch.Tensor:
    states = torch.as_tensor(pupil_states)
    if states.ndim != 2 or states.shape[1] != 2 or len(states) < 1:
        raise ValueError("pupil_states must have shape [ray,2]")
    if states.dtype != torch.float64:
        raise TypeError("pupil_states must use torch.float64")
    if not bool(torch.all(torch.isfinite(states))) or bool(
        torch.any((states < 0.0) | (states > 1.0))
    ):
        raise ValueError("pupil_states must be finite and lie in [0,1]^2")
    return states


def _validated_positive(name: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _validated_step_count(step_count: int) -> int:
    try:
        steps = operator.index(step_count)
    except TypeError as error:
        raise TypeError("step_count must be an integer") from error
    if isinstance(step_count, bool) or steps < 2:
        raise ValueError("step_count must be an integer of at least two")
    return int(steps)


def _field_values_and_central_gradients(
    values_zyx: torch.Tensor,
    flat_positions: torch.Tensor,
    *,
    difference_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate base and six offset points in one vectorized interpolation call."""

    identity = torch.eye(
        3,
        dtype=flat_positions.dtype,
        device=flat_positions.device,
    )
    offsets = torch.cat(
        (
            torch.zeros((1, 3), dtype=flat_positions.dtype, device=flat_positions.device),
            difference_step * identity,
            -difference_step * identity,
        ),
        dim=0,
    )
    query_points = flat_positions[:, None, :] + offsets[None, :, :]
    samples = smoothstep_grid_field(
        values_zyx,
        query_points.reshape(-1, 3),
    ).reshape(len(flat_positions), 7)
    scalar_values = samples[:, 0]
    gradients = (samples[:, 1:4] - samples[:, 4:7]) / (2.0 * difference_step)
    return scalar_values, gradients


def _cell_ids_xyz(
    positions: torch.Tensor,
    grid_shape_zyx: tuple[int, int, int],
) -> torch.Tensor:
    nz, ny, nx = grid_shape_zyx
    sizes_xyz = torch.tensor(
        [nx - 1, ny - 1, nz - 1],
        dtype=positions.dtype,
        device=positions.device,
    )
    maximum_lower = torch.tensor(
        [nx - 2, ny - 2, nz - 2],
        dtype=torch.long,
        device=positions.device,
    )
    scaled = 0.5 * (positions.detach() + 1.0) * sizes_xyz
    lower = torch.floor(scaled).to(torch.long)
    return torch.minimum(torch.maximum(lower, torch.zeros_like(lower)), maximum_lower)


def build_straight_path_state(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    frustum_half_width_u: float,
    frustum_half_width_v: float,
) -> StraightPathState:
    """Construct one shared, differentiable straight-path medium state.

    The output is numerically equivalent to ``straight_ray_deflection`` with
    ``gradient_mode='central'``.  No field query is performed after the single
    vectorized seven-point bundle has been evaluated.
    """

    values = _validated_float64_grid(values_zyx)
    states = _validated_float64_pupil_states(pupil_states)
    delta = _validated_positive("difference_step", difference_step)
    if delta >= 0.25:
        raise ValueError("difference_step must lie in (0,0.25)")
    scale = _validated_positive("refractivity_scale", refractivity_scale)
    steps = _validated_step_count(step_count)
    half_u = _validated_positive("frustum_half_width_u", frustum_half_width_u)
    half_v = _validated_positive("frustum_half_width_v", frustum_half_width_v)

    start, direction, projection_u, projection_v = initial_pupil_rays(states, rig)
    start = start.to(dtype=torch.float64, device=values.device)
    direction = direction.to(dtype=torch.float64, device=values.device)
    projection_u = projection_u.to(dtype=torch.float64, device=values.device)
    projection_v = projection_v.to(dtype=torch.float64, device=values.device)

    total_length = 2.0 * float(rig.path_half_length)
    if not math.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("rig path length must be finite and positive")
    step_size = total_length / steps
    midpoint_distance = (
        torch.arange(steps, dtype=torch.float64, device=values.device) + 0.5
    ) * step_size
    positions = start[:, None, :] + midpoint_distance[None, :, None] * direction[:, None, :]
    domain_margins = 1.0 - torch.amax(torch.abs(positions), dim=-1)
    stencil_margins = domain_margins - delta
    if bool(torch.any(stencil_margins < -_DOMAIN_TOLERANCE)):
        raise RayDomainError(
            "straight_path_midpoints left the central-difference stencil domain"
        )

    ray_count = len(start)
    midpoint_count = ray_count * steps
    scalar_flat, gradient_flat = _field_values_and_central_gradients(
        values,
        positions.reshape(-1, 3),
        difference_step=delta,
    )
    scalar_values = scalar_flat.reshape(ray_count, steps)
    gradients = gradient_flat.reshape(ray_count, steps, 3)
    refractive_indices = 1.0 + scale * scalar_values
    if not bool(torch.all(torch.isfinite(refractive_indices))) or bool(
        torch.any(refractive_indices <= 0.5)
    ):
        raise RayDomainError("refractive index became non-finite or crossed 0.5")

    unit_direction = direction / torch.linalg.vector_norm(
        direction,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-30)
    expanded_direction = unit_direction[:, None, :]
    gradient_n = scale * gradients
    longitudinal = torch.sum(gradient_n * expanded_direction, dim=-1, keepdim=True)
    curvatures = (
        gradient_n - longitudinal * expanded_direction
    ) / refractive_indices[:, :, None]
    projected_integrands = torch.stack(
        (
            torch.sum(curvatures * projection_u[:, None, :], dim=-1),
            torch.sum(curvatures * projection_v[:, None, :], dim=-1),
        ),
        dim=-1,
    )
    projected_outputs = torch.sum(projected_integrands, dim=1) * step_size
    if not bool(torch.all(torch.isfinite(projected_outputs))):
        raise RayDomainError("projected straight-path output became non-finite")

    grid_shape = tuple(int(size) for size in values.shape)
    cell_ids = _cell_ids_xyz(positions, grid_shape)
    frustum_widths = torch.tensor(
        [half_u, half_v],
        dtype=torch.float64,
        device=values.device,
    )
    frustum_margins = frustum_widths.expand(ray_count, steps, 2)
    accounting = StraightPathQueryAccounting(
        ray_count=ray_count,
        step_count=steps,
        midpoint_count=midpoint_count,
        scalar_value_point_queries=midpoint_count,
        central_difference_point_queries=6 * midpoint_count,
        total_field_point_queries=7 * midpoint_count,
        vectorized_interpolation_calls=1,
        projected_output_additional_point_queries=0,
        cell_and_margin_additional_point_queries=0,
    )
    return StraightPathState(
        positions=positions,
        start_positions=start,
        directions=unit_direction,
        projection_u=projection_u,
        projection_v=projection_v,
        scalar_values=scalar_values,
        central_difference_gradients=gradients,
        refractive_indices=refractive_indices,
        curvatures=curvatures,
        projected_integrands=projected_integrands,
        projected_outputs=projected_outputs,
        cell_ids=cell_ids,
        domain_margins=domain_margins,
        stencil_domain_margins=stencil_margins,
        frustum_margins=frustum_margins,
        step_size=step_size,
        difference_step=delta,
        refractivity_scale=scale,
        grid_shape_zyx=grid_shape,
        query_accounting=accounting,
    )


__all__ = [
    "SHARED_STRAIGHT_STATE_SCHEMA",
    "StraightPathQueryAccounting",
    "StraightPathState",
    "build_straight_path_state",
]
