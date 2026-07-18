"""Picard/successive-approximation curved-ray baseline for synthetic BOST.

For the arc-length ray system

    p' = d,
    d' = kappa(p, d)
       = (grad(n) - d (d . grad(n))) / n,

the initial iterate is the calibrated straight ray ``(p^0, d^0)``.  Sweep
``q`` freezes midpoint positions and unit directions from iterate ``q - 1``,
evaluates ``kappa`` there with a seven-point central-difference bundle, and
then performs the direction-first update

    d^q_i = normalize(d_in + h sum_{j < i} kappa^{q}_j),
    p^q_i = p_in + h sum_{j < i} normalize(d^q_j + d^q_{j+1}).

This is a practical Gauss-Seidel form of successive approximation: curvature
is never re-evaluated during a sweep, while the newly updated direction is used
to rebuild that sweep's position path.  One sweep is a bent-path correction of
the straight route; two sweeps re-evaluate curvature on the first bent path.

After the requested path sweeps, curvature is evaluated once more on the final
updated path.  The reported BOST quantity is that final ``h * sum(kappa)``
projected onto the two detector-plane axes.  The extra seven-point bundle is
reported as output cost; without it, a nominal one-sweep result would still be
the straight-path measurement.  The result is an angular deflection in this
normalized synthetic rig, not a calibrated detector displacement.  The module
does not import or call the exact RK4 high-fidelity trace.  It is a strong
iterative baseline, not a convergence proof or a caustic-capable ray solver.
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


PICARD_CURVED_RAY_BASELINE_SCHEMA = "picard-curved-ray-baseline-1.0"
_DOMAIN_TOLERANCE = 1e-12
_DIRECTION_NORM_FLOOR = 1e-14


class PicardRayDomainError(RayDomainError):
    """Raised when a Picard iterate cannot satisfy its declared ray contract."""


@dataclass(frozen=True, slots=True)
class PicardQueryAccounting:
    """Point-query and iteration cost for one returned Picard result."""

    ray_count: int
    step_count: int
    sweep_count: int
    midpoint_curvature_evaluations_per_sweep: int
    scalar_value_point_queries_per_sweep: int
    central_difference_point_queries_per_sweep: int
    total_field_point_queries_per_sweep: int
    total_field_point_queries: int
    vectorized_interpolation_calls: int
    direction_updates: int
    position_updates: int
    output_additional_field_point_queries: int
    exact_high_calls: int
    update_scheme: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["query_unit"] = "scalar_grid_evaluation_at_one_coordinate"
        result["iteration_unit"] = "one_frozen_path_curvature_sweep"
        return result


@dataclass(frozen=True, slots=True)
class PicardCurvedRayResult:
    """Curved trajectory, BOST projection, validity, and auditable metadata.

    ``position_history`` and ``direction_history`` include the straight iterate
    at index zero.  ``curvature_history[q]`` was evaluated only on the frozen
    history entry ``q`` and produced history entry ``q + 1``.
    """

    detector_plane_deflection: torch.Tensor
    exit_direction_deflection: torch.Tensor
    positions: torch.Tensor
    directions: torch.Tensor
    position_history: torch.Tensor
    direction_history: torch.Tensor
    curvature_history: torch.Tensor
    refractive_index_history: torch.Tensor
    output_curvature: torch.Tensor
    output_refractive_index: torch.Tensor
    maximum_position_change_per_sweep: torch.Tensor
    maximum_direction_change_per_sweep: torch.Tensor
    minimum_domain_margin_per_ray: torch.Tensor
    minimum_stencil_margin_per_ray: torch.Tensor
    minimum_refractive_index_per_ray: torch.Tensor
    valid_mask: torch.Tensor
    failure_reasons: tuple[str, ...]
    projection_u: torch.Tensor
    projection_v: torch.Tensor
    step_size: float
    sweep_count: int
    update_scheme: str
    deflection_definition: str
    query_accounting: PicardQueryAccounting


def _require_float64_grid(values_zyx: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values_zyx)
    if values.ndim != 3 or any(int(size) < 3 for size in values.shape):
        raise ValueError("values_zyx must have shape [z,y,x] with size at least three")
    if values.dtype != torch.float64:
        raise TypeError("values_zyx must use torch.float64")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("values_zyx must be finite")
    return values


def _require_float64_pupil_states(pupil_states: torch.Tensor) -> torch.Tensor:
    states = torch.as_tensor(pupil_states)
    if states.ndim != 2 or states.shape[1] != 2 or len(states) < 1:
        raise ValueError("pupil_states must have shape [ray,2]")
    if states.dtype != torch.float64:
        raise TypeError("pupil_states must use torch.float64")
    if not bool(torch.all(torch.isfinite(states))):
        raise ValueError("pupil_states must be finite")
    if bool(torch.any((states < 0.0) | (states > 1.0))):
        raise ValueError("pupil_states must lie in [0,1]^2")
    return states


def _require_integer_at_least(name: str, value: int, minimum: int) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(result)


def _require_positive(name: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _normalize_directions(direction: torch.Tensor, *, label: str) -> torch.Tensor:
    if not bool(torch.all(torch.isfinite(direction))):
        raise PicardRayDomainError(f"{label} contains a non-finite direction")
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if bool(torch.any(~torch.isfinite(norm))) or bool(
        torch.any(norm <= _DIRECTION_NORM_FLOOR)
    ):
        raise PicardRayDomainError(f"{label} contains a degenerate direction")
    normalized = direction / norm
    if not bool(torch.all(torch.isfinite(normalized))):
        raise PicardRayDomainError(f"{label} normalization became non-finite")
    return normalized


def _check_position_domain(
    positions: torch.Tensor,
    *,
    label: str,
    required_margin: float,
    stencil_step: float | None,
) -> torch.Tensor:
    if not bool(torch.all(torch.isfinite(positions))):
        raise PicardRayDomainError(f"{label} contains a non-finite position")
    margins = 1.0 - torch.amax(torch.abs(positions), dim=-1)
    if bool(torch.any(margins < required_margin - _DOMAIN_TOLERANCE)):
        worst = float(torch.min(margins.detach()))
        raise PicardRayDomainError(
            f"{label} violates domain_margin: minimum={worst:.6g}, "
            f"required={required_margin:.6g}"
        )
    if stencil_step is not None and bool(
        torch.any(margins < stencil_step - _DOMAIN_TOLERANCE)
    ):
        worst = float(torch.min((margins - stencil_step).detach()))
        raise PicardRayDomainError(
            f"{label} leaves the central-difference stencil domain: "
            f"minimum_stencil_margin={worst:.6g}"
        )
    return margins


def _field_values_and_central_gradients(
    values_zyx: torch.Tensor,
    flat_positions: torch.Tensor,
    *,
    difference_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate base and six offset samples in one vectorized field call."""

    identity = torch.eye(
        3,
        dtype=torch.float64,
        device=flat_positions.device,
    )
    offsets = torch.cat(
        (
            torch.zeros((1, 3), dtype=torch.float64, device=flat_positions.device),
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
    if not bool(torch.all(torch.isfinite(samples))):
        raise PicardRayDomainError("central-difference field samples became non-finite")
    scalar = samples[:, 0]
    gradient = (samples[:, 1:4] - samples[:, 4:7]) / (2.0 * difference_step)
    if not bool(torch.all(torch.isfinite(gradient))):
        raise PicardRayDomainError("central-difference gradient became non-finite")
    return scalar, gradient


def trace_picard_curved_rays(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    sweep_count: int,
    domain_margin: float = 1e-6,
    refractive_index_floor: float = 0.5,
) -> PicardCurvedRayResult:
    """Run one or more frozen-path curved-ray successive approximations.

    The function fails closed for the whole ray batch.  It never clamps an
    invalid coordinate, refractive index, direction, or non-finite iterate.
    Returning a result therefore means every entry of ``valid_mask`` is true.
    """

    values = _require_float64_grid(values_zyx)
    states = _require_float64_pupil_states(pupil_states)
    if values.device != states.device:
        raise ValueError("values_zyx and pupil_states must share one device")
    delta = _require_positive("difference_step", difference_step)
    if delta >= 0.25:
        raise ValueError("difference_step must lie in (0,0.25)")
    scale = _require_positive("refractivity_scale", refractivity_scale)
    steps = _require_integer_at_least("step_count", step_count, 2)
    sweeps = _require_integer_at_least("sweep_count", sweep_count, 1)
    margin = float(domain_margin)
    if not math.isfinite(margin) or margin < 0.0 or margin >= 1.0:
        raise ValueError("domain_margin must be finite and lie in [0,1)")
    index_floor = _require_positive("refractive_index_floor", refractive_index_floor)

    try:
        start, direction0, projection_u, projection_v = initial_pupil_rays(states, rig)
    except ValueError as error:
        raise PicardRayDomainError(
            "initial calibrated straight ray leaves its declared domain"
        ) from error
    start = start.to(dtype=torch.float64, device=values.device)
    direction0 = _normalize_directions(
        direction0.to(dtype=torch.float64, device=values.device),
        label="initial_direction",
    )
    projection_u = projection_u.to(dtype=torch.float64, device=values.device)
    projection_v = projection_v.to(dtype=torch.float64, device=values.device)
    if not bool(torch.all(torch.isfinite(projection_u))) or not bool(
        torch.all(torch.isfinite(projection_v))
    ):
        raise PicardRayDomainError("detector projection axes are non-finite")

    path_length = 2.0 * float(rig.path_half_length)
    if not math.isfinite(path_length) or path_length <= 0.0:
        raise ValueError("rig.path_half_length must be finite and positive")
    step_size = path_length / steps
    distance = torch.arange(
        steps + 1,
        dtype=torch.float64,
        device=values.device,
    ) * step_size
    positions = start[:, None, :] + distance[None, :, None] * direction0[:, None, :]
    directions = direction0[:, None, :].expand(-1, steps + 1, -1).clone()
    initial_margins = _check_position_domain(
        positions,
        label="initial_straight_path",
        required_margin=margin,
        stencil_step=None,
    )

    ray_count = len(start)
    position_history = [positions]
    direction_history = [directions]
    curvature_history = []
    index_history = []
    position_changes = []
    direction_changes = []
    minimum_domain_margin = torch.amin(initial_margins, dim=1)
    minimum_stencil_margin = torch.full(
        (ray_count,),
        torch.inf,
        dtype=torch.float64,
        device=values.device,
    )
    minimum_index = torch.full_like(minimum_stencil_margin, torch.inf)

    for sweep_index in range(sweeps):
        frozen_midpoint_positions = 0.5 * (positions[:, :-1] + positions[:, 1:])
        frozen_midpoint_directions = _normalize_directions(
            0.5 * (directions[:, :-1] + directions[:, 1:]),
            label=f"sweep={sweep_index + 1},frozen_midpoint_direction",
        )
        midpoint_margins = _check_position_domain(
            frozen_midpoint_positions,
            label=f"sweep={sweep_index + 1},frozen_midpoint_path",
            required_margin=margin,
            stencil_step=delta,
        )
        scalar_flat, gradient_flat = _field_values_and_central_gradients(
            values,
            frozen_midpoint_positions.reshape(-1, 3),
            difference_step=delta,
        )
        scalar = scalar_flat.reshape(ray_count, steps)
        gradient_n = scale * gradient_flat.reshape(ray_count, steps, 3)
        refractive_index = 1.0 + scale * scalar
        if not bool(torch.all(torch.isfinite(refractive_index))):
            raise PicardRayDomainError(
                f"sweep={sweep_index + 1} refractive index became non-finite"
            )
        if bool(torch.any(refractive_index < index_floor)):
            worst = float(torch.min(refractive_index.detach()))
            raise PicardRayDomainError(
                f"sweep={sweep_index + 1} refractive index violates floor: "
                f"minimum={worst:.6g}, required={index_floor:.6g}"
            )

        longitudinal = torch.sum(
            gradient_n * frozen_midpoint_directions,
            dim=-1,
            keepdim=True,
        )
        curvature = (
            gradient_n - longitudinal * frozen_midpoint_directions
        ) / refractive_index[:, :, None]
        if not bool(torch.all(torch.isfinite(curvature))):
            raise PicardRayDomainError(
                f"sweep={sweep_index + 1} curvature became non-finite"
            )

        integrated_curvature = torch.cumsum(curvature, dim=1) * step_size
        raw_directions = torch.cat(
            (
                direction0[:, None, :],
                direction0[:, None, :] + integrated_curvature,
            ),
            dim=1,
        )
        updated_directions = _normalize_directions(
            raw_directions,
            label=f"sweep={sweep_index + 1},updated_direction",
        )
        updated_midpoint_directions = _normalize_directions(
            0.5 * (updated_directions[:, :-1] + updated_directions[:, 1:]),
            label=f"sweep={sweep_index + 1},updated_midpoint_direction",
        )
        integrated_position = torch.cumsum(
            step_size * updated_midpoint_directions,
            dim=1,
        )
        updated_positions = torch.cat(
            (start[:, None, :], start[:, None, :] + integrated_position),
            dim=1,
        )
        updated_margins = _check_position_domain(
            updated_positions,
            label=f"sweep={sweep_index + 1},updated_path",
            required_margin=margin,
            stencil_step=None,
        )

        position_changes.append(
            torch.amax(
                torch.linalg.vector_norm(updated_positions - positions, dim=-1),
                dim=1,
            )
        )
        direction_changes.append(
            torch.amax(
                torch.linalg.vector_norm(updated_directions - directions, dim=-1),
                dim=1,
            )
        )
        minimum_domain_margin = torch.minimum(
            minimum_domain_margin,
            torch.minimum(
                torch.amin(midpoint_margins, dim=1),
                torch.amin(updated_margins, dim=1),
            ),
        )
        minimum_stencil_margin = torch.minimum(
            minimum_stencil_margin,
            torch.amin(midpoint_margins - delta, dim=1),
        )
        minimum_index = torch.minimum(
            minimum_index,
            torch.amin(refractive_index, dim=1),
        )
        curvature_history.append(curvature)
        index_history.append(refractive_index)
        positions = updated_positions
        directions = updated_directions
        position_history.append(positions)
        direction_history.append(directions)

    output_midpoint_positions = 0.5 * (positions[:, :-1] + positions[:, 1:])
    output_midpoint_directions = _normalize_directions(
        0.5 * (directions[:, :-1] + directions[:, 1:]),
        label="output_midpoint_direction",
    )
    output_midpoint_margins = _check_position_domain(
        output_midpoint_positions,
        label="output_midpoint_path",
        required_margin=margin,
        stencil_step=delta,
    )
    output_scalar_flat, output_gradient_flat = _field_values_and_central_gradients(
        values,
        output_midpoint_positions.reshape(-1, 3),
        difference_step=delta,
    )
    output_scalar = output_scalar_flat.reshape(ray_count, steps)
    output_gradient_n = scale * output_gradient_flat.reshape(ray_count, steps, 3)
    output_refractive_index = 1.0 + scale * output_scalar
    if not bool(torch.all(torch.isfinite(output_refractive_index))):
        raise PicardRayDomainError("output refractive index became non-finite")
    if bool(torch.any(output_refractive_index < index_floor)):
        worst = float(torch.min(output_refractive_index.detach()))
        raise PicardRayDomainError(
            "output refractive index violates floor: "
            f"minimum={worst:.6g}, required={index_floor:.6g}"
        )
    output_longitudinal = torch.sum(
        output_gradient_n * output_midpoint_directions,
        dim=-1,
        keepdim=True,
    )
    output_curvature = (
        output_gradient_n - output_longitudinal * output_midpoint_directions
    ) / output_refractive_index[:, :, None]
    if not bool(torch.all(torch.isfinite(output_curvature))):
        raise PicardRayDomainError("output curvature became non-finite")
    minimum_domain_margin = torch.minimum(
        minimum_domain_margin,
        torch.amin(output_midpoint_margins, dim=1),
    )
    minimum_stencil_margin = torch.minimum(
        minimum_stencil_margin,
        torch.amin(output_midpoint_margins - delta, dim=1),
    )
    minimum_index = torch.minimum(
        minimum_index,
        torch.amin(output_refractive_index, dim=1),
    )

    final_integrated_curvature = torch.sum(output_curvature, dim=1) * step_size
    detector_deflection = torch.stack(
        (
            torch.sum(final_integrated_curvature * projection_u, dim=-1),
            torch.sum(final_integrated_curvature * projection_v, dim=-1),
        ),
        dim=-1,
    )
    exit_change = directions[:, -1] - direction0
    exit_deflection = torch.stack(
        (
            torch.sum(exit_change * projection_u, dim=-1),
            torch.sum(exit_change * projection_v, dim=-1),
        ),
        dim=-1,
    )
    output_tensors = (
        detector_deflection,
        exit_deflection,
        positions,
        directions,
        minimum_domain_margin,
        minimum_stencil_margin,
        minimum_index,
    )
    if any(not bool(torch.all(torch.isfinite(tensor))) for tensor in output_tensors):
        raise PicardRayDomainError("returned Picard state contains a non-finite value")

    midpoint_count = ray_count * steps
    point_queries_per_sweep = 7 * midpoint_count
    update_scheme = "direction-first frozen-midpoint Picard/Gauss-Seidel"
    accounting = PicardQueryAccounting(
        ray_count=ray_count,
        step_count=steps,
        sweep_count=sweeps,
        midpoint_curvature_evaluations_per_sweep=midpoint_count,
        scalar_value_point_queries_per_sweep=midpoint_count,
        central_difference_point_queries_per_sweep=6 * midpoint_count,
        total_field_point_queries_per_sweep=point_queries_per_sweep,
        total_field_point_queries=(sweeps + 1) * point_queries_per_sweep,
        vectorized_interpolation_calls=sweeps + 1,
        direction_updates=sweeps,
        position_updates=sweeps,
        output_additional_field_point_queries=point_queries_per_sweep,
        exact_high_calls=0,
        update_scheme=update_scheme,
    )
    valid = torch.ones(ray_count, dtype=torch.bool, device=values.device)
    return PicardCurvedRayResult(
        detector_plane_deflection=detector_deflection,
        exit_direction_deflection=exit_deflection,
        positions=positions,
        directions=directions,
        position_history=torch.stack(position_history, dim=0),
        direction_history=torch.stack(direction_history, dim=0),
        curvature_history=torch.stack(curvature_history, dim=0),
        refractive_index_history=torch.stack(index_history, dim=0),
        output_curvature=output_curvature,
        output_refractive_index=output_refractive_index,
        maximum_position_change_per_sweep=torch.stack(position_changes, dim=0),
        maximum_direction_change_per_sweep=torch.stack(direction_changes, dim=0),
        minimum_domain_margin_per_ray=minimum_domain_margin,
        minimum_stencil_margin_per_ray=minimum_stencil_margin,
        minimum_refractive_index_per_ray=minimum_index,
        valid_mask=valid,
        failure_reasons=("ok",) * ray_count,
        projection_u=projection_u,
        projection_v=projection_v,
        step_size=float(step_size),
        sweep_count=sweeps,
        update_scheme=update_scheme,
        deflection_definition=(
            "detector-axis projection of h*sum(curvature on final updated path)"
        ),
        query_accounting=accounting,
    )


__all__ = [
    "PICARD_CURVED_RAY_BASELINE_SCHEMA",
    "PicardCurvedRayResult",
    "PicardQueryAccounting",
    "PicardRayDomainError",
    "trace_picard_curved_rays",
]
