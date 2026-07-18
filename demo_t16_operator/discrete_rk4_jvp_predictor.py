"""Discrete bend-homotopy JVP predictor for BOST curved-ray defects.

The high-fidelity ray dynamics are augmented with a scalar ``epsilon`` that
multiplies curvature in the *trajectory update only*.  The physical BOST
output integrand remains unchanged.  Therefore ``epsilon=0`` produces the
straight-path medium output, while ``epsilon=1`` recovers the nonlinear
curved trajectory at the same discrete step count.

Differentiating the complete discrete map at ``epsilon=0`` includes every RK4
stage, the per-step direction normalization, and the midpoint output rule.  It
is a first-order trajectory-defect predictor, not an exact curved-ray solve and
not a learned operator.  The implementation is deliberately detached and
fail-closed so it can be used as a development baseline before a trainable
residual model is considered.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Any

import numpy as np
import torch

try:
    from .automatic_discrete_multifidelity import (
        SyntheticRayRig,
        smoothstep_grid_field,
    )
    from .field_dependent_ray import (
        RayDomainError,
        RayTraceResult,
        _ray_rhs,
        initial_pupil_rays,
        path_integrated_deflection,
    )
except ImportError:
    from automatic_discrete_multifidelity import (
        SyntheticRayRig,
        smoothstep_grid_field,
    )
    from field_dependent_ray import (
        RayDomainError,
        RayTraceResult,
        _ray_rhs,
        initial_pupil_rays,
        path_integrated_deflection,
    )


DISCRETE_RK4_JVP_SCHEMA = "discrete-bend-homotopy-rk4-jvp-1.0"


class DiscreteRK4JVPDomainError(RuntimeError):
    """Raised when the primal or predicted tangent path leaves its contract."""


@dataclass(frozen=True)
class DiscreteRK4JVPResult:
    """Detached first-order curved-minus-straight prediction and diagnostics."""

    base_output_uv: torch.Tensor
    residual_prediction_uv: torch.Tensor
    candidate_output_uv: torch.Tensor
    base_positions: torch.Tensor
    base_directions: torch.Tensor
    delta_positions: torch.Tensor
    delta_directions: torch.Tensor
    risk_norm: torch.Tensor
    maximum_position_perturbation: torch.Tensor
    maximum_direction_perturbation: torch.Tensor
    minimum_predicted_domain_margin: torch.Tensor
    minimum_predicted_stencil_margin: torch.Tensor
    direction_tangent_orthogonality_error: torch.Tensor
    valid_mask: torch.Tensor
    failure_reasons: tuple[str, ...]
    step_size: float
    step_count: int
    difference_step: float
    refractivity_scale: float
    bend_linearization_point: float
    query_accounting: dict[str, Any]
    stop_gradient_applied: bool
    risk_definition: str


def _require_float64_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != int(ndim):
        raise ValueError(f"{name} must have {ndim} dimensions")
    if tensor.dtype != torch.float64:
        raise ValueError(f"{name} must use torch.float64")
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must be finite")
    return tensor


def _validate_inputs(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    domain_margin: float,
    refractive_index_floor: float,
    max_position_perturbation: float,
    max_direction_perturbation: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int, float, float, float, float]:
    values = _require_float64_tensor(values_zyx, name="values_zyx", ndim=3)
    if any(int(size) < 3 for size in values.shape):
        raise ValueError("values_zyx axes must each contain at least three samples")
    states = _require_float64_tensor(pupil_states, name="pupil_states", ndim=2)
    if states.shape[1] != 2 or len(states) < 1:
        raise ValueError("pupil_states must have shape [ray,2]")
    if torch.any(states < 0.0) or torch.any(states > 1.0):
        raise ValueError("pupil_states must lie in [0,1]^2")

    delta = float(difference_step)
    scale = float(refractivity_scale)
    try:
        steps = operator.index(step_count)
    except TypeError as error:
        raise TypeError("step_count must be an integer") from error
    margin = float(domain_margin)
    index_floor = float(refractive_index_floor)
    max_position = float(max_position_perturbation)
    max_direction = float(max_direction_perturbation)
    if not np.isfinite(delta) or delta <= 0.0 or delta >= 0.25:
        raise ValueError("difference_step must lie in (0,0.25)")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("refractivity_scale must be finite and positive")
    if isinstance(step_count, bool) or steps < 2:
        raise ValueError("step_count must be an integer of at least two")
    if not np.isfinite(margin) or margin < 0.0 or margin >= 1.0:
        raise ValueError("domain_margin must lie in [0,1)")
    if not np.isfinite(index_floor) or index_floor <= 0.5:
        raise ValueError("refractive_index_floor must be finite and exceed 0.5")
    if not np.isfinite(max_position) or max_position <= 0.0:
        raise ValueError("max_position_perturbation must be finite and positive")
    if not np.isfinite(max_direction) or max_direction <= 0.0:
        raise ValueError("max_direction_perturbation must be finite and positive")
    return (
        values.detach(),
        states.detach(),
        delta,
        scale,
        int(steps),
        margin,
        index_floor,
        max_position,
        max_direction,
    )


def _bend_parameterized_discrete_map(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    bend_strength: torch.Tensor,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    refractive_index_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output and ray states for one discrete bend strength.

    This helper intentionally leaves ``bend_strength`` in the graph.  All
    field and camera inputs are detached by the public entry point.
    """

    if bend_strength.ndim != 0 or bend_strength.dtype != torch.float64:
        raise ValueError("bend_strength must be a scalar torch.float64 tensor")
    values = values_zyx
    try:
        position, direction, projection_u, projection_v = initial_pupil_rays(
            pupil_states,
            rig,
        )
    except ValueError as error:
        raise DiscreteRK4JVPDomainError(
            "initial pupil geometry is outside the declared contract"
        ) from error
    position = position.to(dtype=torch.float64, device=values.device)
    direction = direction.to(dtype=torch.float64, device=values.device)
    projection_u = projection_u.to(dtype=torch.float64, device=values.device)
    projection_v = projection_v.to(dtype=torch.float64, device=values.device)

    path_length = 2.0 * float(rig.path_half_length)
    if not math.isfinite(path_length) or path_length <= 0.0:
        raise ValueError("rig.path_half_length must be finite and positive")
    step_size = path_length / int(step_count)
    positions = [position]
    directions = [direction]

    def rhs(
        p: torch.Tensor,
        d: torch.Tensor,
        *,
        step_index: int,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            tangent, curvature, _, _ = _ray_rhs(
                values,
                p,
                d,
                gradient_mode="central",
                difference_step=float(difference_step),
                refractivity_scale=float(refractivity_scale),
                create_graph=True,
                stage_label=f"bend_jvp_step={step_index},stage={stage}",
            )
        except RayDomainError as error:
            raise DiscreteRK4JVPDomainError(str(error)) from error
        return tangent, bend_strength * curvature

    for index in range(int(step_count)):
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
    trace = RayTraceResult(
        positions=stacked_positions,
        directions=stacked_directions,
        projection_u=projection_u,
        projection_v=projection_v,
        step_size=float(step_size),
        gradient_mode="central",
        minimum_domain_margin=float(
            1.0 - torch.max(torch.abs(stacked_positions.detach()))
        ),
        minimum_stencil_margin=float(
            1.0
            - torch.max(torch.abs(stacked_positions.detach()))
            - float(difference_step)
        ),
        maximum_direction_norm_error=float(
            torch.max(
                torch.abs(
                    torch.linalg.vector_norm(stacked_directions.detach(), dim=-1)
                    - 1.0
                )
            )
        ),
    )
    try:
        output = path_integrated_deflection(
            values,
            trace,
            gradient_mode="central",
            difference_step=float(difference_step),
            refractivity_scale=float(refractivity_scale),
            create_graph=True,
            detach_path=False,
        )
    except RayDomainError as error:
        raise DiscreteRK4JVPDomainError(str(error)) from error

    midpoint_positions = 0.5 * (
        stacked_positions[:, :-1] + stacked_positions[:, 1:]
    )
    midpoint_field = smoothstep_grid_field(
        values,
        midpoint_positions.reshape(-1, 3),
    )
    refractive_index = 1.0 + float(refractivity_scale) * midpoint_field
    if torch.any(~torch.isfinite(refractive_index)) or torch.any(
        refractive_index < float(refractive_index_floor)
    ):
        raise DiscreteRK4JVPDomainError(
            "midpoint refractive index violates the declared positive floor"
        )
    return output, stacked_positions, stacked_directions


def predict_discrete_rk4_jvp_residual(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    domain_margin: float = 1e-6,
    refractive_index_floor: float = 0.500001,
    max_position_perturbation: float = 0.2,
    max_direction_perturbation: float = 0.2,
) -> DiscreteRK4JVPResult:
    """Predict the same-step-count curved-minus-straight BOST defect.

    The exact high output is never evaluated or accepted as an input.  Invalid
    rays retain their detached diagnostic prediction, but receive infinite
    routing risk and ``valid_mask=False`` so downstream code must fail closed.
    """

    (
        values,
        states,
        delta,
        scale,
        steps,
        margin,
        index_floor,
        max_position,
        max_direction,
    ) = _validate_inputs(
        values_zyx,
        pupil_states,
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
        domain_margin=domain_margin,
        refractive_index_floor=refractive_index_floor,
        max_position_perturbation=max_position_perturbation,
        max_direction_perturbation=max_direction_perturbation,
    )

    zero = torch.zeros((), dtype=torch.float64, device=values.device)

    def discrete_map(
        bend_strength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _bend_parameterized_discrete_map(
            values,
            states,
            rig,
            bend_strength,
            difference_step=delta,
            refractivity_scale=scale,
            step_count=steps,
            refractive_index_floor=index_floor,
        )

    try:
        primal, tangent = torch.func.jvp(
            discrete_map,
            (zero,),
            (torch.ones_like(zero),),
        )
    except RuntimeError as error:
        if isinstance(error, DiscreteRK4JVPDomainError):
            raise
        raise DiscreteRK4JVPDomainError(
            "discrete bend-homotopy JVP failed inside the declared domain"
        ) from error
    base_output, base_positions, base_directions = primal
    residual_prediction, delta_positions, delta_directions = tangent
    tensors = (
        base_output,
        base_positions,
        base_directions,
        residual_prediction,
        delta_positions,
        delta_directions,
    )
    if any(torch.any(~torch.isfinite(tensor)) for tensor in tensors):
        raise DiscreteRK4JVPDomainError(
            "non-finite primal or tangent returned by the discrete JVP"
        )

    candidate = base_output + residual_prediction
    predicted_positions = base_positions + delta_positions
    max_position_per_ray = torch.amax(
        torch.linalg.vector_norm(delta_positions, dim=-1),
        dim=1,
    )
    max_direction_per_ray = torch.amax(
        torch.linalg.vector_norm(delta_directions, dim=-1),
        dim=1,
    )
    predicted_domain_margin = torch.amin(
        1.0 - torch.abs(predicted_positions),
        dim=(1, 2),
    )
    predicted_stencil_margin = predicted_domain_margin - delta
    orthogonality_error = torch.amax(
        torch.abs(torch.sum(base_directions * delta_directions, dim=-1)),
        dim=1,
    )
    finite_valid = (
        torch.all(torch.isfinite(residual_prediction), dim=1)
        & torch.all(torch.isfinite(delta_positions), dim=(1, 2))
        & torch.all(torch.isfinite(delta_directions), dim=(1, 2))
    )
    domain_valid = predicted_stencil_margin >= margin - 1e-12
    position_valid = max_position_per_ray <= max_position
    direction_valid = max_direction_per_ray <= max_direction
    tangent_valid = orthogonality_error <= 1e-9
    valid = (
        finite_valid
        & domain_valid
        & position_valid
        & direction_valid
        & tangent_valid
    )
    risk = torch.linalg.vector_norm(residual_prediction, dim=-1)
    risk = torch.where(valid, risk, torch.full_like(risk, torch.inf))

    reasons: list[str] = []
    for ray_index in range(len(valid)):
        ray_reasons = []
        if not bool(finite_valid[ray_index]):
            ray_reasons.append("non_finite_discrete_jvp")
        if not bool(domain_valid[ray_index]):
            ray_reasons.append("predicted_path_stencil_domain")
        if not bool(position_valid[ray_index]):
            ray_reasons.append("position_linearization_radius")
        if not bool(direction_valid[ray_index]):
            ray_reasons.append("direction_linearization_radius")
        if not bool(tangent_valid[ray_index]):
            ray_reasons.append("direction_normalization_tangent")
        reasons.append("ok" if not ray_reasons else ";".join(ray_reasons))

    ray_count = len(states)
    point_queries = 35 * ray_count * steps
    interpolation_dispatches = 35 * steps
    return DiscreteRK4JVPResult(
        base_output_uv=base_output.detach(),
        residual_prediction_uv=residual_prediction.detach(),
        candidate_output_uv=candidate.detach(),
        base_positions=base_positions.detach(),
        base_directions=base_directions.detach(),
        delta_positions=delta_positions.detach(),
        delta_directions=delta_directions.detach(),
        risk_norm=risk.detach(),
        maximum_position_perturbation=max_position_per_ray.detach(),
        maximum_direction_perturbation=max_direction_per_ray.detach(),
        minimum_predicted_domain_margin=predicted_domain_margin.detach(),
        minimum_predicted_stencil_margin=predicted_stencil_margin.detach(),
        direction_tangent_orthogonality_error=orthogonality_error.detach(),
        valid_mask=valid.detach(),
        failure_reasons=tuple(reasons),
        step_size=2.0 * float(rig.path_half_length) / steps,
        step_count=steps,
        difference_step=delta,
        refractivity_scale=scale,
        bend_linearization_point=0.0,
        query_accounting={
            "logical_scalar_grid_point_queries": point_queries,
            "interpolation_dispatches": interpolation_dispatches,
            "forward_mode_bend_jvp_evaluations": 1,
            "exact_high_evaluations": 0,
            "reverse_mode_vjp_evaluations": 0,
            "path_rule": "central-difference RK4 stages plus midpoint output",
        },
        stop_gradient_applied=True,
        risk_definition=(
            "L2 norm of d(output)/d(epsilon) at epsilon=0 for the complete "
            "discrete bend-homotopy map"
        ),
    )
