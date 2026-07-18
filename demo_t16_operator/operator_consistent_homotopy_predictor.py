"""Operator-consistent analytic equivalent of the discrete bend JVP.

Let the curved-ray trajectory update be controlled by a scalar ``epsilon``::

    r' = normalize(d)
    d' = epsilon * F(r, d)

while the measured BOST output remains the physical midpoint integral of
``F``.  At ``epsilon=0`` the primal path is straight.  Differentiating the
trajectory equation gives ``delta_d' = F0``: feedback terms ``A delta_r`` and
``B delta_d`` are second order in this homotopy and must not be inserted into
the first-order trajectory tangent.  The output derivative still requires the
position and direction Jacobians of the *same central-difference operator*
used by the high route.

This module evaluates those operator-consistent Jacobians in seven batched
interpolation dispatches and propagates the exact RK4 tangent analytically.  It
matches the full forward-mode discrete JVP at floating-point precision, but it
remains a first-order development baseline rather than an exact or learned
curved-ray solver.
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
        central_difference_spatial_gradient,
        smoothstep_grid_field,
    )
    from .field_dependent_ray import initial_pupil_rays
    from .trajectory_variational_predictor import integrate_affine_variational_rk4
except ImportError:
    from automatic_discrete_multifidelity import (
        SyntheticRayRig,
        central_difference_spatial_gradient,
        smoothstep_grid_field,
    )
    from field_dependent_ray import initial_pupil_rays
    from trajectory_variational_predictor import integrate_affine_variational_rk4


OPERATOR_CONSISTENT_HOMOTOPY_SCHEMA = "operator-consistent-bend-homotopy-1.0"


class OperatorConsistentHomotopyDomainError(RuntimeError):
    """Raised when the frozen path or first-order path leaves its contract."""


@dataclass(frozen=True)
class OperatorConsistentHomotopyResult:
    """Detached straight output, first-order defect, and safety diagnostics."""

    base_output_uv: torch.Tensor
    residual_prediction_uv: torch.Tensor
    candidate_output_uv: torch.Tensor
    base_positions: torch.Tensor
    base_directions: torch.Tensor
    delta_positions: torch.Tensor
    delta_directions: torch.Tensor
    forcing: torch.Tensor
    output_position_jacobian: torch.Tensor
    output_direction_jacobian: torch.Tensor
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
    coefficient_sample_count: int
    difference_step: float
    refractivity_scale: float
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


def _operator_consistent_coefficients(
    values_zyx: torch.Tensor,
    positions: torch.Tensor,
    direction: torch.Tensor,
    *,
    difference_step: float,
    refractivity_scale: float,
    refractive_index_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``n``, forcing, and exact coordinate/direction Jacobians.

    ``positions`` contains all RK4 endpoint and midpoint nodes.  The central
    gradient Jacobian is obtained by differentiating the six-query operator,
    while the denominator derivative uses the automatic gradient of the scalar
    refractive-index primitive, exactly matching the high route's computation.
    """

    ray_count, sample_count, _ = positions.shape
    points = positions.reshape(-1, 3).detach().requires_grad_(True)
    field = smoothstep_grid_field(values_zyx, points)
    automatic_gradient = torch.autograd.grad(
        field,
        points,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]
    central_gradient = central_difference_spatial_gradient(
        values_zyx,
        points,
        step=float(difference_step),
    )
    central_jacobian_rows = []
    for component in range(3):
        central_jacobian_rows.append(
            torch.autograd.grad(
                central_gradient[:, component],
                points,
                grad_outputs=torch.ones_like(central_gradient[:, component]),
                create_graph=False,
                retain_graph=component < 2,
            )[0]
        )
    central_jacobian = torch.stack(central_jacobian_rows, dim=-2)

    scale = float(refractivity_scale)
    refractive_index = (1.0 + scale * field).reshape(ray_count, sample_count)
    if torch.any(~torch.isfinite(refractive_index)) or torch.any(
        refractive_index < float(refractive_index_floor)
    ):
        raise OperatorConsistentHomotopyDomainError(
            "refractive index violates the declared positive floor"
        )
    gradient_n_auto = (scale * automatic_gradient).reshape(
        ray_count,
        sample_count,
        3,
    )
    gradient_n_central = (scale * central_gradient).reshape(
        ray_count,
        sample_count,
        3,
    )
    jacobian_gradient_n_central = (scale * central_jacobian).reshape(
        ray_count,
        sample_count,
        3,
        3,
    )

    identity = torch.eye(3, dtype=torch.float64, device=values_zyx.device)
    projector = identity[None] - torch.einsum(
        "ri,rj->rij",
        direction,
        direction,
    )
    projector_path = projector[:, None]
    inverse_index = torch.reciprocal(refractive_index)
    forcing = torch.einsum(
        "rsij,rsj->rsi",
        projector_path,
        gradient_n_central,
    ) * inverse_index[:, :, None]

    denominator_outer = torch.einsum(
        "rsi,rsj->rsij",
        gradient_n_central,
        gradient_n_auto,
    )
    position_core = (
        jacobian_gradient_n_central * inverse_index[:, :, None, None]
        - denominator_outer * inverse_index.square()[:, :, None, None]
    )
    position_jacobian = torch.matmul(projector_path, position_core)

    longitudinal_gradient = torch.sum(
        direction[:, None] * gradient_n_central,
        dim=-1,
    )
    raw_direction_jacobian = -(
        longitudinal_gradient[:, :, None, None] * identity[None, None]
        + torch.einsum("ri,rsj->rsij", direction, gradient_n_central)
    ) * inverse_index[:, :, None, None]
    direction_jacobian = torch.matmul(
        raw_direction_jacobian,
        projector_path,
    )
    tensors = (
        forcing,
        position_jacobian,
        direction_jacobian,
    )
    if any(torch.any(~torch.isfinite(tensor)) for tensor in tensors):
        raise OperatorConsistentHomotopyDomainError(
            "non-finite operator-consistent coefficient"
        )
    return (
        refractive_index.detach(),
        forcing.detach(),
        position_jacobian.detach(),
        direction_jacobian.detach(),
    )


def predict_operator_consistent_homotopy_residual(
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
) -> OperatorConsistentHomotopyResult:
    """Return the analytic first derivative of the discrete bend homotopy."""

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
    try:
        start, direction, projection_u, projection_v = initial_pupil_rays(states, rig)
    except ValueError as error:
        raise OperatorConsistentHomotopyDomainError(
            "initial pupil geometry is outside the declared contract"
        ) from error
    start = start.to(dtype=torch.float64, device=values.device).detach()
    direction = direction.to(dtype=torch.float64, device=values.device).detach()
    projection_u = projection_u.to(dtype=torch.float64, device=values.device).detach()
    projection_v = projection_v.to(dtype=torch.float64, device=values.device).detach()
    path_length = 2.0 * float(rig.path_half_length)
    if not math.isfinite(path_length) or path_length <= 0.0:
        raise ValueError("rig.path_half_length must be finite and positive")
    step_size = path_length / steps
    half_step_distance = (
        torch.arange(2 * steps + 1, dtype=torch.float64, device=values.device)
        * (0.5 * step_size)
    )
    positions = start[:, None] + half_step_distance[None, :, None] * direction[:, None]
    base_stencil_margin = torch.amin(
        1.0 - torch.abs(positions),
        dim=(1, 2),
    ) - delta
    if torch.any(base_stencil_margin < margin - 1e-12):
        raise OperatorConsistentHomotopyDomainError(
            "straight RK4 nodes violate the central-difference stencil domain"
        )

    _, forcing, position_jacobian, direction_jacobian = (
        _operator_consistent_coefficients(
            values,
            positions,
            direction,
            difference_step=delta,
            refractivity_scale=scale,
            refractive_index_floor=index_floor,
        )
    )
    zero_matrices = torch.zeros_like(position_jacobian)
    delta_positions, delta_directions = integrate_affine_variational_rk4(
        forcing,
        zero_matrices,
        zero_matrices,
        direction,
        step_size=step_size,
    )

    forcing_midpoint = forcing[:, 1::2]
    integrated_base = torch.sum(forcing_midpoint, dim=1) * step_size
    base_output = torch.stack(
        (
            torch.sum(integrated_base * projection_u, dim=-1),
            torch.sum(integrated_base * projection_v, dim=-1),
        ),
        dim=-1,
    )
    delta_position_midpoint = 0.5 * (
        delta_positions[:, :-1] + delta_positions[:, 1:]
    )
    delta_direction_midpoint = 0.5 * (
        delta_directions[:, :-1] + delta_directions[:, 1:]
    )
    residual_integrand = torch.einsum(
        "rsij,rsj->rsi",
        position_jacobian[:, 1::2],
        delta_position_midpoint,
    ) + torch.einsum(
        "rsij,rsj->rsi",
        direction_jacobian[:, 1::2],
        delta_direction_midpoint,
    )
    residual_xyz = torch.sum(residual_integrand, dim=1) * step_size
    residual_uv = torch.stack(
        (
            torch.sum(residual_xyz * projection_u, dim=-1),
            torch.sum(residual_xyz * projection_v, dim=-1),
        ),
        dim=-1,
    )
    candidate = base_output + residual_uv

    base_endpoint_positions = positions[:, 0::2]
    base_endpoint_directions = direction[:, None].expand(-1, steps + 1, -1)
    predicted_positions = base_endpoint_positions + delta_positions
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
        torch.abs(
            torch.sum(base_endpoint_directions * delta_directions, dim=-1)
        ),
        dim=1,
    )
    finite_valid = (
        torch.all(torch.isfinite(residual_uv), dim=1)
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
    risk = torch.linalg.vector_norm(residual_uv, dim=-1)
    risk = torch.where(valid, risk, torch.full_like(risk, torch.inf))

    reasons: list[str] = []
    for ray_index in range(len(valid)):
        ray_reasons = []
        if not bool(finite_valid[ray_index]):
            ray_reasons.append("non_finite_operator_consistent_tangent")
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
    coefficient_samples = 2 * steps + 1
    return OperatorConsistentHomotopyResult(
        base_output_uv=base_output.detach(),
        residual_prediction_uv=residual_uv.detach(),
        candidate_output_uv=candidate.detach(),
        base_positions=base_endpoint_positions.detach(),
        base_directions=base_endpoint_directions.detach(),
        delta_positions=delta_positions.detach(),
        delta_directions=delta_directions.detach(),
        forcing=forcing.detach(),
        output_position_jacobian=position_jacobian.detach(),
        output_direction_jacobian=direction_jacobian.detach(),
        risk_norm=risk.detach(),
        maximum_position_perturbation=max_position_per_ray.detach(),
        maximum_direction_perturbation=max_direction_per_ray.detach(),
        minimum_predicted_domain_margin=predicted_domain_margin.detach(),
        minimum_predicted_stencil_margin=predicted_stencil_margin.detach(),
        direction_tangent_orthogonality_error=orthogonality_error.detach(),
        valid_mask=valid.detach(),
        failure_reasons=tuple(reasons),
        step_size=float(step_size),
        step_count=steps,
        coefficient_sample_count=coefficient_samples,
        difference_step=delta,
        refractivity_scale=scale,
        query_accounting={
            "logical_scalar_grid_point_queries": 7 * ray_count * coefficient_samples,
            "batched_interpolation_dispatches": 7,
            "coordinate_reverse_sweeps": 4,
            "forward_mode_bend_jvp_evaluations": 0,
            "reverse_mode_field_vjp_evaluations": 0,
            "exact_high_evaluations": 0,
            "coefficient_path": "all RK4 endpoints and midpoints",
        },
        stop_gradient_applied=True,
        risk_definition=(
            "L2 norm of the operator-consistent analytic derivative of the "
            "discrete bend-homotopy output at epsilon=0"
        ),
    )
