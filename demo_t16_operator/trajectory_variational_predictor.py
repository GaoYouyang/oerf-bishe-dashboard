"""Frozen-path first-order trajectory predictor for BOST routing audits.

The field-dependent ray equation used by :mod:`field_dependent_ray` is

    r' = d,
    d' = F(r, d) = (I - d d^T) grad(n(r)) / n(r).

This module linearizes ``F`` along the *straight medium path*
``r0(s) = r_in + s d0``.  With ``P0 = I - d0 d0^T`` and a tangent direction
increment ``delta_d``, the frozen-path affine variational system is

    delta_r' = delta_d,
    delta_d' = F0 + A delta_r + B delta_d,

where

    F0 = P0 g / n,
    A  = P0 (H / n - g g^T / n^2),
    B  = P0 (-(d0.g) I - d0 g^T) P0 / n,

and ``g = grad(n)``, ``H = Hessian(n)`` are sampled on ``r0``.  ``P0`` is
also applied after every RK4 step so ``delta_d`` stays in the tangent plane of
the frozen unit direction up to roundoff.

The predicted high-minus-medium correction uses the same midpoint geometry as
``path_integrated_deflection``:

    Delta_HM ~= sum_j h [A_j delta_r_j + B_j delta_d_j].

This is a development-only routing statistic.  It is not a proof of speed,
accuracy, novelty, experimental validity, or generalization.  The grid,
straight path, coefficients, and outputs are always detached.  In particular,
the returned risk must be treated as stop-gradient metadata; this module must
not silently contribute a reconstruction gradient.

The local model assumes a positive scalar refractive index, a fixed calibrated
ray/field coordinate system, small trajectory perturbations, and no caustic,
visibility, support-topology, or cell-event change.  The smoothstep grid is
piecewise twice differentiable; its Hessian at a cell interface follows the
autograd branch convention and is not a validated interval derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

try:
    from .automatic_discrete_multifidelity import (
        SyntheticRayRig,
        smoothstep_grid_field,
    )
    from .field_dependent_ray import initial_pupil_rays
except ImportError:
    from automatic_discrete_multifidelity import (
        SyntheticRayRig,
        smoothstep_grid_field,
    )
    from field_dependent_ray import initial_pupil_rays


TRAJECTORY_VARIATIONAL_PREDICTOR_SCHEMA = (
    "frozen-straight-path-trajectory-variational-predictor-1.0"
)


class TrajectoryPredictorDomainError(RuntimeError):
    """Raised when the frozen path or its field contract is outside scope."""


@dataclass(frozen=True)
class StraightPathLinearization:
    """Detached coefficients sampled at RK4 half-step nodes.

    Arrays with a path axis have length ``2 * step_count + 1``.  Even indices
    are interval endpoints and odd indices are RK4 midpoints.
    """

    positions: torch.Tensor
    direction: torch.Tensor
    projection_u: torch.Tensor
    projection_v: torch.Tensor
    refractive_index: torch.Tensor
    gradient_n: torch.Tensor
    hessian_n: torch.Tensor
    forcing: torch.Tensor
    position_jacobian: torch.Tensor
    direction_jacobian: torch.Tensor
    step_size: float
    minimum_domain_margin_per_ray: torch.Tensor
    stop_gradient_applied: bool


@dataclass(frozen=True)
class TrajectoryVariationalPrediction:
    """Directional ``H-M`` prediction and fail-closed routing diagnostics."""

    residual_prediction_xyz: torch.Tensor
    residual_prediction_uv: torch.Tensor
    risk_norm: torch.Tensor
    delta_positions: torch.Tensor
    delta_directions: torch.Tensor
    maximum_position_perturbation: torch.Tensor
    maximum_direction_perturbation: torch.Tensor
    minimum_predicted_domain_margin: torch.Tensor
    valid_mask: torch.Tensor
    failure_reasons: tuple[str, ...]
    step_size: float
    coefficient_sample_count: int
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


def _validate_predictor_inputs(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    *,
    refractivity_scale: float,
    step_count: int,
    domain_margin: float,
    refractive_index_floor: float,
    max_position_perturbation: float,
    max_direction_perturbation: float,
) -> tuple[torch.Tensor, torch.Tensor, float, int, float, float, float, float]:
    values = _require_float64_tensor(values_zyx, name="values_zyx", ndim=3)
    if any(int(size) < 3 for size in values.shape):
        raise ValueError("values_zyx axes must each contain at least three samples")
    states = _require_float64_tensor(pupil_states, name="pupil_states", ndim=2)
    if states.shape[1] != 2 or len(states) < 1:
        raise ValueError("pupil_states must have shape [ray,2]")
    if torch.any(states < 0.0) or torch.any(states > 1.0):
        raise ValueError("pupil_states must lie in [0,1]^2")

    scale = float(refractivity_scale)
    steps = int(step_count)
    margin = float(domain_margin)
    index_floor = float(refractive_index_floor)
    max_position = float(max_position_perturbation)
    max_direction = float(max_direction_perturbation)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("refractivity_scale must be finite and positive")
    if steps < 2 or steps != step_count:
        raise ValueError("step_count must be an integer of at least two")
    if not np.isfinite(margin) or margin < 0.0 or margin >= 1.0:
        raise ValueError("domain_margin must be finite and lie in [0,1)")
    if not np.isfinite(index_floor) or index_floor <= 0.0:
        raise ValueError("refractive_index_floor must be finite and positive")
    if not np.isfinite(max_position) or max_position <= 0.0:
        raise ValueError("max_position_perturbation must be finite and positive")
    if not np.isfinite(max_direction) or max_direction <= 0.0:
        raise ValueError("max_direction_perturbation must be finite and positive")
    return (
        values.detach(),
        states.detach(),
        scale,
        steps,
        margin,
        index_floor,
        max_position,
        max_direction,
    )


def _field_value_gradient_hessian(
    values_zyx: torch.Tensor,
    points_xyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached value, coordinate gradient, and coordinate Hessian."""

    points = points_xyz.detach().clone().requires_grad_(True)
    field = smoothstep_grid_field(values_zyx.detach(), points)
    gradient = torch.autograd.grad(
        field,
        points,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]
    hessian_rows = []
    for component in range(3):
        hessian_rows.append(
            torch.autograd.grad(
                gradient[:, component],
                points,
                grad_outputs=torch.ones_like(gradient[:, component]),
                create_graph=False,
                retain_graph=component < 2,
            )[0]
        )
    hessian = torch.stack(hessian_rows, dim=-2)
    return field.detach(), gradient.detach(), hessian.detach()


def linearize_straight_medium_path(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    refractivity_scale: float,
    step_count: int,
    domain_margin: float = 1e-6,
    refractive_index_floor: float = 0.5,
) -> StraightPathLinearization:
    """Build detached ``F0, A, B`` coefficients on a straight ray batch.

    This function deliberately exposes the derivative tensors so their spatial
    finite-difference checks can be audited independently of the integrator.
    It raises instead of extrapolating when any frozen sample leaves the domain
    contract or when ``n`` crosses the declared floor.
    """

    (
        values,
        states,
        scale,
        steps,
        margin,
        index_floor,
        _,
        _,
    ) = _validate_predictor_inputs(
        values_zyx,
        pupil_states,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
        domain_margin=domain_margin,
        refractive_index_floor=refractive_index_floor,
        max_position_perturbation=1.0,
        max_direction_perturbation=1.0,
    )
    try:
        start, direction, projection_u, projection_v = initial_pupil_rays(states, rig)
    except ValueError as error:
        raise TrajectoryPredictorDomainError(
            "straight-medium ray geometry is outside its declared domain"
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
    positions = start[:, None, :] + (
        half_step_distance[None, :, None] * direction[:, None, :]
    )
    domain_margin_per_ray = torch.amin(1.0 - torch.abs(positions), dim=(1, 2))
    if torch.any(domain_margin_per_ray < margin - 1e-12):
        worst = float(torch.min(domain_margin_per_ray))
        raise TrajectoryPredictorDomainError(
            "straight path violates the frozen domain margin: "
            f"minimum={worst:.6g}, required={margin:.6g}"
        )

    ray_count, sample_count, _ = positions.shape
    field, gradient_field, hessian_field = _field_value_gradient_hessian(
        values,
        positions.reshape(-1, 3),
    )
    field = field.reshape(ray_count, sample_count)
    gradient_n = (scale * gradient_field).reshape(ray_count, sample_count, 3)
    hessian_n = (scale * hessian_field).reshape(
        ray_count,
        sample_count,
        3,
        3,
    )
    refractive_index = 1.0 + scale * field
    if torch.any(~torch.isfinite(refractive_index)) or torch.any(
        refractive_index < index_floor
    ):
        worst = float(torch.min(refractive_index))
        raise TrajectoryPredictorDomainError(
            "refractive index violates the frozen positive-index floor: "
            f"minimum={worst:.6g}, required={index_floor:.6g}"
        )

    identity = torch.eye(3, dtype=torch.float64, device=values.device)
    projector = identity[None, :, :] - torch.einsum(
        "ri,rj->rij",
        direction,
        direction,
    )
    projector_path = projector[:, None, :, :]
    inverse_index = torch.reciprocal(refractive_index)
    forcing = torch.einsum(
        "rsij,rsj->rsi",
        projector_path,
        gradient_n,
    ) * inverse_index[:, :, None]

    gradient_outer = torch.einsum(
        "rsi,rsj->rsij",
        gradient_n,
        gradient_n,
    )
    spatial_core = (
        hessian_n * inverse_index[:, :, None, None]
        - gradient_outer * inverse_index.square()[:, :, None, None]
    )
    position_jacobian = torch.matmul(projector_path, spatial_core)

    longitudinal_gradient = torch.sum(
        direction[:, None, :] * gradient_n,
        dim=-1,
    )
    raw_direction_jacobian = -(
        longitudinal_gradient[:, :, None, None] * identity[None, None, :, :]
        + torch.einsum("ri,rsj->rsij", direction, gradient_n)
    ) * inverse_index[:, :, None, None]
    direction_jacobian = torch.matmul(
        torch.matmul(projector_path, raw_direction_jacobian),
        projector_path,
    )

    tensors = (
        gradient_n,
        hessian_n,
        forcing,
        position_jacobian,
        direction_jacobian,
    )
    if any(torch.any(~torch.isfinite(tensor)) for tensor in tensors):
        raise TrajectoryPredictorDomainError(
            "non-finite frozen-path derivative coefficient"
        )
    return StraightPathLinearization(
        positions=positions.detach(),
        direction=direction.detach(),
        projection_u=projection_u.detach(),
        projection_v=projection_v.detach(),
        refractive_index=refractive_index.detach(),
        gradient_n=gradient_n.detach(),
        hessian_n=hessian_n.detach(),
        forcing=forcing.detach(),
        position_jacobian=position_jacobian.detach(),
        direction_jacobian=direction_jacobian.detach(),
        step_size=float(step_size),
        minimum_domain_margin_per_ray=domain_margin_per_ray.detach(),
        stop_gradient_applied=True,
    )


def _validate_variational_coefficients(
    forcing: torch.Tensor,
    position_jacobian: torch.Tensor,
    direction_jacobian: torch.Tensor,
    direction0: torch.Tensor,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    force = _require_float64_tensor(forcing, name="forcing", ndim=3)
    a_matrix = _require_float64_tensor(
        position_jacobian,
        name="position_jacobian",
        ndim=4,
    )
    b_matrix = _require_float64_tensor(
        direction_jacobian,
        name="direction_jacobian",
        ndim=4,
    )
    direction = _require_float64_tensor(direction0, name="direction0", ndim=2)
    if force.shape[-1] != 3:
        raise ValueError("forcing must have shape [ray,2*step+1,3]")
    if a_matrix.shape != force.shape[:2] + (3, 3):
        raise ValueError("position_jacobian shape does not match forcing")
    if b_matrix.shape != a_matrix.shape:
        raise ValueError("direction_jacobian shape does not match forcing")
    if direction.shape != (force.shape[0], 3):
        raise ValueError("direction0 must have shape [ray,3]")
    if force.shape[1] < 5 or force.shape[1] % 2 != 1:
        raise ValueError("coefficient path axis must equal 2*step_count+1")
    if not (force.device == a_matrix.device == b_matrix.device == direction.device):
        raise ValueError("all variational tensors must share one device")
    norms = torch.linalg.vector_norm(direction, dim=-1)
    if torch.any(torch.abs(norms - 1.0) > 1e-10):
        raise ValueError("direction0 must be unit length to 1e-10")
    h = float(step_size)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("step_size must be finite and positive")
    steps = (force.shape[1] - 1) // 2
    return force, a_matrix, b_matrix, direction, h, steps


def integrate_affine_variational_rk4(
    forcing: torch.Tensor,
    position_jacobian: torch.Tensor,
    direction_jacobian: torch.Tensor,
    direction0: torch.Tensor,
    *,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the affine variational state using half-step coefficients.

    For interval ``j``, coefficient indices ``2j``, ``2j+1``, and ``2j+2``
    supply the RK4 start, midpoint, and endpoint stages.  The returned arrays
    contain interval endpoints and therefore have shape ``[ray, step+1, 3]``.
    """

    force, a_matrix, b_matrix, direction, h, steps = (
        _validate_variational_coefficients(
            forcing,
            position_jacobian,
            direction_jacobian,
            direction0,
            step_size,
        )
    )
    identity = torch.eye(3, dtype=torch.float64, device=force.device)
    projector = identity[None, :, :] - torch.einsum(
        "ri,rj->rij",
        direction,
        direction,
    )
    delta_r = torch.zeros_like(direction)
    delta_d = torch.zeros_like(direction)
    delta_r_nodes = [delta_r]
    delta_d_nodes = [delta_d]

    def rhs(
        state_r: torch.Tensor,
        state_d: torch.Tensor,
        coefficient_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acceleration = (
            force[:, coefficient_index]
            + torch.einsum(
                "rij,rj->ri",
                a_matrix[:, coefficient_index],
                state_r,
            )
            + torch.einsum(
                "rij,rj->ri",
                b_matrix[:, coefficient_index],
                state_d,
            )
        )
        acceleration = torch.einsum("rij,rj->ri", projector, acceleration)
        tangent_d = torch.einsum("rij,rj->ri", projector, state_d)
        return tangent_d, acceleration

    for index in range(steps):
        start = 2 * index
        midpoint = start + 1
        end = start + 2
        k1r, k1d = rhs(delta_r, delta_d, start)
        k2r, k2d = rhs(
            delta_r + 0.5 * h * k1r,
            delta_d + 0.5 * h * k1d,
            midpoint,
        )
        k3r, k3d = rhs(
            delta_r + 0.5 * h * k2r,
            delta_d + 0.5 * h * k2d,
            midpoint,
        )
        k4r, k4d = rhs(
            delta_r + h * k3r,
            delta_d + h * k3d,
            end,
        )
        delta_r = delta_r + (h / 6.0) * (
            k1r + 2.0 * k2r + 2.0 * k3r + k4r
        )
        delta_d = delta_d + (h / 6.0) * (
            k1d + 2.0 * k2d + 2.0 * k3d + k4d
        )
        delta_d = torch.einsum("rij,rj->ri", projector, delta_d)
        delta_r_nodes.append(delta_r)
        delta_d_nodes.append(delta_d)
    return (
        torch.stack(delta_r_nodes, dim=1).detach(),
        torch.stack(delta_d_nodes, dim=1).detach(),
    )


def predict_trajectory_variational_residual(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    refractivity_scale: float,
    step_count: int,
    domain_margin: float = 1e-6,
    refractive_index_floor: float = 0.5,
    max_position_perturbation: float = 0.2,
    max_direction_perturbation: float = 0.2,
) -> TrajectoryVariationalPrediction:
    """Predict the directional trajectory part of ``H-M`` on a ray batch.

    ``risk_norm`` is exactly the Euclidean norm of the two detector-plane
    components for in-contract rays.  A ray whose predicted path or direction
    exceeds the declared linearization contract receives ``risk_norm=inf`` and
    ``valid_mask=False``.  The directional diagnostic is retained for audit,
    but it is not authorized as a correction for those rays.
    """

    (
        _,
        _,
        _,
        _,
        margin,
        _,
        max_position,
        max_direction,
    ) = _validate_predictor_inputs(
        values_zyx,
        pupil_states,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
        domain_margin=domain_margin,
        refractive_index_floor=refractive_index_floor,
        max_position_perturbation=max_position_perturbation,
        max_direction_perturbation=max_direction_perturbation,
    )
    linearization = linearize_straight_medium_path(
        values_zyx,
        pupil_states,
        rig,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
        domain_margin=domain_margin,
        refractive_index_floor=refractive_index_floor,
    )
    delta_r, delta_d = integrate_affine_variational_rk4(
        linearization.forcing,
        linearization.position_jacobian,
        linearization.direction_jacobian,
        linearization.direction,
        step_size=linearization.step_size,
    )

    delta_r_midpoint = 0.5 * (delta_r[:, :-1] + delta_r[:, 1:])
    delta_d_midpoint = 0.5 * (delta_d[:, :-1] + delta_d[:, 1:])
    position_jacobian_midpoint = linearization.position_jacobian[:, 1::2]
    direction_jacobian_midpoint = linearization.direction_jacobian[:, 1::2]
    residual_integrand = torch.einsum(
        "rsij,rsj->rsi",
        position_jacobian_midpoint,
        delta_r_midpoint,
    ) + torch.einsum(
        "rsij,rsj->rsi",
        direction_jacobian_midpoint,
        delta_d_midpoint,
    )
    residual_xyz = (
        torch.sum(residual_integrand, dim=1) * float(linearization.step_size)
    )
    residual_uv = torch.stack(
        (
            torch.sum(residual_xyz * linearization.projection_u, dim=-1),
            torch.sum(residual_xyz * linearization.projection_v, dim=-1),
        ),
        dim=-1,
    )

    max_position_per_ray = torch.amax(
        torch.linalg.vector_norm(delta_r, dim=-1),
        dim=1,
    )
    max_direction_per_ray = torch.amax(
        torch.linalg.vector_norm(delta_d, dim=-1),
        dim=1,
    )
    straight_endpoint_positions = linearization.positions[:, 0::2]
    predicted_positions = straight_endpoint_positions + delta_r
    predicted_margin_per_ray = torch.amin(
        1.0 - torch.abs(predicted_positions),
        dim=(1, 2),
    )
    domain_valid = predicted_margin_per_ray >= margin - 1e-12
    position_valid = max_position_per_ray <= max_position
    direction_valid = max_direction_per_ray <= max_direction
    finite_valid = (
        torch.all(torch.isfinite(delta_r), dim=(1, 2))
        & torch.all(torch.isfinite(delta_d), dim=(1, 2))
        & torch.all(torch.isfinite(residual_uv), dim=1)
    )
    valid = domain_valid & position_valid & direction_valid & finite_valid
    risk = torch.linalg.vector_norm(residual_uv, dim=-1)
    risk = torch.where(valid, risk, torch.full_like(risk, torch.inf))

    reasons: list[str] = []
    for ray_index in range(len(valid)):
        ray_reasons = []
        if not bool(finite_valid[ray_index]):
            ray_reasons.append("non_finite_variational_state")
        if not bool(domain_valid[ray_index]):
            ray_reasons.append("predicted_path_domain")
        if not bool(position_valid[ray_index]):
            ray_reasons.append("position_linearization_radius")
        if not bool(direction_valid[ray_index]):
            ray_reasons.append("direction_linearization_radius")
        reasons.append("ok" if not ray_reasons else ";".join(ray_reasons))

    return TrajectoryVariationalPrediction(
        residual_prediction_xyz=residual_xyz.detach(),
        residual_prediction_uv=residual_uv.detach(),
        risk_norm=risk.detach(),
        delta_positions=delta_r.detach(),
        delta_directions=delta_d.detach(),
        maximum_position_perturbation=max_position_per_ray.detach(),
        maximum_direction_perturbation=max_direction_per_ray.detach(),
        minimum_predicted_domain_margin=predicted_margin_per_ray.detach(),
        valid_mask=valid.detach(),
        failure_reasons=tuple(reasons),
        step_size=float(linearization.step_size),
        coefficient_sample_count=int(linearization.positions.shape[1]),
        stop_gradient_applied=True,
        risk_definition="L2 norm of the detector-plane first-order H-M prediction",
    )
