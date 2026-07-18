"""Fail-closed low-route certificates for curved-ray BOST development audits.

The strict part of this module bounds continuous ray curvature, path departure,
the normalized field domain, and a synthetic frustum without evaluating the
curved high-fidelity route.  Support-crossing stability uses a stronger
interval argument with global gradient/Hessian bounds.  It is still a
development certificate for the smoothstep surrogate, not a calibrated-camera
or reacting-flow guarantee.

The returned residual-risk score is deliberately labelled a proxy.  It may be
used to allocate a nonzero high-fidelity sampling probability, but it is not a
proof of residual size and it never overrides a fail-closed unsafe decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

try:
    from .automatic_discrete_multifidelity import (
        SyntheticRayRig,
        automatic_spatial_gradient,
        smoothstep_grid_field,
    )
    from .field_dependent_ray import initial_pupil_rays
except ImportError:
    from automatic_discrete_multifidelity import (
        SyntheticRayRig,
        automatic_spatial_gradient,
        smoothstep_grid_field,
    )
    from field_dependent_ray import initial_pupil_rays


RAY_SAFETY_CERTIFICATE_SCHEMA = "smoothstep-low-route-certificate-1.0"


@dataclass(frozen=True)
class SmoothstepDerivativeBounds:
    """Global derivative bounds for the normalized smoothstep grid field."""

    scalar_minimum: float
    scalar_maximum: float
    gradient_axis_bounds_xyz: tuple[float, float, float]
    gradient_norm_bound: float
    hessian_frobenius_bound: float


@dataclass(frozen=True)
class LowPathSafetyCertificate:
    """Per-ray safety decisions and detached development diagnostics."""

    safe_mask: torch.Tensor
    domain_frustum_safe_mask: torch.Tensor
    support_topology_safe_mask: torch.Tensor
    straight_support_crossings_per_ray: tuple[int, ...]
    failure_codes_per_ray: tuple[tuple[str, ...], ...]
    straight_domain_margin: torch.Tensor
    certified_domain_margin: torch.Tensor
    certified_frustum_margin: torch.Tensor
    support_robustness_margin: torch.Tensor
    local_deviation_proxy: torch.Tensor
    residual_risk_proxy: torch.Tensor
    per_ray_path_deviation_bound: torch.Tensor
    per_ray_direction_change_bound: torch.Tensor
    continuous_path_deviation_bound: float
    continuous_direction_change_bound: float
    curvature_norm_bound: float
    derivative_bounds: SmoothstepDerivativeBounds


@dataclass(frozen=True)
class _SmoothstepCellBounds:
    scalar_minimum: torch.Tensor
    gradient_norm: torch.Tensor
    hessian_frobenius: torch.Tensor


def _validated_grid(values_zyx: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values_zyx)
    if values.ndim != 3 or any(int(size) < 3 for size in values.shape):
        raise ValueError("values_zyx must have shape [z,y,x] with size at least 3")
    if not values.is_floating_point():
        values = values.to(torch.float64)
    if torch.any(~torch.isfinite(values)):
        raise ValueError("values_zyx must be finite")
    return values


def _maximum_absolute(values: torch.Tensor) -> float:
    return float(torch.max(torch.abs(values.detach()))) if values.numel() else 0.0


def smoothstep_derivative_bounds(
    values_zyx: torch.Tensor,
) -> SmoothstepDerivativeBounds:
    """Return conservative global first- and second-derivative bounds.

    For ``w(t)=3t^2-2t^3``, ``max |w'|=3/2`` and ``max |w''|=6``.
    Combining those constants with adjacent and mixed grid differences yields
    a bound valid in every interpolation cell.  The Frobenius Hessian bound is
    also an operator-norm bound.
    """

    values = _validated_grid(values_zyx)
    nz, ny, nx = (int(size) for size in values.shape)
    ax, ay, az = ((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0)

    diff_x = values[:, :, 1:] - values[:, :, :-1]
    diff_y = values[:, 1:, :] - values[:, :-1, :]
    diff_z = values[1:, :, :] - values[:-1, :, :]
    max_dx = _maximum_absolute(diff_x)
    max_dy = _maximum_absolute(diff_y)
    max_dz = _maximum_absolute(diff_z)
    gradient_axis = (
        1.5 * ax * max_dx,
        1.5 * ay * max_dy,
        1.5 * az * max_dz,
    )

    mixed_xy = (
        values[:, 1:, 1:]
        - values[:, 1:, :-1]
        - values[:, :-1, 1:]
        + values[:, :-1, :-1]
    )
    mixed_xz = (
        values[1:, :, 1:]
        - values[1:, :, :-1]
        - values[:-1, :, 1:]
        + values[:-1, :, :-1]
    )
    mixed_yz = (
        values[1:, 1:, :]
        - values[1:, :-1, :]
        - values[:-1, 1:, :]
        + values[:-1, :-1, :]
    )
    second_xx = 6.0 * ax * ax * max_dx
    second_yy = 6.0 * ay * ay * max_dy
    second_zz = 6.0 * az * az * max_dz
    second_xy = 2.25 * ax * ay * _maximum_absolute(mixed_xy)
    second_xz = 2.25 * ax * az * _maximum_absolute(mixed_xz)
    second_yz = 2.25 * ay * az * _maximum_absolute(mixed_yz)
    hessian_frobenius = math.sqrt(
        second_xx**2
        + second_yy**2
        + second_zz**2
        + 2.0 * (second_xy**2 + second_xz**2 + second_yz**2)
    )
    return SmoothstepDerivativeBounds(
        scalar_minimum=float(torch.min(values.detach())),
        scalar_maximum=float(torch.max(values.detach())),
        gradient_axis_bounds_xyz=tuple(float(value) for value in gradient_axis),
        gradient_norm_bound=float(math.sqrt(sum(value**2 for value in gradient_axis))),
        hessian_frobenius_bound=float(hessian_frobenius),
    )


def _smoothstep_cell_bounds(values_zyx: torch.Tensor) -> _SmoothstepCellBounds:
    """Return per-cell bounds used to tighten a globally valid ray tube."""

    values = _validated_grid(values_zyx)
    nz, ny, nx = (int(size) for size in values.shape)
    ax, ay, az = ((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0)
    v000 = values[:-1, :-1, :-1]
    v001 = values[:-1, :-1, 1:]
    v010 = values[:-1, 1:, :-1]
    v011 = values[:-1, 1:, 1:]
    v100 = values[1:, :-1, :-1]
    v101 = values[1:, :-1, 1:]
    v110 = values[1:, 1:, :-1]
    v111 = values[1:, 1:, 1:]
    corners = torch.stack((v000, v001, v010, v011, v100, v101, v110, v111))

    edge_x = torch.stack(
        (v001 - v000, v011 - v010, v101 - v100, v111 - v110)
    ).abs().amax(dim=0)
    edge_y = torch.stack(
        (v010 - v000, v011 - v001, v110 - v100, v111 - v101)
    ).abs().amax(dim=0)
    edge_z = torch.stack(
        (v100 - v000, v101 - v001, v110 - v010, v111 - v011)
    ).abs().amax(dim=0)
    gradient_x = 1.5 * ax * edge_x
    gradient_y = 1.5 * ay * edge_y
    gradient_z = 1.5 * az * edge_z
    gradient_norm = torch.sqrt(
        gradient_x.square() + gradient_y.square() + gradient_z.square()
    )

    mixed_xy = torch.stack(
        (
            v011 - v010 - v001 + v000,
            v111 - v110 - v101 + v100,
        )
    ).abs().amax(dim=0)
    mixed_xz = torch.stack(
        (
            v101 - v100 - v001 + v000,
            v111 - v110 - v011 + v010,
        )
    ).abs().amax(dim=0)
    mixed_yz = torch.stack(
        (
            v110 - v100 - v010 + v000,
            v111 - v101 - v011 + v001,
        )
    ).abs().amax(dim=0)
    second_xx = 6.0 * ax * ax * edge_x
    second_yy = 6.0 * ay * ay * edge_y
    second_zz = 6.0 * az * az * edge_z
    second_xy = 2.25 * ax * ay * mixed_xy
    second_xz = 2.25 * ax * az * mixed_xz
    second_yz = 2.25 * ay * az * mixed_yz
    hessian_frobenius = torch.sqrt(
        second_xx.square()
        + second_yy.square()
        + second_zz.square()
        + 2.0
        * (second_xy.square() + second_xz.square() + second_yz.square())
    )
    return _SmoothstepCellBounds(
        scalar_minimum=corners.amin(dim=0),
        gradient_norm=gradient_norm,
        hessian_frobenius=hessian_frobenius,
    )


def _tube_cell_bounds(
    cells: _SmoothstepCellBounds,
    point_a_xyz: torch.Tensor,
    point_b_xyz: torch.Tensor,
    *,
    radius: float,
) -> tuple[float, float, float]:
    """Bound a line-segment tube by enumerating every intersected grid cell."""

    shape_zyx = cells.gradient_norm.shape
    sizes_xyz = torch.as_tensor(
        [shape_zyx[2], shape_zyx[1], shape_zyx[0]],
        dtype=point_a_xyz.dtype,
        device=point_a_xyz.device,
    )
    lower_xyz = torch.minimum(point_a_xyz, point_b_xyz) - float(radius)
    upper_xyz = torch.maximum(point_a_xyz, point_b_xyz) + float(radius)
    lower_scaled = 0.5 * (torch.clamp(lower_xyz, -1.0, 1.0) + 1.0) * sizes_xyz
    upper_scaled = 0.5 * (torch.clamp(upper_xyz, -1.0, 1.0) + 1.0) * sizes_xyz
    lower_cell = torch.floor(lower_scaled).to(torch.long)
    upper_cell = torch.floor(upper_scaled).to(torch.long)
    maximum = torch.as_tensor(
        [shape_zyx[2] - 1, shape_zyx[1] - 1, shape_zyx[0] - 1],
        dtype=torch.long,
        device=point_a_xyz.device,
    )
    lower_cell = torch.minimum(torch.maximum(lower_cell, torch.zeros_like(lower_cell)), maximum)
    upper_cell = torch.minimum(torch.maximum(upper_cell, torch.zeros_like(upper_cell)), maximum)
    x0, y0, z0 = (int(value) for value in lower_cell)
    x1, y1, z1 = (int(value) for value in upper_cell)
    selection = (
        slice(z0, z1 + 1),
        slice(y0, y1 + 1),
        slice(x0, x1 + 1),
    )
    return (
        float(torch.min(cells.scalar_minimum[selection])),
        float(torch.max(cells.gradient_norm[selection])),
        float(torch.max(cells.hessian_frobenius[selection])),
    )


def _validated_positive(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def low_path_safety_certificate(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    refractivity_scale: float,
    difference_step: float,
    support_threshold: float,
    frustum_half_width_u: float,
    frustum_half_width_v: float,
    support_interval_count: int = 64,
    numerical_path_buffer: float = 0.0,
    robustness_tolerance: float = 1e-12,
) -> LowPathSafetyCertificate:
    """Build a low-only, fail-closed certificate for each pupil ray.

    The continuous path bound follows ``||d'|| <= ||grad(n)|| / n_min``:

    ``||d(s)-d(0)|| <= K s`` and
    ``||r(s)-r(0)-s d(0)|| <= K s^2 / 2``.

    Support topology is certified interval by interval.  Non-crossing
    intervals require a Lipschitz margin.  Crossing intervals require robust
    endpoint signs and a simple-root derivative margin after accounting for
    path and direction perturbations.  Any unresolved interval fails closed.
    """

    values = _validated_grid(values_zyx)
    scale = _validated_positive("refractivity_scale", refractivity_scale)
    stencil = _validated_positive("difference_step", difference_step)
    threshold = float(support_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("support_threshold must be finite and nonnegative")
    half_u = _validated_positive("frustum_half_width_u", frustum_half_width_u)
    half_v = _validated_positive("frustum_half_width_v", frustum_half_width_v)
    interval_count = int(support_interval_count)
    if interval_count < 4:
        raise ValueError("support_interval_count must be at least four")
    path_buffer = float(numerical_path_buffer)
    tolerance = float(robustness_tolerance)
    if not np.isfinite(path_buffer) or path_buffer < 0.0:
        raise ValueError("numerical_path_buffer must be finite and nonnegative")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("robustness_tolerance must be finite and nonnegative")

    bounds = smoothstep_derivative_bounds(values)
    global_minimum_index = 1.0 + scale * bounds.scalar_minimum
    if not np.isfinite(global_minimum_index) or global_minimum_index <= 0.5:
        raise ValueError("global refractive-index lower bound is invalid")
    global_curvature_bound = (
        scale * bounds.gradient_norm_bound / global_minimum_index
    )
    total_length = 2.0 * float(rig.path_half_length)
    if not np.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("rig path length must be finite and positive")
    global_maximum_deviation = (
        0.5 * global_curvature_bound * total_length**2 + path_buffer
    )

    start, direction, projection_u, projection_v = initial_pupil_rays(
        pupil_states,
        rig,
    )
    start = start.to(dtype=values.dtype, device=values.device)
    direction = direction.to(dtype=values.dtype, device=values.device)
    projection_u = projection_u.to(dtype=values.dtype, device=values.device)
    projection_v = projection_v.to(dtype=values.dtype, device=values.device)
    distances = torch.linspace(
        0.0,
        total_length,
        interval_count + 1,
        dtype=values.dtype,
        device=values.device,
    )
    midpoint_distances = 0.5 * (distances[:-1] + distances[1:])
    points = start[:, None, :] + distances[None, :, None] * direction[:, None, :]
    midpoint_points = (
        start[:, None, :] + midpoint_distances[None, :, None] * direction[:, None, :]
    )
    if torch.any(torch.abs(points) > 1.0 + 1e-12):
        raise ValueError("straight low path leaves the normalized field domain")

    ray_count = len(start)
    cells = _smoothstep_cell_bounds(values)
    ray_curvature_bounds: list[float] = []
    ray_deviation_bounds: list[float] = []
    ray_direction_bounds: list[float] = []
    for ray_index in range(ray_count):
        local_scalar_minimum, local_gradient_bound, _ = _tube_cell_bounds(
            cells,
            points[ray_index, 0],
            points[ray_index, -1],
            radius=global_maximum_deviation,
        )
        local_minimum_index = 1.0 + scale * local_scalar_minimum
        if local_minimum_index <= 0.5:
            raise ValueError("local refractive-index lower bound is invalid")
        local_curvature_bound = scale * local_gradient_bound / local_minimum_index
        ray_curvature_bounds.append(local_curvature_bound)
        ray_deviation_bounds.append(
            0.5 * local_curvature_bound * total_length**2 + path_buffer
        )
        ray_direction_bounds.append(min(2.0, local_curvature_bound * total_length))
    ray_curvature_bound = torch.as_tensor(
        ray_curvature_bounds,
        dtype=values.dtype,
        device=values.device,
    )
    ray_deviation_bound = torch.as_tensor(
        ray_deviation_bounds,
        dtype=values.dtype,
        device=values.device,
    )
    ray_direction_bound = torch.as_tensor(
        ray_direction_bounds,
        dtype=values.dtype,
        device=values.device,
    )
    field_endpoints = smoothstep_grid_field(
        values,
        points.reshape(-1, 3),
    ).reshape(ray_count, interval_count + 1)
    field_midpoints = smoothstep_grid_field(
        values,
        midpoint_points.reshape(-1, 3),
    ).reshape(ray_count, interval_count)
    gradient_midpoints = automatic_spatial_gradient(
        values,
        midpoint_points.reshape(-1, 3),
        create_graph=False,
    ).reshape(ray_count, interval_count, 3)

    straight_domain_margin = 1.0 - torch.amax(torch.abs(points), dim=(1, 2))
    certified_domain_margin = straight_domain_margin - ray_deviation_bound - stencil
    frustum_half_width = min(half_u, half_v)
    certified_frustum_margin = frustum_half_width - ray_deviation_bound
    domain_frustum_safe = (certified_domain_margin > tolerance) & (
        certified_frustum_margin > tolerance
    )

    step_length = total_length / interval_count
    endpoint_deviation = (
        0.5 * ray_curvature_bound[:, None] * distances[None, :].square()
        + path_buffer
    )
    midpoint_deviation = (
        0.5
        * ray_curvature_bound[:, None]
        * midpoint_distances[None, :].square()
        + path_buffer
    )
    endpoint_direction_change = torch.clamp(
        ray_curvature_bound[:, None] * distances[None, :],
        max=2.0,
    )
    midpoint_direction_change = torch.clamp(
        ray_curvature_bound[:, None] * midpoint_distances[None, :],
        max=2.0,
    )
    q_endpoints = torch.abs(field_endpoints) - threshold
    q_midpoints = torch.abs(field_midpoints) - threshold
    support_safe_values: list[bool] = []
    support_crossings: list[int] = []
    support_margins: list[float] = []

    for ray_index in range(ray_count):
        ray_safe = True
        crossing_count = 0
        ray_margin = math.inf
        ray_direction = direction[ray_index]
        for interval_index in range(interval_count):
            q_left = float(q_endpoints[ray_index, interval_index])
            q_right = float(q_endpoints[ray_index, interval_index + 1])
            q_mid = float(q_midpoints[ray_index, interval_index])
            field_mid = float(field_midpoints[ray_index, interval_index])
            deviation_left = float(endpoint_deviation[ray_index, interval_index])
            deviation_right = float(
                endpoint_deviation[ray_index, interval_index + 1]
            )
            deviation_mid = float(midpoint_deviation[ray_index, interval_index])
            deviation_max = max(deviation_left, deviation_right, deviation_mid)
            direction_max = max(
                float(endpoint_direction_change[ray_index, interval_index]),
                float(endpoint_direction_change[ray_index, interval_index + 1]),
                float(midpoint_direction_change[ray_index, interval_index]),
            )
            _, gradient_bound, hessian_bound = _tube_cell_bounds(
                cells,
                points[ray_index, interval_index],
                points[ray_index, interval_index + 1],
                radius=deviation_max,
            )
            path_value_error = gradient_bound * deviation_max
            same_strict_sign = q_left * q_right > 0.0
            opposite_strict_sign = q_left * q_right < 0.0
            if same_strict_sign:
                no_cross_margin = (
                    abs(q_mid)
                    - 0.5 * gradient_bound * step_length
                    - path_value_error
                )
                ray_margin = min(ray_margin, no_cross_margin)
                if no_cross_margin <= tolerance:
                    ray_safe = False
            elif opposite_strict_sign:
                crossing_count += 1
                endpoint_margin = min(
                    abs(q_left) - gradient_bound * deviation_left,
                    abs(q_right) - gradient_bound * deviation_right,
                )
                field_sign_margin = (
                    abs(field_mid)
                    - 0.5 * gradient_bound * step_length
                    - path_value_error
                )
                midpoint_gradient = gradient_midpoints[ray_index, interval_index]
                directional_derivative = float(
                    torch.sum(midpoint_gradient * ray_direction)
                )
                if field_mid < 0.0:
                    directional_derivative = -directional_derivative
                baseline_derivative_margin = (
                    abs(directional_derivative)
                    - 0.5 * hessian_bound * step_length
                )
                trajectory_derivative_error = (
                    hessian_bound * deviation_max + gradient_bound * direction_max
                )
                simple_root_margin = (
                    baseline_derivative_margin - trajectory_derivative_error
                )
                crossing_margin = min(
                    endpoint_margin,
                    field_sign_margin,
                    simple_root_margin,
                    threshold - path_value_error,
                )
                ray_margin = min(ray_margin, crossing_margin)
                if crossing_margin <= tolerance:
                    ray_safe = False
            else:
                ray_margin = min(ray_margin, -tolerance)
                ray_safe = False
        support_safe_values.append(ray_safe)
        support_crossings.append(crossing_count)
        support_margins.append(ray_margin if np.isfinite(ray_margin) else 0.0)

    support_topology_safe = torch.as_tensor(
        support_safe_values,
        dtype=torch.bool,
        device=values.device,
    )
    safe_mask = domain_frustum_safe & support_topology_safe
    support_margin_tensor = torch.as_tensor(
        support_margins,
        dtype=values.dtype,
        device=values.device,
    )

    refractive_index_midpoint = 1.0 + scale * field_midpoints
    gradient_n = scale * gradient_midpoints
    longitudinal = torch.sum(
        gradient_n * direction[:, None, :],
        dim=-1,
        keepdim=True,
    )
    curvature = (
        gradient_n - longitudinal * direction[:, None, :]
    ) / refractive_index_midpoint[:, :, None]
    remaining_length = total_length - midpoint_distances
    local_deviation_proxy = torch.sum(
        torch.linalg.vector_norm(curvature, dim=-1)
        * remaining_length[None, :]
        * step_length,
        dim=1,
    )
    projected_curvature_u = torch.sum(curvature * projection_u[:, None, :], dim=-1)
    projected_curvature_v = torch.sum(curvature * projection_v[:, None, :], dim=-1)
    projected_exposure = torch.sqrt(
        torch.sum(projected_curvature_u * step_length, dim=1).square()
        + torch.sum(projected_curvature_v * step_length, dim=1).square()
    )
    crossing_tensor = torch.as_tensor(
        support_crossings,
        dtype=values.dtype,
        device=values.device,
    )
    residual_risk_proxy = (
        local_deviation_proxy / frustum_half_width
        + 0.25
        * projected_exposure
        / torch.clamp(ray_curvature_bound * total_length, min=1e-30)
        + 0.02 * crossing_tensor
        + torch.finfo(values.dtype).eps
    ).detach()

    failure_codes: list[tuple[str, ...]] = []
    for ray_index in range(ray_count):
        codes: list[str] = []
        if float(certified_domain_margin[ray_index]) <= tolerance:
            codes.append("FAIL_DOMAIN_BOUND")
        if float(certified_frustum_margin[ray_index]) <= tolerance:
            codes.append("FAIL_FRUSTUM_BOUND")
        if not bool(support_topology_safe[ray_index]):
            codes.append("FAIL_SUPPORT_TOPOLOGY_BOUND")
        failure_codes.append(tuple(codes))

    return LowPathSafetyCertificate(
        safe_mask=safe_mask.detach(),
        domain_frustum_safe_mask=domain_frustum_safe.detach(),
        support_topology_safe_mask=support_topology_safe.detach(),
        straight_support_crossings_per_ray=tuple(support_crossings),
        failure_codes_per_ray=tuple(failure_codes),
        straight_domain_margin=straight_domain_margin.detach(),
        certified_domain_margin=certified_domain_margin.detach(),
        certified_frustum_margin=certified_frustum_margin.detach(),
        support_robustness_margin=support_margin_tensor.detach(),
        local_deviation_proxy=local_deviation_proxy.detach(),
        residual_risk_proxy=residual_risk_proxy,
        per_ray_path_deviation_bound=ray_deviation_bound.detach(),
        per_ray_direction_change_bound=ray_direction_bound.detach(),
        continuous_path_deviation_bound=float(torch.max(ray_deviation_bound)),
        continuous_direction_change_bound=float(torch.max(ray_direction_bound)),
        curvature_norm_bound=float(torch.max(ray_curvature_bound)),
        derivative_bounds=bounds,
    )
