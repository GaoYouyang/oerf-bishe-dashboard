"""Ordered discrete-program signatures for field-dependent ray derivatives.

The signature records which interpolation cell and support branch each ray
uses at every RK4 stage, curved midpoint, straight midpoint, and central
difference offset.  Continuous trajectory coordinates are deliberately not
hashed: they should vary smoothly under a field perturbation.  The discrete
cell/support/frustum branch must remain unchanged for a local derivative gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

try:
    from .automatic_discrete_multifidelity import SyntheticRayRig, smoothstep_grid_field
    from .field_dependent_ray import (
        RayTraceResult,
        _ray_rhs,
        _validate_trace_parameters,
        initial_pupil_rays,
    )
except ImportError:
    from automatic_discrete_multifidelity import SyntheticRayRig, smoothstep_grid_field
    from field_dependent_ray import (
        RayTraceResult,
        _ray_rhs,
        _validate_trace_parameters,
        initial_pupil_rays,
    )


FIELD_PROGRAM_SIGNATURE_SCHEMA = "field-program-signature-1.0"


@dataclass(frozen=True, slots=True)
class FieldProgramSignature:
    """Canonical hashes and fail-closed geometry diagnostics."""

    digest_sha256: str
    labels_sha256: str
    interpolation_cells_sha256: str
    support_bits_sha256: str
    frustum_margin_signs_sha256: str
    group_count: int
    query_record_count: int
    support_true_count: int
    minimum_domain_margin: float
    minimum_stencil_margin: float
    minimum_frustum_margin: float
    maximum_direction_norm_error: float


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _validated_inputs(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    *,
    difference_step: float,
    support_threshold: float,
    frustum_half_width_u: float,
    frustum_half_width_v: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    values = torch.as_tensor(values_zyx)
    states = torch.as_tensor(pupil_states)
    if values.device.type != "cpu" or values.dtype != torch.float64 or values.ndim != 3:
        raise TypeError("values_zyx must be a CPU float64 [z,y,x] tensor")
    if states.device.type != "cpu" or states.dtype != torch.float64 or states.ndim != 2:
        raise TypeError("pupil_states must be a CPU float64 [ray,2] tensor")
    if states.shape[1] != 2 or len(states) < 1:
        raise ValueError("pupil_states must have shape [ray,2]")
    if not bool(torch.all(torch.isfinite(values))) or not bool(
        torch.all(torch.isfinite(states))
    ):
        raise ValueError("field-program signature inputs must be finite")
    delta = float(difference_step)
    threshold = float(support_threshold)
    half_u = float(frustum_half_width_u)
    half_v = float(frustum_half_width_v)
    if not math.isfinite(delta) or not 0.0 < delta < 0.25:
        raise ValueError("difference_step must lie in (0,0.25)")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("support_threshold must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0.0 for value in (half_u, half_v)):
        raise ValueError("frustum half widths must be finite and positive")
    return values, states, delta, threshold, half_u, half_v


def _central_query_bundle(
    points: torch.Tensor,
    *,
    difference_step: float,
) -> torch.Tensor:
    identity = torch.eye(3, dtype=points.dtype, device=points.device)
    offsets = torch.cat(
        (
            torch.zeros((1, 3), dtype=points.dtype, device=points.device),
            difference_step * identity,
            -difference_step * identity,
        ),
        dim=0,
    )
    return points[:, None, :] + offsets[None, :, :]


def _lower_interpolation_cells(
    queries_xyz: torch.Tensor,
    field_shape_zyx: tuple[int, int, int],
) -> torch.Tensor:
    nz, ny, nx = field_shape_zyx
    sizes_xyz = torch.tensor(
        [nx - 1, ny - 1, nz - 1],
        dtype=queries_xyz.dtype,
        device=queries_xyz.device,
    )
    scaled = 0.5 * (queries_xyz + 1.0) * sizes_xyz
    lower = torch.floor(scaled).to(torch.int64)
    maximum = torch.tensor(
        [nx - 2, ny - 2, nz - 2],
        dtype=torch.int64,
        device=queries_xyz.device,
    )
    return torch.minimum(torch.maximum(lower, torch.zeros_like(lower)), maximum)


def _diagnostic_trace_with_stages(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
) -> tuple[RayTraceResult, list[tuple[str, torch.Tensor]]]:
    """Replay the frozen RK4 program while retaining every stage coordinate.

    The production ray module is already hash-bound by earlier D1-D3 evidence,
    so D4 cannot add an observer argument to it.  This diagnostic replay calls
    the same validated RHS and is independently checked against production
    endpoint traces in tests.
    """

    mode, delta, scale, steps = _validate_trace_parameters(
        gradient_mode="central",
        difference_step=difference_step,
        refractivity_scale=refractivity_scale,
        step_count=step_count,
    )
    values = torch.as_tensor(values_zyx)
    position, direction, projection_u, projection_v = initial_pupil_rays(
        pupil_states, rig
    )
    position = position.to(dtype=values.dtype, device=values.device)
    direction = direction.to(dtype=values.dtype, device=values.device)
    projection_u = projection_u.to(dtype=values.dtype, device=values.device)
    projection_v = projection_v.to(dtype=values.dtype, device=values.device)
    step_size = 2.0 * float(rig.path_half_length) / float(steps)
    positions = [position]
    directions = [direction]
    stages: list[tuple[str, torch.Tensor]] = []
    domain_margins: list[float] = []
    stencil_margins: list[float] = []

    def rhs(
        p: torch.Tensor,
        d: torch.Tensor,
        *,
        step_index: int,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stages.append((f"step={step_index},stage={stage}", p.detach().clone()))
        dp, dd, domain_margin, stencil_margin = _ray_rhs(
            values,
            p,
            d,
            gradient_mode=mode,
            difference_step=delta,
            refractivity_scale=scale,
            create_graph=False,
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
        position = position + (step_size / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        direction = direction + (step_size / 6.0) * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(1e-30)
        positions.append(position)
        directions.append(direction)

    stacked_positions = torch.stack(positions, dim=1)
    stacked_directions = torch.stack(directions, dim=1)
    final_domain_margin = 1.0 - float(torch.max(torch.abs(stacked_positions.detach())))
    domain_margins.append(final_domain_margin)
    stencil_margins.append(final_domain_margin - delta)
    norm_error = torch.max(
        torch.abs(torch.linalg.vector_norm(stacked_directions.detach(), dim=-1) - 1.0)
    )
    trace = RayTraceResult(
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
    return trace, stages


def build_field_program_signature(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    support_threshold: float,
    frustum_half_width_u: float,
    frustum_half_width_v: float,
) -> FieldProgramSignature:
    """Build an ordered cell/support/frustum signature for one field context."""

    values, states, delta, threshold, half_u, half_v = _validated_inputs(
        values_zyx,
        pupil_states,
        difference_step=difference_step,
        support_threshold=support_threshold,
        frustum_half_width_u=frustum_half_width_u,
        frustum_half_width_v=frustum_half_width_v,
    )
    trace, stage_positions = _diagnostic_trace_with_stages(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=float(refractivity_scale),
        step_count=int(step_count),
    )
    groups: list[tuple[str, torch.Tensor]] = list(stage_positions)
    groups.append(
        (
            "curved_path_midpoints",
            (0.5 * (trace.positions[:, :-1] + trace.positions[:, 1:])).reshape(-1, 3),
        )
    )
    start, straight_direction, _, _ = initial_pupil_rays(states, rig)
    midpoint_distance = (
        torch.arange(int(step_count), dtype=torch.float64) + 0.5
    ) * float(trace.step_size)
    straight_midpoints = (
        start[:, None, :]
        + midpoint_distance[None, :, None] * straight_direction[:, None, :]
    )
    groups.append(("straight_path_midpoints", straight_midpoints.reshape(-1, 3)))

    labels: list[dict[str, object]] = []
    cells: list[np.ndarray] = []
    support: list[np.ndarray] = []
    minimum_query_margin = math.inf
    for group_index, (label, points) in enumerate(groups):
        flat = points.reshape(-1, 3).to(dtype=torch.float64, device="cpu")
        bundle = _central_query_bundle(flat, difference_step=delta)
        minimum_query_margin = min(
            minimum_query_margin,
            1.0 - float(torch.max(torch.abs(bundle))),
        )
        if minimum_query_margin < -1e-12:
            raise ValueError("field-program signature query left the grid domain")
        flat_queries = bundle.reshape(-1, 3)
        lower = _lower_interpolation_cells(flat_queries, tuple(values.shape))
        sampled = smoothstep_grid_field(values, flat_queries)
        support_bits = torch.abs(sampled) >= threshold
        labels.append(
            {
                "group_index": group_index,
                "label": label,
                "point_count": len(flat),
                "offset_order": ["base", "+x", "+y", "+z", "-x", "-y", "-z"],
            }
        )
        cells.append(lower.detach().cpu().numpy().astype("<i2", copy=False))
        support.append(support_bits.detach().cpu().numpy().astype(np.uint8, copy=False))

    positions = trace.positions.detach()
    progress = torch.arange(
        positions.shape[1],
        dtype=positions.dtype,
        device=positions.device,
    ) * float(trace.step_size)
    straight_endpoints = (
        positions[:, :1, :]
        + progress[None, :, None] * trace.directions[:, :1, :].detach()
    )
    deviation = positions - straight_endpoints
    offset_u = torch.sum(deviation * trace.projection_u[:, None, :].detach(), dim=-1)
    offset_v = torch.sum(deviation * trace.projection_v[:, None, :].detach(), dim=-1)
    margins = torch.stack((half_u - torch.abs(offset_u), half_v - torch.abs(offset_v)))
    margin_signs = (margins >= 0.0).cpu().numpy().astype(np.uint8, copy=False)

    labels_bytes = _canonical_json(labels)
    cell_bytes = np.concatenate(cells, axis=0).tobytes(order="C")
    support_array = np.concatenate(support, axis=0)
    support_bytes = support_array.tobytes(order="C")
    frustum_bytes = margin_signs.tobytes(order="C")
    component_hashes = {
        "schema": FIELD_PROGRAM_SIGNATURE_SCHEMA,
        "labels_sha256": _sha256_bytes(labels_bytes),
        "interpolation_cells_sha256": _sha256_bytes(cell_bytes),
        "support_bits_sha256": _sha256_bytes(support_bytes),
        "frustum_margin_signs_sha256": _sha256_bytes(frustum_bytes),
    }
    return FieldProgramSignature(
        digest_sha256=_sha256_bytes(_canonical_json(component_hashes)),
        labels_sha256=component_hashes["labels_sha256"],
        interpolation_cells_sha256=component_hashes["interpolation_cells_sha256"],
        support_bits_sha256=component_hashes["support_bits_sha256"],
        frustum_margin_signs_sha256=component_hashes["frustum_margin_signs_sha256"],
        group_count=len(groups),
        query_record_count=int(sum(array.shape[0] for array in cells)),
        support_true_count=int(np.sum(support_array)),
        minimum_domain_margin=float(trace.minimum_domain_margin),
        minimum_stencil_margin=min(
            float(trace.minimum_stencil_margin), minimum_query_margin
        ),
        minimum_frustum_margin=float(torch.min(margins)),
        maximum_direction_norm_error=float(trace.maximum_direction_norm_error),
    )


__all__ = [
    "FIELD_PROGRAM_SIGNATURE_SCHEMA",
    "FieldProgramSignature",
    "build_field_program_signature",
]
