"""Chunked B0 forward/adjoint and CGLS baseline for large PSU ray sets.

The operator uses the same finite-difference, trilinear interpolation, camera
projection, and fixed-denominator primitives as
``psu_b0_reconstruction_interface``. A logical forward or adjoint call is one
complete deterministic traversal of every chunk supplied by the ray store.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import platform
import resource
import time
from typing import Any, Protocol

import torch

from .psu_b0_reconstruction_interface import (
    CompactTrilinearCoordinates,
    build_trilinear_stencil,
    expand_compact_trilinear_coordinates,
    finite_difference_gradient,
    finite_difference_gradient_adjoint,
)


STREAMING_INTERFACE_SCHEMA = "psu-b0-streaming-operator-1.0"


@dataclass(frozen=True)
class StreamingRayChunk:
    """One contiguous output slice of finite-aperture BOST rays."""

    start_index: int
    stop_index: int
    sample_points_xyz: Any
    projection_u_xyz: Any
    projection_v_xyz: Any
    line_length: Any
    system_constant: Any
    observation_uv: Any
    view_id: int
    b0_hit_count: int
    compact_base_indices: Any | None = None
    compact_fractions_xyz: Any | None = None
    compact_valid: Any | None = None
    ray_scale: Any | None = None

    @property
    def ray_count(self) -> int:
        return int(self.stop_index - self.start_index)


class StreamingRayStore(Protocol):
    """Minimal deterministic store contract consumed by the streaming operator."""

    ray_count: int
    sample_count: int

    def iter_chunks(self) -> Any:
        """Yield :class:`StreamingRayChunk` objects in contiguous output order."""

    def load_observations(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return observations with shape ``[1, ray, 2]``."""


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _finite_tensor(
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError("streaming chunk contains non-finite numeric values")
    return tensor


def zero_outer_boundary_support(
    grid_shape: tuple[int, int, int],
    *,
    boundary_width: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the linear gauge projector used by the low-resolution baseline."""

    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("grid_shape must contain three dimensions of at least two")
    width = int(boundary_width)
    if width < 1 or any(value <= 2 * width for value in shape):
        raise ValueError("boundary_width must leave at least one interior voxel")
    support = torch.ones(shape, dtype=dtype)
    support[:width, :, :] = 0
    support[-width:, :, :] = 0
    support[:, :width, :] = 0
    support[:, -width:, :] = 0
    support[:, :, :width] = 0
    support[:, :, -width:] = 0
    return support


@dataclass(frozen=True)
class CGLSResult:
    """Fixed-budget CGLS output and an explicit logical-call history."""

    volume: torch.Tensor
    residual: torch.Tensor
    history: list[dict[str, float | int | bool]]
    breakdown: bool


class PSUB0StreamingOperator(torch.nn.Module):
    """Matrix-free B0 operator that rebuilds bounded chunk stencils on demand."""

    def __init__(
        self,
        *,
        ray_store: StreamingRayStore,
        grid_shape: tuple[int, int, int],
        grid_minimum_xyz: tuple[float, float, float],
        grid_maximum_xyz: tuple[float, float, float],
        support: Any | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        shape = tuple(int(value) for value in grid_shape)
        if len(shape) != 3 or any(value < 2 for value in shape):
            raise ValueError("grid_shape must contain three dimensions of at least two")
        minimum = torch.as_tensor(grid_minimum_xyz, dtype=dtype).reshape(-1)
        maximum = torch.as_tensor(grid_maximum_xyz, dtype=dtype).reshape(-1)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("grid bounds must contain x, y, and z")
        if torch.any(maximum <= minimum):
            raise ValueError("grid maximum must be strictly greater than minimum")
        if int(ray_store.ray_count) < 1 or int(ray_store.sample_count) < 1:
            raise ValueError("ray_store must contain rays and aperture samples")
        cached_shape = getattr(ray_store, "grid_shape", None)
        if cached_shape is not None and tuple(cached_shape) != shape:
            raise ValueError("ray_store compact grid does not match grid_shape")
        cached_minimum = getattr(ray_store, "grid_minimum_xyz", None)
        cached_maximum = getattr(ray_store, "grid_maximum_xyz", None)
        if cached_minimum is not None and tuple(cached_minimum) != tuple(
            float(value) for value in minimum
        ):
            raise ValueError("ray_store compact minimum does not match grid bounds")
        if cached_maximum is not None and tuple(cached_maximum) != tuple(
            float(value) for value in maximum
        ):
            raise ValueError("ray_store compact maximum does not match grid bounds")
        if support is None:
            support_tensor = torch.ones(shape, dtype=dtype)
        else:
            support_tensor = torch.as_tensor(support, dtype=dtype)
            if support_tensor.shape != shape:
                raise ValueError("support must match grid_shape")
            if torch.any((support_tensor < 0) | (support_tensor > 1)):
                raise ValueError("support must lie in [0,1]")

        self.ray_store = ray_store
        self.grid_shape = shape
        self.grid_minimum_xyz = tuple(float(value) for value in minimum)
        self.grid_maximum_xyz = tuple(float(value) for value in maximum)
        self.ray_count = int(ray_store.ray_count)
        self.sample_count = int(ray_store.sample_count)
        nz, ny, nx = shape
        self.spacing_xyz = (
            float((maximum[0] - minimum[0]) / (nx - 1)),
            float((maximum[1] - minimum[1]) / (ny - 1)),
            float((maximum[2] - minimum[2]) / (nz - 1)),
        )
        self.register_buffer("support", support_tensor.contiguous())
        self.forward_calls = 0
        self.adjoint_calls = 0
        self.call_records: list[dict[str, float | int | str]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self.support.dtype

    @property
    def device(self) -> torch.device:
        return self.support.device

    def reset_call_counts(self) -> None:
        self.forward_calls = 0
        self.adjoint_calls = 0
        self.call_records = []

    def call_report(self) -> dict[str, Any]:
        return {
            "forward_calls": int(self.forward_calls),
            "adjoint_calls": int(self.adjoint_calls),
            "records": list(self.call_records),
        }

    def load_observations(self) -> torch.Tensor:
        values = self.ray_store.load_observations(
            dtype=self.dtype,
            device=self.device,
        )
        if values.shape != (1, self.ray_count, 2):
            raise ValueError("ray store observations must have shape [1,ray,2]")
        return values

    def _canonical_volume(self, volume: torch.Tensor) -> torch.Tensor:
        values = volume[:, 0] if volume.ndim == 5 else volume
        if values.ndim != 4 or tuple(values.shape[1:]) != self.grid_shape:
            raise ValueError(
                "volume must have shape [batch,z,y,x] or [batch,1,z,y,x]"
            )
        return values.to(device=self.device, dtype=self.dtype)

    def _chunk_tensors(
        self,
        chunk: StreamingRayChunk,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        compact_values = (
            chunk.compact_base_indices,
            chunk.compact_fractions_xyz,
            chunk.compact_valid,
        )
        if any(value is not None for value in compact_values):
            if not all(value is not None for value in compact_values):
                raise ValueError("cached compact coordinates must be supplied together")
            stencil = expand_compact_trilinear_coordinates(
                CompactTrilinearCoordinates(
                    base_indices=chunk.compact_base_indices,
                    fractions_xyz=chunk.compact_fractions_xyz,
                    valid=chunk.compact_valid,
                    grid_shape=self.grid_shape,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            if chunk.sample_points_xyz is None:
                raise ValueError("chunk needs sample points or compact coordinates")
            stencil = build_trilinear_stencil(
                chunk.sample_points_xyz,
                grid_shape=self.grid_shape,
                grid_minimum_xyz=self.grid_minimum_xyz,
                grid_maximum_xyz=self.grid_maximum_xyz,
                dtype=self.dtype,
            )
        if stencil.ray_count != chunk.ray_count:
            raise ValueError("chunk ray count and sample points disagree")
        if stencil.sample_count != self.sample_count:
            raise ValueError("chunk sample count and ray store disagree")
        indices = stencil.indices.to(device=self.device, dtype=torch.int64)
        weights = stencil.weights.to(device=self.device, dtype=self.dtype)
        projection_u = _finite_tensor(
            chunk.projection_u_xyz,
            dtype=self.dtype,
            device=self.device,
        )
        projection_v = _finite_tensor(
            chunk.projection_v_xyz,
            dtype=self.dtype,
            device=self.device,
        )
        if projection_u.shape != (chunk.ray_count, 3) or projection_v.shape != (
            chunk.ray_count,
            3,
        ):
            raise ValueError("chunk projection vectors must have shape [ray,3]")
        if chunk.ray_scale is None:
            length = _finite_tensor(
                chunk.line_length,
                dtype=self.dtype,
                device=self.device,
            ).reshape(-1)
            constant = _finite_tensor(
                chunk.system_constant,
                dtype=self.dtype,
                device=self.device,
            ).reshape(-1)
            if length.shape != (chunk.ray_count,) or constant.shape != (
                chunk.ray_count,
            ):
                raise ValueError("chunk scale arrays need one value per ray")
            if torch.any(length < 0):
                raise ValueError("chunk line lengths must be nonnegative")
            ray_scale = length * constant / float(self.sample_count)
        else:
            ray_scale = _finite_tensor(
                chunk.ray_scale,
                dtype=self.dtype,
                device=self.device,
            ).reshape(-1)
            if ray_scale.shape != (chunk.ray_count,):
                raise ValueError("chunk ray_scale needs one value per ray")
        projection = torch.stack((projection_u, projection_v), dim=1)
        return indices, weights, projection, ray_scale

    def _forward_chunk(
        self,
        gradient: torch.Tensor,
        chunk: StreamingRayChunk,
    ) -> torch.Tensor:
        indices, weights, projection, ray_scale = self._chunk_tensors(chunk)
        flat = gradient.flatten(2)
        gathered = flat[:, :, indices.reshape(-1)].reshape(
            len(gradient),
            3,
            chunk.ray_count,
            self.sample_count,
            8,
        )
        sampled = torch.sum(
            gathered * weights[None, None, :, :, :],
            dim=-1,
        )
        projected = torch.einsum("bcrs,rkc->brks", sampled, projection)
        return projected.sum(dim=-1) * ray_scale[None, :, None]

    def _scatter_chunk_adjoint(
        self,
        residual_uv: torch.Tensor,
        chunk: StreamingRayChunk,
        gradient_flat: torch.Tensor,
    ) -> None:
        indices, weights, projection, ray_scale = self._chunk_tensors(chunk)
        residual = residual_uv.to(device=self.device, dtype=self.dtype)
        if residual.shape != (len(gradient_flat), chunk.ray_count, 2):
            raise ValueError("chunk residual must have shape [batch,ray,2]")
        component = torch.einsum("brk,rkc->brc", residual, projection)
        component = component * ray_scale[None, :, None]
        contribution = (
            component.permute(0, 2, 1)[:, :, :, None, None]
            * weights[None, None, :, :, :]
        )
        flat_contribution = contribution.reshape(len(residual), 3, -1)
        flat_indices = indices.reshape(-1)
        expanded = flat_indices.reshape(1, 1, -1).expand(
            len(residual),
            3,
            -1,
        )
        gradient_flat.scatter_add_(2, expanded, flat_contribution)

    def _record_call(
        self,
        *,
        operation: str,
        started: float,
        chunk_count: int,
        b0_hit_count: int,
    ) -> None:
        self.call_records.append(
            {
                "operation": operation,
                "wall_seconds": float(time.perf_counter() - started),
                "ray_count": int(self.ray_count),
                "chunk_count": int(chunk_count),
                "b0_hit_count": int(b0_hit_count),
                "max_rss_bytes_after_call": int(_max_rss_bytes()),
            }
        )

    def _forward(self, volume: torch.Tensor, *, record: bool) -> torch.Tensor:
        values = self._canonical_volume(volume) * self.support
        gradient = finite_difference_gradient(
            values,
            spacing_xyz=self.spacing_xyz,
        )
        output = torch.empty(
            (len(values), self.ray_count, 2),
            dtype=self.dtype,
            device=self.device,
        )
        expected_start = 0
        chunk_count = 0
        hit_count = 0
        started = time.perf_counter()
        for chunk in self.ray_store.iter_chunks():
            if chunk.start_index != expected_start:
                raise ValueError("ray chunks must cover contiguous output slices")
            if chunk.stop_index > self.ray_count or chunk.ray_count < 1:
                raise ValueError("ray chunk output bounds are invalid")
            output[:, chunk.start_index : chunk.stop_index] = self._forward_chunk(
                gradient,
                chunk,
            )
            expected_start = chunk.stop_index
            chunk_count += 1
            hit_count += int(chunk.b0_hit_count)
        if expected_start != self.ray_count:
            raise ValueError("ray chunks did not cover the declared ray count")
        if record:
            self._record_call(
                operation="forward",
                started=started,
                chunk_count=chunk_count,
                b0_hit_count=hit_count,
            )
        return output

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self._forward(volume, record=True)

    def _adjoint(self, residual_uv: torch.Tensor, *, record: bool) -> torch.Tensor:
        residual = residual_uv.to(device=self.device, dtype=self.dtype)
        if residual.ndim != 3 or residual.shape[1:] != (self.ray_count, 2):
            raise ValueError("residual_uv must have shape [batch,ray,2]")
        gradient_flat = torch.zeros(
            (len(residual), 3, prod(self.grid_shape)),
            dtype=self.dtype,
            device=self.device,
        )
        expected_start = 0
        chunk_count = 0
        hit_count = 0
        started = time.perf_counter()
        for chunk in self.ray_store.iter_chunks():
            if chunk.start_index != expected_start:
                raise ValueError("ray chunks must cover contiguous output slices")
            if chunk.stop_index > self.ray_count or chunk.ray_count < 1:
                raise ValueError("ray chunk output bounds are invalid")
            self._scatter_chunk_adjoint(
                residual[:, chunk.start_index : chunk.stop_index],
                chunk,
                gradient_flat,
            )
            expected_start = chunk.stop_index
            chunk_count += 1
            hit_count += int(chunk.b0_hit_count)
        if expected_start != self.ray_count:
            raise ValueError("ray chunks did not cover the declared ray count")
        gradient = gradient_flat.reshape(len(residual), 3, *self.grid_shape)
        volume = finite_difference_gradient_adjoint(
            gradient,
            spacing_xyz=self.spacing_xyz,
        )
        output = volume * self.support
        if record:
            self._record_call(
                operation="adjoint",
                started=started,
                chunk_count=chunk_count,
                b0_hit_count=hit_count,
            )
        return output[:, None]

    def adjoint(self, residual_uv: torch.Tensor) -> torch.Tensor:
        self.adjoint_calls += 1
        return self._adjoint(residual_uv, record=True)

    @torch.no_grad()
    def adjoint_relative_error(self, *, seed: int = 0) -> float:
        generator = torch.Generator().manual_seed(int(seed))
        volume = torch.randn(
            (1, 1, *self.grid_shape),
            generator=generator,
            dtype=self.dtype,
        ).to(self.device)
        residual = torch.randn(
            (1, self.ray_count, 2),
            generator=generator,
            dtype=self.dtype,
        ).to(self.device)
        lhs = torch.sum(self._forward(volume, record=False) * residual)
        rhs = torch.sum(volume * self._adjoint(residual, record=False))
        denominator = torch.maximum(torch.abs(lhs), torch.abs(rhs)).clamp_min(1e-18)
        return float(torch.abs(lhs - rhs) / denominator)


@torch.no_grad()
def cgls_solve(
    operator: PSUB0StreamingOperator,
    observation_uv: torch.Tensor,
    *,
    iterations: int,
    initial_adjoint: torch.Tensor | None = None,
    denominator_floor: float = 1e-24,
) -> CGLSResult:
    """Run fixed-budget CGLS on the linear gauge-projected B0 least squares."""

    maximum = int(iterations)
    if maximum < 1:
        raise ValueError("iterations must be positive")
    observation = observation_uv.to(device=operator.device, dtype=operator.dtype)
    if observation.shape != (1, operator.ray_count, 2):
        raise ValueError("observation_uv must have shape [1,ray,2]")
    floor = float(denominator_floor)
    if not floor > 0.0:
        raise ValueError("denominator_floor must be positive")

    current = torch.zeros(
        (1, 1, *operator.grid_shape),
        dtype=operator.dtype,
        device=operator.device,
    )
    residual = observation.clone()
    initial_residual_norm = torch.linalg.vector_norm(residual).clamp_min(floor)
    if initial_adjoint is None:
        normal_residual = operator.adjoint(residual)
    else:
        normal_residual = initial_adjoint.to(
            device=operator.device,
            dtype=operator.dtype,
        )
        if normal_residual.shape != current.shape:
            raise ValueError("initial_adjoint must match the reconstruction volume")
    direction = normal_residual.clone()
    gamma = torch.sum(normal_residual * normal_residual)
    initial_gamma = gamma.clamp_min(floor)
    history: list[dict[str, float | int | bool]] = [
        {
            "iteration": 0,
            "relative_measurement_l2": 1.0,
            "relative_normal_residual_l2": 1.0,
            "volume_l2": 0.0,
            "volume_max_abs": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "breakdown": False,
            "forward_calls": int(operator.forward_calls),
            "adjoint_calls": int(operator.adjoint_calls),
        }
    ]
    breakdown = False

    for iteration in range(1, maximum + 1):
        projected_direction = operator.forward(direction)
        denominator = torch.sum(projected_direction * projected_direction)
        if not torch.isfinite(denominator) or float(denominator) <= floor:
            breakdown = True
            break
        alpha = gamma / denominator
        current = current + alpha * direction
        residual = residual - alpha * projected_direction
        next_normal_residual = operator.adjoint(residual)
        next_gamma = torch.sum(next_normal_residual * next_normal_residual)
        if not torch.isfinite(next_gamma):
            breakdown = True
            break
        beta = next_gamma / gamma.clamp_min(floor)
        direction = next_normal_residual + beta * direction
        normal_residual = next_normal_residual
        gamma = next_gamma
        history.append(
            {
                "iteration": int(iteration),
                "relative_measurement_l2": float(
                    torch.linalg.vector_norm(residual) / initial_residual_norm
                ),
                "relative_normal_residual_l2": float(
                    torch.sqrt(gamma.clamp_min(0.0) / initial_gamma)
                ),
                "volume_l2": float(torch.linalg.vector_norm(current)),
                "volume_max_abs": float(torch.max(torch.abs(current))),
                "alpha": float(alpha),
                "beta": float(beta),
                "breakdown": False,
                "forward_calls": int(operator.forward_calls),
                "adjoint_calls": int(operator.adjoint_calls),
            }
        )

    if breakdown:
        history.append(
            {
                "iteration": int(len(history)),
                "relative_measurement_l2": float(
                    torch.linalg.vector_norm(residual) / initial_residual_norm
                ),
                "relative_normal_residual_l2": float(
                    torch.sqrt(gamma.clamp_min(0.0) / initial_gamma)
                ),
                "volume_l2": float(torch.linalg.vector_norm(current)),
                "volume_max_abs": float(torch.max(torch.abs(current))),
                "alpha": 0.0,
                "beta": 0.0,
                "breakdown": True,
                "forward_calls": int(operator.forward_calls),
                "adjoint_calls": int(operator.adjoint_calls),
            }
        )
    return CGLSResult(
        volume=current,
        residual=residual,
        history=history,
        breakdown=breakdown,
    )
