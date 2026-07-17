"""Matrix-free B0 voxel-gradient reconstruction interface for PSU-style BOST.

The scalar unknown is a refractive-index or density perturbation on a regular
``[z, y, x]`` grid. The forward map applies a declared finite-difference
gradient, trilinear sampling, camera-plane projection, and the fixed Monte
Carlo denominator ``line_length * system_constant / sample_count``.

The adjoint is implemented from the same discrete primitives. This module is a
small, auditable baseline interface; it is not a claim of equivalence to the
released TensorFlow NIRT code or of reconstruction superiority.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import torch


INTERFACE_SCHEMA = "psu-b0-reconstruction-interface-1.0"


def _as_float_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype)
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError("numeric inputs must contain only finite values")
    return tensor


def _validate_grid(
    grid_shape: tuple[int, int, int],
    grid_minimum_xyz: Any,
    grid_maximum_xyz: Any,
    *,
    dtype: torch.dtype,
) -> tuple[tuple[int, int, int], torch.Tensor, torch.Tensor]:
    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("grid_shape must contain three dimensions of at least two")
    minimum = _as_float_tensor(grid_minimum_xyz, dtype=dtype).reshape(-1)
    maximum = _as_float_tensor(grid_maximum_xyz, dtype=dtype).reshape(-1)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("grid bounds must contain x, y, and z")
    if torch.any(maximum <= minimum):
        raise ValueError("grid maximum must be strictly greater than minimum")
    return shape, minimum, maximum


@dataclass(frozen=True)
class TrilinearStencil:
    """Fixed trilinear interpolation stencil for ``[ray, sample]`` points."""

    indices: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor
    grid_shape: tuple[int, int, int]

    @property
    def ray_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.indices.shape[1])


@dataclass(frozen=True)
class CompactTrilinearCoordinates:
    """Compact lower-corner/fraction representation of a fixed stencil."""

    base_indices: torch.Tensor
    fractions_xyz: torch.Tensor
    valid: torch.Tensor
    grid_shape: tuple[int, int, int]

    @property
    def ray_count(self) -> int:
        return int(self.base_indices.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.base_indices.shape[1])


def build_compact_trilinear_coordinates(
    sample_points_xyz: Any,
    *,
    grid_shape: tuple[int, int, int],
    grid_minimum_xyz: Any,
    grid_maximum_xyz: Any,
    sample_valid: Any | None = None,
    dtype: torch.dtype = torch.float64,
) -> CompactTrilinearCoordinates:
    """Build compact coordinates that can regenerate the exact 8-corner stencil."""

    shape, minimum, maximum = _validate_grid(
        grid_shape,
        grid_minimum_xyz,
        grid_maximum_xyz,
        dtype=dtype,
    )
    points = _as_float_tensor(sample_points_xyz, dtype=dtype)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("sample_points_xyz must have shape [ray,sample,3]")
    if points.shape[0] < 1 or points.shape[1] < 1:
        raise ValueError("at least one ray and one sample are required")
    inside = torch.all(
        (points >= minimum.reshape(1, 1, 3))
        & (points <= maximum.reshape(1, 1, 3)),
        dim=-1,
    )
    if sample_valid is not None:
        declared = torch.as_tensor(sample_valid, dtype=torch.bool)
        if declared.shape != inside.shape:
            raise ValueError("sample_valid must have shape [ray,sample]")
        inside = inside & declared

    nz, ny, nx = shape
    counts_xyz = torch.as_tensor([nx, ny, nz], dtype=dtype)
    scaled = (points - minimum) / (maximum - minimum)
    scaled = scaled * (counts_xyz - 1.0)
    scaled = torch.minimum(
        torch.maximum(scaled, torch.zeros_like(scaled)),
        counts_xyz.reshape(1, 1, 3) - 1.0,
    )
    lower = torch.floor(scaled).to(torch.int64)
    maximum_lower = torch.as_tensor([nx - 2, ny - 2, nz - 2], dtype=torch.int64)
    lower = torch.minimum(lower, maximum_lower.reshape(1, 1, 3))
    fraction = scaled - lower.to(dtype)
    base_indices = (
        lower[..., 2] * (ny * nx)
        + lower[..., 1] * nx
        + lower[..., 0]
    )
    base_indices = torch.where(inside, base_indices, torch.zeros_like(base_indices))
    return CompactTrilinearCoordinates(
        base_indices=base_indices.contiguous(),
        fractions_xyz=fraction.contiguous(),
        valid=inside.contiguous(),
        grid_shape=shape,
    )


def expand_compact_trilinear_coordinates(
    compact: CompactTrilinearCoordinates,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> TrilinearStencil:
    """Regenerate an 8-corner stencil from compact cached coordinates."""

    shape = tuple(int(value) for value in compact.grid_shape)
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("compact grid_shape is invalid")
    fractions = torch.as_tensor(compact.fractions_xyz)
    target_dtype = dtype or fractions.dtype
    target_device = torch.device(device) if device is not None else fractions.device
    fractions = fractions.to(device=target_device, dtype=target_dtype)
    base = torch.as_tensor(compact.base_indices).to(
        device=target_device,
        dtype=torch.int64,
    )
    valid = torch.as_tensor(compact.valid).to(
        device=target_device,
        dtype=torch.bool,
    )
    if base.ndim != 2 or fractions.shape != (*base.shape, 3):
        raise ValueError("compact coordinates have incompatible shapes")
    if valid.shape != base.shape:
        raise ValueError("compact valid mask must match base indices")
    _, ny, nx = shape
    linear_offsets = torch.as_tensor(
        [
            0,
            1,
            nx,
            nx + 1,
            ny * nx,
            ny * nx + 1,
            ny * nx + nx,
            ny * nx + nx + 1,
        ],
        device=target_device,
        dtype=torch.int64,
    )
    indices = base[:, :, None] + linear_offsets.reshape(1, 1, 8)
    corner_offsets = torch.as_tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        device=target_device,
        dtype=target_dtype,
    )
    corner_weights = torch.where(
        corner_offsets.reshape(1, 1, 8, 3) > 0.5,
        fractions[:, :, None, :],
        1.0 - fractions[:, :, None, :],
    )
    weights = torch.prod(corner_weights, dim=-1)
    weights = torch.where(valid[:, :, None], weights, torch.zeros_like(weights))
    indices = torch.where(valid[:, :, None], indices, torch.zeros_like(indices))
    return TrilinearStencil(
        indices=indices.contiguous(),
        weights=weights.contiguous(),
        valid=valid.contiguous(),
        grid_shape=shape,
    )


def build_trilinear_stencil(
    sample_points_xyz: Any,
    *,
    grid_shape: tuple[int, int, int],
    grid_minimum_xyz: Any,
    grid_maximum_xyz: Any,
    sample_valid: Any | None = None,
    dtype: torch.dtype = torch.float64,
) -> TrilinearStencil:
    """Build a deterministic fixed-domain trilinear interpolation stencil.

    Points on the closed B0 boundary are valid. Points outside the box retain
    the original sample slot but receive zero interpolation weight, preserving
    the fixed denominator used by the finite-aperture Monte Carlo estimator.
    """

    compact = build_compact_trilinear_coordinates(
        sample_points_xyz,
        grid_shape=grid_shape,
        grid_minimum_xyz=grid_minimum_xyz,
        grid_maximum_xyz=grid_maximum_xyz,
        sample_valid=sample_valid,
        dtype=dtype,
    )
    return expand_compact_trilinear_coordinates(
        compact,
        dtype=dtype,
    )


def _finite_difference_axis(
    values: torch.Tensor,
    *,
    axis: int,
    spacing: float,
) -> torch.Tensor:
    moved = torch.movedim(values, axis, -1)
    output = torch.zeros_like(moved)
    output[..., 0] = (moved[..., 1] - moved[..., 0]) / spacing
    output[..., -1] = (moved[..., -1] - moved[..., -2]) / spacing
    if moved.shape[-1] > 2:
        output[..., 1:-1] = (
            moved[..., 2:] - moved[..., :-2]
        ) / (2.0 * spacing)
    return torch.movedim(output, -1, axis)


def _finite_difference_axis_adjoint(
    values: torch.Tensor,
    *,
    axis: int,
    spacing: float,
) -> torch.Tensor:
    moved = torch.movedim(values, axis, -1)
    output = torch.zeros_like(moved)
    output[..., 0] -= moved[..., 0] / spacing
    output[..., 1] += moved[..., 0] / spacing
    output[..., -2] -= moved[..., -1] / spacing
    output[..., -1] += moved[..., -1] / spacing
    if moved.shape[-1] > 2:
        output[..., :-2] -= moved[..., 1:-1] / (2.0 * spacing)
        output[..., 2:] += moved[..., 1:-1] / (2.0 * spacing)
    return torch.movedim(output, -1, axis)


def _absolute_finite_difference_axis(
    values: torch.Tensor,
    *,
    axis: int,
    spacing: float,
) -> torch.Tensor:
    """Apply the elementwise absolute value of one difference matrix."""

    moved = torch.movedim(values, axis, -1)
    output = torch.zeros_like(moved)
    output[..., 0] = (moved[..., 0] + moved[..., 1]) / spacing
    output[..., -1] = (moved[..., -2] + moved[..., -1]) / spacing
    if moved.shape[-1] > 2:
        output[..., 1:-1] = (
            moved[..., :-2] + moved[..., 2:]
        ) / (2.0 * spacing)
    return torch.movedim(output, -1, axis)


def _absolute_finite_difference_axis_adjoint(
    values: torch.Tensor,
    *,
    axis: int,
    spacing: float,
) -> torch.Tensor:
    """Apply the exact transpose of an absolute difference matrix."""

    moved = torch.movedim(values, axis, -1)
    output = torch.zeros_like(moved)
    output[..., 0] += moved[..., 0] / spacing
    output[..., 1] += moved[..., 0] / spacing
    output[..., -2] += moved[..., -1] / spacing
    output[..., -1] += moved[..., -1] / spacing
    if moved.shape[-1] > 2:
        output[..., :-2] += moved[..., 1:-1] / (2.0 * spacing)
        output[..., 2:] += moved[..., 1:-1] / (2.0 * spacing)
    return torch.movedim(output, -1, axis)


def finite_difference_gradient(
    volume: torch.Tensor,
    *,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Return ``[dx, dy, dz]`` using a declared voxel-centered stencil."""

    if volume.ndim != 4:
        raise ValueError("volume must have shape [batch,z,y,x]")
    dx = _finite_difference_axis(volume, axis=-1, spacing=spacing_xyz[0])
    dy = _finite_difference_axis(volume, axis=-2, spacing=spacing_xyz[1])
    dz = _finite_difference_axis(volume, axis=-3, spacing=spacing_xyz[2])
    return torch.stack((dx, dy, dz), dim=1)


def absolute_finite_difference_gradient(
    volume: torch.Tensor,
    *,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Apply ``|G_c|``, the entrywise absolute difference matrix.

    This is not ``abs(finite_difference_gradient(volume))``. Each signed
    centered or one-sided stencil coefficient is replaced by its absolute
    value before the matrix is applied.
    """

    if volume.ndim != 4:
        raise ValueError("volume must have shape [batch,z,y,x]")
    dx = _absolute_finite_difference_axis(
        volume,
        axis=-1,
        spacing=spacing_xyz[0],
    )
    dy = _absolute_finite_difference_axis(
        volume,
        axis=-2,
        spacing=spacing_xyz[1],
    )
    dz = _absolute_finite_difference_axis(
        volume,
        axis=-3,
        spacing=spacing_xyz[2],
    )
    return torch.stack((dx, dy, dz), dim=1)


def finite_difference_gradient_adjoint(
    gradient: torch.Tensor,
    *,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Apply the exact transpose of :func:`finite_difference_gradient`."""

    if gradient.ndim != 5 or gradient.shape[1] != 3:
        raise ValueError("gradient must have shape [batch,3,z,y,x]")
    return (
        _finite_difference_axis_adjoint(
            gradient[:, 0], axis=-1, spacing=spacing_xyz[0]
        )
        + _finite_difference_axis_adjoint(
            gradient[:, 1], axis=-2, spacing=spacing_xyz[1]
        )
        + _finite_difference_axis_adjoint(
            gradient[:, 2], axis=-3, spacing=spacing_xyz[2]
        )
    )


def absolute_finite_difference_gradient_adjoint(
    gradient: torch.Tensor,
    *,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Apply the exact transpose of the absolute finite-difference gradient."""

    if gradient.ndim != 5 or gradient.shape[1] != 3:
        raise ValueError("gradient must have shape [batch,3,z,y,x]")
    return (
        _absolute_finite_difference_axis_adjoint(
            gradient[:, 0], axis=-1, spacing=spacing_xyz[0]
        )
        + _absolute_finite_difference_axis_adjoint(
            gradient[:, 1], axis=-2, spacing=spacing_xyz[1]
        )
        + _absolute_finite_difference_axis_adjoint(
            gradient[:, 2], axis=-3, spacing=spacing_xyz[2]
        )
    )


def project_dirichlet_gauge(
    volume: torch.Tensor,
    *,
    support: torch.Tensor | None = None,
    boundary_width: int = 1,
) -> torch.Tensor:
    """Fix the additive BOS nullspace by zeroing the outer voxel boundary."""

    if volume.ndim not in {4, 5}:
        raise ValueError("volume must have shape [batch,z,y,x] or [batch,1,z,y,x]")
    if boundary_width < 1:
        raise ValueError("boundary_width must be positive")
    values = volume[:, 0] if volume.ndim == 5 else volume
    if any(size <= 2 * boundary_width for size in values.shape[-3:]):
        raise ValueError("boundary_width leaves no interior voxels")
    mask = torch.ones_like(values)
    width = int(boundary_width)
    mask[..., :width, :, :] = 0
    mask[..., -width:, :, :] = 0
    mask[..., :, :width, :] = 0
    mask[..., :, -width:, :] = 0
    mask[..., :, :, :width] = 0
    mask[..., :, :, -width:] = 0
    if support is not None:
        support_values = torch.as_tensor(
            support,
            dtype=values.dtype,
            device=values.device,
        )
        if support_values.shape == values.shape[-3:]:
            support_values = support_values.reshape(1, *support_values.shape)
        if support_values.shape not in {values.shape, (1, *values.shape[-3:])}:
            raise ValueError("support must match or broadcast over the volume")
        mask = mask * support_values
    projected = values * mask
    return projected[:, None] if volume.ndim == 5 else projected


class PSUB0VoxelGradientOperator(torch.nn.Module):
    """Call-counted matrix-free scalar-field-to-UV BOST operator."""

    def __init__(
        self,
        *,
        stencil: TrilinearStencil,
        projection_u_xyz: Any,
        projection_v_xyz: Any,
        line_length: Any,
        system_constant: Any,
        grid_minimum_xyz: Any,
        grid_maximum_xyz: Any,
        support: Any | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        shape, minimum, maximum = _validate_grid(
            stencil.grid_shape,
            grid_minimum_xyz,
            grid_maximum_xyz,
            dtype=dtype,
        )
        sample_indices = torch.as_tensor(stencil.indices)
        sample_weights = _as_float_tensor(stencil.weights, dtype=dtype)
        sample_valid = torch.as_tensor(stencil.valid)
        if sample_indices.dtype != torch.int64:
            raise ValueError("sample_indices must have dtype int64")
        if sample_valid.dtype != torch.bool:
            raise ValueError("sample_valid must have dtype bool")
        if sample_indices.ndim != 3 or sample_indices.shape[-1] != 8:
            raise ValueError("sample_indices must have shape [ray,sample,8]")
        if sample_weights.shape != sample_indices.shape:
            raise ValueError("sample_weights must have shape [ray,sample,8]")
        expected_valid_shape = sample_indices.shape[:2]
        if sample_valid.shape != expected_valid_shape:
            raise ValueError("sample_valid must have shape [ray,sample]")
        if sample_indices.shape[0] < 1 or sample_indices.shape[1] < 1:
            raise ValueError("at least one ray and one sample are required")
        voxel_count = prod(shape)
        if torch.any((sample_indices < 0) | (sample_indices >= voxel_count)):
            raise ValueError(
                "sample_indices must lie in [0, prod(grid_shape))"
            )
        if torch.any(sample_weights < 0):
            raise ValueError("trilinear interpolation weights must be nonnegative")
        if torch.any(sample_weights.masked_select(~sample_valid[:, :, None])):
            raise ValueError("weights for invalid samples must be exactly zero")

        ray_count = int(sample_indices.shape[0])
        projection_u = _as_float_tensor(projection_u_xyz, dtype=dtype)
        projection_v = _as_float_tensor(projection_v_xyz, dtype=dtype)
        if projection_u.shape != (ray_count, 3) or projection_v.shape != (
            ray_count,
            3,
        ):
            raise ValueError("camera projection vectors must have shape [ray,3]")
        length = _as_float_tensor(line_length, dtype=dtype).reshape(-1)
        constant = _as_float_tensor(system_constant, dtype=dtype).reshape(-1)
        if length.shape != (ray_count,) or constant.shape != (ray_count,):
            raise ValueError("line_length and system_constant need one value per ray")
        if torch.any(length < 0):
            raise ValueError("line_length must be nonnegative")
        if support is None:
            support_tensor = torch.ones(shape, dtype=dtype)
        else:
            support_tensor = _as_float_tensor(support, dtype=dtype)
            if support_tensor.shape != shape:
                raise ValueError("support must match grid_shape")
            if torch.any((support_tensor < 0) | (support_tensor > 1)):
                raise ValueError("support must lie in [0,1]")
        self.grid_shape = shape
        self.grid_minimum_xyz = tuple(float(value) for value in minimum)
        self.grid_maximum_xyz = tuple(float(value) for value in maximum)
        nz, ny, nx = shape
        self.spacing_xyz = (
            float((maximum[0] - minimum[0]) / (nx - 1)),
            float((maximum[1] - minimum[1]) / (ny - 1)),
            float((maximum[2] - minimum[2]) / (nz - 1)),
        )
        self.sample_count = int(sample_indices.shape[1])
        self.ray_count = ray_count
        self.register_buffer("sample_indices", sample_indices)
        self.register_buffer("sample_weights", sample_weights)
        self.register_buffer("sample_valid", sample_valid)
        self.register_buffer("projection_u", projection_u)
        self.register_buffer("projection_v", projection_v)
        self.register_buffer(
            "ray_scale",
            length * constant / float(self.sample_count),
        )
        self.register_buffer("support", support_tensor)
        self.forward_calls = 0
        self.adjoint_calls = 0

    def reset_call_counts(self) -> None:
        self.forward_calls = 0
        self.adjoint_calls = 0

    def call_report(self) -> dict[str, int]:
        return {
            "forward_calls": int(self.forward_calls),
            "adjoint_calls": int(self.adjoint_calls),
        }

    def _canonical_volume(self, volume: torch.Tensor) -> torch.Tensor:
        values = volume[:, 0] if volume.ndim == 5 else volume
        if values.ndim != 4 or tuple(values.shape[1:]) != self.grid_shape:
            raise ValueError(
                "volume must have shape [batch,z,y,x] or [batch,1,z,y,x]"
            )
        return values.to(dtype=self.sample_weights.dtype)

    def trilinear_interpolation(
        self,
        voxel_components: torch.Tensor,
    ) -> torch.Tensor:
        """Apply nonnegative trilinear ``P`` independently per component.

        The input shape is ``[batch,component,z,y,x]`` and the returned shape
        is ``[batch,component,ray,sample]``. Calling this factor directly does
        not increment the logical full-operator call counters.
        """

        values = voxel_components.to(dtype=self.sample_weights.dtype)
        if values.ndim != 5 or tuple(values.shape[-3:]) != self.grid_shape:
            raise ValueError(
                "voxel_components must have shape [batch,component,z,y,x]"
            )
        flat = values.flatten(2)
        indices = self.sample_indices.reshape(-1)
        gathered = flat[:, :, indices].reshape(
            len(values),
            values.shape[1],
            self.ray_count,
            self.sample_count,
            8,
        )
        return torch.sum(
            gathered * self.sample_weights[None, None, :, :, :],
            dim=-1,
        )

    def trilinear_interpolation_adjoint(
        self,
        sampled_components: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exact ``P^T`` to ``[batch,component,ray,sample]`` values."""

        values = sampled_components.to(dtype=self.sample_weights.dtype)
        if values.ndim != 4 or values.shape[2:] != (
            self.ray_count,
            self.sample_count,
        ):
            raise ValueError(
                "sampled_components must have shape "
                "[batch,component,ray,sample]"
            )
        contribution = (
            values[:, :, :, :, None]
            * self.sample_weights[None, None, :, :, :]
        )
        flat_contribution = contribution.reshape(
            len(values),
            values.shape[1],
            -1,
        )
        flat_indices = self.sample_indices.reshape(-1)
        expanded_indices = flat_indices.reshape(1, 1, -1).expand(
            len(values),
            values.shape[1],
            -1,
        )
        voxel_flat = torch.zeros(
            (len(values), values.shape[1], prod(self.grid_shape)),
            dtype=values.dtype,
            device=values.device,
        )
        voxel_flat.scatter_add_(2, expanded_indices, flat_contribution)
        return voxel_flat.reshape(
            len(values),
            values.shape[1],
            *self.grid_shape,
        )

    def _forward(self, volume: torch.Tensor) -> torch.Tensor:
        values = self._canonical_volume(volume) * self.support
        gradient = finite_difference_gradient(
            values,
            spacing_xyz=self.spacing_xyz,
        )
        sampled = self.trilinear_interpolation(gradient)
        u = torch.einsum("bcrs,rc->brs", sampled, self.projection_u)
        v = torch.einsum("bcrs,rc->brs", sampled, self.projection_v)
        projected = torch.stack((u.sum(dim=-1), v.sum(dim=-1)), dim=-1)
        return projected * self.ray_scale[None, :, None]

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self._forward(volume)

    def _adjoint(self, residual_uv: torch.Tensor) -> torch.Tensor:
        residual = residual_uv.to(dtype=self.sample_weights.dtype)
        if residual.ndim != 3 or residual.shape[1:] != (self.ray_count, 2):
            raise ValueError("residual_uv must have shape [batch,ray,2]")
        component = (
            residual[:, :, 0:1] * self.projection_u[None, :, :]
            + residual[:, :, 1:2] * self.projection_v[None, :, :]
        )
        component = component * self.ray_scale[None, :, None]
        sampled_component = component.permute(0, 2, 1)[:, :, :, None].expand(
            -1,
            -1,
            -1,
            self.sample_count,
        )
        gradient = self.trilinear_interpolation_adjoint(sampled_component)
        volume = finite_difference_gradient_adjoint(
            gradient,
            spacing_xyz=self.spacing_xyz,
        )
        return volume * self.support

    def adjoint(self, residual_uv: torch.Tensor) -> torch.Tensor:
        self.adjoint_calls += 1
        return self._adjoint(residual_uv)[:, None]

    def _adjoint_grouped(
        self,
        residual_uv: torch.Tensor,
        *,
        ray_group_index: torch.Tensor,
        group_count: int,
    ) -> torch.Tensor:
        """Retain each ray group's contribution while traversing rays once."""

        residual = residual_uv.to(dtype=self.sample_weights.dtype)
        if residual.ndim != 3 or residual.shape[1:] != (self.ray_count, 2):
            raise ValueError("residual_uv must have shape [batch,ray,2]")
        count = int(group_count)
        if count < 1:
            raise ValueError("group_count must be positive")
        groups = torch.as_tensor(
            ray_group_index,
            dtype=torch.int64,
            device=residual.device,
        ).reshape(-1)
        if groups.shape != (self.ray_count,):
            raise ValueError("ray_group_index must have one entry per ray")
        if torch.any(groups < 0) or torch.any(groups >= count):
            raise ValueError("ray_group_index must lie in [0, group_count)")

        component = (
            residual[:, :, 0:1] * self.projection_u[None, :, :]
            + residual[:, :, 1:2] * self.projection_v[None, :, :]
        )
        component = component * self.ray_scale[None, :, None]
        contribution = (
            component.permute(0, 2, 1)[:, :, :, None, None]
            * self.sample_weights[None, None, :, :, :]
        )
        flat_contribution = contribution.reshape(len(residual), 3, -1)
        voxel_count = prod(self.grid_shape)
        grouped_indices = (
            groups[:, None, None] * voxel_count + self.sample_indices
        ).reshape(-1)
        expanded_indices = grouped_indices.reshape(1, 1, -1).expand(
            len(residual), 3, -1
        )
        grouped_gradient_flat = torch.zeros(
            (len(residual), 3, count * voxel_count),
            dtype=residual.dtype,
            device=residual.device,
        )
        grouped_gradient_flat.scatter_add_(
            2,
            expanded_indices,
            flat_contribution,
        )
        grouped_gradient = grouped_gradient_flat.reshape(
            len(residual),
            3,
            count,
            *self.grid_shape,
        ).permute(0, 2, 1, 3, 4, 5)
        grouped_volume = finite_difference_gradient_adjoint(
            grouped_gradient.reshape(
                len(residual) * count,
                3,
                *self.grid_shape,
            ),
            spacing_xyz=self.spacing_xyz,
        ).reshape(len(residual), count, *self.grid_shape)
        return grouped_volume * self.support[None, None]

    def adjoint_grouped(
        self,
        residual_uv: torch.Tensor,
        *,
        ray_group_index: torch.Tensor,
        group_count: int,
    ) -> torch.Tensor:
        """Return ``[batch,group,1,z,y,x]`` grouped adjoint contributions.

        The ray interpolation/scatter contribution is evaluated once per ray
        and accumulated into disjoint group slots. Retaining every group still
        costs more memory and finite-difference-adjoint work than the pooled
        :meth:`adjoint`, so one recorded invocation must not be described as
        equal-FLOP to one pooled solver adjoint.
        """

        self.adjoint_calls += 1
        return self._adjoint_grouped(
            residual_uv,
            ray_group_index=ray_group_index,
            group_count=group_count,
        )[:, :, None]

    def adjoint_by_view(
        self,
        residual_uv: torch.Tensor,
        *,
        rays_per_view: int,
    ) -> torch.Tensor:
        """Group contiguous ray blocks into camera-view adjoint fields."""

        block = int(rays_per_view)
        if block < 1 or self.ray_count % block != 0:
            raise ValueError(
                "rays_per_view must be positive and divide ray_count"
            )
        view_count = self.ray_count // block
        ray_group_index = torch.arange(
            self.ray_count,
            dtype=torch.int64,
            device=self.sample_indices.device,
        ) // block
        return self.adjoint_grouped(
            residual_uv,
            ray_group_index=ray_group_index,
            group_count=view_count,
        )

    @torch.no_grad()
    def adjoint_relative_error(self, *, seed: int = 0) -> float:
        generator = torch.Generator().manual_seed(int(seed))
        volume = torch.randn(
            (1, 1, *self.grid_shape),
            generator=generator,
            dtype=self.support.dtype,
        ).to(self.support.device)
        residual = torch.randn(
            (1, self.ray_count, 2),
            generator=generator,
            dtype=self.support.dtype,
        ).to(self.support.device)
        lhs = torch.sum(self._forward(volume) * residual)
        rhs = torch.sum(volume * self._adjoint(residual)[:, None])
        denominator = torch.maximum(torch.abs(lhs), torch.abs(rhs)).clamp_min(1e-18)
        return float(torch.abs(lhs - rhs) / denominator)

    @torch.no_grad()
    def estimate_lipschitz(
        self,
        *,
        power_iterations: int = 20,
        boundary_width: int = 1,
        seed: int = 0,
    ) -> float:
        """Estimate ``lambda_max(A^T A)`` with calls recorded explicitly."""

        if power_iterations < 2:
            raise ValueError("power_iterations must be at least two")
        generator = torch.Generator().manual_seed(int(seed))
        current = torch.randn(
            (1, 1, *self.grid_shape),
            generator=generator,
            dtype=self.support.dtype,
        ).to(self.support.device)
        current = project_dirichlet_gauge(
            current,
            support=self.support,
            boundary_width=boundary_width,
        )
        norm = torch.linalg.vector_norm(current).clamp_min(1e-18)
        current = current / norm
        for _ in range(int(power_iterations)):
            normal = self.adjoint(self.forward(current))
            normal = project_dirichlet_gauge(
                normal,
                support=self.support,
                boundary_width=boundary_width,
            )
            norm = torch.linalg.vector_norm(normal).clamp_min(1e-18)
            current = normal / norm
        projected = self.forward(current)
        return float(torch.sum(projected * projected).clamp_min(1e-18))
