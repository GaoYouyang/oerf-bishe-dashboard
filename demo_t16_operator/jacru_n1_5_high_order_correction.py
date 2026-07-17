"""High-order teacher maps and a matched-budget warm-start CGLS solver.

The fourth-order map is used either as a diagnostic direct operator or as a
local approximation-error teacher.  The stable second-order JACRU operator
remains the reconstruction operator in the correction path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from .psu_b0_reconstruction_interface import PSUB0VoxelGradientOperator


Tensor = torch.Tensor
LinearMap = Callable[[Tensor], Tensor]


def fourth_order_difference_matrix(
    size: int, spacing: float, *, dtype: torch.dtype = torch.float64
) -> Tensor:
    """Return a fourth-order centered derivative with second-order boundaries."""

    count = int(size)
    step = float(spacing)
    if count < 5:
        raise ValueError("fourth-order differences require at least five grid points")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("spacing must be finite and positive")
    matrix = torch.zeros((count, count), dtype=dtype)
    matrix[0, :3] = torch.tensor((-3.0, 4.0, -1.0), dtype=dtype) / (2.0 * step)
    matrix[1, (0, 2)] = torch.tensor((-1.0, 1.0), dtype=dtype) / (2.0 * step)
    matrix[-2, (-3, -1)] = torch.tensor((-1.0, 1.0), dtype=dtype) / (2.0 * step)
    matrix[-1, -3:] = torch.tensor((1.0, -4.0, 3.0), dtype=dtype) / (2.0 * step)
    stencil = torch.tensor((1.0, -8.0, 0.0, 8.0, -1.0), dtype=dtype) / (
        12.0 * step
    )
    for index in range(2, count - 2):
        matrix[index, index - 2 : index + 3] = stencil
    return matrix


def _apply_axis(values: Tensor, matrix: Tensor, *, axis: int, adjoint: bool) -> Tensor:
    moved = torch.movedim(values, axis, -1)
    transform = matrix if adjoint else matrix.T
    output = torch.matmul(moved, transform.to(moved))
    return torch.movedim(output, -1, axis)


@dataclass
class HighOrderTeacherMaps:
    """Exact forward/adjoint pair sharing the base operator's ray factors."""

    base: PSUB0VoxelGradientOperator
    forward_calls: int = 0
    adjoint_calls: int = 0

    def __post_init__(self) -> None:
        nz, ny, nx = self.base.grid_shape
        dx, dy, dz = self.base.spacing_xyz
        dtype = self.base.sample_weights.dtype
        self._dx = fourth_order_difference_matrix(nx, dx, dtype=dtype)
        self._dy = fourth_order_difference_matrix(ny, dy, dtype=dtype)
        self._dz = fourth_order_difference_matrix(nz, dz, dtype=dtype)

    def reset_call_counts(self) -> None:
        self.forward_calls = 0
        self.adjoint_calls = 0

    def call_report(self) -> dict[str, int]:
        return {
            "forward_calls": int(self.forward_calls),
            "adjoint_calls": int(self.adjoint_calls),
        }

    def forward(self, field: Tensor) -> Tensor:
        self.forward_calls += 1
        value = torch.as_tensor(field, dtype=self.base.sample_weights.dtype)
        if value.shape != self.base.grid_shape:
            raise ValueError("field must match the base operator grid")
        volume = value[None] * self.base.support
        gradient = torch.stack(
            (
                _apply_axis(volume, self._dx, axis=-1, adjoint=False),
                _apply_axis(volume, self._dy, axis=-2, adjoint=False),
                _apply_axis(volume, self._dz, axis=-3, adjoint=False),
            ),
            dim=1,
        )
        sampled = self.base.trilinear_interpolation(gradient)
        u = torch.einsum("bcrs,rc->brs", sampled, self.base.projection_u)
        v = torch.einsum("bcrs,rc->brs", sampled, self.base.projection_v)
        projected = torch.stack((u.sum(dim=-1), v.sum(dim=-1)), dim=-1)
        return projected[0] * self.base.ray_scale[:, None]

    def adjoint(self, observation: Tensor) -> Tensor:
        self.adjoint_calls += 1
        residual = torch.as_tensor(observation, dtype=self.base.sample_weights.dtype)
        if residual.shape != (self.base.ray_count, 2):
            raise ValueError("observation must have shape [ray,2]")
        component = (
            residual[None, :, 0:1] * self.base.projection_u[None]
            + residual[None, :, 1:2] * self.base.projection_v[None]
        )
        component = component * self.base.ray_scale[None, :, None]
        sampled = component.permute(0, 2, 1)[:, :, :, None].expand(
            -1, -1, -1, self.base.sample_count
        )
        gradient = self.base.trilinear_interpolation_adjoint(sampled)
        volume = (
            _apply_axis(gradient[:, 0], self._dx, axis=-1, adjoint=True)
            + _apply_axis(gradient[:, 1], self._dy, axis=-2, adjoint=True)
            + _apply_axis(gradient[:, 2], self._dz, axis=-3, adjoint=True)
        )
        return (volume * self.base.support)[0]

    def correction(self, field: Tensor, *, low_projection: Tensor | None = None) -> Tensor:
        """Return high-order minus low-order projection for one visible field."""

        high = self.forward(field)
        low = (
            self.base(torch.as_tensor(field)[None, None])[0]
            if low_projection is None
            else torch.as_tensor(low_projection).to(high)
        )
        if low.shape != high.shape:
            raise ValueError("low projection shape does not match high-order projection")
        return high - low


@dataclass(frozen=True)
class WarmStartCGLSResult:
    field: Tensor
    residual: Tensor
    history: tuple[dict[str, float | int | bool | None], ...]
    forward_calls: int
    adjoint_calls: int


@torch.no_grad()
def warm_start_cgls(
    observation: Tensor,
    *,
    forward: LinearMap,
    adjoint: LinearMap,
    support: Tensor,
    initial_field: Tensor,
    initial_projection: Tensor,
    iterations: int,
    denominator_floor: float = 1e-20,
) -> WarmStartCGLSResult:
    """Run CGLS from a supplied field/projection with exactly K forward/K adjoint calls."""

    target = torch.as_tensor(observation, dtype=torch.float64)
    mask = torch.as_tensor(support, dtype=torch.float64)
    field = torch.as_tensor(initial_field, dtype=torch.float64).clone()
    projection = torch.as_tensor(initial_projection, dtype=torch.float64)
    count = int(iterations)
    floor = float(denominator_floor)
    if target.ndim != 2 or target.shape[-1] != 2:
        raise ValueError("observation must have shape [ray,2]")
    if projection.shape != target.shape:
        raise ValueError("initial_projection must match observation")
    if field.shape != mask.shape:
        raise ValueError("initial_field and support must have identical shape")
    if count < 1 or not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("iterations and denominator_floor must be positive")

    forward_calls = 0
    adjoint_calls = 0
    residual = target - projection
    normal = adjoint(residual) * mask
    adjoint_calls += 1
    direction = normal.clone()
    gamma = torch.sum(normal * normal)
    initial_residual_squared = max(float(torch.sum(residual * residual)), floor)
    history: list[dict[str, float | int | bool | None]] = []
    for index in range(count):
        projected = forward(direction)
        forward_calls += 1
        denominator = torch.sum(projected * projected)
        gamma_value = float(gamma)
        denominator_value = float(denominator)
        breakdown = gamma_value <= floor or denominator_value <= floor
        alpha = 0.0 if breakdown else gamma_value / denominator_value
        field = (field + alpha * direction) * mask
        residual = residual - alpha * projected
        residual_squared = float(torch.sum(residual * residual))
        beta: float | None = None
        if index + 1 < count:
            next_normal = adjoint(residual) * mask
            adjoint_calls += 1
            next_gamma = torch.sum(next_normal * next_normal)
            beta = 0.0 if gamma_value <= floor else float(next_gamma) / gamma_value
            direction = (next_normal + beta * direction) * mask
            gamma = next_gamma
        history.append(
            {
                "iteration": index + 1,
                "relative_data_residual_squared": residual_squared
                / initial_residual_squared,
                "alpha": alpha,
                "beta": beta,
                "breakdown": breakdown,
            }
        )
    if forward_calls != count or adjoint_calls != count:
        raise RuntimeError("warm-start CGLS physical call contract drifted")
    if not bool(torch.all(torch.isfinite(field))):
        raise FloatingPointError("warm-start CGLS produced a non-finite field")
    return WarmStartCGLSResult(
        field=field,
        residual=residual,
        history=tuple(history),
        forward_calls=forward_calls,
        adjoint_calls=adjoint_calls,
    )
