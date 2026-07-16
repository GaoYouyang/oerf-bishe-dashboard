"""Source-only shared-field inverse candidate for cross-view supervision.

Unlike v5h/v5i, the learned correction is initialized at exactly zero and is
not bounded by the much smaller analytic residual fit. The target operator is
used only as a physics decoder; one shared three-dimensional correction must
explain every held-out target view.
"""

from __future__ import annotations

import math
from typing import MutableMapping

import torch
from torch import nn

from .model import (
    ResidualInverseOutput,
    VoxelResidualBlock,
    _expanded_source_sigma,
    _expanded_support,
    _group_count,
    physics_decode,
    source_data_consistency_step,
    validate_operator_batch,
)


def _record_operator_calls(
    counter: MutableMapping[str, int] | None,
    name: str,
    batch: int,
) -> None:
    if counter is not None:
        counter[name] = int(counter.get(name, 0)) + int(batch)


def source_adjoint_fisher(
    source_operator: torch.Tensor,
    source_residual: torch.Tensor,
    source_sigma: torch.Tensor,
    *,
    operator_call_counter: MutableMapping[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reusable source adjoint and Fisher diagonal statistics."""

    if source_operator.ndim != 4:
        raise ValueError("source_operator must be [batch,view,measurement,voxel]")
    batch, views, measurements, _ = source_operator.shape
    if source_residual.shape != (batch, views, measurements):
        raise ValueError("source_residual shape disagrees with operator")
    sigma = _expanded_source_sigma(source_sigma, batch, views, measurements)
    whitened_operator = source_operator / sigma[:, :, :, None]
    whitened_residual = source_residual / sigma
    adjoint = torch.einsum(
        "bvmp,bvm->bp", whitened_operator, whitened_residual
    )
    _record_operator_calls(operator_call_counter, "source_adjoint", batch)
    fisher = torch.sum(whitened_operator.square(), dim=(1, 2))
    return adjoint, fisher


def source_krylov_stack(
    source_operator: torch.Tensor,
    source_residual: torch.Tensor,
    source_sigma: torch.Tensor,
    *,
    steps: int = 3,
    ridge_lambda: float = 1.0,
    relaxation: float = 0.45,
    operator_call_counter: MutableMapping[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized-Jacobi Krylov iterates, adjoint and Fisher diagonal."""

    if source_operator.ndim != 4:
        raise ValueError("source_operator must be [batch,view,measurement,voxel]")
    batch, views, measurements, voxels = source_operator.shape
    if source_residual.shape != (batch, views, measurements):
        raise ValueError("source_residual shape disagrees with operator")
    if steps < 1:
        raise ValueError("steps must be positive")
    sigma = _expanded_source_sigma(source_sigma, batch, views, measurements)
    whitened_operator = source_operator / sigma[:, :, :, None]
    whitened_residual = source_residual / sigma
    adjoint, fisher = source_adjoint_fisher(
        source_operator,
        source_residual,
        source_sigma,
        operator_call_counter=operator_call_counter,
    )
    current = torch.zeros(
        batch, voxels, dtype=source_operator.dtype, device=source_operator.device
    )
    iterates: list[torch.Tensor] = []
    for _ in range(int(steps)):
        predicted = torch.einsum("bvmp,bp->bvm", whitened_operator, current)
        _record_operator_calls(operator_call_counter, "source_forward", batch)
        gradient = torch.einsum(
            "bvmp,bvm->bp", whitened_operator, whitened_residual - predicted
        )
        _record_operator_calls(operator_call_counter, "source_adjoint", batch)
        current = current + float(relaxation) * gradient / (
            fisher + float(ridge_lambda)
        ).clamp_min(1e-8)
        scale = torch.sqrt(torch.mean(current.square(), dim=1, keepdim=True) + 1e-10)
        iterates.append(current / scale)
    return torch.stack(iterates, dim=1), adjoint, fisher


class SharedFieldResidualInverseOperator(nn.Module):
    """GC-RIO v2: source operator to one target-independent correction field."""

    predictor_argument_names = (
        "source_operator",
        "target_operator",
        "source_residual",
        "source_sigma",
        "target_sigma",
        "base_field",
        "analytic_correction",
        "support",
        "conditioning_source_operator",
    )

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        *,
        hidden_channels: int = 12,
        residual_blocks: int = 2,
        maximum_base_fraction: float = 0.55,
        ridge_lambda: float = 1.0,
        data_consistency_step: float = 0.0,
        krylov_steps: int = 3,
        use_krylov_features: bool = True,
    ):
        super().__init__()
        self.grid_shape = tuple(int(value) for value in grid_shape)
        if len(self.grid_shape) != 3 or math.prod(self.grid_shape) <= 0:
            raise ValueError("grid_shape must contain three positive values")
        if krylov_steps != 3:
            raise ValueError("v2 freezes exactly three Krylov channels")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        self.maximum_base_fraction = float(maximum_base_fraction)
        self.ridge_lambda = float(ridge_lambda)
        self.data_consistency_step = float(data_consistency_step)
        self.krylov_steps = int(krylov_steps)
        self.use_krylov_features = bool(use_krylov_features)
        if self.maximum_base_fraction <= 0.0 or self.ridge_lambda <= 0.0:
            raise ValueError("field fraction and ridge lambda must be positive")
        if not 0.0 <= self.data_consistency_step <= 1.0:
            raise ValueError("data_consistency_step must lie in [0,1]")

        depth, height, width = self.grid_shape
        z = torch.linspace(-1.0, 1.0, depth)
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        coordinates = torch.stack(torch.meshgrid(z, y, x, indexing="ij"), dim=0)[None]
        self.register_buffer("coordinates", coordinates, persistent=False)
        channels = int(hidden_channels)
        # Seven shared scalar fields, three Krylov channels, three coordinates.
        self.lift = nn.Sequential(
            nn.Conv3d(13, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(VoxelResidualBlock(channels) for _ in range(int(residual_blocks)))
        )
        self.head = nn.Conv3d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _rms(values: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(values.square(), dim=1, keepdim=True) + 1e-10)

    def forward(
        self,
        source_operator: torch.Tensor,
        target_operator: torch.Tensor,
        source_residual: torch.Tensor,
        source_sigma: torch.Tensor,
        target_sigma: torch.Tensor,
        base_field: torch.Tensor,
        analytic_correction: torch.Tensor,
        support: torch.Tensor,
        *,
        conditioning_source_operator: torch.Tensor | None = None,
        conditioning_target_operator: torch.Tensor | None = None,
        precomputed_source_statistics: tuple[torch.Tensor, torch.Tensor] | None = None,
        operator_call_counter: MutableMapping[str, int] | None = None,
    ) -> ResidualInverseOutput:
        batch, _, _, voxels = validate_operator_batch(
            source_operator,
            target_operator,
            source_residual,
            source_sigma,
            target_sigma,
            base_field,
            analytic_correction,
            support,
        )
        if voxels != math.prod(self.grid_shape):
            raise ValueError("operator voxel count disagrees with grid_shape")
        query_source = (
            source_operator
            if conditioning_source_operator is None
            else conditioning_source_operator
        )
        if query_source.shape != source_operator.shape:
            raise ValueError(
                "conditioning_source_operator must match source_operator shape"
            )
        if not torch.all(torch.isfinite(query_source)):
            raise ValueError("conditioning_source_operator must be finite")
        if conditioning_target_operator is not None:
            if conditioning_target_operator.shape != target_operator.shape:
                raise ValueError(
                    "conditioning_target_operator must match target_operator shape"
                )
            # Target geometry is deliberately excluded from the learned field.

        if self.use_krylov_features:
            if precomputed_source_statistics is not None:
                raise ValueError(
                    "precomputed_source_statistics require adjoint-only features"
                )
            krylov, adjoint, source_fisher = source_krylov_stack(
                query_source,
                source_residual,
                source_sigma,
                steps=self.krylov_steps,
                ridge_lambda=self.ridge_lambda,
                operator_call_counter=operator_call_counter,
            )
        else:
            if precomputed_source_statistics is None:
                adjoint, source_fisher = source_adjoint_fisher(
                    query_source,
                    source_residual,
                    source_sigma,
                    operator_call_counter=operator_call_counter,
                )
            else:
                adjoint, source_fisher = precomputed_source_statistics
                expected = (batch, voxels)
                if (
                    adjoint.shape != expected
                    or source_fisher.shape != expected
                    or not torch.all(torch.isfinite(adjoint))
                    or not torch.all(torch.isfinite(source_fisher))
                ):
                    raise ValueError(
                        "precomputed_source_statistics must be two finite [batch,voxel] tensors"
                    )
        adjoint_scaled = adjoint / self._rms(adjoint)
        if not self.use_krylov_features:
            preconditioned = adjoint / (source_fisher + self.ridge_lambda).clamp_min(
                1e-8
            )
            preconditioned = preconditioned / self._rms(preconditioned)
            krylov = preconditioned[:, None, :].expand(-1, self.krylov_steps, -1)
        support_values = _expanded_support(support, batch, voxels).to(base_field)
        base_scale = self._rms(base_field)
        analytic_scale = self._rms(analytic_correction)
        source_mean = source_fisher.mean(dim=1, keepdim=True).clamp_min(1e-10)
        sigma = _expanded_source_sigma(
            source_sigma,
            source_operator.shape[0],
            source_operator.shape[1],
            source_operator.shape[2],
        )
        residual_energy = torch.sqrt(
            torch.mean((source_residual / sigma).square(), dim=(1, 2), keepdim=False)
            + 1e-10
        )[:, None]
        scalar = torch.stack(
            [
                base_field / base_scale,
                analytic_correction / analytic_scale,
                adjoint_scaled,
                torch.log1p(source_fisher / source_mean),
                support_values,
                torch.log1p(residual_energy).expand(-1, voxels),
                torch.log1p(analytic_scale / base_scale).expand(-1, voxels),
            ],
            dim=1,
        )
        flat = torch.cat([scalar, krylov], dim=1)
        volume = flat.reshape(batch, 10, *self.grid_shape)
        coordinates = self.coordinates.to(volume).expand(batch, -1, -1, -1, -1)
        latent = self.blocks(self.lift(torch.cat([volume, coordinates], dim=1)))
        learned = (
            self.maximum_base_fraction
            * base_scale[:, :, None, None, None]
            * torch.tanh(self.head(latent))
        ).reshape(batch, voxels)
        proposed = support_values * learned
        correction = source_data_consistency_step(
            proposed,
            source_operator,
            source_residual,
            source_sigma,
            source_fisher,
            support_values,
            ridge_lambda=self.ridge_lambda,
            step_fraction=self.data_consistency_step,
        )
        corrected_field = support_values * (base_field + correction)
        source_prediction = physics_decode(source_operator, correction)
        target_prediction = physics_decode(target_operator, correction)
        consistency = torch.sqrt(
            torch.mean(
                ((source_prediction - source_residual) / sigma).square(),
                dim=(1, 2),
            )
        )
        return ResidualInverseOutput(
            correction=correction,
            corrected_field=corrected_field,
            source_residual_prediction=source_prediction,
            target_residual_prediction=target_prediction,
            learned_increment=learned,
            source_data_consistency_rms=consistency,
        )
