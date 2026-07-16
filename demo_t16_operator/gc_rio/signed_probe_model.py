"""Signed-probe GC-RIO candidate for preserving target row-space geometry.

The v5h Fisher-diagonal query discarded the signs and detector-space modes of
the target operator. This candidate backprojects a fixed, data-independent
probe bank through the target operator and lets a local voxel corrector
interact those signed maps with the source residual adjoint. Target
observations remain unavailable at inference.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import (
    ResidualInverseOutput,
    VoxelResidualBlock,
    _expanded_source_sigma,
    _expanded_support,
    _expanded_target_sigma,
    _group_count,
    adjoint_fisher_statistics,
    physics_decode,
    source_data_consistency_step,
    validate_operator_batch,
)


def detector_probe_bank(
    depth: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return eight fixed low-order detector modes with unit RMS."""

    if depth < 2 or width < 2:
        raise ValueError("detector probe grid requires depth,width >= 2")
    z = torch.linspace(-1.0, 1.0, int(depth), dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, int(width), dtype=dtype, device=device)
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    probes = torch.stack(
        [
            torch.ones_like(xx),
            xx,
            zz,
            xx * zz,
            torch.sin(math.pi * xx),
            torch.cos(math.pi * xx),
            torch.sin(math.pi * zz),
            torch.cos(math.pi * zz),
        ],
        dim=0,
    ).reshape(8, depth * width)
    return probes / torch.sqrt(torch.mean(probes.square(), dim=1, keepdim=True) + 1e-10)


def signed_operator_probe_maps(
    operator: torch.Tensor,
    sigma: torch.Tensor,
    probes: torch.Tensor,
) -> torch.Tensor:
    """Backproject detector modes through a target operator as signed maps."""

    if operator.ndim != 3:
        raise ValueError("operator must have shape [batch,measurement,voxel]")
    batch, measurements, _ = operator.shape
    if probes.ndim != 2 or probes.shape[1] != measurements:
        raise ValueError("probe measurement dimension disagrees with operator")
    scale = _expanded_target_sigma(sigma, batch, measurements)
    maps = torch.einsum("bmp,km->bkp", operator / scale[:, :, None], probes.to(operator))
    rms = torch.sqrt(torch.mean(maps.square(), dim=2, keepdim=True) + 1e-10)
    return maps / rms


def source_signed_probe_maps(
    operator: torch.Tensor,
    sigma: torch.Tensor,
    probes: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant mean source backprojection for matched controls."""

    if operator.ndim != 4:
        raise ValueError("source operator must have [batch,view,measurement,voxel]")
    batch, views, measurements, _ = operator.shape
    if probes.ndim != 2 or probes.shape[1] != measurements:
        raise ValueError("probe measurement dimension disagrees with source operator")
    scale = _expanded_source_sigma(sigma, batch, views, measurements)
    maps = torch.einsum(
        "bvmp,km->bvkp", operator / scale[:, :, :, None], probes.to(operator)
    ).mean(dim=1)
    rms = torch.sqrt(torch.mean(maps.square(), dim=2, keepdim=True) + 1e-10)
    return maps / rms


class SignedProbeResidualInverseOperator(nn.Module):
    """GC-RIO v1 with signed detector-mode target queries."""

    predictor_argument_names = (
        "source_operator",
        "target_operator",
        "source_residual",
        "source_sigma",
        "target_sigma",
        "base_field",
        "analytic_correction",
        "support",
        "conditioning_target_operator",
    )

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        *,
        hidden_channels: int = 12,
        residual_blocks: int = 2,
        maximum_learned_fraction: float = 0.35,
        ridge_lambda: float = 1.0,
        data_consistency_step: float = 0.12,
        use_target_geometry: bool = True,
    ):
        super().__init__()
        self.grid_shape = tuple(int(value) for value in grid_shape)
        if len(self.grid_shape) != 3 or math.prod(self.grid_shape) <= 0:
            raise ValueError("grid_shape must contain three positive values")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        self.maximum_learned_fraction = float(maximum_learned_fraction)
        self.ridge_lambda = float(ridge_lambda)
        self.data_consistency_step = float(data_consistency_step)
        self.use_target_geometry = bool(use_target_geometry)
        if self.maximum_learned_fraction <= 0.0 or self.ridge_lambda <= 0.0:
            raise ValueError("learned fraction and ridge lambda must be positive")
        if not 0.0 <= self.data_consistency_step <= 1.0:
            raise ValueError("data_consistency_step must lie in [0,1]")

        depth, height, width = self.grid_shape
        probes = detector_probe_bank(depth, width)
        self.register_buffer("detector_probes", probes, persistent=False)
        z = torch.linspace(-1.0, 1.0, depth)
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        coordinates = torch.stack(torch.meshgrid(z, y, x, indexing="ij"), dim=0)[None]
        self.register_buffer("coordinates", coordinates, persistent=False)
        channels = int(hidden_channels)
        # Seven scalar fields, three coordinates, eight signed probe maps and
        # eight explicit source-adjoint/target-query interactions.
        self.lift = nn.Sequential(
            nn.Conv3d(26, channels, kernel_size=3, padding=1),
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

    def _query_maps(
        self,
        source_operator: torch.Tensor,
        target_operator: torch.Tensor,
        source_sigma: torch.Tensor,
        target_sigma: torch.Tensor,
    ) -> torch.Tensor:
        if target_operator.shape[1] != self.detector_probes.shape[1]:
            raise ValueError(
                "target measurement count must equal grid depth times detector width"
            )
        if self.use_target_geometry:
            return signed_operator_probe_maps(
                target_operator, target_sigma, self.detector_probes
            )
        return source_signed_probe_maps(
            source_operator, source_sigma, self.detector_probes
        )

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
        conditioning_target_operator: torch.Tensor | None = None,
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
        query_operator = (
            target_operator
            if conditioning_target_operator is None
            else conditioning_target_operator
        )
        if query_operator.shape != target_operator.shape:
            raise ValueError(
                "conditioning_target_operator must match target_operator shape"
            )
        if not torch.all(torch.isfinite(query_operator)):
            raise ValueError("conditioning_target_operator must be finite")
        adjoint, source_fisher, target_fisher = adjoint_fisher_statistics(
            source_operator,
            query_operator,
            source_residual,
            source_sigma,
            target_sigma,
        )
        query_maps = self._query_maps(
            source_operator,
            query_operator,
            source_sigma,
            target_sigma,
        )
        support_values = _expanded_support(support, batch, voxels).to(base_field)
        selected_target_fisher = (
            target_fisher if self.use_target_geometry else source_fisher
        )
        source_mean = source_fisher.mean(dim=1, keepdim=True).clamp_min(1e-10)
        target_mean = selected_target_fisher.mean(dim=1, keepdim=True).clamp_min(
            1e-10
        )
        base_scaled = base_field / self._rms(base_field)
        analytic_scale = self._rms(analytic_correction)
        analytic_scaled = analytic_correction / analytic_scale
        adjoint_scaled = adjoint / self._rms(adjoint)
        scalar = torch.stack(
            [
                base_scaled,
                analytic_scaled,
                adjoint_scaled,
                torch.log1p(source_fisher / source_mean),
                torch.log1p(selected_target_fisher / target_mean),
                torch.log1p(
                    selected_target_fisher
                    / (source_fisher + self.ridge_lambda).clamp_min(1e-10)
                ),
                support_values,
            ],
            dim=1,
        )
        interaction = torch.tanh(query_maps * adjoint_scaled[:, None, :])
        flat = torch.cat([scalar, query_maps, interaction], dim=1)
        volume = flat.reshape(batch, 23, *self.grid_shape)
        coordinates = self.coordinates.to(volume).expand(batch, -1, -1, -1, -1)
        latent = self.blocks(self.lift(torch.cat([volume, coordinates], dim=1)))
        learned = (
            self.maximum_learned_fraction
            * analytic_scale[:, :, None, None, None]
            * torch.tanh(self.head(latent))
        ).reshape(batch, voxels)
        proposed = support_values * (analytic_correction + learned)
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
        sigma = _expanded_source_sigma(
            source_sigma,
            source_operator.shape[0],
            source_operator.shape[1],
            source_operator.shape[2],
        )
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
