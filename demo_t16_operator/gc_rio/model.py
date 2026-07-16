"""Small operator-aware residual inverse model for the v5h mechanism gate.

The model is deliberately not another camera-token attention network. Source
views are first aggregated by the declared adjoint and Fisher diagonal. A
target-operator query then conditions a compact voxel-space corrector, and an
explicit source data-consistency step closes the prediction through physics.
No target observation or truth field is accepted by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualInverseOutput:
    """Field correction and physics-decoded residual predictions."""

    correction: torch.Tensor
    corrected_field: torch.Tensor
    source_residual_prediction: torch.Tensor
    target_residual_prediction: torch.Tensor
    learned_increment: torch.Tensor
    source_data_consistency_rms: torch.Tensor


def _group_count(channels: int) -> int:
    return next(
        value
        for value in range(min(4, int(channels)), 0, -1)
        if int(channels) % value == 0
    )


def _expanded_support(support: torch.Tensor, batch: int, voxels: int) -> torch.Tensor:
    values = support
    if values.ndim == 1:
        values = values[None].expand(batch, -1)
    if values.shape != (batch, voxels):
        raise ValueError("support must have shape [voxel] or [batch,voxel]")
    return values.to(dtype=torch.float32)


def _expanded_source_sigma(
    sigma: torch.Tensor, batch: int, views: int, measurements: int
) -> torch.Tensor:
    values = sigma
    if values.ndim == 2 and values.shape == (batch, views):
        values = values[:, :, None]
    try:
        values = torch.broadcast_to(values, (batch, views, measurements))
    except RuntimeError as error:
        raise ValueError("source_sigma cannot broadcast to source residuals") from error
    if not torch.all(torch.isfinite(values)) or torch.any(values <= 0.0):
        raise ValueError("source_sigma must be finite and positive")
    return values


def _expanded_target_sigma(
    sigma: torch.Tensor, batch: int, measurements: int
) -> torch.Tensor:
    values = sigma
    if values.ndim == 1 and values.shape == (batch,):
        values = values[:, None]
    try:
        values = torch.broadcast_to(values, (batch, measurements))
    except RuntimeError as error:
        raise ValueError("target_sigma cannot broadcast to target operator rows") from error
    if not torch.all(torch.isfinite(values)) or torch.any(values <= 0.0):
        raise ValueError("target_sigma must be finite and positive")
    return values


def validate_operator_batch(
    source_operator: torch.Tensor,
    target_operator: torch.Tensor,
    source_residual: torch.Tensor,
    source_sigma: torch.Tensor,
    target_sigma: torch.Tensor,
    base_field: torch.Tensor,
    analytic_correction: torch.Tensor,
    support: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate the predictor-only tensor contract."""

    if source_operator.ndim != 4:
        raise ValueError("source_operator must have shape [batch,view,measurement,voxel]")
    batch, views, measurements, voxels = source_operator.shape
    if target_operator.shape != (batch, measurements, voxels):
        raise ValueError("target_operator must have shape [batch,measurement,voxel]")
    if source_residual.shape != (batch, views, measurements):
        raise ValueError("source_residual shape disagrees with source_operator")
    if base_field.shape != (batch, voxels):
        raise ValueError("base_field must have shape [batch,voxel]")
    if analytic_correction.shape != (batch, voxels):
        raise ValueError("analytic_correction must have shape [batch,voxel]")
    _expanded_source_sigma(source_sigma, batch, views, measurements)
    _expanded_target_sigma(target_sigma, batch, measurements)
    _expanded_support(support, batch, voxels)
    tensors = (
        source_operator,
        target_operator,
        source_residual,
        base_field,
        analytic_correction,
    )
    if any(not torch.all(torch.isfinite(value)) for value in tensors):
        raise ValueError("predictor tensors must be finite")
    return batch, views, measurements, voxels


def adjoint_fisher_statistics(
    source_operator: torch.Tensor,
    target_operator: torch.Tensor,
    source_residual: torch.Tensor,
    source_sigma: torch.Tensor,
    target_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute declared adjoint and source/target Fisher diagonals."""

    batch, views, measurements, _ = source_operator.shape
    source_scale = _expanded_source_sigma(
        source_sigma, batch, views, measurements
    )
    target_scale = _expanded_target_sigma(target_sigma, batch, measurements)
    source_whitened = source_operator / source_scale[:, :, :, None]
    residual_whitened = source_residual / source_scale
    target_whitened = target_operator / target_scale[:, :, None]
    adjoint = torch.einsum(
        "bvmp,bvm->bp", source_whitened, residual_whitened
    )
    source_fisher = torch.sum(source_whitened.square(), dim=(1, 2))
    target_fisher = torch.sum(target_whitened.square(), dim=1)
    return adjoint, source_fisher, target_fisher


def physics_decode(operator: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Apply a batched target/source operator to a flattened field."""

    if operator.ndim == 3:
        return torch.einsum("bmp,bp->bm", operator, field)
    if operator.ndim == 4:
        return torch.einsum("bvmp,bp->bvm", operator, field)
    raise ValueError("operator must be a batched target or source operator")


def source_data_consistency_step(
    correction: torch.Tensor,
    source_operator: torch.Tensor,
    source_residual: torch.Tensor,
    source_sigma: torch.Tensor,
    source_fisher: torch.Tensor,
    support: torch.Tensor,
    *,
    ridge_lambda: float,
    step_fraction: float,
) -> torch.Tensor:
    """One diagonally preconditioned source-residual consistency step."""

    batch, views, measurements, voxels = source_operator.shape
    sigma = _expanded_source_sigma(source_sigma, batch, views, measurements)
    support_values = _expanded_support(support, batch, voxels).to(correction)
    mismatch = physics_decode(source_operator, correction) - source_residual
    gradient = torch.einsum(
        "bvmp,bvm->bp",
        source_operator / sigma[:, :, :, None],
        mismatch / sigma,
    )
    denominator = source_fisher + float(ridge_lambda)
    updated = correction - float(step_fraction) * gradient / denominator.clamp_min(1e-8)
    return support_values * updated


class VoxelResidualBlock(nn.Module):
    """Compact local block; no Fourier or per-camera attention components."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


class GeometryConditionedResidualInverseOperator(nn.Module):
    """GC-RIO v0: adjoint/Fisher query with a target physics decoder.

    The target operator affects the latent correction only through its declared
    Fisher sensitivity. Passing ``use_target_geometry=False`` preserves the
    exact parameter count while replacing the target query with the source
    Fisher map, which is the preregistered geometry ablation.
    """

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
        if math.prod(self.grid_shape) <= 0:
            raise ValueError("grid_shape must be positive")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        self.maximum_learned_fraction = float(maximum_learned_fraction)
        self.ridge_lambda = float(ridge_lambda)
        self.data_consistency_step = float(data_consistency_step)
        self.use_target_geometry = bool(use_target_geometry)
        if self.maximum_learned_fraction <= 0.0:
            raise ValueError("maximum_learned_fraction must be positive")
        if self.ridge_lambda <= 0.0:
            raise ValueError("ridge_lambda must be positive")
        if not 0.0 <= self.data_consistency_step <= 1.0:
            raise ValueError("data_consistency_step must lie in [0,1]")

        depth, height, width = self.grid_shape
        z = torch.linspace(-1.0, 1.0, depth)
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        coordinates = torch.stack(
            torch.meshgrid(z, y, x, indexing="ij"), dim=0
        )[None]
        self.register_buffer("coordinates", coordinates, persistent=False)
        channels = int(hidden_channels)
        self.lift = nn.Sequential(
            nn.Conv3d(10, channels, kernel_size=3, padding=1),
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

    def _feature_volume(
        self,
        base_field: torch.Tensor,
        analytic_correction: torch.Tensor,
        adjoint: torch.Tensor,
        source_fisher: torch.Tensor,
        target_fisher: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, voxels = base_field.shape
        support_values = _expanded_support(support, batch, voxels).to(base_field)
        selected_target = target_fisher if self.use_target_geometry else source_fisher
        source_mean = source_fisher.mean(dim=1, keepdim=True).clamp_min(1e-10)
        target_mean = selected_target.mean(dim=1, keepdim=True).clamp_min(1e-10)
        source_log = torch.log1p(source_fisher / source_mean)
        target_log = torch.log1p(selected_target / target_mean)
        leverage = torch.log1p(
            selected_target / (source_fisher + self.ridge_lambda).clamp_min(1e-10)
        )
        base_scaled = base_field / self._rms(base_field)
        analytic_scale = self._rms(analytic_correction)
        analytic_scaled = analytic_correction / analytic_scale
        adjoint_scaled = adjoint / self._rms(adjoint)
        flat = torch.stack(
            [
                base_scaled,
                analytic_scaled,
                adjoint_scaled,
                source_log,
                target_log,
                leverage,
                support_values,
            ],
            dim=1,
        )
        volume = flat.reshape(batch, 7, *self.grid_shape)
        coordinates = self.coordinates.to(volume).expand(batch, -1, -1, -1, -1)
        return torch.cat([volume, coordinates], dim=1), analytic_scale

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
        features, correction_scale = self._feature_volume(
            base_field,
            analytic_correction,
            adjoint,
            source_fisher,
            target_fisher,
            support,
        )
        latent = self.blocks(self.lift(features))
        learned_volume = (
            self.maximum_learned_fraction
            * correction_scale[:, :, None, None, None]
            * torch.tanh(self.head(latent))
        )
        learned = learned_volume.reshape(batch, voxels)
        support_values = _expanded_support(support, batch, voxels).to(learned)
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
