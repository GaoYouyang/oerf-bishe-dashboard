"""Conventional neural comparators for the JACRU M2 residual task.

These compact models are comparison baselines, not novelty claims.  They use
the truth-free :class:`JACRUM2LearnedResidual` call contract and consume only
the CGLS base, support, masked moments of per-view adjoint lifts, and masked
camera-pose moments.  Every correction is support-limited and bounded by a
declared, fixed magnitude.  A zero-initialized residual head makes each model
return the CGLS base exactly at construction time.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import torch
from torch import nn

from .jacru_m2_learned_residual import JACRUM2LearnedResidual


Tensor: TypeAlias = torch.Tensor


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1 or parsed != value:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_finite(value: float, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def _shape3(values: tuple[int, int, int], *, name: str) -> tuple[int, int, int]:
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three positive integers")
    return tuple(
        _positive_integer(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def _group_count(channels: int) -> int:
    return next(
        value
        for value in range(min(4, int(channels)), 0, -1)
        if int(channels) % value == 0
    )


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``model``."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _masked_moments(
    lifted: Tensor,
    poses: Tensor,
    masks: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pool the unordered active-camera set without auxiliary sample metadata."""

    active_views = masks[:, :, None, None, None, None]
    view_denominator = masks.sum(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1.0)
    lift_mean = (lifted * active_views).sum(dim=1) / view_denominator
    centered_lifts = (lifted - lift_mean[:, None]) * active_views
    lift_variance = centered_lifts.square().sum(dim=1) / view_denominator

    active_poses = masks[:, :, None]
    pose_denominator = masks.sum(dim=1, keepdim=True).clamp_min(1.0)
    pose_mean = (poses * active_poses).sum(dim=1) / pose_denominator
    centered_poses = (poses - pose_mean[:, None]) * active_poses
    pose_variance = centered_poses.square().sum(dim=1) / pose_denominator
    pose_moments = torch.cat((pose_mean, pose_variance), dim=1)
    return lift_mean, lift_variance, pose_moments


class _ResidualComparatorBase(JACRUM2LearnedResidual):
    """Reuse the M2 validation/forward contract while replacing its network."""

    def __init__(
        self,
        *,
        pose_feature_count: int,
        maximum_residual_magnitude: float,
    ) -> None:
        nn.Module.__init__(self)
        self.pose_feature_count = _positive_integer(
            pose_feature_count,
            name="pose_feature_count",
        )
        self.maximum_residual_magnitude = _positive_finite(
            maximum_residual_magnitude,
            name="maximum_residual_magnitude",
        )

    @property
    def trainable_parameter_count(self) -> int:
        return parameter_count(self)

    @property
    def parameter_count(self) -> int:
        return parameter_count(self)

    def _raw_residual(
        self,
        base_field: Tensor,
        support: Tensor,
        lift_mean: Tensor,
        lift_variance: Tensor,
        pose_moments: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def _residual_from_validated(
        self,
        base_field: Tensor,
        support: Tensor,
        lifted: Tensor,
        poses: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lift_mean, lift_variance, pose_moments = _masked_moments(
            lifted,
            poses,
            masks,
        )
        raw_residual = self._raw_residual(
            base_field,
            support,
            lift_mean,
            lift_variance,
            pose_moments,
        )
        if raw_residual.shape != base_field.shape:
            raise RuntimeError("comparator residual head returned the wrong field shape")
        gate = torch.ones(
            (base_field.shape[0], 1, 1, 1, 1),
            dtype=base_field.dtype,
            device=base_field.device,
        )
        correction = (
            support
            * self.maximum_residual_magnitude
            * torch.tanh(raw_residual)
        )
        return correction, gate


class PooledCNN3DResidualComparator(_ResidualComparatorBase):
    """Small 3D CNN over masked view moments and the CGLS base field."""

    def __init__(
        self,
        *,
        pose_feature_count: int = 4,
        pose_channels: int = 4,
        hidden_channels: int = 12,
        maximum_residual_magnitude: float = 0.25,
    ) -> None:
        super().__init__(
            pose_feature_count=pose_feature_count,
            maximum_residual_magnitude=maximum_residual_magnitude,
        )
        pose_width = _positive_integer(pose_channels, name="pose_channels")
        hidden = _positive_integer(hidden_channels, name="hidden_channels")
        self.pose_encoder = nn.Sequential(
            nn.Linear(2 * self.pose_feature_count, pose_width),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            nn.Conv3d(4 + pose_width, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.residual_head = nn.Conv3d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _raw_residual(
        self,
        base_field: Tensor,
        support: Tensor,
        lift_mean: Tensor,
        lift_variance: Tensor,
        pose_moments: Tensor,
    ) -> Tensor:
        pose_embedding = self.pose_encoder(pose_moments)
        pose_volume = pose_embedding[:, :, None, None, None].expand(
            -1,
            -1,
            *base_field.shape[-3:],
        )
        features = self.backbone(
            torch.cat(
                (
                    base_field,
                    support,
                    lift_mean,
                    lift_variance,
                    pose_volume,
                ),
                dim=1,
            )
        )
        return self.residual_head(features)


class FixedGridDeepONetResidualComparator(_ResidualComparatorBase):
    """Fixed-grid DeepONet with compact pooled branch observations."""

    def __init__(
        self,
        *,
        grid_shape: tuple[int, int, int] = (12, 12, 12),
        pool_shape: tuple[int, int, int] = (3, 3, 3),
        pose_feature_count: int = 4,
        branch_hidden: int = 48,
        trunk_hidden: int = 32,
        rank: int = 24,
        maximum_residual_magnitude: float = 0.25,
    ) -> None:
        super().__init__(
            pose_feature_count=pose_feature_count,
            maximum_residual_magnitude=maximum_residual_magnitude,
        )
        self.grid_shape = _shape3(grid_shape, name="grid_shape")
        self.pool_shape = _shape3(pool_shape, name="pool_shape")
        if any(
            grid % pooled != 0
            for grid, pooled in zip(self.grid_shape, self.pool_shape)
        ):
            raise ValueError("each grid dimension must be divisible by pool_shape")
        branch_width = _positive_integer(branch_hidden, name="branch_hidden")
        trunk_width = _positive_integer(trunk_hidden, name="trunk_hidden")
        basis_rank = _positive_integer(rank, name="rank")

        branch_inputs = 4 * math.prod(self.pool_shape) + 2 * self.pose_feature_count
        self.branch = nn.Sequential(
            nn.Linear(branch_inputs, branch_width),
            nn.GELU(),
            nn.Linear(branch_width, basis_rank),
        )
        self.trunk = nn.Sequential(
            nn.Linear(3, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, basis_rank),
        )
        self.rank_scale = math.sqrt(float(basis_rank))
        axes = tuple(torch.linspace(-1.0, 1.0, steps=size) for size in self.grid_shape)
        coordinates = torch.stack(
            torch.meshgrid(*axes, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        self.register_buffer("fixed_grid_coordinates", coordinates, persistent=True)
        self.residual_head = nn.Conv3d(1, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _block_average(self, fields: Tensor) -> Tensor:
        batch, channels, depth, height, width = fields.shape
        pooled_depth, pooled_height, pooled_width = self.pool_shape
        return fields.reshape(
            batch,
            channels,
            pooled_depth,
            depth // pooled_depth,
            pooled_height,
            height // pooled_height,
            pooled_width,
            width // pooled_width,
        ).mean(dim=(3, 5, 7))

    def _raw_residual(
        self,
        base_field: Tensor,
        support: Tensor,
        lift_mean: Tensor,
        lift_variance: Tensor,
        pose_moments: Tensor,
    ) -> Tensor:
        if tuple(base_field.shape[-3:]) != self.grid_shape:
            raise ValueError(
                "FixedGridDeepONetResidualComparator requires grid_shape "
                f"{self.grid_shape}, received {tuple(base_field.shape[-3:])}"
            )
        branch_fields = torch.cat(
            (base_field, support, lift_mean, lift_variance),
            dim=1,
        )
        pooled = self._block_average(branch_fields).flatten(start_dim=1)
        coefficients = self.branch(torch.cat((pooled, pose_moments), dim=1))
        basis = self.trunk(self.fixed_grid_coordinates)
        operator_field = torch.einsum(
            "br,nr->bn",
            coefficients,
            basis,
        ) / self.rank_scale
        operator_field = operator_field.reshape(
            base_field.shape[0],
            1,
            *self.grid_shape,
        )
        return self.residual_head(operator_field)


def _official_fno_class() -> type[nn.Module]:
    try:
        from neuralop.models import FNO
    except ImportError as error:
        raise RuntimeError(
            "NeuralOpFNOResidualComparator requires the official "
            "'neuraloperator' package; install it with "
            "'python -m pip install neuraloperator'."
        ) from error
    return FNO


class NeuralOpFNOResidualComparator(_ResidualComparatorBase):
    """Thin residual wrapper around the official ``neuralop.models.FNO``."""

    def __init__(
        self,
        *,
        pose_feature_count: int = 4,
        pose_channels: int = 4,
        hidden_channels: int = 8,
        n_modes: tuple[int, int, int] = (4, 4, 4),
        n_layers: int = 2,
        maximum_residual_magnitude: float = 0.25,
    ) -> None:
        super().__init__(
            pose_feature_count=pose_feature_count,
            maximum_residual_magnitude=maximum_residual_magnitude,
        )
        pose_width = _positive_integer(pose_channels, name="pose_channels")
        hidden = _positive_integer(hidden_channels, name="hidden_channels")
        modes = _shape3(n_modes, name="n_modes")
        layers = _positive_integer(n_layers, name="n_layers")
        self.pose_encoder = nn.Sequential(
            nn.Linear(2 * self.pose_feature_count, pose_width),
            nn.GELU(),
        )
        official_fno = _official_fno_class()
        self.backbone = official_fno(
            n_modes=modes,
            in_channels=4 + pose_width,
            out_channels=1,
            hidden_channels=hidden,
            n_layers=layers,
        )
        self.residual_head = nn.Conv3d(1, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _raw_residual(
        self,
        base_field: Tensor,
        support: Tensor,
        lift_mean: Tensor,
        lift_variance: Tensor,
        pose_moments: Tensor,
    ) -> Tensor:
        pose_embedding = self.pose_encoder(pose_moments)
        pose_volume = pose_embedding[:, :, None, None, None].expand(
            -1,
            -1,
            *base_field.shape[-3:],
        )
        fno_input = torch.cat(
            (
                base_field,
                support,
                lift_mean,
                lift_variance,
                pose_volume,
            ),
            dim=1,
        )
        return self.residual_head(self.backbone(fno_input))


# Short aliases keep experiment configuration names readable.
Pooled3DCNNComparator = PooledCNN3DResidualComparator
FixedGridDeepONetComparator = FixedGridDeepONetResidualComparator
OfficialNeuralOpFNOComparator = NeuralOpFNOResidualComparator


__all__ = [
    "FixedGridDeepONetComparator",
    "FixedGridDeepONetResidualComparator",
    "NeuralOpFNOResidualComparator",
    "OfficialNeuralOpFNOComparator",
    "Pooled3DCNNComparator",
    "PooledCNN3DResidualComparator",
    "parameter_count",
]
