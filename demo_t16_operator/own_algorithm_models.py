"""DeepONet and provisional ray-set models for the T16 v3b benchmark."""

from __future__ import annotations

import math

import torch
from torch import nn


def _group_count(channels: int) -> int:
    return next(value for value in range(min(4, channels), 0, -1) if channels % value == 0)


class GridDeepONetResidual(nn.Module):
    """A fixed-grid DeepONet baseline with a ridge residual skip."""

    def __init__(
        self,
        view_channel_start: int,
        view_count: int,
        mask_channel_start: int,
        angle_sin_channel_start: int,
        angle_cos_channel_start: int,
        coordinate_channels: tuple[int, int, int],
        branch_hidden: int = 64,
        trunk_hidden: int = 64,
        rank: int = 48,
        pool_shape: tuple[int, int, int] = (4, 4, 4),
    ):
        super().__init__()
        self.view_channel_start = int(view_channel_start)
        self.view_count = int(view_count)
        self.mask_channel_start = int(mask_channel_start)
        self.angle_sin_channel_start = int(angle_sin_channel_start)
        self.angle_cos_channel_start = int(angle_cos_channel_start)
        self.coordinate_channels = tuple(int(value) for value in coordinate_channels)
        self.pool_shape = tuple(int(value) for value in pool_shape)
        pooled_values = math.prod(self.pool_shape) * (self.view_count + 1)
        branch_inputs = pooled_values + 3 * self.view_count + 1
        self.branch = nn.Sequential(
            nn.Linear(branch_inputs, int(branch_hidden)),
            nn.GELU(),
            nn.Linear(int(branch_hidden), int(rank)),
        )
        self.trunk = nn.Sequential(
            nn.Linear(3, int(trunk_hidden)),
            nn.GELU(),
            nn.Linear(int(trunk_hidden), int(rank)),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.rank_scale = math.sqrt(float(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ridge = x[:, 0:1]
        support = x[:, 1:2]
        views = x[:, self.view_channel_start : self.view_channel_start + self.view_count]
        masks = x[:, self.mask_channel_start : self.mask_channel_start + self.view_count, 0, 0, 0]
        angle_sin = x[:, self.angle_sin_channel_start : self.angle_sin_channel_start + self.view_count, 0, 0, 0]
        angle_cos = x[:, self.angle_cos_channel_start : self.angle_cos_channel_start + self.view_count, 0, 0, 0]
        batch, _, depth, height, width = views.shape
        pooled_sources = torch.cat([views, ridge], dim=1).reshape(
            batch * (self.view_count + 1), 1, depth, height, width
        )
        pooled_depth, pooled_height, pooled_width = self.pool_shape
        if (
            depth % pooled_depth != 0
            or height % pooled_height != 0
            or width % pooled_width != 0
        ):
            raise ValueError("DeepONet block pooling requires divisible grid dimensions")
        pooled = pooled_sources.reshape(
            batch * (self.view_count + 1),
            1,
            pooled_depth,
            depth // pooled_depth,
            pooled_height,
            height // pooled_height,
            pooled_width,
            width // pooled_width,
        ).mean(dim=(3, 5, 7)).reshape(batch, -1)
        branch_input = torch.cat([pooled, masks, angle_sin, angle_cos, x[:, 2:3, 0, 0, 0]], dim=1)
        coefficients = self.branch(branch_input)
        coordinates = x[:, self.coordinate_channels].permute(0, 2, 3, 4, 1).reshape(batch, -1, 3)
        basis = self.trunk(coordinates)
        residual = torch.einsum("br,bnr->bn", coefficients, basis) / self.rank_scale + self.bias
        residual = residual.reshape(batch, 1, depth, height, width)
        return ridge + support * residual


class RaySetAttentionEncoder(nn.Module):
    """Aggregate active camera lifts into permutation-invariant voxel features."""

    def __init__(
        self,
        view_count: int,
        view_channel_start: int,
        mask_channel_start: int,
        angle_sin_channel_start: int,
        angle_cos_channel_start: int,
        coordinate_channels: tuple[int, int, int],
        view_features: int = 4,
    ):
        super().__init__()
        self.view_channel_start = int(view_channel_start)
        self.view_count = int(view_count)
        self.mask_channel_start = int(mask_channel_start)
        self.angle_sin_channel_start = int(angle_sin_channel_start)
        self.angle_cos_channel_start = int(angle_cos_channel_start)
        self.coordinate_channels = tuple(int(value) for value in coordinate_channels)
        features = int(view_features)
        groups = _group_count(features)
        self.view_encoder = nn.Sequential(
            nn.Conv3d(3, features, kernel_size=3, padding=1),
            nn.GroupNorm(groups, features),
            nn.GELU(),
            nn.Conv3d(features, features, kernel_size=1),
        )
        self.query_encoder = nn.Sequential(
            nn.Conv3d(5, features, kernel_size=3, padding=1),
            nn.GroupNorm(groups, features),
            nn.GELU(),
        )
        self.key = nn.Conv3d(features, features, kernel_size=1)
        self.value = nn.Conv3d(features, features, kernel_size=1)
        self.attention_scale = math.sqrt(float(features))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = x[:, self.view_channel_start : self.view_channel_start + self.view_count]
        masks = x[:, self.mask_channel_start : self.mask_channel_start + self.view_count, 0, 0, 0]
        batch, _, depth, height, width = views.shape
        sin = x[:, self.angle_sin_channel_start : self.angle_sin_channel_start + self.view_count]
        cos = x[:, self.angle_cos_channel_start : self.angle_cos_channel_start + self.view_count]
        encoded_input = torch.stack([views, sin, cos], dim=2).reshape(
            batch * self.view_count, 3, depth, height, width
        )
        encoded = self.view_encoder(encoded_input).reshape(
            batch, self.view_count, -1, depth, height, width
        )
        query_input = torch.cat([x[:, 0:2], x[:, self.coordinate_channels]], dim=1)
        query = self.query_encoder(query_input)
        keys = self.key(encoded.reshape(batch * self.view_count, -1, depth, height, width)).reshape_as(encoded)
        values = self.value(encoded.reshape(batch * self.view_count, -1, depth, height, width)).reshape_as(encoded)
        logits = torch.sum(keys * query[:, None], dim=2) / self.attention_scale
        logits = logits.masked_fill(masks[:, :, None, None, None] < 0.5, -1e4)
        weights = torch.softmax(logits, dim=1)
        aggregate = torch.sum(weights[:, :, None] * values, dim=1)
        variance = torch.sum(weights[:, :, None] * (values - aggregate[:, None]) ** 2, dim=1)
        entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-8)), dim=1, keepdim=True)
        normalizer = torch.log(masks.sum(dim=1, keepdim=True).clamp_min(2.0))[:, :, None, None, None]
        return aggregate, torch.sqrt(variance.clamp_min(1e-8)), entropy / normalizer


class RaySetResidualOperator(nn.Module):
    """Geometry-conditioned per-voxel ray-set attention over camera lifts.

    This is a provisional research hypothesis, not a claimed novel method. Each
    active camera is encoded with shared weights and explicit angle features.
    A ridge-conditioned query aggregates the unordered camera set before a
    compact spectral trunk predicts a support-limited residual field.
    """

    def __init__(
        self,
        view_count: int,
        input_channels: int,
        view_channel_start: int,
        mask_channel_start: int,
        angle_sin_channel_start: int,
        angle_cos_channel_start: int,
        coordinate_channels: tuple[int, int, int],
        view_features: int = 4,
        hidden_channels: int = 8,
        n_modes: tuple[int, int, int] = (4, 6, 6),
        n_layers: int = 3,
    ):
        super().__init__()
        try:
            from neuralop.models import FNO
        except ImportError as exc:
            raise RuntimeError("Install neuraloperator before constructing the ray-set model") from exc
        self.view_channel_start = int(view_channel_start)
        self.view_count = int(view_count)
        self.mask_channel_start = int(mask_channel_start)
        self.angle_sin_channel_start = int(angle_sin_channel_start)
        self.angle_cos_channel_start = int(angle_cos_channel_start)
        self.coordinate_channels = tuple(int(value) for value in coordinate_channels)
        features = int(view_features)
        groups = _group_count(features)
        self.view_encoder = nn.Sequential(
            nn.Conv3d(3, features, kernel_size=3, padding=1),
            nn.GroupNorm(groups, features),
            nn.GELU(),
            nn.Conv3d(features, features, kernel_size=1),
        )
        self.query_encoder = nn.Sequential(
            nn.Conv3d(5, features, kernel_size=3, padding=1),
            nn.GroupNorm(groups, features),
            nn.GELU(),
        )
        self.key = nn.Conv3d(features, features, kernel_size=1)
        self.value = nn.Conv3d(features, features, kernel_size=1)
        trunk_channels = int(input_channels) + 2 * features + 1
        self.trunk = FNO(
            n_modes=tuple(int(value) for value in n_modes),
            in_channels=trunk_channels,
            out_channels=1,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
        self.attention_scale = math.sqrt(float(features))

    def attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = x[:, self.view_channel_start : self.view_channel_start + self.view_count]
        masks = x[:, self.mask_channel_start : self.mask_channel_start + self.view_count, 0, 0, 0]
        batch, _, depth, height, width = views.shape
        sin = x[:, self.angle_sin_channel_start : self.angle_sin_channel_start + self.view_count]
        cos = x[:, self.angle_cos_channel_start : self.angle_cos_channel_start + self.view_count]
        encoded_input = torch.stack([views, sin, cos], dim=2).reshape(
            batch * self.view_count, 3, depth, height, width
        )
        encoded = self.view_encoder(encoded_input).reshape(
            batch, self.view_count, -1, depth, height, width
        )
        query_input = torch.cat([x[:, 0:2], x[:, self.coordinate_channels]], dim=1)
        query = self.query_encoder(query_input)
        keys = self.key(encoded.reshape(batch * self.view_count, -1, depth, height, width)).reshape_as(encoded)
        values = self.value(encoded.reshape(batch * self.view_count, -1, depth, height, width)).reshape_as(encoded)
        logits = torch.sum(keys * query[:, None], dim=2) / self.attention_scale
        logits = logits.masked_fill(masks[:, :, None, None, None] < 0.5, -1e4)
        weights = torch.softmax(logits, dim=1)
        aggregate = torch.sum(weights[:, :, None] * values, dim=1)
        variance = torch.sum(weights[:, :, None] * (values - aggregate[:, None]) ** 2, dim=1)
        entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-8)), dim=1, keepdim=True)
        normalizer = torch.log(masks.sum(dim=1, keepdim=True).clamp_min(2.0))[:, :, None, None, None]
        return aggregate, torch.sqrt(variance.clamp_min(1e-8)), entropy / normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aggregate, spread, entropy = self.attention(x)
        trunk_input = torch.cat([x, aggregate, spread, entropy], dim=1)
        residual = self.trunk(trunk_input)
        return x[:, 0:1] + x[:, 1:2] * residual


class ZeroInitializedRaySetAdapter(nn.Module):
    """Add a bounded ray-set correction that starts exactly at a locked operator.

    The base operator is frozen by default. The final adapter convolution is
    initialized to zero, so the first forward pass is exactly the base FNO
    prediction. This protects the strong baseline while the set branch learns a
    small, support-limited and budget-conditioned correction.
    """

    def __init__(
        self,
        base_operator: nn.Module,
        view_count: int,
        view_channel_start: int,
        mask_channel_start: int,
        angle_sin_channel_start: int,
        angle_cos_channel_start: int,
        coordinate_channels: tuple[int, int, int],
        view_features: int = 6,
        adapter_hidden: int = 8,
        gate_hidden: int = 8,
        maximum_correction_scale: float = 0.25,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_operator = base_operator
        self.maximum_correction_scale = float(maximum_correction_scale)
        features = int(view_features)
        hidden = int(adapter_hidden)
        self.set_encoder = RaySetAttentionEncoder(
            view_count=view_count,
            view_channel_start=view_channel_start,
            mask_channel_start=mask_channel_start,
            angle_sin_channel_start=angle_sin_channel_start,
            angle_cos_channel_start=angle_cos_channel_start,
            coordinate_channels=coordinate_channels,
            view_features=features,
        )
        groups = _group_count(hidden)
        self.adapter = nn.Sequential(
            nn.Conv3d(2 * features + 4, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )
        self.risk_gate = nn.Sequential(
            nn.Linear(3, int(gate_hidden)),
            nn.GELU(),
            nn.Linear(int(gate_hidden), 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.set_base_trainable(not freeze_base)

    def set_base_trainable(self, trainable: bool) -> None:
        self.base_frozen = not bool(trainable)
        for parameter in self.base_operator.parameters():
            parameter.requires_grad_(bool(trainable))

    def train(self, mode: bool = True) -> ZeroInitializedRaySetAdapter:
        super().train(mode)
        if self.base_frozen:
            self.base_operator.eval()
        return self

    def correction(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aggregate, spread, entropy = self.set_encoder(x)
        features = torch.cat([x[:, 0:3], aggregate, spread, entropy], dim=1)
        raw_correction = self.adapter(features)
        view_fraction = x[:, 2:3].mean(dim=(2, 3, 4))
        mean_spread = spread.mean(dim=(1, 2, 3, 4), keepdim=False)[:, None]
        mean_entropy = entropy.mean(dim=(1, 2, 3, 4), keepdim=False)[:, None]
        gate_features = torch.cat([view_fraction, mean_spread, mean_entropy], dim=1)
        gate = self.maximum_correction_scale * torch.sigmoid(self.risk_gate(gate_features))
        gate = gate[:, :, None, None, None]
        return x[:, 1:2] * gate * raw_correction, gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_frozen and any(
            parameter.requires_grad for parameter in self.base_operator.parameters()
        ):
            raise RuntimeError("frozen base operator unexpectedly has trainable parameters")
        base = self.base_operator(x)
        correction, _ = self.correction(x)
        return base + correction


class AcquisitionSetConditioner(nn.Module):
    """Encode an unordered active-camera set into a compact modulation vector."""

    MODES = {"geometry", "mask_only", "static", "shuffled"}

    def __init__(
        self,
        view_count: int,
        mask_channel_start: int,
        angle_sin_channel_start: int,
        angle_cos_channel_start: int,
        hidden_features: int = 16,
        embedding_features: int = 8,
        output_features: int = 6,
    ):
        super().__init__()
        self.view_count = int(view_count)
        self.mask_channel_start = int(mask_channel_start)
        self.angle_sin_channel_start = int(angle_sin_channel_start)
        self.angle_cos_channel_start = int(angle_cos_channel_start)
        hidden = int(hidden_features)
        embedding = int(embedding_features)
        self.camera_encoder = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.set_encoder = nn.Sequential(
            nn.Linear(hidden + 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, embedding),
            nn.GELU(),
        )
        self.modulation = nn.Linear(embedding, int(output_features))

    def components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = x[
            :,
            self.mask_channel_start : self.mask_channel_start + self.view_count,
            0,
            0,
            0,
        ]
        angle_sin = x[
            :,
            self.angle_sin_channel_start : self.angle_sin_channel_start
            + self.view_count,
            0,
            0,
            0,
        ]
        angle_cos = x[
            :,
            self.angle_cos_channel_start : self.angle_cos_channel_start
            + self.view_count,
            0,
            0,
            0,
        ]
        return masks, angle_sin, angle_cos

    @staticmethod
    def transform_components(
        masks: torch.Tensor,
        angle_sin: torch.Tensor,
        angle_cos: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode not in AcquisitionSetConditioner.MODES:
            raise ValueError(f"unknown acquisition descriptor mode: {mode}")
        if mode == "geometry":
            return masks, angle_sin, angle_cos
        if mode == "mask_only":
            return masks, torch.zeros_like(angle_sin), torch.zeros_like(angle_cos)
        if mode == "static":
            fraction = masks.mean(dim=1, keepdim=True)
            static_masks = fraction.expand_as(masks)
            return static_masks, torch.zeros_like(angle_sin), torch.zeros_like(angle_cos)
        if masks.shape[0] < 2:
            raise ValueError("shuffled descriptor control requires batch size >= 2")
        return (
            torch.roll(masks, shifts=1, dims=0),
            torch.roll(angle_sin, shifts=1, dims=0),
            torch.roll(angle_cos, shifts=1, dims=0),
        )

    def encode_components(
        self,
        masks: torch.Tensor,
        angle_sin: torch.Tensor,
        angle_cos: torch.Tensor,
        mode: str = "geometry",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks, angle_sin, angle_cos = self.transform_components(
            masks, angle_sin, angle_cos, mode
        )
        camera_features = torch.stack([masks, angle_sin, angle_cos], dim=-1)
        encoded = self.camera_encoder(camera_features) * masks[:, :, None]
        active_count = masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = encoded.sum(dim=1) / active_count
        first_sin = angle_sin.sum(dim=1, keepdim=True) / active_count
        first_cos = angle_cos.sum(dim=1, keepdim=True) / active_count
        second_sin = (2.0 * angle_sin * angle_cos).sum(dim=1, keepdim=True) / active_count
        second_cos = (angle_cos.square() - angle_sin.square()).sum(
            dim=1, keepdim=True
        ) / active_count
        global_features = torch.cat(
            [masks.mean(dim=1, keepdim=True), first_sin, first_cos, second_sin, second_cos],
            dim=1,
        )
        embedding = self.set_encoder(torch.cat([pooled, global_features], dim=1))
        return embedding, self.modulation(embedding)

    def forward(
        self, x: torch.Tensor, mode: str = "geometry"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_components(*self.components(x), mode=mode)


class LowModeSpectralResidualBlock(nn.Module):
    """Cheap real-valued channel mixing restricted to low 3D Fourier modes."""

    def __init__(self, channels: int, n_modes: tuple[int, int, int]):
        super().__init__()
        channels = int(channels)
        self.n_modes = tuple(int(value) for value in n_modes)
        self.spectral_mix = nn.Parameter(
            torch.randn(channels, channels) / math.sqrt(float(channels))
        )
        self.local = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth, height, width = x.shape[-3:]
        transformed = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        depth_frequency = torch.minimum(
            torch.arange(depth, device=x.device),
            depth - torch.arange(depth, device=x.device),
        )
        height_frequency = torch.minimum(
            torch.arange(height, device=x.device),
            height - torch.arange(height, device=x.device),
        )
        width_frequency = torch.arange(width // 2 + 1, device=x.device)
        mask = (
            (depth_frequency[:, None, None] < min(self.n_modes[0], depth))
            & (height_frequency[None, :, None] < min(self.n_modes[1], height))
            & (width_frequency[None, None, :] < min(self.n_modes[2], width // 2 + 1))
        )
        low_modes = transformed * mask[None, None]
        mixed = torch.complex(
            torch.einsum("bczyx,oc->bozyx", low_modes.real, self.spectral_mix),
            torch.einsum("bczyx,oc->bozyx", low_modes.imag, self.spectral_mix),
        )
        spectral = torch.fft.irfftn(
            mixed,
            s=(depth, height, width),
            dim=(-3, -2, -1),
            norm="ortho",
        )
        return self.activation(self.norm(spectral + self.local(x)))


class GeometryConditionedSpectralResidualOperator(nn.Module):
    """GC-SRO v0: frozen FNO plus a zero-init geometry-conditioned spectral residual.

    This protocol model is a falsifiable working hypothesis, not a novelty claim.
    Descriptor modes share exactly the same parameters so geometry, mask-only,
    static and shuffled controls differ only in the information they receive.
    """

    def __init__(
        self,
        base_operator: nn.Module,
        view_count: int,
        mask_channel_start: int,
        angle_sin_channel_start: int,
        angle_cos_channel_start: int,
        coordinate_channels: tuple[int, int, int],
        descriptor_hidden: int = 16,
        descriptor_embedding: int = 8,
        adapter_hidden: int = 6,
        spectral_modes: tuple[int, int, int] = (4, 6, 6),
        maximum_correction_scale: float = 0.25,
        descriptor_mode: str = "geometry",
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_operator = base_operator
        self.coordinate_channels = tuple(int(value) for value in coordinate_channels)
        self.maximum_correction_scale = float(maximum_correction_scale)
        self.descriptor_mode = str(descriptor_mode)
        hidden = int(adapter_hidden)
        self.conditioner = AcquisitionSetConditioner(
            view_count=view_count,
            mask_channel_start=mask_channel_start,
            angle_sin_channel_start=angle_sin_channel_start,
            angle_cos_channel_start=angle_cos_channel_start,
            hidden_features=int(descriptor_hidden),
            embedding_features=int(descriptor_embedding),
            output_features=hidden,
        )
        self.lift = nn.Conv3d(7, hidden, kernel_size=1)
        self.spectral_adapter = LowModeSpectralResidualBlock(hidden, spectral_modes)
        self.head = nn.Conv3d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.set_base_trainable(not freeze_base)

    def set_base_trainable(self, trainable: bool) -> None:
        self.base_frozen = not bool(trainable)
        for parameter in self.base_operator.parameters():
            parameter.requires_grad_(bool(trainable))

    def set_descriptor_mode(self, mode: str) -> None:
        if mode not in AcquisitionSetConditioner.MODES:
            raise ValueError(f"unknown acquisition descriptor mode: {mode}")
        self.descriptor_mode = str(mode)

    def train(self, mode: bool = True) -> GeometryConditionedSpectralResidualOperator:
        super().train(mode)
        if self.base_frozen:
            self.base_operator.eval()
        return self

    def descriptor_embedding(
        self,
        x: torch.Tensor,
        mode: str | None = None,
        descriptor_components: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_mode = mode or self.descriptor_mode
        if descriptor_components is None:
            return self.conditioner(x, mode=selected_mode)
        return self.conditioner.encode_components(
            *descriptor_components, mode=selected_mode
        )

    def correction(
        self,
        x: torch.Tensor,
        base_prediction: torch.Tensor | None = None,
        descriptor_components: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_prediction is None:
            base_prediction = self.base_operator(x)
        embedding, modulation = self.descriptor_embedding(
            x, descriptor_components=descriptor_components
        )
        adapter_input = torch.cat(
            [base_prediction, x[:, 0:3], x[:, self.coordinate_channels]], dim=1
        )
        features = self.lift(adapter_input)
        features = features * (1.0 + 0.25 * torch.tanh(modulation)[:, :, None, None, None])
        features = self.spectral_adapter(features)
        correction = (
            x[:, 1:2]
            * self.maximum_correction_scale
            * torch.tanh(self.head(features))
        )
        return correction, embedding

    def forward(
        self,
        x: torch.Tensor,
        descriptor_components: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        if self.base_frozen and any(
            parameter.requires_grad for parameter in self.base_operator.parameters()
        ):
            raise RuntimeError("frozen base operator unexpectedly has trainable parameters")
        base = self.base_operator(x)
        correction, _ = self.correction(
            x,
            base_prediction=base,
            descriptor_components=descriptor_components,
        )
        return base + correction


class VoxelRaySetResidualOperator(nn.Module):
    """Frozen operator plus a voxel-local, permutation-invariant camera set branch.

    Each voxel receives one token per camera containing its normalized
    backprojection, activity mask, and angular coordinates. A shared token MLP
    and masked query attention aggregate the active camera set before a small
    spatial residual block. This is a mechanism candidate, not a novelty claim.
    """

    MODES = {"ray_set", "geometry_only", "pooled_static"}

    def __init__(
        self,
        base_operator: nn.Module,
        view_count: int,
        mask_channel_start: int,
        ray_channel_start: int,
        angle_sin_channel_start: int,
        angle_cos_channel_start: int,
        coordinate_channels: tuple[int, int, int],
        token_hidden: int = 18,
        latent_features: int = 10,
        adapter_hidden: int = 8,
        spectral_modes: tuple[int, int, int] = (4, 6, 6),
        maximum_correction_scale: float = 0.25,
        acquisition_mode: str = "ray_set",
        freeze_base: bool = True,
    ):
        super().__init__()
        if acquisition_mode not in self.MODES:
            raise ValueError(f"unknown voxel ray-set mode: {acquisition_mode}")
        self.base_operator = base_operator
        self.view_count = int(view_count)
        self.mask_channel_start = int(mask_channel_start)
        self.ray_channel_start = int(ray_channel_start)
        self.angle_sin_channel_start = int(angle_sin_channel_start)
        self.angle_cos_channel_start = int(angle_cos_channel_start)
        self.coordinate_channels = tuple(int(value) for value in coordinate_channels)
        self.maximum_correction_scale = float(maximum_correction_scale)
        self.acquisition_mode = str(acquisition_mode)
        latent = int(latent_features)
        hidden = int(adapter_hidden)
        self.token_encoder = nn.Sequential(
            nn.Linear(4, int(token_hidden)),
            nn.GELU(),
            nn.Linear(int(token_hidden), latent),
            nn.GELU(),
        )
        self.query_encoder = nn.Conv3d(7, latent, kernel_size=1)
        self.key = nn.Linear(latent, latent, bias=False)
        self.value = nn.Linear(latent, latent)
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * latent, hidden, kernel_size=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.spatial_adapter = LowModeSpectralResidualBlock(hidden, spectral_modes)
        self.head = nn.Conv3d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.set_base_trainable(not freeze_base)

    def set_base_trainable(self, trainable: bool) -> None:
        self.base_frozen = not bool(trainable)
        for parameter in self.base_operator.parameters():
            parameter.requires_grad_(bool(trainable))

    def set_acquisition_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown voxel ray-set mode: {mode}")
        self.acquisition_mode = str(mode)

    def train(self, mode: bool = True) -> VoxelRaySetResidualOperator:
        super().train(mode)
        if self.base_frozen:
            self.base_operator.eval()
        return self

    def components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = x[
            :,
            self.mask_channel_start : self.mask_channel_start + self.view_count,
            0,
            0,
            0,
        ]
        angle_sin = x[
            :,
            self.angle_sin_channel_start : self.angle_sin_channel_start
            + self.view_count,
            0,
            0,
            0,
        ]
        angle_cos = x[
            :,
            self.angle_cos_channel_start : self.angle_cos_channel_start
            + self.view_count,
            0,
            0,
            0,
        ]
        rays = x[
            :,
            self.ray_channel_start : self.ray_channel_start + self.view_count,
        ]
        return masks, angle_sin, angle_cos, rays

    @staticmethod
    def transform_components(
        masks: torch.Tensor,
        angle_sin: torch.Tensor,
        angle_cos: torch.Tensor,
        rays: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode not in VoxelRaySetResidualOperator.MODES:
            raise ValueError(f"unknown voxel ray-set mode: {mode}")
        if mode == "ray_set":
            return masks, angle_sin, angle_cos, rays
        if mode == "geometry_only":
            return masks, angle_sin, angle_cos, torch.zeros_like(rays)
        active = (masks > 1e-6).to(rays.dtype)
        active_count = active.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_ray = (rays * active[:, :, None, None, None]).sum(
            dim=1, keepdim=True
        ) / active_count[:, :, None, None, None]
        fraction = masks.mean(dim=1, keepdim=True)
        static_masks = fraction.expand_as(masks)
        static_rays = pooled_ray.expand(-1, masks.shape[1], -1, -1, -1)
        return (
            static_masks,
            torch.zeros_like(angle_sin),
            torch.zeros_like(angle_cos),
            static_rays,
        )

    def ray_set_features(
        self,
        x: torch.Tensor,
        base_prediction: torch.Tensor,
        acquisition_components: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        components = acquisition_components or self.components(x)
        masks, angle_sin, angle_cos, rays = self.transform_components(
            *components, mode=self.acquisition_mode
        )
        if torch.any((masks > 1e-6).sum(dim=1) == 0):
            raise ValueError("voxel ray-set branch received an empty camera set")
        depth, height, width = rays.shape[-3:]
        mask_volume = masks[:, :, None, None, None].expand(
            -1, -1, depth, height, width
        )
        sin_volume = angle_sin[:, :, None, None, None].expand_as(mask_volume)
        cos_volume = angle_cos[:, :, None, None, None].expand_as(mask_volume)
        tokens = torch.stack([rays, mask_volume, sin_volume, cos_volume], dim=-1)
        active = (masks > 1e-6)[:, :, None, None, None, None]
        encoded = self.token_encoder(tokens) * active.to(tokens.dtype)
        query_input = torch.cat(
            [base_prediction, x[:, 0:3], x[:, self.coordinate_channels]], dim=1
        )
        query = self.query_encoder(query_input).permute(0, 2, 3, 4, 1)
        keys = self.key(encoded)
        logits = torch.sum(keys * query[:, None], dim=-1) / math.sqrt(
            float(query.shape[-1])
        )
        logits = logits.masked_fill(~active.squeeze(-1), -1e4)
        weights = torch.softmax(logits, dim=1)
        aggregate = torch.sum(weights[..., None] * self.value(encoded), dim=1)
        aggregate = aggregate.permute(0, 4, 1, 2, 3)
        query = query.permute(0, 4, 1, 2, 3)
        return torch.cat([query, aggregate], dim=1), weights

    def correction(
        self,
        x: torch.Tensor,
        base_prediction: torch.Tensor | None = None,
        acquisition_components: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_prediction is None:
            base_prediction = self.base_operator(x)
        ray_features, weights = self.ray_set_features(
            x, base_prediction, acquisition_components=acquisition_components
        )
        features = self.fuse(ray_features)
        features = self.spatial_adapter(features)
        correction = (
            x[:, 1:2]
            * self.maximum_correction_scale
            * torch.tanh(self.head(features))
        )
        return correction, weights

    def forward(
        self,
        x: torch.Tensor,
        acquisition_components: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]
        | None = None,
    ) -> torch.Tensor:
        if self.base_frozen and any(
            parameter.requires_grad for parameter in self.base_operator.parameters()
        ):
            raise RuntimeError("frozen base operator unexpectedly has trainable parameters")
        base = self.base_operator(x)
        correction, _ = self.correction(
            x,
            base_prediction=base,
            acquisition_components=acquisition_components,
        )
        return base + correction
