"""Residual 3D FNO and parameter-conscious 3D U-Net baselines."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = next(
            candidate
            for candidate in range(min(4, out_channels), 0, -1)
            if out_channels % candidate == 0
        )
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet3D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 8):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, 2 * b)
        self.center = ConvBlock(2 * b, 4 * b)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(4 * b, 2 * b, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose3d(2 * b, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(2 * b, b)
        self.out = nn.Conv3d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        center = self.center(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(center), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class FixedViewGate(nn.Module):
    def __init__(self, view_channel: int, reference_fraction: float):
        super().__init__()
        self.view_channel = int(view_channel)
        self.reference_fraction = float(reference_fraction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        view_fraction = x[:, self.view_channel : self.view_channel + 1].mean(dim=(2, 3, 4), keepdim=True)
        return torch.clamp(view_fraction / self.reference_fraction, min=0.0, max=1.0)


class LearnedReliabilityGate(nn.Module):
    def __init__(self, feature_channels: list[int], hidden_channels: int = 8, max_scale: float = 1.25):
        super().__init__()
        self.feature_channels = [int(channel) for channel in feature_channels]
        self.max_scale = float(max_scale)
        self.network = nn.Sequential(
            nn.Linear(len(self.feature_channels), int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), 1),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        initial_probability = min(1.0 / self.max_scale, 1.0 - 1e-4)
        nn.init.constant_(final.bias, torch.logit(torch.tensor(initial_probability)).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x[:, self.feature_channels].mean(dim=(2, 3, 4))
        alpha = self.max_scale * torch.sigmoid(self.network(features))
        return alpha[:, :, None, None, None]


def make_gate(config: dict | None) -> nn.Module | None:
    if config is None:
        return None
    gate_type = str(config["type"])
    if gate_type == "fixed_view":
        return FixedViewGate(
            view_channel=int(config["view_channel"]),
            reference_fraction=float(config["reference_fraction"]),
        )
    if gate_type == "learned":
        return LearnedReliabilityGate(
            feature_channels=[int(value) for value in config["feature_channels"]],
            hidden_channels=int(config.get("hidden_channels", 8)),
            max_scale=float(config.get("max_scale", 1.25)),
        )
    raise ValueError(f"Unknown gate type: {gate_type}")


class Reconstructor(nn.Module):
    """Optionally add the network correction to a declared input channel."""

    def __init__(
        self,
        backbone: nn.Module,
        residual_channel: int | None = 0,
        gate: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.residual_channel = residual_channel
        self.gate = gate

    def gate_values(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.residual_channel is None:
            return None
        if self.gate is None:
            return torch.ones((x.shape[0], 1, 1, 1, 1), dtype=x.dtype, device=x.device)
        return self.gate(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.backbone(x)
        if self.residual_channel is None:
            return prediction
        start = self.residual_channel
        alpha = self.gate_values(x)
        return alpha * x[:, start : start + 1] + prediction


class DualBranchOperator(nn.Module):
    """Shared operator trunk with residual/absolute experts and a small router."""

    def __init__(self, backbone: nn.Module, router_features: int, router_hidden: int = 12):
        super().__init__()
        self.backbone = backbone
        self.router = nn.Sequential(
            nn.Linear(int(router_features), int(router_hidden)),
            nn.GELU(),
            nn.Linear(int(router_hidden), 1),
        )
        final = self.router[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def experts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.backbone(x)
        if raw.shape[1] != 2:
            raise RuntimeError(f"DualBranchOperator expects two backbone outputs, received {raw.shape[1]}")
        residual = x[:, 0:1] + raw[:, 0:1]
        absolute = raw[:, 1:2]
        return residual, absolute

    def route(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.router(features))[:, :, None, None, None]

    @staticmethod
    def combine(residual: torch.Tensor, absolute: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return weight * residual + (1.0 - weight) * absolute


class IndependentDualBranchOperator(nn.Module):
    """Independent residual/absolute operators with the same observable router."""

    def __init__(
        self,
        residual_backbone: nn.Module,
        absolute_backbone: nn.Module,
        router_features: int,
        router_hidden: int = 12,
    ):
        super().__init__()
        self.residual_backbone = residual_backbone
        self.absolute_backbone = absolute_backbone
        self.router = nn.Sequential(
            nn.Linear(int(router_features), int(router_hidden)),
            nn.GELU(),
            nn.Linear(int(router_hidden), 1),
        )
        final = self.router[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def experts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x[:, 0:1] + self.residual_backbone(x)
        absolute = self.absolute_backbone(x)
        return residual, absolute

    def route(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.router(features))[:, :, None, None, None]

    @staticmethod
    def combine(residual: torch.Tensor, absolute: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return weight * residual + (1.0 - weight) * absolute


def make_model(
    name: str,
    config: dict,
    in_channels: int,
    residual: bool = True,
    gate_config: dict | None = None,
) -> nn.Module:
    if name == "unet":
        backbone = SmallUNet3D(in_channels, int(config["base_channels"]))
    elif name == "fno":
        try:
            from neuralop.models import FNO
        except ImportError as exc:
            raise RuntimeError("Install neuraloperator before constructing the FNO model") from exc
        backbone = FNO(
            n_modes=tuple(int(value) for value in config["n_modes"]),
            in_channels=in_channels,
            out_channels=1,
            hidden_channels=int(config["hidden_channels"]),
            n_layers=int(config["n_layers"]),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Reconstructor(
        backbone,
        residual_channel=0 if residual else None,
        gate=make_gate(gate_config),
    )


def make_dual_branch_model(
    config: dict,
    in_channels: int,
    router_features: int,
    router_hidden: int = 12,
    expert_sharing: str = "shared",
) -> DualBranchOperator | IndependentDualBranchOperator:
    try:
        from neuralop.models import FNO
    except ImportError as exc:
        raise RuntimeError("Install neuraloperator before constructing the FNO model") from exc
    kwargs = {
        "n_modes": tuple(int(value) for value in config["n_modes"]),
        "in_channels": in_channels,
        "hidden_channels": int(config["hidden_channels"]),
        "n_layers": int(config["n_layers"]),
    }
    if expert_sharing == "shared":
        backbone = FNO(out_channels=2, **kwargs)
        return DualBranchOperator(
            backbone,
            router_features=router_features,
            router_hidden=router_hidden,
        )
    if expert_sharing == "independent":
        return IndependentDualBranchOperator(
            FNO(out_channels=1, **kwargs),
            FNO(out_channels=1, **kwargs),
            router_features=router_features,
            router_hidden=router_hidden,
        )
    raise ValueError(f"Unknown expert sharing mode: {expert_sharing}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
