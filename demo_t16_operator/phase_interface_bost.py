"""Minimal falsifiable phase/interface reconstruction candidate for BOST.

The candidate separates a smooth background from at most two explicit level
set fields.  Its inverse routine consumes observations plus injectable linear
forward/adjoint callables.  It deliberately contains no evaluation target and
makes no performance claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import nn

from .psu_b0_reconstruction_interface import finite_difference_gradient


TensorMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PhaseInterfaceConfig:
    """Frozen model, regularization, and optimization budget."""

    grid_shape: tuple[int, int, int]
    max_interfaces: int = 2
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    epsilon: float = 0.12
    optimization_steps: int = 32
    learning_rate: float = 3e-2
    seed: int = 0
    initial_gate_logit: float = -1.5
    deployment_gate_threshold: float = 0.5
    minimum_interface_rms: float = 1e-4
    adjoint_initialization_scale: float = 1.0
    data_floor: float = 1e-8
    data_weight: float = 1.0
    background_smoothness_weight: float = 1e-3
    phase_smoothness_weight: float = 1e-4
    eikonal_weight: float = 1e-3
    gate_sparsity_weight: float = 1e-3
    amplitude_weight: float = 1e-5
    maximum_gradient_norm: float | None = 10.0

    def validated(self) -> "PhaseInterfaceConfig":
        shape = tuple(int(value) for value in self.grid_shape)
        if len(shape) != 3 or any(value < 3 for value in shape):
            raise ValueError("grid_shape must contain three dimensions of at least three")
        if self.max_interfaces not in {0, 1, 2}:
            raise ValueError("max_interfaces must be zero, one, or two")
        if len(self.spacing_xyz) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.spacing_xyz
        ):
            raise ValueError("spacing_xyz must contain three positive finite values")
        positive = {
            "epsilon": self.epsilon,
            "optimization_steps": self.optimization_steps,
            "learning_rate": self.learning_rate,
            "data_floor": self.data_floor,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive.values()):
            raise ValueError("epsilon, budget, learning rate, and data floor must be positive")
        if not 0.0 < float(self.deployment_gate_threshold) < 1.0:
            raise ValueError("deployment_gate_threshold must lie strictly inside (0,1)")
        nonnegative = (
            self.minimum_interface_rms,
            self.data_weight,
            self.background_smoothness_weight,
            self.phase_smoothness_weight,
            self.eikonal_weight,
            self.gate_sparsity_weight,
            self.amplitude_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in nonnegative):
            raise ValueError("regularization weights and thresholds must be finite and nonnegative")
        if not math.isfinite(float(self.initial_gate_logit)):
            raise ValueError("initial_gate_logit must be finite")
        if not math.isfinite(float(self.adjoint_initialization_scale)):
            raise ValueError("adjoint_initialization_scale must be finite")
        if self.maximum_gradient_norm is not None and (
            not math.isfinite(float(self.maximum_gradient_norm))
            or float(self.maximum_gradient_norm) <= 0.0
        ):
            raise ValueError("maximum_gradient_norm must be positive when supplied")
        return self


def _canonical_volume(
    value: torch.Tensor,
    *,
    batch_size: int,
    grid_shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 4:
        tensor = tensor[:, None]
    if tensor.shape != (batch_size, 1, *grid_shape):
        raise ValueError(f"{name} must have shape [batch,1,z,y,x] or [batch,z,y,x]")
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _canonical_support(
    support: torch.Tensor | None,
    *,
    batch_size: int,
    grid_shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if support is None:
        return torch.ones((batch_size, 1, *grid_shape), dtype=dtype, device=device)
    values = torch.as_tensor(support, dtype=dtype, device=device)
    if values.shape == grid_shape:
        values = values.reshape(1, 1, *grid_shape)
    elif values.ndim == 4 and values.shape[-3:] == grid_shape:
        values = values[:, None]
    if values.shape == (1, 1, *grid_shape):
        values = values.expand(batch_size, -1, -1, -1, -1)
    if values.shape != (batch_size, 1, *grid_shape):
        raise ValueError("support must match the spatial grid and broadcast over the batch")
    if torch.any(~torch.isfinite(values)) or torch.any((values < 0.0) | (values > 1.0)):
        raise ValueError("support must be finite and lie in [0,1]")
    return values


def _initial_phase_fields(
    config: PhaseInterfaceConfig,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    count = int(config.max_interfaces)
    shape = tuple(int(value) for value in config.grid_shape)
    if count == 0:
        return torch.empty((batch_size, 0, *shape), dtype=dtype, device=device)
    nz, ny, nx = shape
    sx, sy, sz = (float(value) for value in config.spacing_xyz)
    x = (torch.arange(nx, dtype=dtype) - 0.5 * (nx - 1)) * sx
    y = (torch.arange(ny, dtype=dtype) - 0.5 * (ny - 1)) * sy
    z = (torch.arange(nz, dtype=dtype) - 0.5 * (nz - 1)) * sz
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    extent_x = max(0.5 * (nx - 1) * sx, sx)
    offsets = (0.0,) if count == 1 else (-0.22 * extent_x, 0.22 * extent_x)
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    noise_scale = 0.015 * min(sx, sy, sz)
    fields = []
    for index, offset in enumerate(offsets):
        base = xx - float(offset)
        if index == 1:
            base = base + 0.08 * yy - 0.05 * zz
        noise = torch.randn(shape, generator=generator, dtype=dtype)
        fields.append(base + noise_scale * noise)
    phases = torch.stack(fields, dim=0)[None].expand(batch_size, -1, -1, -1, -1)
    return phases.clone().to(device=device)


class PhaseInterfaceField(nn.Module):
    """Represent ``q = b + sum gate * amplitude * tanh(phi / epsilon)``."""

    def __init__(
        self,
        config: PhaseInterfaceConfig,
        *,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        support: torch.Tensor | None = None,
        initial_background: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config.validated()
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        target_device = torch.device(device)
        shape = tuple(int(value) for value in self.config.grid_shape)
        support_values = _canonical_support(
            support,
            batch_size=int(batch_size),
            grid_shape=shape,
            dtype=dtype,
            device=target_device,
        )
        if initial_background is None:
            background = torch.zeros_like(support_values)
        else:
            background = _canonical_volume(
                initial_background,
                batch_size=int(batch_size),
                grid_shape=shape,
                dtype=dtype,
                device=target_device,
                name="initial_background",
            )
        count = int(self.config.max_interfaces)
        self.register_buffer("support", support_values.clone())
        self.background = nn.Parameter((background * support_values).clone())
        self.gate_logits = nn.Parameter(
            torch.full(
                (int(batch_size), count),
                float(self.config.initial_gate_logit),
                dtype=dtype,
                device=target_device,
            )
        )
        self.amplitudes = nn.Parameter(
            torch.zeros((int(batch_size), count), dtype=dtype, device=target_device)
        )
        self.phase_fields = nn.Parameter(
            _initial_phase_fields(
                self.config,
                batch_size=int(batch_size),
                dtype=dtype,
                device=target_device,
            )
        )

    @property
    def interface_count(self) -> int:
        return int(self.config.max_interfaces)

    def gate_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def soft_interface_components(self) -> torch.Tensor:
        transition = torch.tanh(self.phase_fields / float(self.config.epsilon))
        scale = self.gate_probabilities() * self.amplitudes
        return scale[:, :, None, None, None] * transition * self.support

    def interface_rms(self) -> torch.Tensor:
        components = self.soft_interface_components()
        if self.interface_count == 0:
            return self.amplitudes.clone()
        return torch.sqrt(torch.mean(components.square(), dim=(-3, -2, -1)))

    def active_interface_mask(self) -> torch.Tensor:
        return (self.gate_probabilities() >= float(self.config.deployment_gate_threshold)) & (
            self.interface_rms() >= float(self.config.minimum_interface_rms)
        )

    def forward(self, *, hard_gate: bool = False) -> torch.Tensor:
        components = self.soft_interface_components()
        if hard_gate and self.interface_count:
            components = components * self.active_interface_mask()[:, :, None, None, None]
        return (self.background * self.support) + components.sum(dim=1, keepdim=True)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = torch.broadcast_to(mask.to(values), values.shape)
    denominator = expanded.sum().clamp_min(1.0)
    return torch.sum(values * expanded) / denominator


def _curvature_energy(values: torch.Tensor, spacing_xyz: tuple[float, float, float]) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    energies = []
    for axis, spacing in zip((-1, -2, -3), spacing_xyz, strict=True):
        second = torch.diff(values, n=2, dim=axis) / float(spacing) ** 2
        energies.append(torch.mean(second.square()))
    return torch.stack(energies).mean()


@dataclass(frozen=True)
class PhaseInterfaceObjective:
    total: torch.Tensor
    data: torch.Tensor
    background_smoothness: torch.Tensor
    phase_smoothness: torch.Tensor
    eikonal: torch.Tensor
    gate_sparsity: torch.Tensor
    amplitude: torch.Tensor


def phase_interface_objective(
    model: PhaseInterfaceField,
    observation: torch.Tensor,
    *,
    forward: TensorMap,
    observation_weight: torch.Tensor | None = None,
) -> PhaseInterfaceObjective:
    """Build one differentiable, observation-only inverse objective."""

    measured = torch.as_tensor(
        observation,
        dtype=model.background.dtype,
        device=model.background.device,
    ).detach()
    if torch.any(~torch.isfinite(measured)):
        raise ValueError("observation must contain only finite values")
    prediction = forward(model(hard_gate=False))
    if prediction.shape != measured.shape:
        raise ValueError("forward output must have the same shape as observation")
    if torch.any(~torch.isfinite(prediction)):
        raise RuntimeError("forward produced non-finite values")
    if observation_weight is None:
        weight = torch.ones_like(measured)
    else:
        weight = torch.as_tensor(
            observation_weight,
            dtype=measured.dtype,
            device=measured.device,
        )
        try:
            weight = torch.broadcast_to(weight, measured.shape)
        except RuntimeError as exc:
            raise ValueError("observation_weight must broadcast to observation") from exc
        if torch.any(~torch.isfinite(weight)) or torch.any(weight < 0.0):
            raise ValueError("observation_weight must be finite and nonnegative")
    residual = (prediction - measured) * weight
    scale = torch.mean((measured * weight).square()).detach().clamp_min(
        float(model.config.data_floor)
    )
    data = torch.mean(residual.square()) / scale

    background_gradient = finite_difference_gradient(
        (model.background * model.support)[:, 0],
        spacing_xyz=model.config.spacing_xyz,
    )
    background_smoothness = _masked_mean(
        background_gradient.square(),
        model.support,
    )

    if model.interface_count:
        phases = model.phase_fields.reshape(-1, *model.config.grid_shape)
        phase_gradient = finite_difference_gradient(
            phases,
            spacing_xyz=model.config.spacing_xyz,
        )
        phase_norm = torch.sqrt(torch.sum(phase_gradient.square(), dim=1) + 1e-12)
        phase_mask = model.support[:, 0].repeat_interleave(model.interface_count, dim=0)
        eikonal = _masked_mean((phase_norm - 1.0).square(), phase_mask)
        phase_smoothness = _curvature_energy(phases, model.config.spacing_xyz)
        gate_sparsity = model.gate_probabilities().sum(dim=1).mean()
        amplitude = torch.mean(model.amplitudes.square())
    else:
        zero = model.background.sum() * 0.0
        eikonal = zero
        phase_smoothness = zero
        gate_sparsity = zero
        amplitude = zero

    total = (
        float(model.config.data_weight) * data
        + float(model.config.background_smoothness_weight) * background_smoothness
        + float(model.config.phase_smoothness_weight) * phase_smoothness
        + float(model.config.eikonal_weight) * eikonal
        + float(model.config.gate_sparsity_weight) * gate_sparsity
        + float(model.config.amplitude_weight) * amplitude
    )
    return PhaseInterfaceObjective(
        total=total,
        data=data,
        background_smoothness=background_smoothness,
        phase_smoothness=phase_smoothness,
        eikonal=eikonal,
        gate_sparsity=gate_sparsity,
        amplitude=amplitude,
    )


@dataclass(frozen=True)
class ObjectiveSnapshot:
    step: int
    total: float
    data: float
    background_smoothness: float
    phase_smoothness: float
    eikonal: float
    gate_sparsity: float
    amplitude: float


@dataclass(frozen=True)
class PhaseInterfaceResult:
    model: PhaseInterfaceField
    prediction: torch.Tensor
    soft_prediction: torch.Tensor
    background: torch.Tensor
    phase_fields: torch.Tensor
    amplitudes: torch.Tensor
    gate_probabilities: torch.Tensor
    active_interface_mask: torch.Tensor
    interface_rms: torch.Tensor
    history: tuple[ObjectiveSnapshot, ...]
    optimization_steps: int
    forward_evaluations: int
    adjoint_evaluations: int
    seed: int


def optimize_phase_interface_bost(
    observation: torch.Tensor,
    *,
    forward: TensorMap,
    config: PhaseInterfaceConfig,
    adjoint: TensorMap | None = None,
    support: torch.Tensor | None = None,
    initial_background: torch.Tensor | None = None,
    observation_weight: torch.Tensor | None = None,
) -> PhaseInterfaceResult:
    """Run exactly ``optimization_steps`` Adam updates on observable inputs."""

    config = config.validated()
    measured = torch.as_tensor(observation)
    if not measured.is_floating_point() or measured.ndim < 2 or measured.shape[0] < 1:
        raise ValueError("observation must be a floating tensor with a batch dimension")
    if torch.any(~torch.isfinite(measured)):
        raise ValueError("observation must contain only finite values")
    batch_size = int(measured.shape[0])
    shape = tuple(int(value) for value in config.grid_shape)
    adjoint_evaluations = 0
    if initial_background is None and adjoint is not None:
        with torch.no_grad():
            initial_background = adjoint(measured)
        adjoint_evaluations = 1
        initial_background = _canonical_volume(
            initial_background,
            batch_size=batch_size,
            grid_shape=shape,
            dtype=measured.dtype,
            device=measured.device,
            name="adjoint initialization",
        ) * float(config.adjoint_initialization_scale)

    model = PhaseInterfaceField(
        config,
        batch_size=batch_size,
        dtype=measured.dtype,
        device=measured.device,
        support=support,
        initial_background=initial_background,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    forward_evaluations = 0

    def counted_forward(volume: torch.Tensor) -> torch.Tensor:
        nonlocal forward_evaluations
        forward_evaluations += 1
        return forward(volume)

    history: list[ObjectiveSnapshot] = []
    for step in range(1, int(config.optimization_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        objective = phase_interface_objective(
            model,
            measured,
            forward=counted_forward,
            observation_weight=observation_weight,
        )
        if not torch.isfinite(objective.total):
            raise RuntimeError(f"non-finite objective at optimization step {step}")
        objective.total.backward()
        if config.maximum_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(config.maximum_gradient_norm),
            )
        optimizer.step()
        if any(torch.any(~torch.isfinite(parameter)) for parameter in model.parameters()):
            raise RuntimeError(f"non-finite parameter at optimization step {step}")
        history.append(
            ObjectiveSnapshot(
                step=step,
                total=float(objective.total.detach()),
                data=float(objective.data.detach()),
                background_smoothness=float(objective.background_smoothness.detach()),
                phase_smoothness=float(objective.phase_smoothness.detach()),
                eikonal=float(objective.eikonal.detach()),
                gate_sparsity=float(objective.gate_sparsity.detach()),
                amplitude=float(objective.amplitude.detach()),
            )
        )

    model.eval()
    with torch.no_grad():
        prediction = model(hard_gate=True).detach().clone()
        soft_prediction = model(hard_gate=False).detach().clone()
        active = model.active_interface_mask().detach().clone()
        rms = model.interface_rms().detach().clone()
    if forward_evaluations != int(config.optimization_steps):
        raise RuntimeError("fixed forward budget was not respected")
    return PhaseInterfaceResult(
        model=model,
        prediction=prediction,
        soft_prediction=soft_prediction,
        background=(model.background * model.support).detach().clone(),
        phase_fields=model.phase_fields.detach().clone(),
        amplitudes=model.amplitudes.detach().clone(),
        gate_probabilities=model.gate_probabilities().detach().clone(),
        active_interface_mask=active,
        interface_rms=rms,
        history=tuple(history),
        optimization_steps=int(config.optimization_steps),
        forward_evaluations=int(forward_evaluations),
        adjoint_evaluations=int(adjoint_evaluations),
        seed=int(config.seed),
    )


__all__ = [
    "ObjectiveSnapshot",
    "PhaseInterfaceConfig",
    "PhaseInterfaceField",
    "PhaseInterfaceObjective",
    "PhaseInterfaceResult",
    "optimize_phase_interface_bost",
    "phase_interface_objective",
]
