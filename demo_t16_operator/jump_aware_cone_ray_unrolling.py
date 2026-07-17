"""Minimal jump-aware cone-ray unrolling candidate for BOST.

This module is an intentionally small research candidate, not a claimed
state-of-the-art method.  It parameterizes one interface as an upstream field,
a downstream jump field, and a level set.  The cone-ray forward consumes an
exactly closing discrete gradient split so that the reported jump component
cannot silently disagree with the scalar field used by the inverse problem.

The optimization entry point accepts observations and operators only.  It has
no evaluation target or ground-truth argument.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import nn

from .psu_b0_reconstruction_interface import finite_difference_gradient


GradientMap = Callable[[torch.Tensor], torch.Tensor]
ScalarAdjoint = Callable[[torch.Tensor], torch.Tensor]


def voxel_operator_gradient_forward(
    operator: nn.Module,
    gradient_xyz: torch.Tensor,
) -> torch.Tensor:
    """Apply the public factors of a PSU voxel operator to a supplied gradient.

    This bypasses only the scalar finite-difference stage.  It retains the
    operator's trilinear sampling, camera projections, line integration, and
    physical scale, which lets JACRU expose an exactly closing jump-gradient
    split without maintaining a second cone-ray implementation.
    """

    required = (
        "trilinear_interpolation",
        "projection_u",
        "projection_v",
        "ray_scale",
        "grid_shape",
    )
    missing = [name for name in required if not hasattr(operator, name)]
    if missing:
        raise TypeError(f"operator is missing required gradient factors: {missing}")
    gradient = torch.as_tensor(gradient_xyz)
    if gradient.ndim != 5 or gradient.shape[1] != 3:
        raise ValueError("gradient_xyz must have shape [batch,3,z,y,x]")
    if tuple(gradient.shape[-3:]) != tuple(operator.grid_shape):
        raise ValueError("gradient_xyz must match the operator grid_shape")
    sampled = operator.trilinear_interpolation(gradient)
    projection_u = operator.projection_u.to(sampled)
    projection_v = operator.projection_v.to(sampled)
    ray_scale = operator.ray_scale.to(sampled)
    u = torch.einsum("bcrs,rc->brs", sampled, projection_u)
    v = torch.einsum("bcrs,rc->brs", sampled, projection_v)
    projected = torch.stack((u.sum(dim=-1), v.sum(dim=-1)), dim=-1)
    output = projected * ray_scale[None, :, None]
    if torch.any(~torch.isfinite(output)):
        raise RuntimeError("voxel operator gradient forward produced non-finite output")
    return output


@dataclass(frozen=True)
class JumpAwareConfig:
    """Frozen parameterization, regularization, and block-update budget."""

    grid_shape: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    epsilon: float = 0.12
    outer_steps: int = 16
    field_updates_per_outer: int = 1
    interface_updates_per_outer: int = 1
    bias_updates_per_outer: int = 0
    field_learning_rate: float = 2e-2
    interface_learning_rate: float = 1e-2
    bias_learning_rate: float = 1e-2
    seed: int = 0
    initial_phase_mode: str = "fixed_x"
    initial_gate_logit: float = 0.0
    initial_jump_amplitude: float = 0.05
    deployment_gate_threshold: float = 0.5
    minimum_jump_rms: float = 1e-4
    adjoint_initialization_scale: float = 1.0
    learn_upstream_field: bool = True
    data_floor: float = 1e-8
    data_weight: float = 1.0
    side_smoothness_weight: float = 1e-4
    jump_smoothness_weight: float = 1e-4
    jump_amplitude_weight: float = 1e-5
    eikonal_weight: float = 1e-3
    curvature_weight: float = 1e-4
    interface_localization_weight: float = 1e-3
    gate_sparsity_weight: float = 1e-4
    camera_bias_weight: float = 1e-3
    maximum_gradient_norm: float | None = 10.0

    def validated(self) -> "JumpAwareConfig":
        shape = tuple(int(value) for value in self.grid_shape)
        if len(shape) != 3 or any(value < 3 for value in shape):
            raise ValueError("grid_shape must contain three dimensions of at least three")
        if len(self.spacing_xyz) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.spacing_xyz
        ):
            raise ValueError("spacing_xyz must contain three positive finite values")
        positive = {
            "epsilon": self.epsilon,
            "outer_steps": self.outer_steps,
            "field_updates_per_outer": self.field_updates_per_outer,
            "interface_updates_per_outer": self.interface_updates_per_outer,
            "field_learning_rate": self.field_learning_rate,
            "interface_learning_rate": self.interface_learning_rate,
            "bias_learning_rate": self.bias_learning_rate,
            "data_floor": self.data_floor,
        }
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in positive.values()
        ):
            raise ValueError("budgets, learning rates, epsilon, and data floor must be positive")
        if int(self.bias_updates_per_outer) < 0:
            raise ValueError("bias_updates_per_outer must be nonnegative")
        if self.initial_phase_mode not in {"fixed_x", "random_plane"}:
            raise ValueError("initial_phase_mode must be 'fixed_x' or 'random_plane'")
        if not 0.0 < float(self.deployment_gate_threshold) < 1.0:
            raise ValueError("deployment_gate_threshold must lie strictly inside (0,1)")
        nonnegative = (
            self.minimum_jump_rms,
            self.data_weight,
            self.side_smoothness_weight,
            self.jump_smoothness_weight,
            self.jump_amplitude_weight,
            self.eikonal_weight,
            self.curvature_weight,
            self.interface_localization_weight,
            self.gate_sparsity_weight,
            self.camera_bias_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in nonnegative):
            raise ValueError("regularization weights and thresholds must be finite and nonnegative")
        finite = (
            self.initial_gate_logit,
            self.initial_jump_amplitude,
            self.adjoint_initialization_scale,
        )
        if any(not math.isfinite(float(value)) for value in finite):
            raise ValueError("initialization values must be finite")
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


def _initial_phase(
    config: JumpAwareConfig,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    nz, ny, nx = (int(value) for value in config.grid_shape)
    sx, sy, sz = (float(value) for value in config.spacing_xyz)
    x = (torch.arange(nx, dtype=dtype) - 0.5 * (nx - 1)) * sx
    y = (torch.arange(ny, dtype=dtype) - 0.5 * (ny - 1)) * sy
    z = (torch.arange(nz, dtype=dtype) - 0.5 * (nz - 1)) * sz
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    if config.initial_phase_mode == "fixed_x":
        base = xx + 0.05 * yy - 0.03 * zz
    else:
        normal = torch.randn((3,), generator=generator, dtype=dtype)
        normal = normal / torch.linalg.vector_norm(normal).clamp_min(1e-12)
        minimum_extent = min((nx - 1) * sx, (ny - 1) * sy, (nz - 1) * sz)
        offset = (
            torch.rand((), generator=generator, dtype=dtype) - 0.5
        ) * (0.3 * minimum_extent)
        base = normal[0] * xx + normal[1] * yy + normal[2] * zz - offset
    noise = torch.randn((nz, ny, nx), generator=generator, dtype=dtype)
    phase = base + 0.01 * min(sx, sy, sz) * noise
    return phase.reshape(1, 1, nz, ny, nx).expand(batch_size, -1, -1, -1, -1).clone().to(device)


def _curvature_energy(values: torch.Tensor, spacing_xyz: tuple[float, float, float]) -> torch.Tensor:
    energies = []
    for axis, spacing in zip((-1, -2, -3), spacing_xyz, strict=True):
        second = torch.diff(values, n=2, dim=axis) / float(spacing) ** 2
        energies.append(torch.mean(second.square()))
    return torch.stack(energies).mean()


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = torch.broadcast_to(weight.to(values), values.shape)
    return torch.sum(values * expanded) / expanded.sum().clamp_min(1.0)


@dataclass(frozen=True)
class DiscreteGradientSplit:
    total: torch.Tensor
    smooth_side: torch.Tensor
    jump: torch.Tensor
    closure_rms: torch.Tensor


class JumpAwareConeRayField(nn.Module):
    """One-interface upstream-plus-jump scalar field with exact gradient closure."""

    def __init__(
        self,
        config: JumpAwareConfig,
        *,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        support: torch.Tensor | None = None,
        initial_upstream: torch.Tensor | None = None,
        camera_group_count: int = 0,
    ) -> None:
        super().__init__()
        self.config = config.validated()
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if int(camera_group_count) < 0:
            raise ValueError("camera_group_count must be nonnegative")
        target_device = torch.device(device)
        shape = tuple(int(value) for value in self.config.grid_shape)
        support_values = _canonical_support(
            support,
            batch_size=int(batch_size),
            grid_shape=shape,
            dtype=dtype,
            device=target_device,
        )
        if initial_upstream is None:
            upstream = torch.zeros_like(support_values)
        else:
            upstream = _canonical_volume(
                initial_upstream,
                batch_size=int(batch_size),
                grid_shape=shape,
                dtype=dtype,
                device=target_device,
                name="initial_upstream",
            )
        self.register_buffer("support", support_values.clone())
        self.upstream = nn.Parameter(
            (upstream * support_values).clone(),
            requires_grad=bool(self.config.learn_upstream_field),
        )
        self.jump_field = nn.Parameter(
            torch.full_like(upstream, float(self.config.initial_jump_amplitude))
            * support_values
        )
        self.phase_field = nn.Parameter(
            _initial_phase(
                self.config,
                batch_size=int(batch_size),
                dtype=dtype,
                device=target_device,
            )
        )
        self.gate_logit = nn.Parameter(
            torch.full(
                (int(batch_size), 1),
                float(self.config.initial_gate_logit),
                dtype=dtype,
                device=target_device,
            )
        )
        self.camera_group_count = int(camera_group_count)
        if self.camera_group_count:
            self.camera_bias = nn.Parameter(
                torch.zeros(
                    (int(batch_size), self.camera_group_count, 2),
                    dtype=dtype,
                    device=target_device,
                )
            )
        else:
            self.register_parameter("camera_bias", None)

    def gate_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def soft_heaviside(self) -> torch.Tensor:
        return torch.sigmoid(self.phase_field / float(self.config.epsilon))

    def jump_rms(self) -> torch.Tensor:
        contribution = self.gate_probability()[:, :, None, None, None] * self.soft_heaviside()
        contribution = contribution * self.jump_field * self.support
        return torch.sqrt(torch.mean(contribution.square(), dim=(-3, -2, -1))).reshape(-1)

    def active_gate(self) -> torch.Tensor:
        return (self.gate_probability().reshape(-1) >= float(self.config.deployment_gate_threshold)) & (
            self.jump_rms() >= float(self.config.minimum_jump_rms)
        )

    def mixing_fraction(self, *, hard: bool = False) -> torch.Tensor:
        if hard:
            gate = self.active_gate().to(self.phase_field)[:, None, None, None, None]
            return gate * (self.phase_field >= 0.0).to(self.phase_field)
        return self.gate_probability()[:, :, None, None, None] * self.soft_heaviside()

    def volume(self, *, hard: bool = False) -> torch.Tensor:
        upstream = self.upstream * self.support
        jump = self.jump_field * self.support
        return upstream + self.mixing_fraction(hard=hard) * jump

    def downstream(self) -> torch.Tensor:
        return (self.upstream + self.jump_field) * self.support

    def gradient_split(self, *, hard: bool = False) -> DiscreteGradientSplit:
        upstream = (self.upstream * self.support)[:, 0]
        jump = (self.jump_field * self.support)[:, 0]
        mixing = self.mixing_fraction(hard=hard)[:, 0]
        volume = upstream + mixing * jump
        total = finite_difference_gradient(volume, spacing_xyz=self.config.spacing_xyz)
        upstream_gradient = finite_difference_gradient(
            upstream,
            spacing_xyz=self.config.spacing_xyz,
        )
        jump_gradient = finite_difference_gradient(
            jump,
            spacing_xyz=self.config.spacing_xyz,
        )
        smooth = upstream_gradient + mixing[:, None] * jump_gradient
        discontinuity = total - smooth
        closure = torch.sqrt(torch.mean((total - smooth - discontinuity).square()))
        return DiscreteGradientSplit(
            total=total,
            smooth_side=smooth,
            jump=discontinuity,
            closure_rms=closure,
        )

    def centered_camera_bias(self) -> torch.Tensor | None:
        if self.camera_bias is None:
            return None
        return self.camera_bias - self.camera_bias.mean(dim=1, keepdim=True)

    def bias_for_rays(self, ray_group_index: torch.Tensor) -> torch.Tensor:
        centered = self.centered_camera_bias()
        if centered is None:
            raise ValueError("camera bias was not enabled")
        groups = torch.as_tensor(
            ray_group_index,
            dtype=torch.int64,
            device=centered.device,
        ).reshape(-1)
        if torch.any(groups < 0) or torch.any(groups >= self.camera_group_count):
            raise ValueError("ray_group_index lies outside the configured camera groups")
        return centered[:, groups, :]

    def apply_gauge_constraints(self) -> None:
        with torch.no_grad():
            self.upstream.mul_(self.support)
            self.jump_field.mul_(self.support)


@dataclass(frozen=True)
class JumpAwareObjective:
    total: torch.Tensor
    data: torch.Tensor
    side_smoothness: torch.Tensor
    jump_smoothness: torch.Tensor
    jump_amplitude: torch.Tensor
    eikonal: torch.Tensor
    curvature: torch.Tensor
    interface_localization: torch.Tensor
    gate_sparsity: torch.Tensor
    camera_bias: torch.Tensor
    closure_rms: torch.Tensor


def jump_aware_objective(
    model: JumpAwareConeRayField,
    observation: torch.Tensor,
    *,
    gradient_forward: GradientMap,
    ray_group_index: torch.Tensor | None = None,
    observation_weight: torch.Tensor | None = None,
) -> JumpAwareObjective:
    measured = torch.as_tensor(
        observation,
        dtype=model.upstream.dtype,
        device=model.upstream.device,
    ).detach()
    if torch.any(~torch.isfinite(measured)):
        raise ValueError("observation must contain only finite values")
    split = model.gradient_split(hard=False)
    prediction = gradient_forward(split.total)
    if prediction.shape != measured.shape:
        raise ValueError("gradient_forward output must have the same shape as observation")
    if model.camera_bias is not None:
        if ray_group_index is None:
            raise ValueError("ray_group_index is required when camera bias is enabled")
        if prediction.shape[-1] != 2:
            raise ValueError("camera bias requires a two-component UV observation")
        prediction = prediction + model.bias_for_rays(ray_group_index)
    if torch.any(~torch.isfinite(prediction)):
        raise RuntimeError("gradient_forward produced non-finite values")
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

    upstream = (model.upstream * model.support)[:, 0]
    downstream = ((model.upstream + model.jump_field) * model.support)[:, 0]
    grad_upstream = finite_difference_gradient(
        upstream,
        spacing_xyz=model.config.spacing_xyz,
    )
    grad_downstream = finite_difference_gradient(
        downstream,
        spacing_xyz=model.config.spacing_xyz,
    )
    heaviside = model.soft_heaviside()
    support = model.support
    side_smoothness = _weighted_mean(
        grad_upstream.square(),
        (1.0 - heaviside) * support,
    ) + _weighted_mean(
        grad_downstream.square(),
        heaviside * support,
    )
    jump_gradient = finite_difference_gradient(
        (model.jump_field * model.support)[:, 0],
        spacing_xyz=model.config.spacing_xyz,
    )
    jump_smoothness = _weighted_mean(jump_gradient.square(), support)
    jump_amplitude = _weighted_mean(model.jump_field.square(), support)

    phase = model.phase_field[:, 0]
    phase_gradient = finite_difference_gradient(
        phase,
        spacing_xyz=model.config.spacing_xyz,
    )
    phase_norm = torch.sqrt(torch.sum(phase_gradient.square(), dim=1) + 1e-12)
    eikonal = _weighted_mean((phase_norm - 1.0).square(), support[:, 0])
    curvature = _curvature_energy(phase, model.config.spacing_xyz)
    interface_band = (4.0 * heaviside * (1.0 - heaviside)).clamp(0.0, 1.0)
    jump_energy = torch.mean(split.jump.square())
    interface_localization = torch.mean(
        split.jump.square() * (1.0 - interface_band)
    ) / jump_energy.detach().clamp_min(float(model.config.data_floor))
    gate_sparsity = model.gate_probability().mean()
    centered_bias = model.centered_camera_bias()
    camera_bias = (
        centered_bias.square().mean()
        if centered_bias is not None
        else model.upstream.sum() * 0.0
    )
    total = (
        float(model.config.data_weight) * data
        + float(model.config.side_smoothness_weight) * side_smoothness
        + float(model.config.jump_smoothness_weight) * jump_smoothness
        + float(model.config.jump_amplitude_weight) * jump_amplitude
        + float(model.config.eikonal_weight) * eikonal
        + float(model.config.curvature_weight) * curvature
        + float(model.config.interface_localization_weight) * interface_localization
        + float(model.config.gate_sparsity_weight) * gate_sparsity
        + float(model.config.camera_bias_weight) * camera_bias
    )
    return JumpAwareObjective(
        total=total,
        data=data,
        side_smoothness=side_smoothness,
        jump_smoothness=jump_smoothness,
        jump_amplitude=jump_amplitude,
        eikonal=eikonal,
        curvature=curvature,
        interface_localization=interface_localization,
        gate_sparsity=gate_sparsity,
        camera_bias=camera_bias,
        closure_rms=split.closure_rms,
    )


@dataclass(frozen=True)
class JumpAwareSnapshot:
    outer_step: int
    block: str
    total: float
    data: float
    side_smoothness: float
    jump_smoothness: float
    jump_amplitude: float
    eikonal: float
    curvature: float
    interface_localization: float
    gate_sparsity: float
    camera_bias: float
    closure_rms: float


@dataclass(frozen=True)
class JumpAwareResult:
    model: JumpAwareConeRayField
    soft_volume: torch.Tensor
    hard_volume: torch.Tensor
    soft_observation: torch.Tensor
    hard_observation: torch.Tensor
    soft_gradient_split: DiscreteGradientSplit
    hard_gradient_split: DiscreteGradientSplit
    history: tuple[JumpAwareSnapshot, ...]
    active_gate: torch.Tensor
    gate_probability: torch.Tensor
    jump_rms: torch.Tensor
    optimization_forward_evaluations: int
    implicit_data_vjp_evaluations: int
    reporting_forward_evaluations: int
    total_forward_evaluations: int
    adjoint_evaluations: int
    seed: int


def _snapshot(
    objective: JumpAwareObjective,
    *,
    outer_step: int,
    block: str,
) -> JumpAwareSnapshot:
    return JumpAwareSnapshot(
        outer_step=int(outer_step),
        block=str(block),
        total=float(objective.total.detach()),
        data=float(objective.data.detach()),
        side_smoothness=float(objective.side_smoothness.detach()),
        jump_smoothness=float(objective.jump_smoothness.detach()),
        jump_amplitude=float(objective.jump_amplitude.detach()),
        eikonal=float(objective.eikonal.detach()),
        curvature=float(objective.curvature.detach()),
        interface_localization=float(objective.interface_localization.detach()),
        gate_sparsity=float(objective.gate_sparsity.detach()),
        camera_bias=float(objective.camera_bias.detach()),
        closure_rms=float(objective.closure_rms.detach()),
    )


def optimize_jump_aware_cone_ray(
    observation: torch.Tensor,
    *,
    gradient_forward: GradientMap,
    config: JumpAwareConfig,
    scalar_adjoint: ScalarAdjoint | None = None,
    support: torch.Tensor | None = None,
    initial_upstream: torch.Tensor | None = None,
    ray_group_index: torch.Tensor | None = None,
    camera_group_count: int = 0,
    observation_weight: torch.Tensor | None = None,
) -> JumpAwareResult:
    """Run a fixed observable-only block-coordinate unrolling budget."""

    config = config.validated()
    measured = torch.as_tensor(observation)
    if not measured.is_floating_point() or measured.ndim < 2 or measured.shape[0] < 1:
        raise ValueError("observation must be a floating tensor with a batch dimension")
    if torch.any(~torch.isfinite(measured)):
        raise ValueError("observation must contain only finite values")
    batch_size = int(measured.shape[0])
    shape = tuple(int(value) for value in config.grid_shape)
    adjoint_evaluations = 0
    if initial_upstream is None and scalar_adjoint is not None:
        with torch.no_grad():
            initial_upstream = scalar_adjoint(measured)
        adjoint_evaluations = 1
        initial_upstream = _canonical_volume(
            initial_upstream,
            batch_size=batch_size,
            grid_shape=shape,
            dtype=measured.dtype,
            device=measured.device,
            name="adjoint initialization",
        ) * float(config.adjoint_initialization_scale)
    if int(camera_group_count) == 0 and ray_group_index is not None:
        groups = torch.as_tensor(ray_group_index, dtype=torch.int64).reshape(-1)
        if groups.numel() < 1 or torch.any(groups < 0):
            raise ValueError("ray_group_index must contain nonnegative group labels")
        camera_group_count = int(groups.max()) + 1
    if int(camera_group_count) > 0 and ray_group_index is None:
        raise ValueError("ray_group_index is required when camera groups are enabled")

    model = JumpAwareConeRayField(
        config,
        batch_size=batch_size,
        dtype=measured.dtype,
        device=measured.device,
        support=support,
        initial_upstream=initial_upstream,
        camera_group_count=int(camera_group_count),
    )
    field_parameters = [model.jump_field]
    if model.upstream.requires_grad:
        field_parameters.insert(0, model.upstream)
    field_optimizer = torch.optim.Adam(
        field_parameters,
        lr=float(config.field_learning_rate),
    )
    interface_optimizer = torch.optim.Adam(
        [model.phase_field, model.gate_logit],
        lr=float(config.interface_learning_rate),
    )
    bias_optimizer = None
    if model.camera_bias is not None and int(config.bias_updates_per_outer) > 0:
        bias_optimizer = torch.optim.Adam(
            [model.camera_bias],
            lr=float(config.bias_learning_rate),
        )
    all_optimizers = [field_optimizer, interface_optimizer]
    if bias_optimizer is not None:
        all_optimizers.append(bias_optimizer)

    optimization_forward_evaluations = 0

    def counted_forward(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal optimization_forward_evaluations
        optimization_forward_evaluations += 1
        return gradient_forward(gradient)

    history: list[JumpAwareSnapshot] = []
    schedule: list[tuple[str, torch.optim.Optimizer, list[nn.Parameter]]] = []
    schedule.extend(
        ("field", field_optimizer, field_parameters)
        for _ in range(int(config.field_updates_per_outer))
    )
    schedule.extend(
        ("interface", interface_optimizer, [model.phase_field, model.gate_logit])
        for _ in range(int(config.interface_updates_per_outer))
    )
    if bias_optimizer is not None:
        schedule.extend(
            ("bias", bias_optimizer, [model.camera_bias])
            for _ in range(int(config.bias_updates_per_outer))
        )

    for outer_step in range(1, int(config.outer_steps) + 1):
        for block, optimizer, block_parameters in schedule:
            for active_optimizer in all_optimizers:
                active_optimizer.zero_grad(set_to_none=True)
            objective = jump_aware_objective(
                model,
                measured,
                gradient_forward=counted_forward,
                ray_group_index=ray_group_index,
                observation_weight=observation_weight,
            )
            if not torch.isfinite(objective.total):
                raise RuntimeError(
                    f"non-finite objective at outer step {outer_step}, block {block}"
                )
            objective.total.backward()
            if config.maximum_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    block_parameters,
                    max_norm=float(config.maximum_gradient_norm),
                )
            optimizer.step()
            model.apply_gauge_constraints()
            if any(
                torch.any(~torch.isfinite(parameter))
                for parameter in model.parameters()
            ):
                raise RuntimeError(
                    f"non-finite parameter at outer step {outer_step}, block {block}"
                )
            history.append(
                _snapshot(objective, outer_step=outer_step, block=block)
            )

    expected = int(config.outer_steps) * len(schedule)
    if optimization_forward_evaluations != expected:
        raise RuntimeError("fixed optimization forward budget was not respected")
    model.eval()
    with torch.no_grad():
        soft_split = model.gradient_split(hard=False)
        hard_split = model.gradient_split(hard=True)
        soft_observation = gradient_forward(soft_split.total)
        hard_observation = gradient_forward(hard_split.total)
        if model.camera_bias is not None:
            assert ray_group_index is not None
            if soft_observation.shape[-1] != 2 or hard_observation.shape[-1] != 2:
                raise ValueError("camera bias requires a two-component UV observation")
            bias = model.bias_for_rays(ray_group_index)
            soft_observation = soft_observation + bias
            hard_observation = hard_observation + bias
        soft_volume = model.volume(hard=False).detach().clone()
        hard_volume = model.volume(hard=True).detach().clone()
        active = model.active_gate().detach().clone()
        gate_probability = model.gate_probability().detach().clone()
        jump_rms = model.jump_rms().detach().clone()
    reporting_forward_evaluations = 2
    return JumpAwareResult(
        model=model,
        soft_volume=soft_volume,
        hard_volume=hard_volume,
        soft_observation=soft_observation.detach().clone(),
        hard_observation=hard_observation.detach().clone(),
        soft_gradient_split=DiscreteGradientSplit(
            total=soft_split.total.detach().clone(),
            smooth_side=soft_split.smooth_side.detach().clone(),
            jump=soft_split.jump.detach().clone(),
            closure_rms=soft_split.closure_rms.detach().clone(),
        ),
        hard_gradient_split=DiscreteGradientSplit(
            total=hard_split.total.detach().clone(),
            smooth_side=hard_split.smooth_side.detach().clone(),
            jump=hard_split.jump.detach().clone(),
            closure_rms=hard_split.closure_rms.detach().clone(),
        ),
        history=tuple(history),
        active_gate=active,
        gate_probability=gate_probability,
        jump_rms=jump_rms,
        optimization_forward_evaluations=int(optimization_forward_evaluations),
        # Each scalar objective backward traverses the differentiable linear
        # data map once.  Record that reverse traversal explicitly instead of
        # hiding it behind autograd when comparing solver budgets.
        implicit_data_vjp_evaluations=int(optimization_forward_evaluations),
        reporting_forward_evaluations=int(reporting_forward_evaluations),
        total_forward_evaluations=int(
            optimization_forward_evaluations + reporting_forward_evaluations
        ),
        adjoint_evaluations=int(adjoint_evaluations),
        seed=int(config.seed),
    )


__all__ = [
    "DiscreteGradientSplit",
    "JumpAwareConfig",
    "JumpAwareConeRayField",
    "JumpAwareObjective",
    "JumpAwareResult",
    "JumpAwareSnapshot",
    "jump_aware_objective",
    "optimize_jump_aware_cone_ray",
    "voxel_operator_gradient_forward",
]
