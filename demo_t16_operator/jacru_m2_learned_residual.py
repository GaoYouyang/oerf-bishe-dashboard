"""Laptop-sized learned residual on top of a truth-free CGLS base.

This module is an implementation component, not a novelty claim.  The
physical preparation path runs the existing matrix-free CGLS baseline and
lifts its terminal data residual back to one voxel field per camera.  The
learned path then applies shared per-camera weights and symmetric set pooling,
so jointly permuting camera lifts, poses, and masks does not change the model.

The final convolution is initialized to zero.  Consequently a newly created
model returns the CGLS base exactly, while later corrections remain inside the
declared support and a fixed absolute magnitude bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .interface_baselines import cgls_baseline
from .jacru_synthetic_fixture import JACRUInferencePayload


Tensor = torch.Tensor


def _group_count(channels: int) -> int:
    return next(
        value
        for value in range(min(4, int(channels)), 0, -1)
        if int(channels) % value == 0
    )


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


@dataclass(frozen=True)
class JACRUM2Batch:
    """Truth-free tensors consumed by :class:`JACRUM2LearnedResidual`."""

    base_field: Tensor
    support: Tensor
    adjoint_lifted_data_residual: Tensor
    camera_pose_features: Tensor
    camera_mask: Tensor

    def model_kwargs(self) -> dict[str, Tensor]:
        return {
            "base_field": self.base_field,
            "support": self.support,
            "adjoint_lifted_data_residual": self.adjoint_lifted_data_residual,
            "camera_pose_features": self.camera_pose_features,
            "camera_mask": self.camera_mask,
        }


@torch.no_grad()
def prepare_jacru_m2_batch(
    inference: JACRUInferencePayload,
    *,
    support: Tensor | None = None,
    cgls_iterations: int = 4,
    model_dtype: torch.dtype = torch.float32,
    model_device: torch.device | str | None = None,
) -> JACRUM2Batch:
    """Prepare one JACRU case using only its inference-side payload.

    CGLS and the physical residual lifts run on the operator's own device and
    dtype.  The returned tensors are then converted to ``model_dtype`` and
    ``model_device`` so the compact learned branch can train in float32 on CPU
    or MPS.  One extra forward and one grouped adjoint are used after CGLS to
    form ``A_v^T (y - A x_base)`` for every camera.
    """

    iterations = _positive_integer(cgls_iterations, name="cgls_iterations")
    if not isinstance(model_dtype, torch.dtype) or not model_dtype.is_floating_point:
        raise TypeError("model_dtype must be a floating torch dtype")

    observation = inference.observations_uv
    geometry = inference.geometry
    operator = inference.operator
    if observation.ndim != 3 or observation.shape[0] != 1:
        raise ValueError("prepare_jacru_m2_batch expects one [1,ray,2] observation")
    if observation.shape[1:] != (operator.ray_count, 2):
        raise ValueError("observation shape must match the physical operator")
    if geometry.camera_count < 1:
        raise ValueError("geometry must contain at least one camera")
    if geometry.camera_index.shape != (operator.ray_count,):
        raise ValueError("geometry camera_index must contain one entry per ray")

    physical_device = operator.support.device
    physical_dtype = operator.support.dtype
    observation = observation.to(device=physical_device, dtype=physical_dtype)
    physical_support = operator.support if support is None else torch.as_tensor(support)
    physical_support = physical_support.to(
        device=physical_device,
        dtype=physical_dtype,
    )
    if tuple(physical_support.shape) != tuple(operator.grid_shape):
        raise ValueError("support must match the operator grid_shape")
    if not bool(torch.all(torch.isfinite(physical_support))):
        raise ValueError("support must contain only finite values")
    if not bool(
        torch.all((physical_support == 0.0) | (physical_support == 1.0))
    ):
        raise ValueError("support must be binary")
    if not bool(torch.any(physical_support > 0.5)):
        raise ValueError("support must retain at least one voxel")

    # The same support must govern the physical forward, physical adjoint,
    # CGLS projection, and learned correction.  Otherwise a caller-supplied
    # zero boundary would affect only the recurrence mask while residual lifts
    # were still produced by the operator's previous support.
    operator.support.copy_(physical_support)

    def physical_forward(field: Tensor) -> Tensor:
        return operator.forward(field[None, None])

    def physical_adjoint(measurement: Tensor) -> Tensor:
        return operator.adjoint(measurement)[0, 0]

    cgls = cgls_baseline(
        observation,
        forward=physical_forward,
        adjoint=physical_adjoint,
        support=physical_support,
        spacing_xyz=operator.spacing_xyz,
        iterations=iterations,
    )
    base_field = cgls.field[None, None]
    data_residual = observation - operator.forward(base_field)
    camera_index = geometry.camera_index.to(
        device=physical_device,
        dtype=torch.int64,
    )
    lifted = operator.adjoint_grouped(
        data_residual,
        ray_group_index=camera_index,
        group_count=geometry.camera_count,
    )

    azimuth = torch.as_tensor(
        geometry.camera_azimuth_degrees,
        device=physical_device,
        dtype=physical_dtype,
    )
    elevation = torch.as_tensor(
        geometry.camera_elevation_degrees,
        device=physical_device,
        dtype=physical_dtype,
    )
    if azimuth.shape != (geometry.camera_count,) or elevation.shape != (
        geometry.camera_count,
    ):
        raise ValueError("camera pose arrays must match geometry.camera_count")
    azimuth = torch.deg2rad(azimuth)
    elevation = torch.deg2rad(elevation)
    pose_features = torch.stack(
        (
            torch.sin(azimuth),
            torch.cos(azimuth),
            torch.sin(elevation),
            torch.cos(elevation),
        ),
        dim=-1,
    )[None]
    camera_mask = torch.ones(
        (1, geometry.camera_count),
        device=physical_device,
        dtype=physical_dtype,
    )

    output_device = (
        physical_device if model_device is None else torch.device(model_device)
    )

    def model_tensor(value: Tensor) -> Tensor:
        return value.detach().to(
            device=output_device,
            dtype=model_dtype,
        ).contiguous()

    return JACRUM2Batch(
        base_field=model_tensor(base_field),
        support=model_tensor(physical_support[None, None]),
        adjoint_lifted_data_residual=model_tensor(lifted),
        camera_pose_features=model_tensor(pose_features),
        camera_mask=model_tensor(camera_mask),
    )


class JACRUM2LearnedResidual(nn.Module):
    """Small geometry-conditioned residual operator for 12-cubed fields.

    Camera tokens combine an adjoint-lifted data residual with a pose
    embedding.  Shared token weights followed by masked mean and variance
    pooling make the camera axis an unordered set.  ``forward`` returns the
    corrected field and can additionally return a scalar gate per sample.
    """

    def __init__(
        self,
        *,
        pose_feature_count: int = 4,
        set_channels: int = 6,
        hidden_channels: int = 8,
        gate_hidden: int = 8,
        maximum_residual_magnitude: float = 0.25,
    ) -> None:
        super().__init__()
        pose_count = _positive_integer(
            pose_feature_count,
            name="pose_feature_count",
        )
        set_width = _positive_integer(set_channels, name="set_channels")
        hidden = _positive_integer(hidden_channels, name="hidden_channels")
        gate_width = _positive_integer(gate_hidden, name="gate_hidden")
        self.pose_feature_count = pose_count
        self.maximum_residual_magnitude = _positive_finite(
            maximum_residual_magnitude,
            name="maximum_residual_magnitude",
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_count, set_width),
            nn.GELU(),
            nn.Linear(set_width, set_width),
            nn.GELU(),
        )
        self.view_encoder = nn.Sequential(
            nn.Conv3d(1 + set_width, set_width, kernel_size=1),
            nn.GroupNorm(_group_count(set_width), set_width),
            nn.GELU(),
            nn.Conv3d(set_width, set_width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        trunk_inputs = 4 + 2 * set_width
        self.spatial_trunk = nn.Sequential(
            nn.Conv3d(trunk_inputs, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(set_width + 3, gate_width),
            nn.GELU(),
            nn.Linear(gate_width, 1),
        )
        self.residual_head = nn.Conv3d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _canonical_support(self, support: Tensor, *, base_field: Tensor) -> Tensor:
        batch = base_field.shape[0]
        spatial_shape = base_field.shape[-3:]
        values = torch.as_tensor(support, device=base_field.device)
        if values.ndim == 3:
            values = values[None, None].expand(batch, -1, -1, -1, -1)
        elif values.ndim == 4 and tuple(values.shape[-3:]) == spatial_shape:
            if values.shape[0] not in {1, batch}:
                raise ValueError("four-dimensional support must use batch size 1 or B")
            values = values[:, None]
            if values.shape[0] == 1 and batch > 1:
                values = values.expand(batch, -1, -1, -1, -1)
        elif values.ndim == 5:
            if values.shape[1] != 1 or tuple(values.shape[-3:]) != spatial_shape:
                raise ValueError("support must have one channel and match the field grid")
            if values.shape[0] not in {1, batch}:
                raise ValueError("support batch size must be 1 or match base_field")
            if values.shape[0] == 1 and batch > 1:
                values = values.expand(batch, -1, -1, -1, -1)
        else:
            raise ValueError("support must have shape [z,y,x], [B,z,y,x], or [B,1,z,y,x]")
        if values.is_complex():
            raise TypeError("support must be real")
        values = values.to(dtype=base_field.dtype)
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError("support must contain only finite values")
        if not bool(torch.all((values == 0.0) | (values == 1.0))):
            raise ValueError("support must be binary")
        if bool(torch.any(values.sum(dim=(1, 2, 3, 4)) < 1.0)):
            raise ValueError("every sample support must retain at least one voxel")
        return values

    def _validated_inputs(
        self,
        base_field: Tensor,
        support: Tensor,
        adjoint_lifted_data_residual: Tensor,
        camera_pose_features: Tensor,
        camera_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(base_field, torch.Tensor):
            raise TypeError("base_field must be a torch.Tensor")
        if (
            base_field.ndim != 5
            or base_field.shape[1] != 1
            or min(base_field.shape[-3:]) < 2
        ):
            raise ValueError("base_field must have shape [B,1,z,y,x]")
        if not base_field.is_floating_point() or base_field.is_complex():
            raise TypeError("base_field must have a real floating dtype")
        if not bool(torch.all(torch.isfinite(base_field))):
            raise ValueError("base_field must contain only finite values")
        parameter = next(self.parameters())
        if base_field.device != parameter.device or base_field.dtype != parameter.dtype:
            raise ValueError("base_field dtype and device must match the model parameters")

        batch = base_field.shape[0]
        spatial_shape = base_field.shape[-3:]
        lifted = torch.as_tensor(
            adjoint_lifted_data_residual,
            device=base_field.device,
        ).to(dtype=base_field.dtype)
        if (
            lifted.ndim != 6
            or lifted.shape[0] != batch
            or lifted.shape[2] != 1
            or tuple(lifted.shape[-3:]) != spatial_shape
        ):
            raise ValueError(
                "adjoint_lifted_data_residual must have shape [B,V,1,z,y,x]"
            )
        view_count = lifted.shape[1]
        if view_count < 1:
            raise ValueError("at least one camera lift is required")
        if not bool(torch.all(torch.isfinite(lifted))):
            raise ValueError("adjoint residual lifts must contain only finite values")

        poses = torch.as_tensor(camera_pose_features, device=base_field.device).to(
            dtype=base_field.dtype
        )
        if poses.ndim == 2:
            poses = poses[None].expand(batch, -1, -1)
        if poses.shape != (batch, view_count, self.pose_feature_count):
            raise ValueError(
                "camera_pose_features must have shape [V,P] or [B,V,P]"
            )
        if not bool(torch.all(torch.isfinite(poses))):
            raise ValueError("camera pose features must contain only finite values")

        masks = torch.as_tensor(camera_mask, device=base_field.device)
        if masks.ndim == 1:
            masks = masks[None].expand(batch, -1)
        if masks.shape != (batch, view_count):
            raise ValueError("camera_mask must have shape [V] or [B,V]")
        if masks.is_complex():
            raise TypeError("camera_mask must be real")
        masks = masks.to(dtype=base_field.dtype)
        if not bool(torch.all(torch.isfinite(masks))):
            raise ValueError("camera_mask must contain only finite values")
        if not bool(torch.all((masks == 0.0) | (masks == 1.0))):
            raise ValueError("camera_mask must be binary")
        if bool(torch.any(masks.sum(dim=1) < 1.0)):
            raise ValueError("every sample must retain at least one camera")

        canonical_support = self._canonical_support(support, base_field=base_field)
        return base_field, canonical_support, lifted, poses, masks

    def _residual_from_validated(
        self,
        base_field: Tensor,
        support: Tensor,
        lifted: Tensor,
        poses: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, view_count, _, depth, height, width = lifted.shape
        pose_embedding = self.pose_encoder(poses)
        pose_volume = pose_embedding[:, :, :, None, None, None].expand(
            -1,
            -1,
            -1,
            depth,
            height,
            width,
        )
        view_input = torch.cat((lifted, pose_volume), dim=2).reshape(
            batch * view_count,
            1 + pose_embedding.shape[-1],
            depth,
            height,
            width,
        )
        encoded = self.view_encoder(view_input).reshape(
            batch,
            view_count,
            -1,
            depth,
            height,
            width,
        )
        active = masks[:, :, None, None, None, None]
        active_count = masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        denominator = active_count[:, :, None, None, None]
        set_mean = torch.sum(encoded * active, dim=1) / denominator
        centered = encoded - set_mean[:, None]
        set_spread = torch.sqrt(
            (
                torch.sum(centered.square() * active, dim=1) / denominator
            ).clamp_min(1e-12)
        )
        pooled_lift = torch.sum(lifted * active, dim=1) / denominator
        pooled_abs_lift = torch.sum(lifted.abs() * active, dim=1) / denominator

        trunk_input = torch.cat(
            (
                base_field,
                support,
                pooled_lift,
                pooled_abs_lift,
                set_mean,
                set_spread,
            ),
            dim=1,
        )
        spatial_features = self.spatial_trunk(trunk_input)

        set_pose = torch.sum(pose_embedding * masks[:, :, None], dim=1) / active_count
        support_voxels = support.sum(dim=(2, 3, 4)).clamp_min(1.0)
        base_level = (base_field.abs() * support).sum(dim=(2, 3, 4)) / support_voxels
        lifted_support = support[:, None]
        lift_denominator = support_voxels * active_count
        lift_level = (
            lifted.abs() * active * lifted_support
        ).sum(dim=(1, 2, 3, 4, 5))[:, None] / lift_denominator
        active_fraction = masks.mean(dim=1, keepdim=True)
        gate_features = torch.cat(
            (set_pose, base_level, lift_level, active_fraction),
            dim=1,
        )
        gate = torch.sigmoid(self.gate_network(gate_features))
        gate_volume = gate[:, :, None, None, None]
        residual = (
            support
            * self.maximum_residual_magnitude
            * gate_volume
            * torch.tanh(self.residual_head(spatial_features))
        )
        return residual, gate_volume

    def residual(
        self,
        base_field: Tensor,
        support: Tensor,
        adjoint_lifted_data_residual: Tensor,
        camera_pose_features: Tensor,
        camera_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the bounded support-limited correction and sample gate."""

        values = self._validated_inputs(
            base_field,
            support,
            adjoint_lifted_data_residual,
            camera_pose_features,
            camera_mask,
        )
        return self._residual_from_validated(*values)

    def forward(
        self,
        base_field: Tensor,
        support: Tensor,
        adjoint_lifted_data_residual: Tensor,
        camera_pose_features: Tensor,
        camera_mask: Tensor,
        *,
        return_gate: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        values = self._validated_inputs(
            base_field,
            support,
            adjoint_lifted_data_residual,
            camera_pose_features,
            camera_mask,
        )
        correction, gate = self._residual_from_validated(*values)
        output = base_field + correction
        if return_gate:
            return output, gate
        return output


__all__ = [
    "JACRUM2Batch",
    "JACRUM2LearnedResidual",
    "prepare_jacru_m2_batch",
]
