"""Geometry-conditioned fixed-SPD preconditioning for PSU B0 PCGLS.

The learned component materializes one positive Fourier multiplier before the
linear solve. The same multiplier is reused at every PCGLS stage, so the
preconditioner is fixed within a reconstruction and standard PCG structure is
not silently replaced by a variable-preconditioner recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .psu_b0_spectral_preconditioner import (
    _frequency_basis,
    _frequency_components,
)


CONDITIONED_PCGLS_SCHEMA = "psu-b0-geometry-conditioned-spd-pcgls-1.0"


def view_geometry_features_from_operator(
    operator: Any,
    *,
    rays_per_view: int,
) -> torch.Tensor:
    """Summarize each camera view using only declared operator geometry."""

    per_view = int(rays_per_view)
    if per_view < 1:
        raise ValueError("rays_per_view must be positive")
    projection_u = operator.projection_u.detach()
    projection_v = operator.projection_v.detach()
    ray_scale = operator.ray_scale.detach()
    if projection_u.ndim != 2 or projection_u.shape[1] != 3:
        raise ValueError("operator projection_u must have shape [ray,3]")
    if projection_v.shape != projection_u.shape:
        raise ValueError("operator projection vectors must align")
    if ray_scale.shape != (len(projection_u),):
        raise ValueError("operator ray_scale must have one value per ray")
    if len(projection_u) % per_view:
        raise ValueError("rays_per_view must exactly divide the ray count")
    view_count = len(projection_u) // per_view
    u = projection_u.reshape(view_count, per_view, 3)
    v = projection_v.reshape(view_count, per_view, 3)
    normal = torch.linalg.cross(u, v, dim=-1)
    normal = normal / torch.linalg.vector_norm(
        normal,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    scale = torch.log(torch.abs(ray_scale).clamp_min(1e-20)).reshape(
        view_count,
        per_view,
    )
    raw = torch.cat(
        (
            u.mean(dim=1),
            v.mean(dim=1),
            normal.mean(dim=1),
            normal.std(dim=1, unbiased=False),
            scale.mean(dim=1, keepdim=True),
            scale.std(dim=1, keepdim=True, unbiased=False),
        ),
        dim=1,
    )
    center = raw.mean(dim=0, keepdim=True)
    spread = raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (raw - center) / spread
    if not torch.all(torch.isfinite(normalized)):
        raise ValueError("view geometry features are not finite")
    return normalized.to(dtype=torch.float32, device="cpu")


@dataclass
class MaterializedPositiveSpectralDirection:
    """A batch-specific positive multiplier held fixed across solver stages."""

    gain: torch.Tensor
    controller_coefficients: torch.Tensor
    log_correction: torch.Tensor

    def __call__(
        self,
        gradient: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if gradient.ndim != 5:
            raise ValueError("gradient must have shape [batch,1,z,y,x]")
        if self.gain.ndim != 4 or len(self.gain) != len(gradient):
            raise ValueError("materialized gain must align with the batch")
        if tuple(self.gain.shape[1:]) != (
            gradient.shape[-3],
            gradient.shape[-2],
            gradient.shape[-1] // 2 + 1,
        ):
            raise ValueError("materialized gain does not match the grid")
        gain = self.gain.to(gradient)
        spectrum = torch.fft.rfftn(gradient, dim=(-3, -2, -1))
        direction = torch.fft.irfftn(
            spectrum * gain[:, None],
            s=gradient.shape[-3:],
            dim=(-3, -2, -1),
        )
        return direction, {
            "gain_minimum": gain.amin(dim=(1, 2, 3)),
            "gain_maximum": gain.amax(dim=(1, 2, 3)),
            "gain_geometric_mean": torch.exp(
                torch.mean(torch.log(gain), dim=(1, 2, 3))
            ),
            "correction_log_abs_maximum": torch.amax(
                torch.abs(self.log_correction.to(gradient)),
                dim=(1, 2, 3),
            ),
            "controller_coefficients": self.controller_coefficients.to(
                gradient
            ),
            "fixed_within_solve": torch.ones(
                len(gradient),
                dtype=gradient.dtype,
                device=gradient.device,
            ),
        }


class GeometryConditionedSPDPreconditioner(nn.Module):
    """Materialize a low-dimensional positive spectral PCGLS preconditioner."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        *,
        view_geometry_features: torch.Tensor,
        hidden: int = 24,
        base_sobolev_strength: float = 4.0,
        base_epsilon: float = 0.05,
        maximum_log_correction: float = 0.5,
    ) -> None:
        super().__init__()
        shape = tuple(int(value) for value in grid_shape)
        geometry = torch.as_tensor(
            view_geometry_features,
            dtype=torch.float32,
        )
        if len(shape) != 3 or min(shape) < 4:
            raise ValueError("grid_shape must contain three dimensions >= 4")
        if geometry.ndim != 2 or geometry.shape[0] < 2:
            raise ValueError(
                "view_geometry_features must have shape [view,feature]"
            )
        if int(hidden) < 4:
            raise ValueError("hidden must be at least four")
        if float(base_sobolev_strength) < 0.0:
            raise ValueError("base_sobolev_strength must be nonnegative")
        if float(base_epsilon) <= 0.0:
            raise ValueError("base_epsilon must be positive")
        if not 0.0 < float(maximum_log_correction) <= 1.5:
            raise ValueError(
                "maximum_log_correction must lie in (0,1.5]"
            )
        self.grid_shape = shape
        self.view_count = int(geometry.shape[0])
        self.maximum_log_correction = float(maximum_log_correction)
        self.register_buffer("view_geometry_features", geometry)
        basis = _frequency_basis(shape)
        self.register_buffer("frequency_basis", basis)
        x2, y2, z2 = _frequency_components(shape)
        base_gain = (
            float(base_epsilon) + x2 + y2 + z2
        ).pow(-float(base_sobolev_strength))
        base_gain = base_gain / torch.exp(
            torch.mean(torch.log(base_gain.clamp_min(1e-12)))
        )
        self.register_buffer("base_gain", base_gain)
        observation_width = 5
        self.view_encoder = nn.Sequential(
            nn.Linear(geometry.shape[1] + observation_width, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
        )
        self.controller = nn.Sequential(
            nn.Linear(2 * int(hidden) + 4, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), basis.shape[0]),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def _controller_features(
        self,
        observation_uv: torch.Tensor,
        *,
        sigma_by_view: torch.Tensor,
        view_mask: torch.Tensor,
        rays_per_view: int,
    ) -> torch.Tensor:
        if observation_uv.ndim != 3 or observation_uv.shape[-1] != 2:
            raise ValueError(
                "observation_uv must have shape [batch,ray,2]"
            )
        batch, ray_count, _ = observation_uv.shape
        if sigma_by_view.shape != (batch, self.view_count):
            raise ValueError("sigma_by_view has the wrong shape")
        if view_mask.shape != sigma_by_view.shape:
            raise ValueError("view_mask and sigma_by_view must align")
        if ray_count != self.view_count * int(rays_per_view):
            raise ValueError("observations do not match the view layout")
        if torch.any(sigma_by_view <= 0.0):
            raise ValueError("sigma_by_view must be strictly positive")
        active = view_mask > 0.5
        if torch.any(active.sum(dim=1) < 1):
            raise ValueError("each sample needs at least one active view")
        residual = observation_uv.reshape(
            batch,
            self.view_count,
            int(rays_per_view),
            2,
        )
        white = residual / sigma_by_view[:, :, None, None]
        component_rms = torch.sqrt(
            torch.mean(white.square(), dim=2).clamp_min(1e-20)
        )
        cross = torch.mean(white[..., 0] * white[..., 1], dim=2)
        correlation = cross / (
            component_rms[..., 0] * component_rms[..., 1]
        ).clamp_min(1e-12)
        active_count = active.sum(dim=1).clamp_min(1)
        log_sigma = torch.log(sigma_by_view.clamp_min(1e-12))
        sigma_center = torch.sum(
            torch.where(active, log_sigma, torch.zeros_like(log_sigma)),
            dim=1,
        ) / active_count
        sigma_relative = log_sigma - sigma_center[:, None]
        geometry = self.view_geometry_features.to(observation_uv)
        geometry = geometry[None].expand(batch, -1, -1)
        per_view = torch.cat(
            (
                geometry,
                torch.log1p(component_rms),
                correlation[:, :, None],
                sigma_relative[:, :, None],
                view_mask[:, :, None].to(observation_uv),
            ),
            dim=2,
        )
        encoded = self.view_encoder(per_view)
        selector = active[:, :, None]
        active_float = selector.to(encoded)
        pooled_mean = torch.sum(encoded * active_float, dim=1)
        pooled_mean = pooled_mean / active_count[:, None]
        pooled_max = torch.max(
            torch.where(
                selector,
                encoded,
                torch.full_like(encoded, -torch.inf),
            ),
            dim=1,
        ).values
        view_white_rms = torch.sqrt(
            torch.mean(white.square(), dim=(2, 3)).clamp_min(1e-20)
        )
        global_white_rms = torch.sum(
            view_white_rms * active.to(view_white_rms),
            dim=1,
        ) / active_count
        centered_white = torch.where(
            active,
            torch.log1p(view_white_rms)
            - (
                torch.sum(
                    torch.log1p(view_white_rms) * active.to(view_white_rms),
                    dim=1,
                )
                / active_count
            )[:, None],
            torch.zeros_like(view_white_rms),
        )
        white_spread = torch.sqrt(
            torch.sum(centered_white.square(), dim=1)
            / active_count
        )
        global_features = torch.stack(
            (
                active_count.to(observation_uv) / self.view_count,
                torch.log1p(global_white_rms),
                white_spread,
                sigma_center,
            ),
            dim=1,
        )
        return torch.cat(
            (pooled_mean, pooled_max, global_features),
            dim=1,
        )

    def materialize(
        self,
        observation_uv: torch.Tensor,
        *,
        sigma_by_view: torch.Tensor,
        view_mask: torch.Tensor,
        rays_per_view: int,
    ) -> MaterializedPositiveSpectralDirection:
        """Create one batch-specific SPD map before PCGLS starts."""

        features = self._controller_features(
            observation_uv,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
        )
        coefficients = self.controller(features)
        raw = torch.einsum(
            "bc,czyp->bzyp",
            coefficients,
            self.frequency_basis.to(observation_uv),
        )
        raw = raw - raw.mean(dim=(1, 2, 3), keepdim=True)
        log_correction = self.maximum_log_correction * torch.tanh(raw)
        log_correction = log_correction - log_correction.mean(
            dim=(1, 2, 3),
            keepdim=True,
        )
        gain = self.base_gain.to(observation_uv)[None] * torch.exp(
            log_correction
        )
        return MaterializedPositiveSpectralDirection(
            gain=gain,
            controller_coefficients=coefficients,
            log_correction=log_correction,
        )


__all__ = [
    "CONDITIONED_PCGLS_SCHEMA",
    "GeometryConditionedSPDPreconditioner",
    "MaterializedPositiveSpectralDirection",
    "view_geometry_features_from_operator",
]
