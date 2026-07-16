"""Call-matched spectral preconditioning for the PSU B0 linear inverse.

The learned component is restricted to a bounded positive Fourier multiplier
applied to the exact adjoint gradient. An analytic line search follows every
direction, so the network cannot bypass the declared forward/adjoint pair or
directly synthesize a reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from .psu_b0_reconstruction_interface import finite_difference_gradient


SPECTRAL_PRECONDITIONER_SCHEMA = "psu-b0-positive-spectral-preconditioner-1.0"


def _view_expansion(
    values: torch.Tensor,
    *,
    rays_per_view: int,
    ray_count: int,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("view values must have shape [batch,view]")
    expanded = values.repeat_interleave(int(rays_per_view), dim=1)
    if expanded.shape[1] != int(ray_count):
        raise ValueError("view count and rays_per_view do not cover the operator")
    return expanded[:, :, None]


def _weighted_measurement_terms(
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if observation_uv.ndim != 3 or observation_uv.shape[-1] != 2:
        raise ValueError("observation_uv must have shape [batch,ray,2]")
    batch = len(observation_uv)
    if sigma_by_view.shape != view_mask.shape or sigma_by_view.shape[0] != batch:
        raise ValueError("sigma_by_view and view_mask must align with observations")
    if torch.any(sigma_by_view <= 0.0):
        raise ValueError("sigma_by_view must be strictly positive")
    if torch.any((view_mask < 0.0) | (view_mask > 1.0)):
        raise ValueError("view_mask must lie in [0,1]")
    if torch.any(torch.sum(view_mask > 0.5, dim=1) < 1):
        raise ValueError("each sample needs at least one active view")
    active = _view_expansion(
        view_mask.to(observation_uv),
        rays_per_view=rays_per_view,
        ray_count=observation_uv.shape[1],
    )
    sigma = _view_expansion(
        sigma_by_view.to(observation_uv),
        rays_per_view=rays_per_view,
        ray_count=observation_uv.shape[1],
    )
    return active, sigma


def _relative_objective(
    residual: torch.Tensor,
    sigma: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    value = torch.sum((residual / sigma).square(), dim=(1, 2))
    return value / initial.clamp_min(1e-20)


class SearchDirection(Protocol):
    def __call__(
        self,
        gradient: torch.Tensor,
        *,
        residual_uv: torch.Tensor,
        sigma_by_view: torch.Tensor,
        view_mask: torch.Tensor,
        rays_per_view: int,
        stage_fraction: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return an SPD-preconditioned direction and diagnostics."""


class IdentityDirection:
    """Unpreconditioned exact-gradient direction."""

    def __call__(
        self,
        gradient: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        one = torch.ones(
            len(gradient),
            dtype=gradient.dtype,
            device=gradient.device,
        )
        return gradient, {
            "gain_minimum": one,
            "gain_maximum": one,
            "gain_geometric_mean": one,
        }


def _frequency_components(
    grid_shape: tuple[int, int, int],
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nz, ny, nx = (int(value) for value in grid_shape)
    if min(nz, ny, nx) < 4:
        raise ValueError("grid_shape must contain dimensions of at least four")
    fz = torch.fft.fftfreq(nz, dtype=dtype)
    fy = torch.fft.fftfreq(ny, dtype=dtype)
    fx = torch.fft.rfftfreq(nx, dtype=dtype)
    zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
    scale = torch.as_tensor(0.5, dtype=dtype)
    x2 = (xx / scale).square()
    y2 = (yy / scale).square()
    z2 = (zz / scale).square()
    return x2, y2, z2


def _frequency_basis(
    grid_shape: tuple[int, int, int],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x2, y2, z2 = _frequency_components(grid_shape, dtype=dtype)
    radius2 = x2 + y2 + z2
    basis = torch.stack(
        (
            x2,
            y2,
            z2,
            x2 * y2,
            x2 * z2,
            y2 * z2,
            radius2.square(),
        )
    )
    centered = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
    scale_by_channel = torch.sqrt(
        torch.mean(centered.square(), dim=(1, 2, 3), keepdim=True)
    ).clamp_min(1e-8)
    return centered / scale_by_channel


class FixedSobolevDirection(nn.Module):
    """Positive inverse-Sobolev reference selected only on validation data."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        *,
        strength: float,
    ) -> None:
        super().__init__()
        if float(strength) < 0.0:
            raise ValueError("strength must be nonnegative")
        x2, y2, z2 = _frequency_components(grid_shape)
        radius2 = x2 + y2 + z2
        gain = (0.05 + radius2).pow(-float(strength))
        geometric = torch.exp(torch.mean(torch.log(gain.clamp_min(1e-8))))
        self.register_buffer("gain", gain / geometric)

    def forward(
        self,
        gradient: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spectrum = torch.fft.rfftn(gradient, dim=(-3, -2, -1))
        gain = self.gain.to(gradient)
        direction = torch.fft.irfftn(
            spectrum * gain[None, None],
            s=gradient.shape[-3:],
            dim=(-3, -2, -1),
        )
        minimum = gain.amin().expand(len(gradient))
        maximum = gain.amax().expand(len(gradient))
        mean = torch.exp(torch.mean(torch.log(gain))).expand(len(gradient))
        return direction, {
            "gain_minimum": minimum,
            "gain_maximum": maximum,
            "gain_geometric_mean": mean,
        }


class PositiveSpectralDirection(nn.Module):
    """Set-conditioned bounded SPD correction to a Sobolev reference."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        *,
        view_count: int,
        hidden: int = 24,
        embedding_width: int = 4,
        maximum_log_gain: float = 1.25,
        base_sobolev_strength: float = 0.0,
    ) -> None:
        super().__init__()
        if int(view_count) < 2:
            raise ValueError("view_count must be at least two")
        if int(hidden) < 4 or int(embedding_width) < 1:
            raise ValueError("network widths are too small")
        if not 0.0 < float(maximum_log_gain) <= 2.0:
            raise ValueError("maximum_log_gain must lie in (0,2]")
        if float(base_sobolev_strength) < 0.0:
            raise ValueError("base_sobolev_strength must be nonnegative")
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.view_count = int(view_count)
        self.maximum_log_gain = float(maximum_log_gain)
        self.base_sobolev_strength = float(base_sobolev_strength)
        basis = _frequency_basis(self.grid_shape)
        self.register_buffer("frequency_basis", basis)
        x2, y2, z2 = _frequency_components(self.grid_shape)
        base_gain = (0.05 + x2 + y2 + z2).pow(
            -self.base_sobolev_strength
        )
        base_geometric = torch.exp(
            torch.mean(torch.log(base_gain.clamp_min(1e-8)))
        )
        self.register_buffer("base_gain", base_gain / base_geometric)
        self.view_embedding = nn.Embedding(self.view_count, int(embedding_width))
        view_input = 3 + int(embedding_width)
        self.view_encoder = nn.Sequential(
            nn.Linear(view_input, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
        )
        self.controller = nn.Sequential(
            nn.Linear(2 * int(hidden) + 2, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), basis.shape[0]),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def _features(
        self,
        residual_uv: torch.Tensor,
        sigma_by_view: torch.Tensor,
        view_mask: torch.Tensor,
        rays_per_view: int,
        stage_fraction: float,
    ) -> torch.Tensor:
        batch, ray_count, components = residual_uv.shape
        if components != 2 or ray_count != self.view_count * int(rays_per_view):
            raise ValueError("residual rays do not match the declared view layout")
        if sigma_by_view.shape != (batch, self.view_count):
            raise ValueError("sigma_by_view has the wrong shape")
        if view_mask.shape != sigma_by_view.shape:
            raise ValueError("view_mask and sigma_by_view must align")
        residual = residual_uv.reshape(
            batch,
            self.view_count,
            int(rays_per_view),
            2,
        )
        white_rms = torch.sqrt(
            torch.mean(
                (residual / sigma_by_view[:, :, None, None]).square(),
                dim=(2, 3),
            ).clamp_min(1e-20)
        )
        active = view_mask > 0.5
        active_count = active.sum(dim=1).clamp_min(1)
        log_sigma = torch.log(sigma_by_view.clamp_min(1e-12))
        sigma_center = torch.sum(
            torch.where(active, log_sigma, torch.zeros_like(log_sigma)),
            dim=1,
        ) / active_count
        sigma_relative = log_sigma - sigma_center[:, None]
        view_ids = torch.arange(self.view_count, device=residual_uv.device)
        embedding = self.view_embedding(view_ids)[None].expand(batch, -1, -1)
        per_view = torch.cat(
            (
                torch.log1p(white_rms)[:, :, None],
                sigma_relative[:, :, None],
                view_mask[:, :, None].to(residual_uv),
                embedding.to(residual_uv),
            ),
            dim=2,
        )
        encoded = self.view_encoder(per_view)
        active_float = active[:, :, None].to(encoded)
        mean = torch.sum(encoded * active_float, dim=1) / active_count[:, None]
        maximum = torch.max(
            torch.where(
                active[:, :, None],
                encoded,
                torch.full_like(encoded, -torch.inf),
            ),
            dim=1,
        ).values
        stage = torch.full(
            (batch, 1),
            float(stage_fraction),
            dtype=residual_uv.dtype,
            device=residual_uv.device,
        )
        active_fraction = active_count[:, None].to(residual_uv) / self.view_count
        return torch.cat((mean, maximum, stage, active_fraction), dim=1)

    def forward(
        self,
        gradient: torch.Tensor,
        *,
        residual_uv: torch.Tensor,
        sigma_by_view: torch.Tensor,
        view_mask: torch.Tensor,
        rays_per_view: int,
        stage_fraction: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if gradient.ndim != 5 or tuple(gradient.shape[-3:]) != self.grid_shape:
            raise ValueError("gradient must match the configured 3D grid")
        features = self._features(
            residual_uv,
            sigma_by_view,
            view_mask,
            rays_per_view,
            stage_fraction,
        )
        coefficients = self.controller(features)
        raw = torch.einsum(
            "bc,czyp->bzyp",
            coefficients,
            self.frequency_basis.to(gradient),
        )
        raw = raw - raw.mean(dim=(1, 2, 3), keepdim=True)
        log_gain = self.maximum_log_gain * torch.tanh(raw)
        gain = self.base_gain.to(gradient)[None] * torch.exp(log_gain)
        spectrum = torch.fft.rfftn(gradient, dim=(-3, -2, -1))
        filtered = torch.fft.irfftn(
            spectrum * gain[:, None],
            s=self.grid_shape,
            dim=(-3, -2, -1),
        )
        return filtered, {
            "gain_minimum": gain.amin(dim=(1, 2, 3)),
            "gain_maximum": gain.amax(dim=(1, 2, 3)),
            "gain_geometric_mean": torch.exp(
                torch.mean(torch.log(gain), dim=(1, 2, 3))
            ),
            "controller_coefficients": coefficients,
        }


class ActiveViewSupportEnvelopeDirection(nn.Module):
    """Fall back exactly to a fixed direction outside a view-count support."""

    def __init__(
        self,
        *,
        candidate: nn.Module,
        fallback: nn.Module,
        minimum_active_views: int,
        maximum_active_views: int,
    ) -> None:
        super().__init__()
        minimum = int(minimum_active_views)
        maximum = int(maximum_active_views)
        if minimum < 1 or maximum < minimum:
            raise ValueError("active-view support is invalid")
        self.candidate = candidate
        self.fallback = fallback
        self.minimum_active_views = minimum
        self.maximum_active_views = maximum

    def forward(
        self,
        gradient: torch.Tensor,
        *,
        residual_uv: torch.Tensor,
        sigma_by_view: torch.Tensor,
        view_mask: torch.Tensor,
        rays_per_view: int,
        stage_fraction: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        candidate, candidate_diagnostics = self.candidate(
            gradient,
            residual_uv=residual_uv,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
            stage_fraction=stage_fraction,
        )
        fallback, fallback_diagnostics = self.fallback(
            gradient,
            residual_uv=residual_uv,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
            stage_fraction=stage_fraction,
        )
        active_count = torch.sum(view_mask > 0.5, dim=1)
        trust = (
            (active_count >= self.minimum_active_views)
            & (active_count <= self.maximum_active_views)
        ).to(gradient)
        selector = (trust > 0.5)[:, None, None, None, None]
        direction = torch.where(selector, candidate, fallback)
        diagnostics = {
            key: value
            for key, value in candidate_diagnostics.items()
            if key not in {"gain_minimum", "gain_maximum", "gain_geometric_mean"}
        }
        diagnostics.update(
            {
                "gain_minimum": torch.where(
                    trust > 0.5,
                    candidate_diagnostics["gain_minimum"],
                    fallback_diagnostics["gain_minimum"],
                ),
                "gain_maximum": torch.where(
                    trust > 0.5,
                    candidate_diagnostics["gain_maximum"],
                    fallback_diagnostics["gain_maximum"],
                ),
                "gain_geometric_mean": torch.where(
                    trust > 0.5,
                    candidate_diagnostics["gain_geometric_mean"],
                    fallback_diagnostics["gain_geometric_mean"],
                ),
                "support_envelope_trust": trust,
                "active_view_count": active_count.to(gradient),
            }
        )
        return direction, diagnostics


@dataclass(frozen=True)
class IterativeReconstruction:
    volume: torch.Tensor
    residual_uv: torch.Tensor
    history: list[dict[str, torch.Tensor]]
    forward_calls: int
    adjoint_calls: int


def exact_line_search_reconstruction(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    stages: int,
    direction: SearchDirection,
    denominator_floor: float = 1e-20,
) -> IterativeReconstruction:
    """Run fixed-depth SPD-preconditioned steepest descent."""

    count = int(stages)
    if count < 1:
        raise ValueError("stages must be positive")
    active, sigma = _weighted_measurement_terms(
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )
    support = operator.support[None, None].to(observation_uv)
    current = torch.zeros(
        (len(observation_uv), 1, *operator.grid_shape),
        dtype=observation_uv.dtype,
        device=observation_uv.device,
    )
    residual = active * observation_uv
    initial_objective = torch.sum(
        (residual / sigma).square(),
        dim=(1, 2),
    ).clamp_min(float(denominator_floor))
    history: list[dict[str, torch.Tensor]] = []
    for stage in range(count):
        weighted_residual = residual / sigma.square()
        gradient = operator.adjoint(weighted_residual)
        proposed, diagnostics = direction(
            gradient,
            residual_uv=residual,
            sigma_by_view=sigma_by_view,
            view_mask=view_mask,
            rays_per_view=rays_per_view,
            stage_fraction=(stage + 1) / count,
        )
        search = proposed * support
        projected = active * operator(search)
        numerator = torch.sum(weighted_residual * projected, dim=(1, 2))
        denominator = torch.sum(
            (projected / sigma).square(),
            dim=(1, 2),
        ).clamp_min(float(denominator_floor))
        alpha = torch.clamp_min(numerator / denominator, 0.0)
        objective_before = _relative_objective(
            residual,
            sigma,
            initial_objective,
        )
        current = current + alpha[:, None, None, None, None] * search
        residual = residual - alpha[:, None, None] * projected
        objective_after = _relative_objective(
            residual,
            sigma,
            initial_objective,
        )
        history.append(
            {
                "stage": torch.full_like(alpha, stage + 1, dtype=torch.int64),
                "alpha": alpha,
                "directional_derivative": numerator,
                "relative_objective_before": objective_before,
                "relative_objective_after": objective_after,
                **diagnostics,
            }
        )
    return IterativeReconstruction(
        volume=current,
        residual_uv=residual,
        history=history,
        forward_calls=count,
        adjoint_calls=count,
    )


def weighted_cgls_reconstruction(
    operator: Any,
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
    stages: int,
    denominator_floor: float = 1e-20,
) -> IterativeReconstruction:
    """Run batched CGLS on the same masked, whitened least-squares objective."""

    count = int(stages)
    if count < 1:
        raise ValueError("stages must be positive")
    active, sigma = _weighted_measurement_terms(
        observation_uv,
        sigma_by_view=sigma_by_view,
        view_mask=view_mask,
        rays_per_view=rays_per_view,
    )
    current = torch.zeros(
        (len(observation_uv), 1, *operator.grid_shape),
        dtype=observation_uv.dtype,
        device=observation_uv.device,
    )
    residual_white = active * observation_uv / sigma
    initial_objective = torch.sum(
        residual_white.square(),
        dim=(1, 2),
    ).clamp_min(float(denominator_floor))
    normal = operator.adjoint(active * residual_white / sigma)
    direction = normal.clone()
    gamma = torch.sum(normal.square(), dim=(1, 2, 3, 4))
    history: list[dict[str, torch.Tensor]] = []
    for stage in range(count):
        projected_white = active * operator(direction) / sigma
        denominator = torch.sum(
            projected_white.square(),
            dim=(1, 2),
        ).clamp_min(float(denominator_floor))
        alpha = gamma / denominator
        current = current + alpha[:, None, None, None, None] * direction
        residual_white = residual_white - alpha[:, None, None] * projected_white
        next_normal = operator.adjoint(active * residual_white / sigma)
        next_gamma = torch.sum(next_normal.square(), dim=(1, 2, 3, 4))
        beta = next_gamma / gamma.clamp_min(float(denominator_floor))
        direction = next_normal + beta[:, None, None, None, None] * direction
        gamma = next_gamma
        history.append(
            {
                "stage": torch.full_like(alpha, stage + 1, dtype=torch.int64),
                "alpha": alpha,
                "beta": beta,
                "relative_objective_after": torch.sum(
                    residual_white.square(),
                    dim=(1, 2),
                )
                / initial_objective,
            }
        )
    return IterativeReconstruction(
        volume=current,
        residual_uv=residual_white * sigma,
        history=history,
        forward_calls=count,
        adjoint_calls=count + 1,
    )


def normalized_field_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    gradient_weight: float = 0.25,
) -> torch.Tensor:
    """Return per-sample field plus gradient relative-MSE loss."""

    if prediction.shape != truth.shape or prediction.ndim != 5:
        raise ValueError("prediction and truth must be aligned 3D batches")
    field = torch.mean((prediction - truth).square(), dim=(1, 2, 3, 4))
    field = field / torch.mean(truth.square(), dim=(1, 2, 3, 4)).clamp_min(1e-12)
    spacing = tuple(2.0 / (size - 1) for size in truth.shape[-1:-4:-1])
    predicted_gradient = finite_difference_gradient(
        prediction[:, 0],
        spacing_xyz=spacing,
    )
    truth_gradient = finite_difference_gradient(
        truth[:, 0],
        spacing_xyz=spacing,
    )
    gradient = torch.mean(
        (predicted_gradient - truth_gradient).square(),
        dim=(1, 2, 3, 4),
    )
    gradient = gradient / torch.mean(
        truth_gradient.square(),
        dim=(1, 2, 3, 4),
    ).clamp_min(1e-12)
    return field + float(gradient_weight) * gradient
