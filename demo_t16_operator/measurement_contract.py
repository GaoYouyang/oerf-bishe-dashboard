"""A dataset-neutral measurement contract for BOST inverse operators.

The contract keeps the learned reconstructor independent from one camera rig.
It carries observations, active views, calibrated noise, view geometry and the
volume support separately.  A future OERF loader only needs to populate this
object and provide a compatible forward/adjoint implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BOSTBatch:
    """One batch of displacement-domain BOST measurements.

    Shapes:
      observation: ``[batch, depth, view, detector]``
      view_mask: ``[batch, view]``
      noise_std: broadcastable to ``observation``
      view_angles_degrees: ``[batch, view]``
      support: ``[batch, 1, depth, height, width]`` or one shared volume
      truth: optional ``[batch, 1, depth, height, width]``
    """

    observation: torch.Tensor
    view_mask: torch.Tensor
    noise_std: torch.Tensor
    view_angles_degrees: torch.Tensor
    support: torch.Tensor
    geometry_ids: tuple[str, ...]
    truth: torch.Tensor | None = None

    def validate(self) -> "BOSTBatch":
        if self.observation.ndim != 4:
            raise ValueError("observation must have shape [batch,depth,view,detector]")
        batch, _, views, _ = self.observation.shape
        if self.view_mask.shape != (batch, views):
            raise ValueError("view_mask must have shape [batch,view]")
        if self.view_angles_degrees.shape != (batch, views):
            raise ValueError("view_angles_degrees must have shape [batch,view]")
        if len(self.geometry_ids) != batch:
            raise ValueError("geometry_ids must contain one identifier per sample")
        if self.support.ndim == 4:
            support = self.support[:, None]
        elif self.support.ndim == 5:
            support = self.support
        else:
            raise ValueError("support must have four or five dimensions")
        if support.shape[0] not in {1, batch} or support.shape[1] != 1:
            raise ValueError("support must be shared or provide one volume per sample")
        if torch.any(~torch.isfinite(self.observation)):
            raise ValueError("observation contains non-finite values")
        if torch.any(~torch.isfinite(self.view_mask)):
            raise ValueError("view_mask contains non-finite values")
        if torch.any(self.view_mask.sum(dim=1) < 1):
            raise ValueError("each sample needs at least one active view")
        try:
            sigma = torch.broadcast_to(self.noise_std, self.observation.shape)
        except RuntimeError as exc:
            raise ValueError("noise_std is not broadcastable to observation") from exc
        active = self.active_observation_mask()
        if torch.any(~torch.isfinite(sigma[active])) or torch.any(sigma[active] <= 0):
            raise ValueError("active observations require finite positive noise_std")
        if self.truth is not None:
            if self.truth.ndim != 5 or self.truth.shape[:2] != (batch, 1):
                raise ValueError("truth must have shape [batch,1,depth,height,width]")
            if self.truth.shape[2:] != support.shape[2:]:
                raise ValueError("truth and support volume shapes disagree")
        return self

    def expanded_support(self) -> torch.Tensor:
        support = self.support[:, None] if self.support.ndim == 4 else self.support
        return support.expand(len(self.geometry_ids), -1, -1, -1, -1)

    def expanded_noise_std(self) -> torch.Tensor:
        return torch.broadcast_to(self.noise_std, self.observation.shape)

    def active_observation_mask(self) -> torch.Tensor:
        return (self.view_mask[:, None, :, None] > 0.5).expand_as(self.observation)

    def whitened(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape != self.observation.shape:
            raise ValueError("values and observation shapes disagree")
        return torch.where(
            self.active_observation_mask(),
            values / self.expanded_noise_std(),
            torch.zeros_like(values),
        )


def robust_noise_std_from_repeats(
    repeats: torch.Tensor,
    *,
    view_mask: torch.Tensor | None = None,
    shrinkage: float = 0.2,
    floor_fraction: float = 0.05,
) -> torch.Tensor:
    """Estimate a positive diagonal noise model from repeated flow-off data.

    ``repeats`` has shape ``[repeat,batch,depth,view,detector]``. The median
    absolute deviation is robust to occasional bad frames. Camera-level
    shrinkage and a small floor prevent coincident pixels from receiving
    unrealistically large precision.
    """

    if repeats.ndim != 5 or repeats.shape[0] < 3:
        raise ValueError("repeats must have shape [repeat,batch,depth,view,detector]")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not 0.0 < floor_fraction <= 1.0:
        raise ValueError("floor_fraction must lie in (0, 1]")
    if torch.any(~torch.isfinite(repeats)):
        raise ValueError("repeats contain non-finite values")
    _, batch, _, views, _ = repeats.shape
    if view_mask is not None and view_mask.shape != (batch, views):
        raise ValueError("view_mask must have shape [batch,view]")
    center = torch.median(repeats, dim=0).values
    mad = torch.median(torch.abs(repeats - center.unsqueeze(0)), dim=0).values
    pixel_std = 1.4826 * mad
    camera_rms = torch.sqrt(torch.mean(pixel_std.square(), dim=(1, 3), keepdim=True))
    estimate = torch.sqrt(
        (1.0 - float(shrinkage)) * pixel_std.square()
        + float(shrinkage) * camera_rms.square()
    )
    estimate = torch.maximum(estimate, float(floor_fraction) * camera_rms)
    tiny = torch.finfo(estimate.dtype).tiny
    estimate = estimate.clamp_min(tiny)
    if view_mask is not None:
        active = view_mask[:, None, :, None] > 0.5
        estimate = torch.where(active, estimate, camera_rms.clamp_min(tiny))
    return estimate


def estimated_relative_noise(batch: BOSTBatch, *, maximum: float = 10.0) -> torch.Tensor:
    """Estimate noise-to-signal ratio from declared observations and sigma.

    Under ``y=s+e`` with independent zero-mean noise, ``sigma/RMS(y)`` equals
    ``q/sqrt(1+q^2)`` for ``q=RMS(e)/RMS(s)``. This inversion is deployable
    when ``noise_std`` came from independent flow-off repeats.
    """

    batch.validate()
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    active = batch.active_observation_mask()
    count = active.flatten(1).sum(dim=1).clamp_min(1)
    observation_energy = torch.where(
        active, batch.observation.square(), torch.zeros_like(batch.observation)
    ).flatten(1).sum(dim=1) / count
    noise_energy = torch.where(
        active,
        batch.expanded_noise_std().square(),
        torch.zeros_like(batch.observation),
    ).flatten(1).sum(dim=1) / count
    ratio2 = (noise_energy / observation_energy.clamp_min(1e-24)).clamp(
        0.0, 1.0 - 1e-6
    )
    estimate = torch.sqrt(ratio2 / (1.0 - ratio2))
    return estimate.clamp_max(float(maximum))


def inverse_variance_trust_budget(
    batch: BOSTBatch,
    *,
    reference_noise: float,
    power: float = 2.0,
    minimum: float = 0.0,
) -> torch.Tensor:
    """Shrink learned updates outside the calibrated noise envelope."""

    if reference_noise <= 0 or power <= 0:
        raise ValueError("reference_noise and power must be positive")
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum must lie in [0, 1]")
    q_hat = estimated_relative_noise(batch)
    raw = (float(reference_noise) / q_hat.clamp_min(float(reference_noise))) ** float(power)
    return raw.clamp(min=float(minimum), max=1.0)


def geometry_noise_features(
    batch: BOSTBatch,
    *,
    period_degrees: float = 180.0,
) -> torch.Tensor:
    """Return deployable geometry/noise descriptors for a controller.

    The features are invariant to the order in which camera records are stored:
    active-view fraction, largest angular gap, gap coefficient of variation,
    first angular resultant, mean/std/max active log-noise and support fraction.
    """

    batch.validate()
    period = float(period_degrees)
    if period <= 0:
        raise ValueError("period_degrees must be positive")
    rows: list[torch.Tensor] = []
    sigma = batch.expanded_noise_std()
    support = batch.expanded_support()
    for index in range(len(batch.geometry_ids)):
        active = batch.view_mask[index] > 0.5
        angles = torch.remainder(batch.view_angles_degrees[index, active], period)
        angles = torch.sort(angles).values
        if len(angles) == 1:
            gaps = torch.as_tensor([period], dtype=angles.dtype, device=angles.device)
        else:
            gaps = torch.diff(torch.cat([angles, angles[:1] + period]))
        phase = 2.0 * torch.pi * angles / period
        resultant = torch.abs(torch.mean(torch.complex(torch.cos(phase), torch.sin(phase))))
        active_sigma = sigma[index, :, active, :]
        log_sigma = torch.log(torch.clamp(active_sigma, min=torch.finfo(active_sigma.dtype).tiny))
        mean_log_sigma = torch.mean(log_sigma)
        std_log_sigma = torch.std(log_sigma, correction=0)
        max_log_sigma = torch.max(log_sigma)
        row = torch.stack(
            [
                active.to(batch.observation.dtype).mean(),
                torch.max(gaps) / period,
                torch.std(gaps, correction=0) / (torch.mean(gaps) + 1e-12),
                resultant.to(batch.observation.dtype),
                mean_log_sigma,
                std_log_sigma,
                max_log_sigma - mean_log_sigma,
                support[index].to(batch.observation.dtype).mean(),
            ]
        )
        rows.append(row)
    return torch.stack(rows)


class DepthSeparableLinearBOST(torch.nn.Module):
    """Small linear BOST adapter used by the current controlled experiments.

    ``operator`` has shape ``[view, detector, height*width]``.  Real OERF data
    should replace this class with a calibrated sparse-ray or differentiable
    curved-ray operator while preserving the same ``forward`` and ``adjoint``
    methods.
    """

    def __init__(self, operator: torch.Tensor, volume_shape: tuple[int, int, int]):
        super().__init__()
        matrix = torch.as_tensor(operator)
        if matrix.ndim != 3:
            raise ValueError("operator must have shape [view,detector,pixel]")
        if matrix.shape[-1] != int(volume_shape[-2] * volume_shape[-1]):
            raise ValueError("operator pixels and volume shape disagree")
        self.register_buffer("operator", matrix)
        self.volume_shape = tuple(int(value) for value in volume_shape)

    def forward(self, volume: torch.Tensor, batch: BOSTBatch) -> torch.Tensor:
        batch.validate()
        values = volume[:, 0] if volume.ndim == 5 else volume
        if values.ndim != 4 or tuple(values.shape[1:]) != self.volume_shape:
            raise ValueError("volume has an incompatible shape")
        flat = values.reshape(values.shape[0], values.shape[1], -1)
        projected = torch.einsum("vnp,bdp->bdvn", self.operator, flat)
        return projected * batch.view_mask[:, None, :, None]

    def adjoint(self, residual: torch.Tensor, batch: BOSTBatch) -> torch.Tensor:
        batch.validate()
        if residual.shape != batch.observation.shape:
            raise ValueError("residual and observation shapes disagree")
        weighted = residual * batch.view_mask[:, None, :, None]
        flat = torch.einsum("vnp,bdvn->bdp", self.operator, weighted)
        return flat.reshape(len(residual), 1, *self.volume_shape)

    def weighted_gradient(
        self,
        volume: torch.Tensor,
        batch: BOSTBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = (batch.observation - self.forward(volume, batch))
        active = batch.active_observation_mask()
        sigma = batch.expanded_noise_std()
        weighted = torch.where(active, residual / (sigma * sigma), torch.zeros_like(residual))
        return self.adjoint(weighted, batch), residual

    @torch.no_grad()
    def weighted_lipschitz(
        self,
        batch: BOSTBatch,
        *,
        power_iterations: int = 16,
    ) -> torch.Tensor:
        """Estimate ``lambda_max(A^T C^-1 A)`` for each sample."""

        if power_iterations < 2:
            raise ValueError("power_iterations must be at least two")
        support = batch.expanded_support().to(self.operator)
        current = torch.ones_like(support) * support
        current /= torch.linalg.vector_norm(current.flatten(1), dim=1)[:, None, None, None, None]
        sigma = batch.expanded_noise_std()
        active = batch.active_observation_mask()
        for _ in range(int(power_iterations)):
            projected = self.forward(current, batch)
            weighted = torch.where(active, projected / (sigma * sigma), torch.zeros_like(projected))
            candidate = self.adjoint(weighted, batch) * support
            norm = torch.linalg.vector_norm(candidate.flatten(1), dim=1).clamp_min(1e-12)
            current = candidate / norm[:, None, None, None, None]
        projected = self.forward(current, batch)
        weighted = torch.where(active, projected / (sigma * sigma), torch.zeros_like(projected))
        normal = self.adjoint(weighted, batch) * support
        rayleigh = torch.sum(current * normal, dim=(1, 2, 3, 4))
        return rayleigh.clamp_min(1e-8)

    @torch.no_grad()
    def weighted_lipschitz_exact(self, batch: BOSTBatch) -> torch.Tensor:
        """Return the small-matrix weighted normal-operator spectral norm.

        This expensive path is intended only for the tiny audit grids in this
        repository. It avoids treating a finite power-iteration estimate as a
        mathematical upper bound.
        """

        batch.validate()
        sigma = batch.expanded_noise_std().to(self.operator)
        active = batch.active_observation_mask().to(self.operator)
        rows = []
        for sample in range(len(batch.geometry_ids)):
            depth_norms = []
            for depth_index in range(batch.observation.shape[1]):
                scale = active[sample, depth_index] / sigma[sample, depth_index]
                weighted = self.operator * scale[:, :, None]
                singular = torch.linalg.svdvals(weighted.flatten(0, 1))[0]
                depth_norms.append(singular.square())
            rows.append(torch.stack(depth_norms).max())
        return torch.stack(rows).clamp_min(1e-8)

    @torch.no_grad()
    def adjoint_relative_error(self, batch: BOSTBatch, seed: int = 0) -> float:
        generator = torch.Generator(device=self.operator.device).manual_seed(int(seed))
        volume = torch.randn(
            (len(batch.geometry_ids), 1, *self.volume_shape),
            generator=generator,
            device=self.operator.device,
            dtype=self.operator.dtype,
        )
        residual = torch.randn(
            batch.observation.shape,
            generator=generator,
            device=self.operator.device,
            dtype=self.operator.dtype,
        )
        residual = residual * batch.view_mask[:, None, :, None]
        lhs = torch.sum(self.forward(volume, batch) * residual)
        rhs = torch.sum(volume * self.adjoint(residual, batch))
        return float(torch.abs(lhs - rhs) / torch.maximum(torch.abs(lhs), torch.abs(rhs)).clamp_min(1e-12))


class DenseVolumeLinearBOST(torch.nn.Module):
    """Small fully three-dimensional BOST adapter for independent audits.

    ``operator`` has shape ``[detector_z, view, detector_x, voxel]``. Unlike
    :class:`DepthSeparableLinearBOST`, a ray may interpolate across multiple z
    planes, so this adapter can represent cone, tilted, or prescribed curved
    paths while retaining an exact declared adjoint.
    """

    def __init__(self, operator: torch.Tensor, volume_shape: tuple[int, int, int]):
        super().__init__()
        matrix = torch.as_tensor(operator)
        if matrix.ndim != 4:
            raise ValueError(
                "operator must have shape [detector_z,view,detector_x,voxel]"
            )
        if matrix.shape[-1] != int(volume_shape[0] * volume_shape[1] * volume_shape[2]):
            raise ValueError("operator voxels and volume shape disagree")
        if matrix.shape[0] != int(volume_shape[0]):
            raise ValueError("detector_z and volume depth must agree in this adapter")
        self.register_buffer("operator", matrix)
        self.volume_shape = tuple(int(value) for value in volume_shape)

    def forward(self, volume: torch.Tensor, batch: BOSTBatch) -> torch.Tensor:
        batch.validate()
        values = volume[:, 0] if volume.ndim == 5 else volume
        if values.ndim != 4 or tuple(values.shape[1:]) != self.volume_shape:
            raise ValueError("volume has an incompatible shape")
        if batch.observation.shape[1:4] != self.operator.shape[:3]:
            raise ValueError("batch observation and dense operator shapes disagree")
        projected = torch.einsum("dvnp,bp->bdvn", self.operator, values.flatten(1))
        return projected * batch.view_mask[:, None, :, None]

    def adjoint(self, residual: torch.Tensor, batch: BOSTBatch) -> torch.Tensor:
        batch.validate()
        if residual.shape != batch.observation.shape:
            raise ValueError("residual and observation shapes disagree")
        weighted = residual * batch.view_mask[:, None, :, None]
        flat = torch.einsum("dvnp,bdvn->bp", self.operator, weighted)
        return flat.reshape(len(residual), 1, *self.volume_shape)

    def weighted_gradient(
        self,
        volume: torch.Tensor,
        batch: BOSTBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = batch.observation - self.forward(volume, batch)
        active = batch.active_observation_mask()
        sigma = batch.expanded_noise_std()
        weighted = torch.where(active, residual / (sigma * sigma), torch.zeros_like(residual))
        return self.adjoint(weighted, batch), residual

    @torch.no_grad()
    def weighted_lipschitz(
        self,
        batch: BOSTBatch,
        *,
        power_iterations: int = 20,
    ) -> torch.Tensor:
        if power_iterations < 2:
            raise ValueError("power_iterations must be at least two")
        support = batch.expanded_support().to(self.operator)
        current = torch.ones_like(support) * support
        norm = torch.linalg.vector_norm(current.flatten(1), dim=1).clamp_min(1e-12)
        current = current / norm[:, None, None, None, None]
        sigma = batch.expanded_noise_std()
        active = batch.active_observation_mask()
        for _ in range(int(power_iterations)):
            projected = self.forward(current, batch)
            weighted = torch.where(active, projected / (sigma * sigma), torch.zeros_like(projected))
            candidate = self.adjoint(weighted, batch) * support
            norm = torch.linalg.vector_norm(candidate.flatten(1), dim=1).clamp_min(1e-12)
            current = candidate / norm[:, None, None, None, None]
        projected = self.forward(current, batch)
        weighted = torch.where(active, projected / (sigma * sigma), torch.zeros_like(projected))
        normal = self.adjoint(weighted, batch) * support
        return torch.sum(current * normal, dim=(1, 2, 3, 4)).clamp_min(1e-8)

    @torch.no_grad()
    def weighted_lipschitz_exact(self, batch: BOSTBatch) -> torch.Tensor:
        """Return the exact small-matrix weighted normal spectral norm."""

        batch.validate()
        sigma = batch.expanded_noise_std().to(self.operator)
        active = batch.active_observation_mask().to(self.operator)
        rows = []
        for sample in range(len(batch.geometry_ids)):
            scale = active[sample] / sigma[sample]
            weighted = self.operator * scale[:, :, :, None]
            singular = torch.linalg.svdvals(weighted.flatten(0, 2))[0]
            rows.append(singular.square())
        return torch.stack(rows).clamp_min(1e-8)

    @torch.no_grad()
    def adjoint_relative_error(self, batch: BOSTBatch, seed: int = 0) -> float:
        generator = torch.Generator(device=self.operator.device).manual_seed(int(seed))
        volume = torch.randn(
            (len(batch.geometry_ids), 1, *self.volume_shape),
            generator=generator,
            device=self.operator.device,
            dtype=self.operator.dtype,
        )
        residual = torch.randn(
            batch.observation.shape,
            generator=generator,
            device=self.operator.device,
            dtype=self.operator.dtype,
        )
        residual = residual * batch.view_mask[:, None, :, None]
        lhs = torch.sum(self.forward(volume, batch) * residual)
        rhs = torch.sum(volume * self.adjoint(residual, batch))
        return float(
            torch.abs(lhs - rhs)
            / torch.maximum(torch.abs(lhs), torch.abs(rhs)).clamp_min(1e-12)
        )
