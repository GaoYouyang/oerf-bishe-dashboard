"""Observable descriptors of the first BOST normal-equation residual."""

from __future__ import annotations

from typing import Iterable

import torch


INITIAL_NORMAL_FEATURE_SCHEMA = "psu-b0-initial-normal-features-1.0"


def _masked_moments(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=1).clamp_min(1)
    masked = torch.where(mask, values, torch.zeros_like(values))
    mean = masked.sum(dim=1) / count
    centered = torch.where(mask, values - mean[:, None], torch.zeros_like(values))
    variance = centered.square().sum(dim=1) / count
    standard = torch.sqrt(variance.clamp_min(1e-12))
    minimum = torch.where(
        mask,
        values,
        torch.full_like(values, torch.inf),
    ).amin(dim=1)
    maximum = torch.where(
        mask,
        values,
        torch.full_like(values, -torch.inf),
    ).amax(dim=1)
    return mean, standard, minimum, maximum


def measurement_metadata_features(
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    rays_per_view: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Describe available cameras and whitened measurements without truth."""

    if observation_uv.ndim != 3 or observation_uv.shape[-1] != 2:
        raise ValueError("observation_uv must have shape [batch,ray,2]")
    batch, ray_count, _ = observation_uv.shape
    if sigma_by_view.shape != view_mask.shape:
        raise ValueError("sigma_by_view and view_mask must align")
    if sigma_by_view.shape[0] != batch:
        raise ValueError("view metadata must align with observations")
    view_count = int(sigma_by_view.shape[1])
    if ray_count != view_count * int(rays_per_view):
        raise ValueError("rays_per_view does not cover the observations")
    if torch.any(sigma_by_view <= 0.0):
        raise ValueError("sigma_by_view must be positive")
    active = view_mask > 0.5
    if torch.any(active.sum(dim=1) < 1):
        raise ValueError("each sample requires at least one active view")
    images = observation_uv.reshape(
        batch,
        view_count,
        int(rays_per_view),
        2,
    )
    white = images / sigma_by_view[:, :, None, None]
    component_rms = torch.sqrt(
        torch.mean(white.square(), dim=2).clamp_min(1e-20)
    )
    correlation = torch.mean(
        white[..., 0] * white[..., 1],
        dim=2,
    ) / (
        component_rms[..., 0] * component_rms[..., 1]
    ).clamp_min(1e-12)
    log_sigma = torch.log(sigma_by_view.clamp_min(1e-20))
    sigma_stats = _masked_moments(log_sigma, active)
    rms_u_stats = _masked_moments(
        torch.log1p(component_rms[..., 0]),
        active,
    )
    rms_v_stats = _masked_moments(
        torch.log1p(component_rms[..., 1]),
        active,
    )
    correlation_stats = _masked_moments(correlation, active)
    active_count = active.sum(dim=1).to(observation_uv) / view_count
    features = torch.cat(
        (
            view_mask.to(observation_uv),
            active_count[:, None],
            *[value[:, None] for value in sigma_stats],
            *[value[:, None] for value in rms_u_stats],
            *[value[:, None] for value in rms_v_stats],
            *[value[:, None] for value in correlation_stats],
        ),
        dim=1,
    )
    statistic_names = ("mean", "std", "min", "max")
    names = (
        *(f"view_mask_{index}" for index in range(view_count)),
        "active_view_fraction",
        *(f"log_sigma_{name}" for name in statistic_names),
        *(f"log_white_u_rms_{name}" for name in statistic_names),
        *(f"log_white_v_rms_{name}" for name in statistic_names),
        *(f"white_uv_correlation_{name}" for name in statistic_names),
    )
    if not torch.all(torch.isfinite(features)):
        raise ValueError("measurement metadata features are not finite")
    return features, tuple(names)


def _fraction_by_masks(
    weights: torch.Tensor,
    masks: Iterable[torch.Tensor],
) -> torch.Tensor:
    total = weights.sum(dim=(1, 2, 3)).clamp_min(1e-20)
    output = []
    for mask in masks:
        output.append(
            (weights * mask.to(weights)[None]).sum(dim=(1, 2, 3)) / total
        )
    return torch.stack(output, dim=1)


def initial_normal_spectral_features(
    initial_normal: torch.Tensor,
    *,
    radial_bins: int = 6,
    axis_bins: int = 4,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Summarize 3-D morphology visible in the shared first adjoint field."""

    if initial_normal.ndim != 5 or initial_normal.shape[1] != 1:
        raise ValueError("initial_normal must have shape [batch,1,z,y,x]")
    if min(initial_normal.shape[-3:]) < 4:
        raise ValueError("initial_normal grid dimensions must be at least four")
    if int(radial_bins) < 3 or int(axis_bins) < 2:
        raise ValueError("feature bin counts are too small")
    volume = initial_normal[:, 0]
    batch, nz, ny, nx = volume.shape
    rms = torch.sqrt(
        torch.mean(volume.square(), dim=(1, 2, 3)).clamp_min(1e-20)
    )
    normalized = volume / rms[:, None, None, None]
    spectrum = torch.fft.rfftn(normalized, dim=(-3, -2, -1))
    power = spectrum.abs().square()
    total_power = power.sum(dim=(1, 2, 3)).clamp_min(1e-20)
    probability = power / total_power[:, None, None, None]

    fz = torch.fft.fftfreq(nz, device=volume.device, dtype=volume.dtype)
    fy = torch.fft.fftfreq(ny, device=volume.device, dtype=volume.dtype)
    fx = torch.fft.rfftfreq(nx, device=volume.device, dtype=volume.dtype)
    zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square() + zz.square())
    radius_normalized = radius / radius.max().clamp_min(1e-12)
    radial_edges = torch.linspace(
        0.0,
        1.0,
        int(radial_bins) + 1,
        device=volume.device,
        dtype=volume.dtype,
    )
    radial_masks = []
    for index in range(int(radial_bins)):
        lower = radial_edges[index]
        upper = radial_edges[index + 1]
        if index + 1 == int(radial_bins):
            radial_masks.append(
                (radius_normalized >= lower) & (radius_normalized <= upper)
            )
        else:
            radial_masks.append(
                (radius_normalized >= lower) & (radius_normalized < upper)
            )
    radial_fraction = _fraction_by_masks(power, radial_masks)

    axis_fraction_parts = []
    axis_names = []
    for axis_name, frequency in (("x", xx), ("y", yy), ("z", zz)):
        normalized_frequency = torch.abs(frequency) / 0.5
        edges = torch.linspace(
            0.0,
            1.0,
            int(axis_bins) + 1,
            device=volume.device,
            dtype=volume.dtype,
        )
        masks = []
        for index in range(int(axis_bins)):
            lower = edges[index]
            upper = edges[index + 1]
            if index + 1 == int(axis_bins):
                masks.append(
                    (normalized_frequency >= lower)
                    & (normalized_frequency <= upper)
                )
            else:
                masks.append(
                    (normalized_frequency >= lower)
                    & (normalized_frequency < upper)
                )
            axis_names.append(f"spectral_{axis_name}_bin_{index}")
        axis_fraction_parts.append(_fraction_by_masks(power, masks))
    axis_fraction = torch.cat(axis_fraction_parts, dim=1)

    radius_mean = (
        probability * radius_normalized[None]
    ).sum(dim=(1, 2, 3))
    radius_variance = (
        probability
        * (radius_normalized[None] - radius_mean[:, None, None, None]).square()
    ).sum(dim=(1, 2, 3))
    nonzero_radius = radius.square().clamp_min(1e-12)
    directional = torch.stack(
        [
            (
                probability * xx.square()[None] / nonzero_radius[None]
            ).sum(dim=(1, 2, 3)),
            (
                probability * yy.square()[None] / nonzero_radius[None]
            ).sum(dim=(1, 2, 3)),
            (
                probability * zz.square()[None] / nonzero_radius[None]
            ).sum(dim=(1, 2, 3)),
            (
                probability * torch.abs(xx * yy)[None] / nonzero_radius[None]
            ).sum(dim=(1, 2, 3)),
            (
                probability * torch.abs(xx * zz)[None] / nonzero_radius[None]
            ).sum(dim=(1, 2, 3)),
            (
                probability * torch.abs(yy * zz)[None] / nonzero_radius[None]
            ).sum(dim=(1, 2, 3)),
        ],
        dim=1,
    )
    entropy = -(
        probability
        * torch.log(probability.clamp_min(1e-20))
    ).sum(dim=(1, 2, 3)) / torch.log(
        torch.as_tensor(
            probability[0].numel(),
            dtype=volume.dtype,
            device=volume.device,
        )
    )

    energy = normalized.square()
    energy_total = energy.sum(dim=(1, 2, 3)).clamp_min(1e-20)
    z_coord = torch.linspace(-1.0, 1.0, nz, device=volume.device, dtype=volume.dtype)
    y_coord = torch.linspace(-1.0, 1.0, ny, device=volume.device, dtype=volume.dtype)
    x_coord = torch.linspace(-1.0, 1.0, nx, device=volume.device, dtype=volume.dtype)
    z_grid, y_grid, x_grid = torch.meshgrid(
        z_coord,
        y_coord,
        x_coord,
        indexing="ij",
    )
    spatial_probability = energy / energy_total[:, None, None, None]
    centers = torch.stack(
        [
            (spatial_probability * x_grid[None]).sum(dim=(1, 2, 3)),
            (spatial_probability * y_grid[None]).sum(dim=(1, 2, 3)),
            (spatial_probability * z_grid[None]).sum(dim=(1, 2, 3)),
        ],
        dim=1,
    )
    centered_x = x_grid[None] - centers[:, 0, None, None, None]
    centered_y = y_grid[None] - centers[:, 1, None, None, None]
    centered_z = z_grid[None] - centers[:, 2, None, None, None]
    covariance = torch.stack(
        [
            (spatial_probability * centered_x.square()).sum(dim=(1, 2, 3)),
            (spatial_probability * centered_y.square()).sum(dim=(1, 2, 3)),
            (spatial_probability * centered_z.square()).sum(dim=(1, 2, 3)),
            (spatial_probability * centered_x * centered_y).sum(dim=(1, 2, 3)),
            (spatial_probability * centered_x * centered_z).sum(dim=(1, 2, 3)),
            (spatial_probability * centered_y * centered_z).sum(dim=(1, 2, 3)),
        ],
        dim=1,
    )
    gradient_energy = torch.stack(
        [
            torch.mean(torch.diff(normalized, dim=3).square(), dim=(1, 2, 3)),
            torch.mean(torch.diff(normalized, dim=2).square(), dim=(1, 2, 3)),
            torch.mean(torch.diff(normalized, dim=1).square(), dim=(1, 2, 3)),
        ],
        dim=1,
    )
    gradient_energy = gradient_energy / gradient_energy.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-20)
    flat = normalized.flatten(1)
    signed = torch.stack(
        [
            flat.mean(dim=1),
            torch.mean(flat.pow(3), dim=1),
            torch.mean(flat.pow(4), dim=1),
            torch.mean((flat > 0.0).to(flat), dim=1),
        ],
        dim=1,
    )
    features = torch.cat(
        (
            torch.log(rms.clamp_min(1e-20))[:, None],
            radial_fraction,
            axis_fraction,
            radius_mean[:, None],
            torch.sqrt(radius_variance.clamp_min(0.0))[:, None],
            entropy[:, None],
            directional,
            centers,
            covariance,
            gradient_energy,
            signed,
        ),
        dim=1,
    )
    names = (
        "log_initial_normal_rms",
        *(f"radial_power_bin_{index}" for index in range(int(radial_bins))),
        *axis_names,
        "spectral_radius_mean",
        "spectral_radius_std",
        "spectral_entropy",
        "spectral_direction_xx",
        "spectral_direction_yy",
        "spectral_direction_zz",
        "spectral_direction_abs_xy",
        "spectral_direction_abs_xz",
        "spectral_direction_abs_yz",
        "spatial_energy_center_x",
        "spatial_energy_center_y",
        "spatial_energy_center_z",
        "spatial_energy_cov_xx",
        "spatial_energy_cov_yy",
        "spatial_energy_cov_zz",
        "spatial_energy_cov_xy",
        "spatial_energy_cov_xz",
        "spatial_energy_cov_yz",
        "gradient_energy_x_fraction",
        "gradient_energy_y_fraction",
        "gradient_energy_z_fraction",
        "normalized_mean",
        "normalized_third_moment",
        "normalized_fourth_moment",
        "positive_fraction",
    )
    if features.shape != (batch, len(names)):
        raise RuntimeError("initial-normal feature schema drift")
    if not torch.all(torch.isfinite(features)):
        raise ValueError("initial-normal features are not finite")
    return features, tuple(names)


__all__ = [
    "INITIAL_NORMAL_FEATURE_SCHEMA",
    "initial_normal_spectral_features",
    "measurement_metadata_features",
]
