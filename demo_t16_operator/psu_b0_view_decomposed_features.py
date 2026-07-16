"""Permutation-invariant diagnostics for camera-wise BOST adjoint fields."""

from __future__ import annotations

import torch


VIEW_DECOMPOSED_FEATURE_SCHEMA = "psu-b0-view-decomposed-features-1.0"


def _masked_moments(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("masked moments require aligned [batch,item] tensors")
    count = mask.sum(dim=1)
    safe_count = count.clamp_min(1)
    masked = torch.where(mask, values, torch.zeros_like(values))
    mean = masked.sum(dim=1) / safe_count
    centered = torch.where(
        mask,
        values - mean[:, None],
        torch.zeros_like(values),
    )
    standard = torch.sqrt(
        (centered.square().sum(dim=1) / safe_count).clamp_min(0.0)
    )
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
    has_value = count > 0
    minimum = torch.where(has_value, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_value, maximum, torch.zeros_like(maximum))
    return mean, standard, minimum, maximum


def view_adjoint_conflict_features(
    per_view_adjoint: torch.Tensor,
    *,
    view_mask: torch.Tensor,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Summarize agreement and cancellation among per-camera adjoint fields.

    The returned descriptor is invariant to a simultaneous permutation of the
    camera fields and mask. It uses no truth field, reconstructed target, or
    post-iteration residual.
    """

    if per_view_adjoint.ndim != 6 or per_view_adjoint.shape[2] != 1:
        raise ValueError(
            "per_view_adjoint must have shape [batch,view,1,z,y,x]"
        )
    batch, view_count = per_view_adjoint.shape[:2]
    if view_mask.shape != (batch, view_count):
        raise ValueError("view_mask must align with per_view_adjoint")
    if min(per_view_adjoint.shape[-3:]) < 2:
        raise ValueError("adjoint grid dimensions must be at least two")
    if torch.any(~torch.isfinite(per_view_adjoint)):
        raise ValueError("per_view_adjoint must be finite")
    if torch.any(~torch.isfinite(view_mask)):
        raise ValueError("view_mask must be finite")

    active = view_mask > 0.5
    active_count = active.sum(dim=1)
    if torch.any(active_count < 1):
        raise ValueError("each sample requires at least one active view")
    volume = per_view_adjoint[:, :, 0]
    volume = torch.where(
        active[:, :, None, None, None],
        volume,
        torch.zeros_like(volume),
    )
    flat = volume.flatten(2)
    voxel_count = flat.shape[-1]
    norm = torch.linalg.vector_norm(flat, dim=2)
    rms = norm / float(voxel_count) ** 0.5
    positive = norm > torch.finfo(norm.dtype).eps
    valid_view = active & positive
    log_rms_stats = _masked_moments(torch.log1p(rms), active)

    norm_sum = norm.sum(dim=1).clamp_min(1e-20)
    share = norm / norm_sum[:, None]
    share_entropy = -(
        torch.where(
            active,
            share * torch.log(share.clamp_min(1e-20)),
            torch.zeros_like(share),
        ).sum(dim=1)
    )
    effective_view_fraction = (
        torch.exp(share_entropy) / active_count.to(norm)
    )
    maximum_share = torch.where(
        active,
        share,
        torch.zeros_like(share),
    ).amax(dim=1)

    normalized = flat / norm[:, :, None].clamp_min(1e-20)
    cosine = torch.einsum("bvi,bwi->bvw", normalized, normalized)
    upper = torch.triu(
        torch.ones(
            (view_count, view_count),
            dtype=torch.bool,
            device=volume.device,
        ),
        diagonal=1,
    )
    pair_mask = (
        valid_view[:, :, None]
        & valid_view[:, None, :]
        & upper[None]
    )
    pair_values = cosine.reshape(batch, -1)
    pair_mask_flat = pair_mask.reshape(batch, -1)
    pair_stats = _masked_moments(pair_values, pair_mask_flat)
    pair_count = pair_mask_flat.sum(dim=1).clamp_min(1)
    negative_pair_fraction = (
        ((pair_values < 0.0) & pair_mask_flat).sum(dim=1) / pair_count
    )

    pooled = flat.sum(dim=1)
    pooled_norm = torch.linalg.vector_norm(pooled, dim=1)
    cancellation_ratio = pooled_norm / norm_sum
    squared_norm_sum = norm.square().sum(dim=1).clamp_min(1e-20)
    coherent_energy_ratio = (
        pooled_norm.square()
        / squared_norm_sum
        / active_count.to(norm)
    )
    pooled_cosine = torch.einsum(
        "bvi,bi->bv",
        normalized,
        pooled / pooled_norm[:, None].clamp_min(1e-20),
    )
    pooled_mask = valid_view & (pooled_norm[:, None] > 1e-20)
    pooled_stats = _masked_moments(pooled_cosine, pooled_mask)

    features = torch.stack(
        (
            *log_rms_stats,
            share_entropy,
            effective_view_fraction,
            maximum_share,
            cancellation_ratio,
            coherent_energy_ratio,
            *pair_stats,
            negative_pair_fraction,
            *pooled_stats,
        ),
        dim=1,
    )
    statistic_names = ("mean", "std", "min", "max")
    names = (
        *(f"view_log_adjoint_rms_{name}" for name in statistic_names),
        "view_norm_share_entropy",
        "view_effective_fraction",
        "view_maximum_norm_share",
        "view_pooled_to_sum_norm_ratio",
        "view_coherent_energy_ratio",
        *(f"view_pair_cosine_{name}" for name in statistic_names),
        "view_negative_pair_fraction",
        *(f"view_to_pooled_cosine_{name}" for name in statistic_names),
    )
    if not torch.all(torch.isfinite(features)):
        raise ValueError("view adjoint conflict features are not finite")
    return features, tuple(names)
