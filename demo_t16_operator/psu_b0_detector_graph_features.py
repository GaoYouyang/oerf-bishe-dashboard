"""Detector-neighborhood and pose-aware descriptors for sparse BOST rays."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


DETECTOR_GRAPH_FEATURE_SCHEMA = "psu-b0-detector-graph-features-1.0"


def build_detector_knn_graph(
    detector_xy: np.ndarray,
    *,
    view_count: int,
    rays_per_view: int,
    neighbor_count: int = 8,
    least_squares_ridge: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Build local weighted least-squares gradients on irregular pixels."""

    views = int(view_count)
    rays = int(rays_per_view)
    neighbors = int(neighbor_count)
    coordinates = np.asarray(detector_xy, dtype=np.float64)
    if coordinates.shape == (views * rays, 2):
        coordinates = coordinates.reshape(views, rays, 2)
    if coordinates.shape != (views, rays, 2):
        raise ValueError("detector_xy must align with view and ray counts")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("detector_xy must be finite")
    if neighbors < 3 or neighbors >= rays:
        raise ValueError("neighbor_count must lie in [3, rays_per_view)")
    if float(least_squares_ridge) <= 0.0:
        raise ValueError("least_squares_ridge must be positive")

    neighbor_index = np.empty((views, rays, neighbors), dtype=np.int64)
    neighbor_weight = np.empty((views, rays, neighbors), dtype=np.float64)
    gradient_coefficients = np.empty(
        (views, rays, 2, neighbors),
        dtype=np.float64,
    )
    neighbor_distance = np.empty((views, rays, neighbors), dtype=np.float64)
    for view in range(views):
        xy = coordinates[view]
        displacement = xy[None, :, :] - xy[:, None, :]
        squared_distance = np.sum(displacement**2, axis=-1)
        np.fill_diagonal(squared_distance, np.inf)
        candidate = np.argpartition(
            squared_distance,
            kth=neighbors - 1,
            axis=1,
        )[:, :neighbors]
        candidate_distance = np.take_along_axis(
            squared_distance,
            candidate,
            axis=1,
        )
        order = np.argsort(candidate_distance, axis=1)
        selected = np.take_along_axis(candidate, order, axis=1)
        selected_squared = np.take_along_axis(
            squared_distance,
            selected,
            axis=1,
        )
        selected_distance = np.sqrt(selected_squared)
        local_scale = np.maximum(selected_distance[:, -1:], 1e-12)
        weight = np.exp(-selected_squared / local_scale**2)
        weight = weight / np.sum(weight, axis=1, keepdims=True)
        coefficient = np.empty((rays, 2, neighbors), dtype=np.float64)
        for ray in range(rays):
            delta = xy[selected[ray]] - xy[ray]
            weighted_design = delta.T * weight[ray][None, :]
            normal = weighted_design @ delta
            local_ridge = float(least_squares_ridge) * max(
                float(np.trace(normal)) / 2.0,
                1e-12,
            )
            coefficient[ray] = np.linalg.solve(
                normal + local_ridge * np.eye(2),
                weighted_design,
            )
        neighbor_index[view] = selected
        neighbor_weight[view] = weight
        gradient_coefficients[view] = coefficient
        neighbor_distance[view] = selected_distance

    return {
        "detector_xy": torch.as_tensor(coordinates, dtype=torch.float64),
        "neighbor_index": torch.as_tensor(
            neighbor_index,
            dtype=torch.int64,
        ),
        "neighbor_weight": torch.as_tensor(
            neighbor_weight,
            dtype=torch.float64,
        ),
        "gradient_coefficients": torch.as_tensor(
            gradient_coefficients,
            dtype=torch.float64,
        ),
        "neighbor_distance": torch.as_tensor(
            neighbor_distance,
            dtype=torch.float64,
        ),
    }


def detector_graph_diagnostics(
    graph: dict[str, torch.Tensor],
) -> dict[str, Any]:
    distance = graph["neighbor_distance"].detach().cpu().numpy()
    coordinates = graph["detector_xy"].detach().cpu().numpy()
    return {
        "view_count": int(distance.shape[0]),
        "rays_per_view": int(distance.shape[1]),
        "neighbor_count": int(distance.shape[2]),
        "nearest_neighbor_distance_median": float(
            np.median(distance[..., 0])
        ),
        "furthest_selected_neighbor_distance_median": float(
            np.median(distance[..., -1])
        ),
        "detector_x_minimum": float(np.min(coordinates[..., 0])),
        "detector_x_maximum": float(np.max(coordinates[..., 0])),
        "detector_y_minimum": float(np.min(coordinates[..., 1])),
        "detector_y_maximum": float(np.max(coordinates[..., 1])),
    }


def _masked_moments(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("masked moments require aligned [batch,view] tensors")
    count = mask.sum(dim=1)
    safe_count = count.clamp_min(1)
    selected = torch.where(mask, values, torch.zeros_like(values))
    mean = selected.sum(dim=1) / safe_count
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
    return mean, standard, minimum, maximum


def _projection_frame(
    projection_u_xyz: torch.Tensor,
    projection_v_xyz: torch.Tensor,
    *,
    view_count: int,
    rays_per_view: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    views = int(view_count)
    rays = int(rays_per_view)
    if projection_u_xyz.shape != (views * rays, 3):
        raise ValueError("projection_u_xyz does not match the ray layout")
    if projection_v_xyz.shape != projection_u_xyz.shape:
        raise ValueError("projection vectors must align")
    u = projection_u_xyz.reshape(views, rays, 3).mean(dim=1)
    v = projection_v_xyz.reshape(views, rays, 3).mean(dim=1)
    u = u / torch.linalg.vector_norm(u, dim=1, keepdim=True).clamp_min(1e-12)
    v = v - torch.sum(u * v, dim=1, keepdim=True) * u
    v = v / torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-12)
    return u, v


def detector_graph_front_features(
    observation_uv: torch.Tensor,
    *,
    sigma_by_view: torch.Tensor,
    view_mask: torch.Tensor,
    graph: dict[str, torch.Tensor],
    projection_u_xyz: torch.Tensor,
    projection_v_xyz: torch.Tensor,
    rays_per_view: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Summarize observable detector fronts and their pose consistency.

    The descriptor uses only measured displacement vectors, declared noise
    scales, view availability, detector coordinates, and forward projection
    vectors. It does not use a reconstructed volume, truth field, morphology
    label, or post-iteration residual.
    """

    if observation_uv.ndim != 3 or observation_uv.shape[-1] != 2:
        raise ValueError("observation_uv must have shape [batch,ray,2]")
    batch, ray_count, _ = observation_uv.shape
    if sigma_by_view.shape != view_mask.shape:
        raise ValueError("sigma_by_view and view_mask must align")
    if sigma_by_view.shape[0] != batch:
        raise ValueError("view metadata must align with observations")
    view_count = int(sigma_by_view.shape[1])
    rays = int(rays_per_view)
    if ray_count != view_count * rays:
        raise ValueError("rays_per_view does not cover the observations")
    if torch.any(sigma_by_view <= 0.0):
        raise ValueError("sigma_by_view must be positive")
    active = view_mask > 0.5
    if torch.any(active.sum(dim=1) < 1):
        raise ValueError("each sample requires at least one active view")

    values = observation_uv.reshape(batch, view_count, rays, 2)
    white = values / sigma_by_view[:, :, None, None]
    white = torch.where(
        active[:, :, None, None],
        white,
        torch.zeros_like(white),
    )
    neighbor_index = graph["neighbor_index"].to(white.device)
    neighbor_weight = graph["neighbor_weight"].to(white)
    gradient_coefficients = graph["gradient_coefficients"].to(white)
    if neighbor_index.shape[:2] != (view_count, rays):
        raise ValueError("detector graph does not match the observation layout")

    neighbor_parts = [
        white[:, view][:, neighbor_index[view]]
        for view in range(view_count)
    ]
    neighbor_values = torch.stack(neighbor_parts, dim=1)
    difference = neighbor_values - white[:, :, :, None, :]
    jacobian = torch.einsum(
        "vrak,bvrkc->bvrac",
        gradient_coefficients,
        difference,
    )
    weighted_difference_energy = torch.sum(
        neighbor_weight[None, :, :, :, None]
        * difference.square(),
        dim=(2, 3, 4),
    ) / float(rays)
    signal_rms = torch.sqrt(
        torch.mean(white.square(), dim=(2, 3)).clamp_min(1e-20)
    )
    contrast_rms = torch.sqrt(weighted_difference_energy.clamp_min(1e-20))
    jacobian_energy = torch.mean(
        jacobian.square(),
        dim=(2, 3, 4),
    )
    jacobian_rms = torch.sqrt(jacobian_energy.clamp_min(1e-20))
    divergence = jacobian[..., 0, 0] + jacobian[..., 1, 1]
    curl = jacobian[..., 0, 1] - jacobian[..., 1, 0]
    divergence_energy = torch.mean(divergence.square(), dim=2)
    curl_energy = torch.mean(curl.square(), dim=2)
    divergence_curl_balance = (
        divergence_energy - curl_energy
    ) / (divergence_energy + curl_energy).clamp_min(1e-20)

    front_energy = torch.sum(jacobian.square(), dim=(3, 4))
    top_count = max(1, int(np.ceil(0.10 * rays)))
    top_energy = torch.topk(
        front_energy,
        k=top_count,
        dim=2,
        sorted=False,
    ).values.sum(dim=2)
    front_concentration = top_energy / front_energy.sum(dim=2).clamp_min(1e-20)
    structure = torch.einsum(
        "bvrkc,bvrlc->bvkl",
        jacobian,
        jacobian,
    ) / float(rays)
    structure_xx = structure[..., 0, 0]
    structure_xy = structure[..., 0, 1]
    structure_yy = structure[..., 1, 1]
    trace = structure_xx + structure_yy
    discriminant = torch.sqrt(
        (
            (structure_xx - structure_yy).square()
            + 4.0 * structure_xy.square()
        ).clamp_min(0.0)
    )
    eigenvalues = torch.stack(
        (
            0.5 * (trace - discriminant),
            0.5 * (trace + discriminant),
        ),
        dim=-1,
    )
    anisotropy = (
        eigenvalues[..., 1] - eigenvalues[..., 0]
    ) / eigenvalues.sum(dim=-1).clamp_min(1e-20)
    principal_angle = 0.5 * torch.atan2(
        2.0 * structure_xy,
        structure_xx - structure_yy,
    )
    principal_image = torch.stack(
        (torch.cos(principal_angle), torch.sin(principal_angle)),
        dim=-1,
    )

    per_view_metrics = (
        ("log_white_vector_rms", torch.log1p(signal_rms)),
        (
            "neighbor_contrast_to_signal_ratio",
            contrast_rms / signal_rms.clamp_min(1e-12),
        ),
        ("log_local_jacobian_rms", torch.log1p(jacobian_rms)),
        ("front_top10_energy_share", front_concentration),
        ("front_structure_anisotropy", anisotropy),
        ("divergence_curl_energy_balance", divergence_curl_balance),
    )
    statistic_names = ("mean", "std", "min", "max")
    feature_parts = []
    feature_names: list[str] = []
    for metric_name, metric in per_view_metrics:
        feature_parts.extend(_masked_moments(metric, active))
        feature_names.extend(
            f"{metric_name}_{statistic}" for statistic in statistic_names
        )

    frame_u, frame_v = _projection_frame(
        projection_u_xyz.to(white),
        projection_v_xyz.to(white),
        view_count=view_count,
        rays_per_view=rays,
    )
    principal_world = (
        principal_image[..., 0, None] * frame_u[None]
        + principal_image[..., 1, None] * frame_v[None]
    )
    principal_world = principal_world / torch.linalg.vector_norm(
        principal_world,
        dim=2,
        keepdim=True,
    ).clamp_min(1e-20)
    orientation_weight = (
        active.to(white)
        * anisotropy.clamp_min(0.0)
        * jacobian_energy
    )
    weight_total = orientation_weight.sum(dim=1, keepdim=True)
    normalized_weight = orientation_weight / weight_total.clamp_min(1e-20)
    world_dyadic = torch.einsum(
        "bv,bvi,bvj->bij",
        normalized_weight,
        principal_world,
        principal_world,
    )
    # MPS does not currently implement symmetric eigendecomposition. These
    # matrices are only [batch,3,3], so a tiny CPU decomposition avoids moving
    # the observation or graph feature computation off the accelerator.
    world_eigenvalues = torch.linalg.eigvalsh(
        world_dyadic.detach().cpu()
    ).flip(dims=(-1,)).to(white)
    valid_orientation = weight_total[:, 0] > 1e-20
    world_eigenvalues = torch.where(
        valid_orientation[:, None],
        world_eigenvalues,
        torch.zeros_like(world_eigenvalues),
    )
    orientation_entropy = -torch.sum(
        torch.where(
            world_eigenvalues > 0.0,
            world_eigenvalues
            * torch.log(world_eigenvalues.clamp_min(1e-20)),
            torch.zeros_like(world_eigenvalues),
        ),
        dim=1,
    )
    view_weight_entropy = -torch.sum(
        torch.where(
            normalized_weight > 0.0,
            normalized_weight
            * torch.log(normalized_weight.clamp_min(1e-20)),
            torch.zeros_like(normalized_weight),
        ),
        dim=1,
    )
    active_count = active.sum(dim=1).to(white)
    effective_front_view_fraction = torch.where(
        valid_orientation,
        torch.exp(view_weight_entropy) / active_count.clamp_min(1.0),
        torch.zeros_like(view_weight_entropy),
    )
    active_fraction = active_count / float(view_count)
    pose_features = (
        active_fraction,
        world_eigenvalues[:, 0],
        world_eigenvalues[:, 1],
        world_eigenvalues[:, 2],
        orientation_entropy,
        effective_front_view_fraction,
    )
    pose_names = (
        "active_view_fraction",
        "world_front_orientation_eigenvalue_1",
        "world_front_orientation_eigenvalue_2",
        "world_front_orientation_eigenvalue_3",
        "world_front_orientation_entropy",
        "effective_front_view_fraction",
    )
    feature_parts.extend(pose_features)
    feature_names.extend(pose_names)
    features = torch.stack(feature_parts, dim=1)
    if not torch.all(torch.isfinite(features)):
        raise ValueError("detector graph front features are not finite")
    return features, tuple(feature_names)


__all__ = [
    "DETECTOR_GRAPH_FEATURE_SCHEMA",
    "build_detector_knn_graph",
    "detector_graph_diagnostics",
    "detector_graph_front_features",
]
