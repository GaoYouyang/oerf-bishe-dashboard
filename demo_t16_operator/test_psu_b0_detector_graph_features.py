from __future__ import annotations

import numpy as np
import torch

from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
    detector_graph_front_features,
)


def _fixture(
    *,
    seed: int = 4,
    views: int = 3,
    rays: int = 24,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(-0.4, 0.4, size=(views, rays, 2))
    projection_u = torch.zeros((views, rays, 3), dtype=torch.float64)
    projection_v = torch.zeros_like(projection_u)
    frames = (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    for view, (u, v) in enumerate(frames):
        projection_u[view] = torch.tensor(u)
        projection_v[view] = torch.tensor(v)
    sigma = torch.full((2, views), 0.2, dtype=torch.float64)
    mask = torch.ones((2, views), dtype=torch.float64)
    return (
        coordinates,
        projection_u.reshape(-1, 3),
        projection_v.reshape(-1, 3),
        sigma,
        mask,
    )


def test_detector_features_are_invariant_to_within_view_ray_permutation() -> None:
    coordinates, projection_u, projection_v, sigma, mask = _fixture()
    views, rays = coordinates.shape[:2]
    rng = np.random.default_rng(9)
    observation = torch.as_tensor(
        rng.normal(size=(2, views * rays, 2)),
        dtype=torch.float64,
    )
    graph = build_detector_knn_graph(
        coordinates,
        view_count=views,
        rays_per_view=rays,
        neighbor_count=6,
    )
    reference, names = detector_graph_front_features(
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        graph=graph,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        rays_per_view=rays,
    )

    permutation = np.stack(
        [rng.permutation(rays) for _ in range(views)],
        axis=0,
    )
    permuted_coordinates = np.stack(
        [coordinates[view, permutation[view]] for view in range(views)]
    )
    observation_by_view = observation.reshape(2, views, rays, 2)
    permuted_observation = torch.stack(
        [
            observation_by_view[:, view, permutation[view]]
            for view in range(views)
        ],
        dim=1,
    ).reshape(2, views * rays, 2)
    projection_u_by_view = projection_u.reshape(views, rays, 3)
    projection_v_by_view = projection_v.reshape(views, rays, 3)
    permuted_u = torch.stack(
        [
            projection_u_by_view[view, permutation[view]]
            for view in range(views)
        ]
    ).reshape(-1, 3)
    permuted_v = torch.stack(
        [
            projection_v_by_view[view, permutation[view]]
            for view in range(views)
        ]
    ).reshape(-1, 3)
    permuted_graph = build_detector_knn_graph(
        permuted_coordinates,
        view_count=views,
        rays_per_view=rays,
        neighbor_count=6,
    )
    candidate, candidate_names = detector_graph_front_features(
        permuted_observation,
        sigma_by_view=sigma,
        view_mask=mask,
        graph=permuted_graph,
        projection_u_xyz=permuted_u,
        projection_v_xyz=permuted_v,
        rays_per_view=rays,
    )
    assert names == candidate_names
    assert torch.allclose(reference, candidate, atol=1e-10, rtol=1e-10)


def test_linear_front_has_high_structure_anisotropy() -> None:
    coordinates, projection_u, projection_v, sigma, mask = _fixture()
    views, rays = coordinates.shape[:2]
    graph = build_detector_knn_graph(
        coordinates,
        view_count=views,
        rays_per_view=rays,
        neighbor_count=8,
    )
    vector = np.zeros((1, views, rays, 2), dtype=np.float64)
    vector[0, ..., 0] = coordinates[..., 0]
    observation = torch.as_tensor(vector.reshape(1, views * rays, 2))
    features, names = detector_graph_front_features(
        observation,
        sigma_by_view=sigma[:1],
        view_mask=mask[:1],
        graph=graph,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        rays_per_view=rays,
    )
    anisotropy = features[
        0,
        names.index("front_structure_anisotropy_mean"),
    ]
    assert float(anisotropy) > 0.95
    assert torch.all(torch.isfinite(features))


def test_inactive_view_values_do_not_change_features() -> None:
    coordinates, projection_u, projection_v, sigma, mask = _fixture()
    views, rays = coordinates.shape[:2]
    graph = build_detector_knn_graph(
        coordinates,
        view_count=views,
        rays_per_view=rays,
        neighbor_count=6,
    )
    rng = np.random.default_rng(18)
    first = torch.as_tensor(
        rng.normal(size=(1, views, rays, 2)),
        dtype=torch.float64,
    )
    second = first.clone()
    second[:, 2] = 1e6
    partial_mask = mask[:1].clone()
    partial_mask[:, 2] = 0.0
    first_features, _ = detector_graph_front_features(
        first.reshape(1, views * rays, 2),
        sigma_by_view=sigma[:1],
        view_mask=partial_mask,
        graph=graph,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        rays_per_view=rays,
    )
    second_features, _ = detector_graph_front_features(
        second.reshape(1, views * rays, 2),
        sigma_by_view=sigma[:1],
        view_mask=partial_mask,
        graph=graph,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        rays_per_view=rays,
    )
    assert torch.allclose(first_features, second_features)
