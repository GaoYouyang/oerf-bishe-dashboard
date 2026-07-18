from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from demo_t16_operator.jacru_n1_9_global_contrast import (
    SUPPORTED_CONTRAST_BASIS_KINDS,
    build_global_contrast_basis,
    camera_contrast_vectors,
    three_camera_helmert_contrasts,
)
from demo_t16_operator.jacru_synthetic_fixture import (
    JACRUSyntheticFixtureConfig,
    build_jacru_geometry,
)


def _geometry():
    return build_jacru_geometry(
        split="development",
        base_seed=1009,
        config=JACRUSyntheticFixtureConfig(detector_shape=(3, 3)),
    )


def _maps(observation_shape: tuple[int, ...], field_shape: tuple[int, ...]):
    calls = {"forward": 0, "adjoint": 0}
    generator = torch.Generator().manual_seed(41)
    matrix = torch.randn(
        (int(torch.tensor(observation_shape).prod()), int(torch.tensor(field_shape).prod())),
        generator=generator,
        dtype=torch.float64,
    )
    matrix = matrix / torch.linalg.vector_norm(matrix)

    def forward(field: torch.Tensor) -> torch.Tensor:
        calls["forward"] += 1
        return (matrix @ field.reshape(-1)).reshape(observation_shape)

    def adjoint(observation: torch.Tensor) -> torch.Tensor:
        calls["adjoint"] += 1
        return (matrix.T @ observation.reshape(-1)).reshape(field_shape)

    return forward, adjoint, calls


def test_three_camera_contrasts_are_centered_and_orthonormal() -> None:
    contrasts = three_camera_helmert_contrasts(_geometry())
    assert contrasts.shape == (2, 3)
    assert torch.allclose(
        torch.sum(contrasts, dim=1), torch.zeros(2, dtype=torch.float64), atol=1e-14
    )
    assert torch.allclose(
        contrasts @ contrasts.T, torch.eye(2, dtype=torch.float64), atol=1e-14
    )


@pytest.mark.parametrize(
    ("kind", "contrast_quantity"),
    (
        ("residual_contrast_global_k6_total", "warm_residual"),
        ("damping_contrast_global_k6_total", "damping"),
    ),
)
def test_global_contrast_basis_has_exact_rank_and_budget(
    kind: str, contrast_quantity: str
) -> None:
    assert SUPPORTED_CONTRAST_BASIS_KINDS == (
        "residual_contrast_global_k6_total",
        "damping_contrast_global_k6_total",
    )
    geometry = _geometry()
    observation_shape = (geometry.ray_count, 2)
    field_shape = (3, 3, 3)
    forward, adjoint, calls = _maps(observation_shape, field_shape)
    generator = torch.Generator().manual_seed(59)
    basis = build_global_contrast_basis(
        kind=kind,
        damping=torch.randn(observation_shape, generator=generator, dtype=torch.float64),
        warm_residual=torch.randn(
            observation_shape, generator=generator, dtype=torch.float64
        ),
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(field_shape, dtype=torch.float64),
        geometry=geometry,
    )
    assert basis.rank == 6
    assert basis.names == (
        "damping",
        "warm_residual",
        f"camera_contrast_1_{contrast_quantity}",
        f"camera_contrast_2_{contrast_quantity}",
        "normal_damping",
        "normal_warm_residual",
    )
    assert calls == {"forward": 2, "adjoint": 2}
    assert (basis.setup_forward_calls, basis.setup_adjoint_calls) == (2, 2)
    assert basis.uses_evaluated_case_truth is False
    assert basis.orthonormality_defect < 1e-12


def test_camera_contrast_weights_are_constant_within_camera() -> None:
    geometry = _geometry()
    anchor = torch.ones((geometry.ray_count, 2), dtype=torch.float64)
    vectors = camera_contrast_vectors(
        anchor, geometry=geometry, quantity_name="visible"
    )
    for _, vector in vectors:
        for camera in range(geometry.camera_count):
            values = vector[geometry.camera_index == camera]
            assert torch.unique(values[:, 0]).numel() == 1
            assert torch.equal(values[:, 0], values[:, 1])


def test_non_three_camera_geometry_fails_closed_before_operator_calls() -> None:
    geometry = _geometry()
    invalid = replace(
        geometry,
        camera_azimuth_degrees=geometry.camera_azimuth_degrees[:2],
        camera_elevation_degrees=geometry.camera_elevation_degrees[:2],
    )
    observation_shape = (geometry.ray_count, 2)
    forward, adjoint, calls = _maps(observation_shape, (3, 3, 3))
    with pytest.raises(ValueError, match="exactly three cameras"):
        build_global_contrast_basis(
            kind="residual_contrast_global_k6_total",
            damping=torch.ones(observation_shape, dtype=torch.float64),
            warm_residual=torch.ones(observation_shape, dtype=torch.float64),
            forward=forward,
            adjoint=adjoint,
            support=torch.ones((3, 3, 3), dtype=torch.float64),
            geometry=invalid,
        )
    assert calls == {"forward": 0, "adjoint": 0}


def test_unbalanced_camera_ray_counts_fail_closed() -> None:
    geometry = _geometry()
    labels = geometry.camera_index.clone()
    first_camera_two_ray = int(torch.nonzero(labels == 2, as_tuple=False)[0, 0])
    labels[first_camera_two_ray] = 1
    invalid = replace(geometry, camera_index=labels)
    with pytest.raises(ValueError, match="equal valid ray counts"):
        three_camera_helmert_contrasts(invalid)
