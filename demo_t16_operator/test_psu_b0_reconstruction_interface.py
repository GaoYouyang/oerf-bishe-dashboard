from __future__ import annotations

from math import prod

import pytest
import torch

from .psu_b0_reconstruction_interface import (
    CompactTrilinearCoordinates,
    PSUB0VoxelGradientOperator,
    build_compact_trilinear_coordinates,
    build_trilinear_stencil,
    expand_compact_trilinear_coordinates,
    finite_difference_gradient,
    finite_difference_gradient_adjoint,
    project_dirichlet_gauge,
)


def _operator(
    *,
    points: torch.Tensor | None = None,
    sample_valid: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> PSUB0VoxelGradientOperator:
    if points is None:
        points = torch.tensor(
            [
                [[-0.8, -0.6, -0.4], [-0.2, 0.1, 0.3], [0.7, 0.5, 0.8]],
                [[-0.9, 0.4, 0.2], [0.0, -0.3, 0.6], [0.9, 0.2, -0.7]],
                [[-0.5, -0.8, 0.7], [0.3, 0.8, -0.1], [1.2, 0.0, 0.0]],
            ],
            dtype=dtype,
        )
    stencil = build_trilinear_stencil(
        points,
        grid_shape=(5, 6, 7),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        sample_valid=sample_valid,
        dtype=dtype,
    )
    return PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8]],
            dtype=dtype,
        ),
        projection_v_xyz=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [-0.8, 0.0, 0.6]],
            dtype=dtype,
        ),
        line_length=torch.tensor([2.0, 1.6, 1.2], dtype=dtype),
        system_constant=torch.tensor([0.8, 1.1, 0.6], dtype=dtype),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=dtype,
    )


def test_finite_difference_gradient_has_exact_declared_adjoint() -> None:
    generator = torch.Generator().manual_seed(37)
    volume = torch.randn((2, 4, 5, 6), generator=generator, dtype=torch.float64)
    dual = torch.randn((2, 3, 4, 5, 6), generator=generator, dtype=torch.float64)
    spacing = (0.3, 0.4, 0.5)
    lhs = torch.sum(finite_difference_gradient(volume, spacing_xyz=spacing) * dual)
    rhs = torch.sum(
        volume * finite_difference_gradient_adjoint(dual, spacing_xyz=spacing)
    )
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_matrix_free_b0_operator_passes_dot_product_identity() -> None:
    operator = _operator()
    assert operator.adjoint_relative_error(seed=91) < 1e-12
    assert operator.call_report() == {"forward_calls": 0, "adjoint_calls": 0}


def test_matrix_free_forward_matches_tiny_materialized_matrix() -> None:
    operator = _operator()
    voxel_count = prod(operator.grid_shape)
    identity = torch.eye(voxel_count, dtype=torch.float64).reshape(
        voxel_count, 1, *operator.grid_shape
    )
    matrix = (
        operator.forward(identity)
        .permute(1, 2, 0)
        .reshape(2 * operator.ray_count, voxel_count)
    )
    generator = torch.Generator().manual_seed(103)
    volume = torch.randn(
        (2, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    matrix_free = operator.forward(volume).reshape(2, -1)
    materialized = volume.flatten(1) @ matrix.T
    assert torch.allclose(matrix_free, materialized, atol=1e-12, rtol=1e-12)


def test_fixed_denominator_keeps_invalid_sample_slots() -> None:
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    stencil = build_trilinear_stencil(
        points,
        grid_shape=(5, 5, 5),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
    )
    operator = PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=[[1.0, 0.0, 0.0]],
        projection_v_xyz=[[0.0, 1.0, 0.0]],
        line_length=[2.0],
        system_constant=[1.0],
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
    )
    x_axis = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    volume = x_axis.reshape(1, 1, 1, 1, 5).expand(1, 1, 5, 5, 5)
    predicted = operator(volume)
    assert predicted[0, 0, 0].item() == 1.0
    assert predicted[0, 0, 1].item() == 0.0
    assert stencil.valid.tolist() == [[True, False]]


def test_compact_coordinates_regenerate_boundary_and_invalid_stencil() -> None:
    points = torch.tensor(
        [
            [
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
                [0.25, -0.4, 0.75],
                [1.01, 0.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    compact = build_compact_trilinear_coordinates(
        points,
        grid_shape=(5, 6, 7),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    regenerated = expand_compact_trilinear_coordinates(
        CompactTrilinearCoordinates(
            base_indices=compact.base_indices.to(torch.uint16),
            fractions_xyz=compact.fractions_xyz,
            valid=compact.valid.to(torch.uint8),
            grid_shape=compact.grid_shape,
        ),
        dtype=torch.float64,
    )
    direct = build_trilinear_stencil(
        points,
        grid_shape=(5, 6, 7),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    assert torch.equal(regenerated.indices, direct.indices)
    assert torch.equal(regenerated.valid, direct.valid)
    assert torch.equal(regenerated.weights, direct.weights)
    assert regenerated.valid.tolist() == [[True, True, True, False]]


def test_compact_coordinates_honor_declared_sample_validity() -> None:
    points = torch.zeros((2, 3, 3), dtype=torch.float64)
    declared = torch.tensor(
        [[True, False, True], [False, True, False]],
        dtype=torch.bool,
    )
    compact = build_compact_trilinear_coordinates(
        points,
        grid_shape=(4, 4, 4),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        sample_valid=declared,
        dtype=torch.float64,
    )
    regenerated = expand_compact_trilinear_coordinates(compact)
    assert torch.equal(regenerated.valid, declared)
    assert torch.count_nonzero(regenerated.weights[~declared]) == 0
    assert torch.count_nonzero(regenerated.indices[~declared]) == 0


def test_forward_and_adjoint_calls_are_counted_logically() -> None:
    operator = _operator()
    volume = torch.zeros((1, 1, *operator.grid_shape), dtype=torch.float64)
    residual = operator.forward(volume)
    operator.adjoint(residual)
    assert operator.call_report() == {"forward_calls": 1, "adjoint_calls": 1}
    operator.reset_call_counts()
    assert operator.call_report() == {"forward_calls": 0, "adjoint_calls": 0}


def test_grouped_adjoint_matches_individual_groups_and_pooled_sum() -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(1907)
    residual = torch.randn(
        (2, operator.ray_count, 2),
        generator=generator,
        dtype=torch.float64,
    )
    group_index = torch.tensor([0, 1, 0], dtype=torch.int64)
    references = []
    for group in range(2):
        masked = residual * (group_index == group)[None, :, None]
        references.append(operator.adjoint(masked))
    pooled = operator.adjoint(residual)

    operator.reset_call_counts()
    grouped = operator.adjoint_grouped(
        residual,
        ray_group_index=group_index,
        group_count=2,
    )

    assert grouped.shape == (2, 2, 1, *operator.grid_shape)
    assert operator.call_report() == {"forward_calls": 0, "adjoint_calls": 1}
    assert torch.allclose(grouped[:, 0], references[0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(grouped[:, 1], references[1], atol=1e-12, rtol=1e-12)
    assert torch.allclose(grouped.sum(dim=1), pooled, atol=1e-12, rtol=1e-12)


def test_view_adjoint_uses_contiguous_blocks_and_validates_block_size() -> None:
    operator = _operator()
    residual = torch.zeros(
        (1, operator.ray_count, 2),
        dtype=torch.float64,
    )
    residual[:, 1, 0] = 1.0
    grouped = operator.adjoint_by_view(residual, rays_per_view=1)

    assert grouped.shape == (1, 3, 1, *operator.grid_shape)
    assert torch.count_nonzero(grouped[:, 0]) == 0
    assert torch.count_nonzero(grouped[:, 1]) > 0
    assert torch.count_nonzero(grouped[:, 2]) == 0
    with pytest.raises(ValueError, match="divide ray_count"):
        operator.adjoint_by_view(residual, rays_per_view=2)


def test_dirichlet_gauge_removes_boundary_and_respects_support() -> None:
    volume = torch.ones((1, 1, 6, 6, 6), dtype=torch.float64)
    support = torch.ones((6, 6, 6), dtype=torch.float64)
    support[3, 3, 3] = 0
    projected = project_dirichlet_gauge(volume, support=support)
    assert torch.count_nonzero(projected[..., 0, :, :]) == 0
    assert torch.count_nonzero(projected[..., -1, :, :]) == 0
    assert projected[0, 0, 3, 3, 3] == 0
    assert projected[0, 0, 2, 2, 2] == 1
