"""Mechanical CPU/float64 checks for the PSU-B0 P and |G_c| factors."""

from __future__ import annotations

from math import prod
from pathlib import Path
import sys

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from demo_t16_operator.psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    TrilinearStencil,
    absolute_finite_difference_gradient,
    absolute_finite_difference_gradient_adjoint,
    build_trilinear_stencil,
    finite_difference_gradient,
    finite_difference_gradient_adjoint,
)


DTYPE = torch.float64
GRID_SHAPE = (3, 4, 5)
SPACING_XYZ = (0.5, 0.7, 1.1)


def _operator() -> PSUB0VoxelGradientOperator:
    points = torch.tensor(
        [
            [
                [-1.0, -1.4, -1.1],
                [-0.35, 0.17, 0.28],
                [1.0, 1.4, 1.1],
            ],
            [
                [0.63, -0.72, 0.51],
                [1.01, 0.0, 0.0],
                [-0.81, 0.93, -0.44],
            ],
        ],
        dtype=DTYPE,
    )
    stencil = build_trilinear_stencil(
        points,
        grid_shape=GRID_SHAPE,
        grid_minimum_xyz=(-1.0, -1.4, -1.1),
        grid_maximum_xyz=(1.0, 1.4, 1.1),
        sample_valid=torch.tensor(
            [[True, True, True], [True, True, False]],
            dtype=torch.bool,
        ),
        dtype=DTYPE,
    )
    support = torch.linspace(
        0.2,
        1.0,
        prod(GRID_SHAPE),
        dtype=DTYPE,
    ).reshape(GRID_SHAPE)
    return PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=[[0.0, 1.0, 0.0], [0.6, -0.2, 0.7]],
        projection_v_xyz=[[0.3, 0.0, 0.9], [-0.4, 0.8, 0.1]],
        line_length=[1.7, 2.1],
        system_constant=[0.8, 1.2],
        grid_minimum_xyz=(-1.0, -1.4, -1.1),
        grid_maximum_xyz=(1.0, 1.4, 1.1),
        support=support,
        dtype=DTYPE,
    )


def _explicit_interpolation_matrix(
    operator: PSUB0VoxelGradientOperator,
) -> torch.Tensor:
    sample_total = operator.ray_count * operator.sample_count
    matrix = torch.zeros(
        (sample_total, prod(operator.grid_shape)),
        dtype=DTYPE,
    )
    for row, (indices, weights) in enumerate(
        zip(
            operator.sample_indices.reshape(sample_total, 8),
            operator.sample_weights.reshape(sample_total, 8),
        )
    ):
        matrix[row].scatter_add_(0, indices, weights)
    return matrix


def _explicit_absolute_gradient_matrix(
    shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    nz, ny, nx = shape
    voxel_count = prod(shape)
    matrix = torch.zeros((3 * voxel_count, voxel_count), dtype=DTYPE)

    def flat_index(z: int, y: int, x: int) -> int:
        return z * ny * nx + y * nx + x

    for component, (axis_size, spacing) in enumerate(
        zip((nx, ny, nz), spacing_xyz)
    ):
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    coordinate = (x, y, z)[component]
                    if coordinate == 0:
                        neighbors = (0, 1)
                        coefficient = 1.0 / spacing
                    elif coordinate == axis_size - 1:
                        neighbors = (axis_size - 2, axis_size - 1)
                        coefficient = 1.0 / spacing
                    else:
                        neighbors = (coordinate - 1, coordinate + 1)
                        coefficient = 1.0 / (2.0 * spacing)
                    row = component * voxel_count + flat_index(z, y, x)
                    for neighbor in neighbors:
                        location = [x, y, z]
                        location[component] = neighbor
                        column = flat_index(
                            location[2],
                            location[1],
                            location[0],
                        )
                        matrix[row, column] = coefficient
    return matrix


def _legacy_forward(
    operator: PSUB0VoxelGradientOperator,
    volume: torch.Tensor,
) -> torch.Tensor:
    values = volume[:, 0] * operator.support
    gradient = finite_difference_gradient(
        values,
        spacing_xyz=operator.spacing_xyz,
    )
    gathered = gradient.flatten(2)[:, :, operator.sample_indices.reshape(-1)]
    gathered = gathered.reshape(
        len(values),
        3,
        operator.ray_count,
        operator.sample_count,
        8,
    )
    sampled = torch.sum(
        gathered * operator.sample_weights[None, None],
        dim=-1,
    )
    u = torch.einsum("bcrs,rc->brs", sampled, operator.projection_u)
    v = torch.einsum("bcrs,rc->brs", sampled, operator.projection_v)
    projected = torch.stack((u.sum(dim=-1), v.sum(dim=-1)), dim=-1)
    return projected * operator.ray_scale[None, :, None]


def _legacy_adjoint(
    operator: PSUB0VoxelGradientOperator,
    residual: torch.Tensor,
) -> torch.Tensor:
    component = (
        residual[:, :, 0:1] * operator.projection_u[None]
        + residual[:, :, 1:2] * operator.projection_v[None]
    )
    component = component * operator.ray_scale[None, :, None]
    contribution = (
        component.permute(0, 2, 1)[:, :, :, None, None]
        * operator.sample_weights[None, None]
    )
    flat_contribution = contribution.reshape(len(residual), 3, -1)
    flat_indices = operator.sample_indices.reshape(-1)
    expanded_indices = flat_indices.reshape(1, 1, -1).expand(
        len(residual), 3, -1
    )
    gradient_flat = torch.zeros(
        (len(residual), 3, prod(operator.grid_shape)),
        dtype=DTYPE,
    )
    gradient_flat.scatter_add_(2, expanded_indices, flat_contribution)
    gradient = gradient_flat.reshape(len(residual), 3, *operator.grid_shape)
    return finite_difference_gradient_adjoint(
        gradient,
        spacing_xyz=operator.spacing_xyz,
    ) * operator.support


def test_trilinear_p_is_nonnegative_and_matches_dense_matrix() -> None:
    operator = _operator()
    voxel_count = prod(operator.grid_shape)
    identity = torch.eye(voxel_count, dtype=DTYPE).reshape(
        voxel_count, 1, *operator.grid_shape
    )
    materialized = operator.trilinear_interpolation(identity).reshape(
        voxel_count, -1
    ).T
    explicit = _explicit_interpolation_matrix(operator)

    assert torch.all(operator.sample_weights >= 0.0)
    assert torch.all(explicit >= 0.0)
    assert torch.equal(materialized, explicit)
    positive_output = operator.trilinear_interpolation(torch.ones_like(identity))
    assert torch.all(positive_output >= 0.0)

    sample_total = operator.ray_count * operator.sample_count
    sample_identity = torch.eye(sample_total, dtype=DTYPE).reshape(
        sample_total,
        1,
        operator.ray_count,
        operator.sample_count,
    )
    transpose_materialized = operator.trilinear_interpolation_adjoint(
        sample_identity
    ).reshape(sample_total, voxel_count)
    assert torch.equal(transpose_materialized, explicit)


def test_operator_rejects_a_signed_interpolation_stencil() -> None:
    operator = _operator()
    signed_weights = operator.sample_weights.clone()
    signed_weights[0, 0, 0] = -signed_weights[0, 0, 0]
    signed_stencil = TrilinearStencil(
        indices=operator.sample_indices,
        weights=signed_weights,
        valid=operator.sample_valid,
        grid_shape=operator.grid_shape,
    )
    with pytest.raises(ValueError, match="weights must be nonnegative"):
        PSUB0VoxelGradientOperator(
            stencil=signed_stencil,
            projection_u_xyz=operator.projection_u,
            projection_v_xyz=operator.projection_v,
            line_length=torch.ones(operator.ray_count, dtype=DTYPE),
            system_constant=torch.ones(operator.ray_count, dtype=DTYPE),
            grid_minimum_xyz=operator.grid_minimum_xyz,
            grid_maximum_xyz=operator.grid_maximum_xyz,
            dtype=DTYPE,
        )


def _construct_with_stencil(stencil: TrilinearStencil) -> None:
    ray_count = stencil.indices.shape[0]
    PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=torch.zeros((ray_count, 3), dtype=DTYPE),
        projection_v_xyz=torch.zeros((ray_count, 3), dtype=DTYPE),
        line_length=torch.ones(ray_count, dtype=DTYPE),
        system_constant=torch.ones(ray_count, dtype=DTYPE),
        grid_minimum_xyz=(-1.0, -1.4, -1.1),
        grid_maximum_xyz=(1.0, 1.4, 1.1),
        dtype=DTYPE,
    )


def _valid_stencil() -> TrilinearStencil:
    return TrilinearStencil(
        indices=torch.zeros((1, 2, 8), dtype=torch.int64),
        weights=torch.zeros((1, 2, 8), dtype=DTYPE),
        valid=torch.tensor([[True, False]], dtype=torch.bool),
        grid_shape=GRID_SHAPE,
    )


@pytest.mark.parametrize("bad_index", [-1, prod(GRID_SHAPE)])
def test_operator_rejects_out_of_domain_sample_indices(bad_index: int) -> None:
    stencil = _valid_stencil()
    indices = stencil.indices.clone()
    indices[0, 0, 0] = bad_index
    with pytest.raises(ValueError, match=r"must lie in \[0, prod\(grid_shape\)\)"):
        _construct_with_stencil(
            TrilinearStencil(indices, stencil.weights, stencil.valid, GRID_SHAPE)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("indices", torch.zeros((1, 2, 7), dtype=torch.int64), "sample_indices"),
        ("weights", torch.zeros((1, 2, 7), dtype=DTYPE), "sample_weights"),
        ("valid", torch.ones((1, 2, 1), dtype=torch.bool), "sample_valid"),
    ],
)
def test_operator_rejects_stencil_shape_mismatch(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    stencil = _valid_stencil()
    values = {
        "indices": stencil.indices,
        "weights": stencil.weights,
        "valid": stencil.valid,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        _construct_with_stencil(
            TrilinearStencil(
                values["indices"],
                values["weights"],
                values["valid"],
                GRID_SHAPE,
            )
        )


def test_operator_rejects_non_int64_indices_and_non_bool_validity() -> None:
    stencil = _valid_stencil()
    with pytest.raises(ValueError, match="sample_indices must have dtype int64"):
        _construct_with_stencil(
            TrilinearStencil(
                stencil.indices.to(torch.int32),
                stencil.weights,
                stencil.valid,
                GRID_SHAPE,
            )
        )
    with pytest.raises(ValueError, match="sample_valid must have dtype bool"):
        _construct_with_stencil(
            TrilinearStencil(
                stencil.indices,
                stencil.weights,
                stencil.valid.to(torch.int64),
                GRID_SHAPE,
            )
        )


def test_operator_rejects_nonzero_weight_in_invalid_sample_slot() -> None:
    stencil = _valid_stencil()
    weights = stencil.weights.clone()
    weights[0, 1, 3] = 0.25
    with pytest.raises(ValueError, match="invalid samples must be exactly zero"):
        _construct_with_stencil(
            TrilinearStencil(stencil.indices, weights, stencil.valid, GRID_SHAPE)
        )


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), -float("inf")])
def test_operator_rejects_nonfinite_sample_weights(bad_weight: float) -> None:
    stencil = _valid_stencil()
    weights = stencil.weights.clone()
    weights[0, 0, 0] = bad_weight
    with pytest.raises(ValueError, match="only finite values"):
        _construct_with_stencil(
            TrilinearStencil(stencil.indices, weights, stencil.valid, GRID_SHAPE)
        )


def test_trilinear_p_and_pt_pass_dot_product_identity_without_full_calls() -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(1701)
    field = torch.randn(
        (2, 4, *operator.grid_shape),
        generator=generator,
        dtype=DTYPE,
    )
    dual = torch.randn(
        (2, 4, operator.ray_count, operator.sample_count),
        generator=generator,
        dtype=DTYPE,
    )
    lhs = torch.sum(operator.trilinear_interpolation(field) * dual)
    rhs = torch.sum(field * operator.trilinear_interpolation_adjoint(dual))

    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)
    assert operator.call_report() == {"forward_calls": 0, "adjoint_calls": 0}


def test_absolute_gradient_and_transpose_match_explicit_dense_matrix() -> None:
    voxel_count = prod(GRID_SHAPE)
    explicit = _explicit_absolute_gradient_matrix(GRID_SHAPE, SPACING_XYZ)
    identity = torch.eye(voxel_count, dtype=DTYPE).reshape(
        voxel_count, *GRID_SHAPE
    )
    materialized = absolute_finite_difference_gradient(
        identity,
        spacing_xyz=SPACING_XYZ,
    ).flatten(1).T
    assert torch.equal(materialized, explicit)

    gradient_identity = torch.eye(3 * voxel_count, dtype=DTYPE).reshape(
        3 * voxel_count, 3, *GRID_SHAPE
    )
    transpose_materialized = absolute_finite_difference_gradient_adjoint(
        gradient_identity,
        spacing_xyz=SPACING_XYZ,
    ).flatten(1)
    assert torch.equal(transpose_materialized, explicit)


def test_absolute_gradient_transpose_and_signed_dominance() -> None:
    generator = torch.Generator().manual_seed(1702)
    volume = torch.randn((2, *GRID_SHAPE), generator=generator, dtype=DTYPE)
    dual = torch.randn((2, 3, *GRID_SHAPE), generator=generator, dtype=DTYPE)
    absolute_gradient = absolute_finite_difference_gradient(
        volume,
        spacing_xyz=SPACING_XYZ,
    )
    lhs = torch.sum(absolute_gradient * dual)
    rhs = torch.sum(
        volume
        * absolute_finite_difference_gradient_adjoint(
            dual,
            spacing_xyz=SPACING_XYZ,
        )
    )
    signed_magnitude = torch.abs(
        finite_difference_gradient(volume, spacing_xyz=SPACING_XYZ)
    )
    majorizer = absolute_finite_difference_gradient(
        torch.abs(volume),
        spacing_xyz=SPACING_XYZ,
    )

    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)
    assert torch.all(signed_magnitude <= majorizer + 1e-14)
    assert torch.any(
        absolute_gradient
        != torch.abs(finite_difference_gradient(volume, spacing_xyz=SPACING_XYZ))
    )


def test_full_forward_and_adjoint_regress_exactly_and_keep_call_counts() -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(1703)
    volume = torch.randn(
        (2, 1, *operator.grid_shape),
        generator=generator,
        dtype=DTYPE,
    )
    residual = torch.randn(
        (2, operator.ray_count, 2),
        generator=generator,
        dtype=DTYPE,
    )
    expected_forward = _legacy_forward(operator, volume)
    expected_adjoint = _legacy_adjoint(operator, residual)[:, None]

    actual_forward = operator.forward(volume)
    actual_adjoint = operator.adjoint(residual)

    assert torch.equal(actual_forward, expected_forward)
    assert torch.equal(actual_adjoint, expected_adjoint)
    assert operator.call_report() == {"forward_calls": 1, "adjoint_calls": 1}
