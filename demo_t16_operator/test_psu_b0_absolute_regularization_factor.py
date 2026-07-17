"""Exact-matrix tests for the forward-Neumann TV absolute factor."""

from __future__ import annotations

import torch

from .psu_b0_primal_dual import (
    ForwardNeumannRegularizationOperator,
    regularization_gradient,
    regularization_gradient_adjoint,
    regularization_site_mask,
)


SHAPE_ZYX = (2, 3, 4)
SPACING_XYZ = (0.5, 0.25, 2.0)


def _flat_index(z: int, y: int, x: int) -> int:
    _, size_y, size_x = SHAPE_ZYX
    return (z * size_y + y) * size_x + x


def _dense_forward_neumann() -> torch.Tensor:
    size_z, size_y, size_x = SHAPE_ZYX
    voxel_count = size_z * size_y * size_x
    dense = torch.zeros((3 * voxel_count, voxel_count), dtype=torch.float64)
    sizes_xyz = (size_x, size_y, size_z)
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                coordinates = (x, y, z)
                column = _flat_index(z, y, x)
                site = _flat_index(z, y, x)
                for axis, (size, spacing) in enumerate(
                    zip(sizes_xyz, SPACING_XYZ, strict=True)
                ):
                    if coordinates[axis] == size - 1:
                        continue
                    neighbor = list(coordinates)
                    neighbor[axis] += 1
                    neighbor_column = _flat_index(
                        neighbor[2], neighbor[1], neighbor[0]
                    )
                    # Dense/oracle rows are site-major: (z,y,x,component).
                    row = 3 * site + axis
                    dense[row, column] = -1.0 / spacing
                    dense[row, neighbor_column] = 1.0 / spacing
    return dense


def test_absolute_factor_matches_entrywise_dense_matrix_and_transpose() -> None:
    operator = ForwardNeumannRegularizationOperator(SPACING_XYZ)
    dense_absolute = _dense_forward_neumann().abs()
    volume = torch.arange(1, 49, dtype=torch.float64).reshape(
        2, *SHAPE_ZYX
    )
    dual = torch.arange(1, 145, dtype=torch.float64).reshape(
        2, 3, *SHAPE_ZYX
    )

    expected_forward_site_major = (
        dense_absolute @ volume.flatten(1).T
    ).T.reshape(2, *SHAPE_ZYX, 3)
    expected_forward = expected_forward_site_major.permute(0, 4, 1, 2, 3)
    dual_site_major = dual.permute(0, 2, 3, 4, 1).reshape(2, -1)
    expected_adjoint = (dense_absolute.T @ dual_site_major.T).T.reshape(
        2, *SHAPE_ZYX
    )

    torch.testing.assert_close(
        operator.absolute_forward(volume),
        expected_forward,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        operator.absolute_adjoint(dual),
        expected_adjoint,
        atol=0.0,
        rtol=0.0,
    )


def test_site_major_dense_rows_reject_component_major_cross_interface_order() -> None:
    operator = ForwardNeumannRegularizationOperator(SPACING_XYZ)
    dense = _dense_forward_neumann()
    volume = torch.arange(1, 25, dtype=torch.float64).reshape(1, *SHAPE_ZYX)

    tensor_gradient = operator(volume)
    site_major = tensor_gradient.permute(0, 2, 3, 4, 1).reshape(1, -1)
    component_major = tensor_gradient.reshape(1, -1)
    expected = (dense @ volume.flatten(1).T).T

    torch.testing.assert_close(site_major, expected, atol=0.0, rtol=0.0)
    assert not torch.equal(component_major, expected)


def test_terminal_isolated_corner_has_no_forward_neumann_site_row() -> None:
    support = torch.zeros(SHAPE_ZYX, dtype=torch.bool)
    support[-1, -1, -1] = True

    sites = regularization_site_mask(support)

    assert not bool(sites[-1, -1, -1])
    assert bool(sites[-1, -1, -2])
    assert bool(sites[-1, -2, -1])
    assert bool(sites[-2, -1, -1])
    assert int(torch.count_nonzero(sites)) == 3


def test_absolute_factor_has_exact_dot_product_identity() -> None:
    generator = torch.Generator().manual_seed(1707)
    operator = ForwardNeumannRegularizationOperator(SPACING_XYZ)
    volume = torch.randn(
        (2, *SHAPE_ZYX), generator=generator, dtype=torch.float64
    )
    dual = torch.randn(
        (2, 3, *SHAPE_ZYX), generator=generator, dtype=torch.float64
    )

    lhs = torch.sum(operator.absolute_forward(volume) * dual)
    rhs = torch.sum(volume * operator.absolute_adjoint(dual))

    torch.testing.assert_close(lhs, rhs, atol=1e-13, rtol=1e-13)


def test_signed_gradient_is_dominated_by_matrix_absolute_factor() -> None:
    generator = torch.Generator().manual_seed(1708)
    operator = ForwardNeumannRegularizationOperator(SPACING_XYZ)
    volume = torch.randn(
        (3, *SHAPE_ZYX), generator=generator, dtype=torch.float64
    )

    signed = operator(volume)
    majorant = operator.absolute_forward(volume.abs())

    assert torch.all(signed.abs() <= majorant + 1e-14)


def test_absolute_row_and_column_sums_hold_at_every_xyz_site() -> None:
    operator = ForwardNeumannRegularizationOperator(SPACING_XYZ)
    ones_volume = torch.ones((1, *SHAPE_ZYX), dtype=torch.float64)
    ones_gradient = torch.ones((1, 3, *SHAPE_ZYX), dtype=torch.float64)

    row_sums = operator.absolute_forward(ones_volume)[0]
    expected_rows = torch.empty_like(row_sums)
    expected_rows[0].fill_(2.0 / SPACING_XYZ[0])
    expected_rows[0, :, :, -1] = 0.0
    expected_rows[1].fill_(2.0 / SPACING_XYZ[1])
    expected_rows[1, :, -1, :] = 0.0
    expected_rows[2].fill_(2.0 / SPACING_XYZ[2])
    expected_rows[2, -1, :, :] = 0.0

    column_sums = operator.absolute_adjoint(ones_gradient)[0]
    expected_columns = torch.zeros_like(column_sums)
    for z in range(SHAPE_ZYX[0]):
        for y in range(SHAPE_ZYX[1]):
            for x in range(SHAPE_ZYX[2]):
                degree_x = int(x > 0) + int(x < SHAPE_ZYX[2] - 1)
                degree_y = int(y > 0) + int(y < SHAPE_ZYX[1] - 1)
                degree_z = int(z > 0) + int(z < SHAPE_ZYX[0] - 1)
                expected_columns[z, y, x] = (
                    degree_x / SPACING_XYZ[0]
                    + degree_y / SPACING_XYZ[1]
                    + degree_z / SPACING_XYZ[2]
                )

    torch.testing.assert_close(row_sums, expected_rows, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        column_sums, expected_columns, atol=0.0, rtol=0.0
    )
    assert torch.count_nonzero(row_sums[0, :, :, -1]) == 0
    assert torch.count_nonzero(row_sums[1, :, -1, :]) == 0
    assert torch.count_nonzero(row_sums[2, -1, :, :]) == 0


def test_absolute_and_signed_call_ledgers_are_independent() -> None:
    operator = ForwardNeumannRegularizationOperator(SPACING_XYZ)
    volume = torch.zeros((1, *SHAPE_ZYX), dtype=torch.float64)
    gradient = torch.zeros((1, 3, *SHAPE_ZYX), dtype=torch.float64)

    assert operator.call_report() == {
        "gradient_calls": 0,
        "gradient_adjoint_calls": 0,
    }
    assert operator.absolute_call_report() == {
        "absolute_forward_calls": 0,
        "absolute_adjoint_calls": 0,
    }

    operator.absolute_forward(volume)
    operator.absolute_adjoint(gradient)
    operator.absolute_adjoint(gradient)
    assert operator.call_report() == {
        "gradient_calls": 0,
        "gradient_adjoint_calls": 0,
    }
    assert operator.absolute_call_report() == {
        "absolute_forward_calls": 1,
        "absolute_adjoint_calls": 2,
    }

    operator(volume)
    operator.adjoint(gradient)
    assert operator.call_report() == {
        "gradient_calls": 1,
        "gradient_adjoint_calls": 1,
    }
    assert operator.absolute_call_report() == {
        "absolute_forward_calls": 1,
        "absolute_adjoint_calls": 2,
    }


def test_existing_signed_operator_remains_the_declared_gradient_pair() -> None:
    generator = torch.Generator().manual_seed(1709)
    operator = ForwardNeumannRegularizationOperator(SPACING_XYZ)
    volume = torch.randn(
        (2, *SHAPE_ZYX), generator=generator, dtype=torch.float64
    )
    gradient = torch.randn(
        (2, 3, *SHAPE_ZYX), generator=generator, dtype=torch.float64
    )

    torch.testing.assert_close(
        operator(volume),
        regularization_gradient(volume, spacing_xyz=SPACING_XYZ),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        operator.adjoint(gradient),
        regularization_gradient_adjoint(
            gradient, spacing_xyz=SPACING_XYZ
        ),
        atol=0.0,
        rtol=0.0,
    )
    assert operator.call_report() == {
        "gradient_calls": 1,
        "gradient_adjoint_calls": 1,
    }
    assert operator.absolute_call_report() == {
        "absolute_forward_calls": 0,
        "absolute_adjoint_calls": 0,
    }
