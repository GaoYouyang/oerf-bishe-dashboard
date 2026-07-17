"""Tiny exact-matrix tests for the composed PSU-B0 measurement factor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from .psu_b0_absolute_measurement_factor import (
    ExactAbsoluteMeasurementFactor,
    WHITENING_SUPPORT_CONTRACT_SCHEMA,
)
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
    finite_difference_gradient,
    finite_difference_gradient_adjoint,
)


class _WhiteningFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.view_count = 2
        self.rays_per_view = 2
        self.register_buffer(
            "matrix",
            torch.tensor(
                [
                    [
                        [1.0, -0.5, 0.25, 0.0],
                        [0.4, 0.8, 0.0, -0.3],
                        [-0.2, 0.1, 1.1, 0.5],
                        [0.0, -0.6, 0.2, 0.9],
                    ],
                    [
                        [0.7, 0.2, -0.4, 0.1],
                        [-0.1, 1.2, 0.3, 0.0],
                        [0.5, -0.3, 0.8, -0.2],
                        [0.2, 0.0, -0.5, 1.0],
                    ],
                ],
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "scale_by_view",
            torch.tensor([[1.5, 0.8], [0.75, 1.6]], dtype=torch.float64),
        )


def _primitives() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projection_u = torch.tensor(
        [
            [1.0, -0.5, 0.2],
            [-0.4, 0.8, 0.3],
            [0.6, 0.1, -0.7],
            [-0.2, -0.9, 0.5],
        ],
        dtype=torch.float64,
    )
    projection_v = torch.tensor(
        [
            [-0.3, 0.7, 0.4],
            [0.9, -0.2, -0.6],
            [0.2, -0.8, 0.5],
            [0.4, 0.3, -0.1],
        ],
        dtype=torch.float64,
    )
    ray_scale = torch.tensor([0.8, -1.1, 0.6, 1.3], dtype=torch.float64)
    return projection_u, projection_v, ray_scale


def _factor() -> ExactAbsoluteMeasurementFactor:
    projection_u, projection_v, ray_scale = _primitives()
    return ExactAbsoluteMeasurementFactor(
        _WhiteningFixture(),
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        ray_scale=ray_scale,
        sample_count=3,
        measurement_scale=0.25,
    )


def _physical_operator() -> PSUB0VoxelGradientOperator:
    projection_u, projection_v, ray_scale = _primitives()
    points = np.array(
        [
            [[-0.7, -0.4, -0.2], [-0.1, 0.2, 0.4], [0.6, 0.4, 0.1]],
            [[-0.5, 0.3, -0.6], [0.0, -0.2, 0.5], [0.7, 0.1, 0.6]],
            [[-0.6, 0.6, 0.2], [0.2, 0.1, -0.4], [0.5, -0.5, 0.3]],
            [[-0.4, -0.6, 0.5], [0.1, 0.5, 0.0], [0.6, -0.1, -0.5]],
        ],
        dtype=np.float64,
    )
    stencil = build_trilinear_stencil(
        points,
        grid_shape=(5, 5, 5),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    return PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        line_length=torch.full((4,), 3.0, dtype=torch.float64),
        system_constant=ray_scale,
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )


def _manual_dense(
    whitening: _WhiteningFixture,
    *,
    view: int,
    batch: int,
) -> torch.Tensor:
    projection_u, projection_v, ray_scale = _primitives()
    projection = torch.stack((projection_u, projection_v), dim=1)
    rays_per_view = whitening.rays_per_view
    samples = 3
    camera = torch.zeros(
        (2 * rays_per_view, 3 * rays_per_view * samples),
        dtype=torch.float64,
    )
    start = view * rays_per_view
    for ray in range(rays_per_view):
        global_ray = start + ray
        for component in range(2):
            for axis in range(3):
                for sample in range(samples):
                    column = axis * rays_per_view * samples + ray * samples + sample
                    camera[2 * ray + component, column] = (
                        ray_scale[global_ray]
                        * projection[global_ray, component, axis]
                    )
    return (
        0.25
        * whitening.matrix[view]
        @ camera
        / whitening.scale_by_view[batch, view]
    )


def test_dense_block_is_exact_composition_before_absolute_value() -> None:
    whitening = _WhiteningFixture()
    factor = _factor()
    for batch in range(2):
        for view in range(2):
            expected = _manual_dense(whitening, view=view, batch=batch)
            torch.testing.assert_close(
                factor.dense_block(view, batch_index=batch),
                expected,
                atol=1e-14,
                rtol=1e-14,
            )
            torch.testing.assert_close(
                factor.dense_block(view, batch_index=batch, absolute=True),
                expected.abs(),
                atol=1e-14,
                rtol=1e-14,
            )

    projection_u, projection_v, ray_scale = _primitives()
    projection = torch.stack((projection_u, projection_v), dim=1)[:2]
    loose_camera = torch.zeros((4, 18), dtype=torch.float64)
    for ray in range(2):
        for component in range(2):
            for axis in range(3):
                for sample in range(3):
                    loose_camera[2 * ray + component, axis * 6 + ray * 3 + sample] = (
                        ray_scale[ray].abs() * projection[ray, component, axis].abs()
                    )
    loose = 0.25 * whitening.matrix[0].abs() @ loose_camera / whitening.scale_by_view[0, 0]
    exact_absolute = factor.dense_block(0, absolute=True)
    assert torch.all(loose >= exact_absolute - 1e-14)
    assert torch.any(loose > exact_absolute + 1e-8)


def test_signed_and_absolute_calls_match_dense_blocks_and_transposes() -> None:
    factor = _factor()
    generator = torch.Generator().manual_seed(23)
    sampled = torch.randn((2, 3, 4, 3), generator=generator, dtype=torch.float64)
    detector = torch.randn((2, 4, 2), generator=generator, dtype=torch.float64)

    signed_expected = torch.empty_like(detector)
    absolute_expected = torch.empty_like(detector)
    transpose_expected = torch.empty_like(sampled)
    absolute_transpose_expected = torch.empty_like(sampled)
    for batch in range(2):
        for view in range(2):
            ray_slice = slice(2 * view, 2 * (view + 1))
            x = sampled[batch, :, ray_slice, :].reshape(-1)
            y = detector[batch, ray_slice, :].reshape(-1)
            signed = factor.dense_block(view, batch_index=batch)
            absolute = factor.dense_block(
                view,
                batch_index=batch,
                absolute=True,
            )
            signed_expected[batch, ray_slice, :] = (signed @ x).reshape(2, 2)
            absolute_expected[batch, ray_slice, :] = (absolute @ x).reshape(2, 2)
            transpose_expected[batch, :, ray_slice, :] = (signed.T @ y).reshape(
                3, 2, 3
            )
            absolute_transpose_expected[batch, :, ray_slice, :] = (
                absolute.T @ y
            ).reshape(3, 2, 3)

    torch.testing.assert_close(factor.signed_forward(sampled), signed_expected)
    torch.testing.assert_close(factor.absolute_forward(sampled), absolute_expected)
    torch.testing.assert_close(
        factor.signed_transpose(detector),
        transpose_expected,
    )
    torch.testing.assert_close(
        factor.absolute_transpose(detector),
        absolute_transpose_expected,
    )


def test_signed_dot_product_and_elementwise_dominance_hold() -> None:
    factor = _factor()
    generator = torch.Generator().manual_seed(91)
    sampled = torch.randn((2, 3, 4, 3), generator=generator, dtype=torch.float64)
    detector = torch.randn((2, 4, 2), generator=generator, dtype=torch.float64)
    lhs = torch.sum(factor.signed_forward(sampled) * detector)
    rhs = torch.sum(sampled * factor.signed_transpose(detector))
    torch.testing.assert_close(lhs, rhs, atol=1e-13, rtol=1e-13)
    assert torch.all(
        factor.signed_forward(sampled).abs()
        <= factor.absolute_forward(sampled.abs()) + 1e-13
    )


def test_factorized_signed_chain_matches_original_physical_composition() -> None:
    base = _physical_operator()
    whitening = _WhiteningFixture()
    projection_u, projection_v, ray_scale = _primitives()
    factor = ExactAbsoluteMeasurementFactor(
        whitening,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        ray_scale=ray_scale,
        sample_count=base.sample_count,
        measurement_scale=0.25,
    )
    generator = torch.Generator().manual_seed(1717)
    volume = torch.randn(
        (2, 1, *base.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    gradient = finite_difference_gradient(
        volume[:, 0] * base.support,
        spacing_xyz=base.spacing_xyz,
    )
    sampled = base.trilinear_interpolation(gradient)
    physical = base(volume).reshape(2, 2, 4)
    expected = torch.einsum(
        "vij,bvj->bvi",
        whitening.matrix,
        physical,
    ) / whitening.scale_by_view[:, :, None]
    expected = 0.25 * expected.reshape(2, 4, 2)
    torch.testing.assert_close(
        factor.signed_forward(sampled),
        expected,
        atol=1e-13,
        rtol=1e-13,
    )

    detector = torch.randn((2, 4, 2), generator=generator, dtype=torch.float64)
    sampled_adjoint = factor.signed_transpose(detector)
    gradient_adjoint = base.trilinear_interpolation_adjoint(sampled_adjoint)
    factorized_adjoint = finite_difference_gradient_adjoint(
        gradient_adjoint,
        spacing_xyz=base.spacing_xyz,
    )
    factorized_adjoint = factorized_adjoint[:, None] * base.support

    canonical = detector.reshape(2, 2, 4) / whitening.scale_by_view[:, :, None]
    unwhitened = torch.einsum(
        "vji,bvj->bvi",
        whitening.matrix,
        canonical,
    ).reshape(2, 4, 2)
    expected_adjoint = 0.25 * base.adjoint(unwhitened)
    torch.testing.assert_close(
        factorized_adjoint,
        expected_adjoint,
        atol=1e-13,
        rtol=1e-13,
    )


def test_one_pass_sums_use_exactly_one_absolute_call_each_way() -> None:
    factor = _factor()
    factor.reset_call_counts()
    rows, columns = factor.one_pass_sums(batch_index=1)
    assert factor.call_report() == {
        "signed_forward_calls": 0,
        "signed_transpose_calls": 0,
        "absolute_forward_calls": 1,
        "absolute_transpose_calls": 1,
    }
    for view in range(2):
        dense = factor.dense_block(view, batch_index=1, absolute=True)
        ray_slice = slice(2 * view, 2 * (view + 1))
        torch.testing.assert_close(
            rows[ray_slice].reshape(-1),
            dense.sum(dim=1),
        )
        torch.testing.assert_close(
            columns[:, ray_slice, :].reshape(-1),
            dense.sum(dim=0),
        )


def test_invalid_scale_and_shapes_fail_closed() -> None:
    projection_u, projection_v, ray_scale = _primitives()
    whitening = _WhiteningFixture()
    whitening.scale_by_view[0, 0] = 0.0
    try:
        ExactAbsoluteMeasurementFactor(
            whitening,
            projection_u_xyz=projection_u,
            projection_v_xyz=projection_v,
            ray_scale=ray_scale,
            sample_count=3,
        )
    except ValueError as error:
        assert "scale_by_view" in str(error)
    else:
        raise AssertionError("zero whitening scale must fail closed")

    factor = _factor()
    try:
        factor.absolute_forward(torch.ones((1, 3, 3, 3), dtype=torch.float64))
    except ValueError as error:
        assert "sampled_gradient" in str(error)
    else:
        raise AssertionError("wrong sampled shape must fail closed")


@pytest.mark.parametrize(
    "field",
    ["cross_view_covariance", "cross_view_coupling"],
)
def test_cross_view_whitening_metadata_is_rejected(field: str) -> None:
    projection_u, projection_v, ray_scale = _primitives()
    whitening = _WhiteningFixture()
    whitening.whitening_metadata = {field: True}

    with pytest.raises(ValueError, match=rf"{field}=True.*view-local"):
        ExactAbsoluteMeasurementFactor(
            whitening,
            projection_u_xyz=projection_u,
            projection_v_xyz=projection_v,
            ray_scale=ray_scale,
            sample_count=3,
        )


def test_non_view_local_covariance_block_ids_are_rejected() -> None:
    projection_u, projection_v, ray_scale = _primitives()
    whitening = _WhiteningFixture()
    whitening.covariance_block_ids = ("shared", "shared")

    with pytest.raises(ValueError, match="unique per view"):
        ExactAbsoluteMeasurementFactor(
            whitening,
            projection_u_xyz=projection_u,
            projection_v_xyz=projection_v,
            ray_scale=ray_scale,
            sample_count=3,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("whitening_block_scope", "cross_view", "exactly 'view_local'"),
        (
            "independent_whitening_blocks",
            False,
            "must be exactly True",
        ),
        (
            "cross_view_covariance_supported",
            True,
            "unsupported.*view-local",
        ),
        (
            "cross_view_coupling_supported",
            True,
            "unsupported.*view-local",
        ),
    ],
)
def test_contradictory_contract_vocabulary_is_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    projection_u, projection_v, ray_scale = _primitives()
    whitening = _WhiteningFixture()
    whitening.contract_metadata = {field: value}
    with pytest.raises(ValueError, match=message):
        ExactAbsoluteMeasurementFactor(
            whitening,
            projection_u_xyz=projection_u,
            projection_v_xyz=projection_v,
            ray_scale=ray_scale,
            sample_count=3,
        )


def test_factor_exposes_view_local_whitening_contract_metadata() -> None:
    assert _factor().contract_metadata == {
        "schema_version": WHITENING_SUPPORT_CONTRACT_SCHEMA,
        "whitening_block_scope": "view_local",
        "independent_whitening_blocks": True,
        "cross_view_covariance_supported": False,
        "cross_view_coupling_supported": False,
        "covariance_block_ids": (0, 1),
    }
