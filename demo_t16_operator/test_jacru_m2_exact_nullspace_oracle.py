"""Tests for the independent 12^3 exact dense null-space oracle."""

from __future__ import annotations

import pytest
import torch

from .jacru_m2_exact_nullspace_oracle import (
    TOY_VOXEL_LIMIT,
    assemble_dense_operator_matrix,
    build_exact_dense_nullspace_projector,
    exact_dense_nullspace_oracle,
)


def test_exact_rank_deficient_matrix_separates_known_components() -> None:
    matrix = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        dtype=torch.float64,
    )
    support = torch.ones((1, 1, 3), dtype=torch.bool)
    correction = torch.tensor([[[2.0, -3.0, 4.0]]], dtype=torch.float64)

    result = exact_dense_nullspace_oracle(
        correction=correction,
        support=support,
        dense_matrix=matrix,
        rank_rtol=1e-14,
        rank_atol=0.0,
    )

    assert result.rank == 2
    torch.testing.assert_close(
        result.singular_values,
        torch.tensor((2.0, 1.0), dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.row_space_correction,
        torch.tensor([[[2.0, -3.0, 0.0]]], dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.null_space_correction,
        torch.tensor([[[0.0, 0.0, 4.0]]], dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    )
    assert result.internal_projection_residual < 1e-14


def test_full_column_rank_has_no_null_space_component() -> None:
    matrix = torch.diag(torch.tensor((1.0, 2.0, 4.0), dtype=torch.float64))
    support = torch.ones((1, 1, 3), dtype=torch.int64)
    correction = torch.tensor([[[1.5, -0.25, 2.0]]], dtype=torch.float32)

    result = exact_dense_nullspace_oracle(
        correction=correction,
        support=support,
        dense_matrix=matrix,
    )

    assert result.rank == 3
    assert result.row_space_correction.dtype == torch.float64
    assert result.row_space_correction.device.type == "cpu"
    torch.testing.assert_close(
        result.row_space_correction,
        correction.to(torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(result.null_space_correction) == 0
    assert result.internal_projection_residual == 0.0


def test_binary_support_selects_full_matrix_columns_and_zeroes_inactive_voxels() -> None:
    support = torch.tensor(
        (((1, 0), (1, 0)), ((0, 1), (0, 0))),
        dtype=torch.float64,
    )
    full_matrix = torch.tensor(
        (
            (1.0, 8.0, 0.0, 7.0, 6.0, 0.0, 5.0, 4.0),
            (0.0, 3.0, 1.0, 2.0, 1.0, 0.0, 9.0, 8.0),
        ),
        dtype=torch.float64,
    )
    correction = torch.zeros_like(support)
    correction[support == 1.0] = torch.tensor(
        (2.0, -1.0, 5.0),
        dtype=torch.float64,
    )

    result = exact_dense_nullspace_oracle(
        correction=correction,
        support=support,
        dense_matrix=full_matrix,
    )

    active_matrix = full_matrix[:, torch.tensor((0, 2, 5))]
    active_null = result.null_space_correction[support == 1.0]
    assert result.active_voxel_count == 3
    assert result.measurement_count == 2
    assert torch.count_nonzero(result.row_space_correction[support == 0.0]) == 0
    assert torch.count_nonzero(result.null_space_correction[support == 0.0]) == 0
    torch.testing.assert_close(
        active_matrix @ active_null,
        torch.zeros(2, dtype=torch.float64),
        atol=1e-13,
        rtol=0.0,
    )


def test_forward_and_operator_assembly_use_only_active_support_columns() -> None:
    support = torch.tensor([[[1, 0], [1, 1]]], dtype=torch.bool)
    full_matrix = torch.tensor(
        ((1.0, 20.0, 2.0, 3.0), (0.0, 30.0, 4.0, 5.0)),
        dtype=torch.float64,
    )

    def spatial_forward(field: torch.Tensor) -> torch.Tensor:
        return full_matrix @ field.reshape(-1)

    class BatchChannelOperator:
        def forward(self, field: torch.Tensor) -> torch.Tensor:
            assert field.shape == (1, 1, 1, 2, 2)
            return (full_matrix @ field.reshape(-1)).reshape(1, 2)

    expected = full_matrix[:, torch.tensor((0, 2, 3))]
    from_forward = assemble_dense_operator_matrix(
        support=support,
        forward=spatial_forward,
    )
    from_operator = assemble_dense_operator_matrix(
        support=support,
        operator=BatchChannelOperator(),
        forward_input_layout="batch_channel",
    )

    torch.testing.assert_close(from_forward, expected)
    torch.testing.assert_close(from_operator, expected)


def test_reconstruction_decomposition_and_orthogonality_hold() -> None:
    matrix = torch.tensor(
        (
            (1.0, 2.0, -1.0, 0.0),
            (0.0, 1.0, 1.0, 2.0),
        ),
        dtype=torch.float64,
    )
    support = torch.ones((1, 2, 2), dtype=torch.bool)
    correction = torch.tensor([[[0.3, -1.2], [2.1, 0.7]]], dtype=torch.float64)

    result = exact_dense_nullspace_oracle(
        correction=correction,
        support=support,
        dense_matrix=matrix,
    )
    row = result.row_space_correction.reshape(-1)
    null = result.null_space_correction.reshape(-1)

    torch.testing.assert_close(
        result.row_space_correction + result.null_space_correction,
        correction,
        atol=2e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        matrix @ null,
        torch.zeros(2, dtype=torch.float64),
        atol=2e-14,
        rtol=0.0,
    )
    assert abs(float(torch.dot(row, null))) < 2e-14
    assert result.internal_projection_residual < 1e-13


def test_reusable_projector_matches_single_call_oracle() -> None:
    matrix = torch.tensor(
        ((1.0, 2.0, 0.0, -1.0), (0.0, 1.0, 1.0, 0.0)),
        dtype=torch.float64,
    )
    support = torch.ones((1, 2, 2), dtype=torch.bool)
    correction = torch.tensor([[[0.2, -0.3], [1.4, 0.8]]], dtype=torch.float64)
    projector = build_exact_dense_nullspace_projector(
        support=support,
        dense_matrix=matrix,
    )
    reusable = projector.project(correction)
    direct = exact_dense_nullspace_oracle(
        correction=correction,
        support=support,
        dense_matrix=matrix,
    )
    torch.testing.assert_close(reusable.row_space_correction, direct.row_space_correction)
    torch.testing.assert_close(reusable.null_space_correction, direct.null_space_correction)
    assert reusable.rank == direct.rank
    assert reusable.internal_projection_residual < 1e-13


def test_exact_affine_projection_matches_target_and_preserves_kernel_component() -> None:
    support = torch.ones((1, 2, 3), dtype=torch.float64)
    matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    projector = build_exact_dense_nullspace_projector(
        support=support,
        dense_matrix=matrix,
        rank_rtol=1e-12,
    )
    field = torch.tensor(
        [[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]],
        dtype=torch.float64,
    )
    target = torch.tensor([1.0, -2.0], dtype=torch.float64)
    result = projector.project_field_to_observation(
        field=field,
        observation=target,
    )
    active = result.projected_field.reshape(-1)
    torch.testing.assert_close(matrix @ active, target, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(active[2:], field.reshape(-1)[2:])
    assert result.rank == 2
    assert result.relative_target_residual < 1e-12
    assert result.internal_projection_residual < 1e-12


def test_exact_affine_projection_retains_unreachable_measurement_component() -> None:
    support = torch.ones((1, 1, 2), dtype=torch.float64)
    matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    projector = build_exact_dense_nullspace_projector(
        support=support,
        dense_matrix=matrix,
        rank_rtol=1e-12,
    )
    field = torch.tensor([[[3.0, 4.0]]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result = projector.project_field_to_observation(
        field=field,
        observation=target,
    )
    torch.testing.assert_close(
        result.projected_field,
        torch.tensor([[[1.0, 4.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.target_residual,
        torch.tensor([0.0, -2.0], dtype=torch.float64),
    )
    assert result.normal_equation_residual < 1e-12


def test_exact_affine_projection_rejects_wrong_observation_size() -> None:
    support = torch.ones((1, 1, 2), dtype=torch.float64)
    projector = build_exact_dense_nullspace_projector(
        support=support,
        dense_matrix=torch.eye(2, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="observation size"):
        projector.project_field_to_observation(
            field=torch.zeros_like(support),
            observation=torch.zeros(3, dtype=torch.float64),
        )


def test_explicit_rank_tolerance_controls_numerical_rank() -> None:
    matrix = torch.diag(torch.tensor((1.0, 1e-10, 0.0), dtype=torch.float64))
    support = torch.ones((1, 1, 3), dtype=torch.bool)
    correction = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float64)

    loose = exact_dense_nullspace_oracle(
        correction=correction,
        support=support,
        dense_matrix=matrix,
        rank_rtol=1e-8,
    )
    tight = exact_dense_nullspace_oracle(
        correction=correction,
        support=support,
        dense_matrix=matrix,
        rank_rtol=1e-12,
    )

    assert loose.rank == 1
    assert loose.rank_tolerance == pytest.approx(1e-8)
    assert tight.rank == 2
    assert tight.rank_tolerance == pytest.approx(1e-12)


def test_invalid_inputs_are_rejected() -> None:
    matrix = torch.eye(2, dtype=torch.float64)
    correction = torch.ones((1, 1, 2), dtype=torch.float64)
    support = torch.ones((1, 1, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="strictly binary"):
        exact_dense_nullspace_oracle(
            correction=correction,
            support=torch.tensor([[[1.0, 0.5]]]),
            dense_matrix=matrix,
        )
    with pytest.raises(ValueError, match="zero outside"):
        exact_dense_nullspace_oracle(
            correction=correction,
            support=torch.tensor([[[1, 0]]]),
            dense_matrix=matrix,
        )
    with pytest.raises(ValueError, match="columns"):
        exact_dense_nullspace_oracle(
            correction=correction,
            support=support,
            dense_matrix=torch.ones((2, 3), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="exactly one matrix source"):
        exact_dense_nullspace_oracle(
            correction=correction,
            support=support,
            dense_matrix=matrix,
            forward=lambda field: field,
        )
    with pytest.raises(ValueError, match="rank_rtol"):
        exact_dense_nullspace_oracle(
            correction=correction,
            support=support,
            dense_matrix=matrix,
            rank_rtol=-1.0,
        )
    with pytest.raises(ValueError, match=r"12\^3"):
        exact_dense_nullspace_oracle(
            correction=torch.zeros((1, 1, TOY_VOXEL_LIMIT + 1)),
            support=torch.ones((1, 1, TOY_VOXEL_LIMIT + 1)),
            dense_matrix=torch.ones((1, TOY_VOXEL_LIMIT + 1)),
        )
    with pytest.raises(ValueError, match="map zero to zero"):
        assemble_dense_operator_matrix(
            support=support,
            forward=lambda field: field.reshape(-1) + 1.0,
            zero_atol=0.0,
        )
