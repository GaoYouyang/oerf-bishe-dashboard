"""CPU/float64 tests for exact PSU-B0 coordinate reduction utilities."""

from __future__ import annotations

import pytest
import torch

from .psu_b0_active_coordinates import (
    CoordinateSupportGauge,
    build_coordinate_support_gauge,
    reduce_zero_coupling_system,
)


def _support() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )


def test_coordinate_embedding_has_stable_indices_and_explicit_dense_E() -> None:
    gauge = build_coordinate_support_gauge(_support())
    assert gauge.active_indices.tolist() == [1, 3, 5, 6, 10]
    assert gauge.E.shape == (12, 5)
    assert gauge.ET.shape == (5, 12)
    assert torch.equal(gauge.ET, gauge.E.T)
    assert torch.equal(gauge.E.T @ gauge.E, torch.eye(5, dtype=torch.float64))

    active = torch.tensor(
        [[2.0, -1.0, 3.0, 4.0, 5.0], [0.5, 0.25, -0.5, 1.5, 2.5]],
        dtype=torch.float64,
    )
    embedded = gauge.embed_active(active)
    assert embedded.shape == (2, 1, 2, 2, 3)
    torch.testing.assert_close(embedded.flatten(1), active @ gauge.ET)
    torch.testing.assert_close(gauge.restrict_active(embedded), active)
    assert (
        torch.count_nonzero(embedded.flatten(1)[:, [0, 2, 4, 7, 8, 9, 11]])
        == 0
    )


def test_coordinate_E_and_ET_satisfy_dot_product_identity() -> None:
    gauge = CoordinateSupportGauge.from_support(_support())
    generator = torch.Generator().manual_seed(1601)
    active = torch.randn((3, gauge.n_active), generator=generator, dtype=torch.float64)
    full = torch.randn((3, 1, *gauge.grid_shape), generator=generator, dtype=torch.float64)
    lhs = torch.sum(gauge.embed_active(active) * full)
    rhs = torch.sum(active * gauge.restrict_active(full))
    torch.testing.assert_close(lhs, rhs, atol=1e-14, rtol=1e-14)


def test_zero_rows_columns_constant_and_recovery_mapping_are_exact() -> None:
    K = torch.tensor(
        [
            [0.0, 2.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -3.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([1.0, 5.0, -2.0, -3.0], dtype=torch.float64)
    ledger = reduce_zero_coupling_system(K, target)

    assert ledger.original_data_indices.tolist() == [0, 1, 2, 3]
    assert ledger.active_data_indices.tolist() == [0, 2]
    assert ledger.deleted_data_indices.tolist() == [1, 3]
    assert ledger.original_primal_indices.tolist() == [0, 1, 2, 3, 4]
    assert ledger.active_primal_indices.tolist() == [1, 3]
    assert ledger.deleted_primal_indices.tolist() == [0, 2, 4]
    torch.testing.assert_close(
        ledger.K_active,
        torch.tensor([[2.0, -1.0], [-3.0, 4.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(ledger.target_active, target[[0, 2]])
    assert ledger.deleted_data_objective_constant.item() == 17.0

    active = torch.tensor([0.25, -1.5], dtype=torch.float64)
    recovered = ledger.recover_full_primal(active)
    torch.testing.assert_close(
        recovered,
        torch.tensor([0.0, 0.25, 0.0, -1.5, 0.0], dtype=torch.float64),
    )
    full_objective = 0.5 * torch.sum((K @ recovered - target).square())
    torch.testing.assert_close(ledger.reduced_objective(active), full_objective)


def test_nonzero_fixed_offset_shifts_target_and_recovery_origin() -> None:
    K = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    target = torch.tensor([7.0, -4.0, 1.0], dtype=torch.float64)
    fixed = torch.tensor([0.5, 9.0, -1.0, -3.0], dtype=torch.float64)
    ledger = reduce_zero_coupling_system(K, target, fixed_full=fixed)
    expected_shifted = target - K @ fixed
    torch.testing.assert_close(ledger.fixed_data_offset, K @ fixed)
    torch.testing.assert_close(ledger.target_shifted, expected_shifted)
    torch.testing.assert_close(ledger.target_active, expected_shifted[[0, 2]])
    torch.testing.assert_close(ledger.deleted_target_shifted, expected_shifted[[1]])
    assert ledger.deleted_data_objective_constant.item() == 8.0

    correction = torch.tensor([1.5, 0.25], dtype=torch.float64)
    full = ledger.recover_full_primal(correction)
    torch.testing.assert_close(
        full,
        torch.tensor([2.0, 9.0, -0.75, -3.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        ledger.reduced_objective(correction),
        0.5 * torch.sum((K @ full - target).square()),
    )

    batch = torch.stack((correction, -correction))
    recovered_batch = ledger.recover_full_primal(batch)
    assert recovered_batch.shape == (2, 4)
    torch.testing.assert_close(
        ledger.reduced_objective(batch),
        0.5 * torch.sum((recovered_batch @ K.T - target).square(), dim=1),
    )


@pytest.mark.parametrize(
    "support, message",
    [
        (torch.tensor([[[0.0, 0.5]]], dtype=torch.float64), "strictly binary"),
        (torch.tensor([[[0.0, float("nan")]]], dtype=torch.float64), "finite"),
        (torch.zeros((1, 1, 2), dtype=torch.float64), "active coordinate"),
        (torch.ones((2, 2), dtype=torch.float64), r"\[Z,Y,X\]"),
    ],
)
def test_non_coordinate_or_empty_support_fails_closed(
    support: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_coordinate_support_gauge(support)


def test_coordinate_shape_mismatches_fail_closed() -> None:
    gauge = build_coordinate_support_gauge(_support())
    with pytest.raises(ValueError, match=r"\[B,n_active\]"):
        gauge.embed_active(torch.ones((2, gauge.n_active + 1), dtype=torch.float64))
    with pytest.raises(ValueError, match=r"\[B,1,Z,Y,X\]"):
        gauge.restrict_active(torch.ones((2, *gauge.grid_shape), dtype=torch.float64))


@pytest.mark.parametrize(
    "K, target, fixed, message",
    [
        (torch.ones(3, dtype=torch.float64), torch.ones(3), None, "matrix"),
        (torch.eye(2, dtype=torch.float64), torch.ones(3), None, "target"),
        (torch.eye(2, dtype=torch.float64), torch.ones(2), torch.ones(3), "fixed_full"),
        (torch.zeros((2, 3), dtype=torch.float64), torch.ones(2), None, "empty active"),
        (
            torch.tensor([[1.0, float("inf")]], dtype=torch.float64),
            torch.ones(1),
            None,
            "finite",
        ),
    ],
)
def test_reduction_invalid_inputs_fail_closed(
    K: torch.Tensor,
    target: torch.Tensor,
    fixed: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reduce_zero_coupling_system(K, target, fixed_full=fixed)


def test_strict_zero_activity_does_not_use_an_epsilon() -> None:
    K = torch.tensor([[1e-300, 0.0], [0.0, 0.0]], dtype=torch.float64)
    ledger = reduce_zero_coupling_system(K, torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert ledger.active_data_indices.tolist() == [0]
    assert ledger.active_primal_indices.tolist() == [0]
    assert ledger.K_active.item() == 1e-300
