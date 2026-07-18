from __future__ import annotations

import math

import pytest
import torch

from demo_t16_operator.field_jvp_vjp_gate import (
    audit_tensor_closure,
    central_difference_sweep,
    compare_residual_structure,
    evaluate_jvp_vjp,
)


def _analytic_closure(value: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.sum(value.square()),
            value[0] * value[1] + torch.sin(value[2]),
        )
    )


def test_analytic_closure_matches_jvp_vjp_dot_and_fixed_multi_h() -> None:
    point = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
    tangent = torch.tensor([0.4, -0.2, 0.5], dtype=torch.float64)
    cotangent = torch.tensor([0.6, -0.8], dtype=torch.float64)
    h_values = (0.2, 0.1, 0.05)

    audit = audit_tensor_closure(
        _analytic_closure,
        point,
        tangent,
        cotangent,
        h_values,
    )
    jacobian = torch.stack(
        (
            2.0 * point,
            torch.stack((point[1], point[0], torch.cos(point[2]))),
        )
    )
    expected_jvp = jacobian @ tangent
    expected_vjp = jacobian.T @ cotangent

    assert torch.allclose(audit.autodiff.jvp, expected_jvp, atol=1e-15, rtol=1e-14)
    assert torch.allclose(audit.autodiff.vjp, expected_vjp, atol=1e-15, rtol=1e-14)
    assert audit.autodiff.dot_relative_defect < 1e-14
    assert audit.autodiff.repeated_primal_relative_defect == 0.0
    assert audit.finite
    assert audit.nondegenerate
    assert tuple(level.h for level in audit.finite_difference.levels) == h_values
    errors = [level.relative_error for level in audit.finite_difference.levels]
    assert errors[0] > errors[1] > errors[2]
    assert audit.finite_difference.best_index == 2
    assert audit.finite_difference.best_h == 0.05
    assert audit.finite_difference.count_at_or_below(1e-4) == 2
    assert audit.passes(
        dot_relative_tolerance=1e-12,
        finite_difference_relative_tolerance=1e-4,
        minimum_finite_difference_levels=2,
    )
    assert not audit.passes(
        dot_relative_tolerance=1e-12,
        finite_difference_relative_tolerance=1e-8,
        minimum_finite_difference_levels=2,
    )


def test_zero_derivative_cannot_fake_a_dot_product_pass() -> None:
    point = torch.tensor([0.2, -0.4], dtype=torch.float64)
    tangent = torch.tensor([0.6, 0.8], dtype=torch.float64)
    cotangent = torch.tensor([1.0, -0.5], dtype=torch.float64)

    audit = audit_tensor_closure(
        lambda value: torch.zeros_like(value),
        point,
        tangent,
        cotangent,
        (0.1, 0.01),
    )

    assert audit.autodiff.dot_relative_defect == 0.0
    assert audit.finite
    assert not audit.autodiff.nondegenerate_derivative
    assert not audit.autodiff.nondegenerate_dot
    assert not audit.nondegenerate
    assert not audit.passes(
        dot_relative_tolerance=1.0,
        finite_difference_relative_tolerance=1.0,
    )


def test_nonfinite_closure_fails_closed_without_reporting_a_best_h() -> None:
    point = torch.tensor([0.3, -0.2], dtype=torch.float64)
    tangent = torch.tensor([0.4, 0.7], dtype=torch.float64)
    cotangent = torch.tensor([1.0, -0.5], dtype=torch.float64)

    def nonfinite(value: torch.Tensor) -> torch.Tensor:
        return value / torch.zeros((), dtype=value.dtype)

    audit = audit_tensor_closure(
        nonfinite,
        point,
        tangent,
        cotangent,
        (0.1, 0.01),
    )

    assert not audit.finite
    assert not audit.nondegenerate
    assert math.isinf(audit.autodiff.dot_relative_defect)
    assert not audit.finite_difference.any_finite
    assert audit.finite_difference.best_index is None
    assert audit.finite_difference.best_h is None
    assert math.isinf(audit.finite_difference.best_relative_error)
    assert not audit.passes(
        dot_relative_tolerance=1.0,
        finite_difference_relative_tolerance=1.0,
    )


def test_residual_structure_accepts_difference_and_rejects_wrong_direct_map() -> None:
    point = torch.tensor([0.4, -0.6, 0.9], dtype=torch.float64)
    tangent = torch.tensor([0.2, 0.5, -0.3], dtype=torch.float64)
    cotangent = torch.tensor([0.7, -0.4], dtype=torch.float64)
    curved_matrix = torch.tensor(
        [[1.2, -0.3, 0.7], [0.1, 0.8, -0.5]], dtype=torch.float64
    )
    straight_matrix = torch.tensor(
        [[0.9, -0.1, 0.2], [-0.2, 0.4, -0.1]], dtype=torch.float64
    )

    def curved(value: torch.Tensor) -> torch.Tensor:
        return curved_matrix @ value + torch.stack((value[0] ** 2, value[2] ** 2))

    def straight(value: torch.Tensor) -> torch.Tensor:
        return straight_matrix @ value

    correct = compare_residual_structure(
        lambda value: curved(value) - straight(value),
        curved,
        straight,
        point,
        tangent,
        cotangent,
    )
    wrong = compare_residual_structure(
        lambda value: curved(value) + straight(value),
        curved,
        straight,
        point,
        tangent,
        cotangent,
    )

    assert correct.finite
    assert correct.nondegenerate
    assert correct.consistent
    assert correct.primal.within_tolerance
    assert correct.jvp.within_tolerance
    assert correct.vjp.within_tolerance
    assert wrong.finite
    assert wrong.nondegenerate
    assert not wrong.consistent
    assert not wrong.primal.within_tolerance
    assert not wrong.jvp.within_tolerance
    assert not wrong.vjp.within_tolerance


@pytest.mark.parametrize("h_values", [(), (0.1, 0.1), (0.0,), (-0.1,), (math.nan,)])
def test_invalid_fixed_h_grid_is_rejected(h_values: tuple[float, ...]) -> None:
    point = torch.tensor([0.2, 0.5], dtype=torch.float64)
    tangent = torch.tensor([0.3, -0.4], dtype=torch.float64)
    reference = evaluate_jvp_vjp(
        lambda value: value.square(),
        point,
        tangent,
        torch.tensor([0.6, -0.8], dtype=torch.float64),
    ).jvp

    with pytest.raises(ValueError):
        central_difference_sweep(
            lambda value: value.square(),
            point,
            tangent,
            reference,
            h_values,
        )


def test_cpu_float64_and_tensor_output_contracts_are_enforced() -> None:
    with pytest.raises(TypeError, match="CPU torch.float64"):
        evaluate_jvp_vjp(
            lambda value: value.square(),
            torch.tensor([0.2], dtype=torch.float32),
            torch.tensor([0.3], dtype=torch.float32),
            torch.tensor([0.4], dtype=torch.float32),
        )

    point = torch.tensor([0.2], dtype=torch.float64)
    with pytest.raises(TypeError, match="torch.Tensor"):
        evaluate_jvp_vjp(
            lambda _value: 1.0,  # type: ignore[return-value]
            point,
            torch.ones_like(point),
            torch.ones_like(point),
        )
