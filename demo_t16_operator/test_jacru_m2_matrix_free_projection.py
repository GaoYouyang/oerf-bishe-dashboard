import inspect

import pytest
import torch

from demo_t16_operator.jacru_m2_exact_nullspace_oracle import (
    exact_dense_nullspace_oracle,
)
from demo_t16_operator.jacru_m2_matrix_free_projection import (
    matrix_free_measurement_projection_path,
)


DTYPE = torch.float64


def _maps(matrix: torch.Tensor, shape: tuple[int, int, int]):
    counts = {"forward": 0, "adjoint": 0}

    def forward(field: torch.Tensor) -> torch.Tensor:
        counts["forward"] += 1
        return matrix @ field.reshape(-1)

    def adjoint(measurement: torch.Tensor) -> torch.Tensor:
        counts["adjoint"] += 1
        return (matrix.mT @ measurement.reshape(-1)).reshape(shape)

    return forward, adjoint, counts


def test_signature_exposes_no_truth_or_metric_input() -> None:
    names = set(inspect.signature(matrix_free_measurement_projection_path).parameters)
    assert "truth" not in names
    assert "reference_truth" not in names
    assert "metric" not in names
    assert "score" not in names


def test_fixed_call_budget_and_requested_snapshots() -> None:
    shape = (1, 2, 2)
    matrix = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    forward, adjoint, counts = _maps(matrix, shape)
    reference = torch.zeros(shape, dtype=DTYPE)
    learned = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=DTYPE)
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=DTYPE),
        snapshot_iterations=(0, 1, 3),
    )
    assert result.forward_calls == 4
    assert result.adjoint_calls == 3
    assert counts == {"forward": 4, "adjoint": 3}
    assert set(result.fields_by_iteration) == {0, 1, 3}
    assert len(result.history) == 4
    assert torch.equal(result.fields_by_iteration[0], learned)


def test_converged_small_system_matches_dense_exact_oracle() -> None:
    shape = (1, 2, 3)
    matrix = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 3.0, 0.0],
        ],
        dtype=DTYPE,
    )
    forward, adjoint, _ = _maps(matrix, shape)
    reference = torch.tensor([[[0.1, -0.2, 0.3], [0.2, 0.0, -0.1]]], dtype=DTYPE)
    learned = torch.tensor([[[1.1, 0.5, -0.4], [0.7, 0.9, 1.2]]], dtype=DTYPE)
    correction = learned - reference
    exact = exact_dense_nullspace_oracle(
        correction=correction,
        support=torch.ones(shape, dtype=DTYPE),
        dense_matrix=matrix,
        rank_rtol=1e-12,
    )
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=DTYPE),
        snapshot_iterations=(0, 3, 6),
    )
    expected = reference + exact.null_space_correction
    assert torch.allclose(result.fields_by_iteration[3], expected, atol=1e-11, rtol=1e-11)
    assert torch.allclose(result.fields_by_iteration[6], expected, atol=1e-11, rtol=1e-11)
    assert torch.linalg.vector_norm(forward(result.retained_corrections_by_iteration[6])) < 1e-11


def test_identity_row_map_removes_all_visible_correction_in_one_step() -> None:
    shape = (1, 1, 3)
    matrix = torch.eye(3, dtype=DTYPE)
    forward, adjoint, _ = _maps(matrix, shape)
    reference = torch.tensor([[[0.2, -0.3, 0.4]]], dtype=DTYPE)
    learned = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=DTYPE)
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=DTYPE),
        snapshot_iterations=(0, 1, 4),
    )
    assert torch.allclose(result.fields_by_iteration[1], reference)
    assert torch.allclose(result.fields_by_iteration[4], reference)
    assert result.history[1]["converged"] is True
    assert result.history[-1]["breakdown"] is False


def test_positive_diagonal_preconditioner_preserves_solution() -> None:
    shape = (1, 1, 3)
    matrix = torch.diag(torch.tensor([1.0, 2.0, 4.0], dtype=DTYPE))
    forward, adjoint, _ = _maps(matrix, shape)
    reference = torch.zeros(shape, dtype=DTYPE)
    learned = torch.tensor([[[1.0, -2.0, 0.5]]], dtype=DTYPE)
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=DTYPE),
        snapshot_iterations=(0, 1, 3),
        preconditioner_diagonal=torch.tensor([1.0, 4.0, 16.0], dtype=DTYPE),
    )
    assert result.preconditioner == "supplied_positive_diagonal"
    assert result.preconditioner_applications == 4
    assert torch.allclose(result.fields_by_iteration[1], reference, atol=1e-12)
    assert torch.allclose(result.fields_by_iteration[3], reference, atol=1e-12)


def test_damping_solves_regularized_system_and_is_not_exact_nullspace() -> None:
    shape = (1, 1, 2)
    matrix = torch.eye(2, dtype=DTYPE)
    forward, adjoint, _ = _maps(matrix, shape)
    reference = torch.zeros(shape, dtype=DTYPE)
    learned = torch.tensor([[[1.0, -2.0]]], dtype=DTYPE)
    damping = 0.5
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=DTYPE),
        snapshot_iterations=(0, 1),
        damping=damping,
    )
    expected = learned * (damping / (1.0 + damping))
    assert torch.allclose(result.fields_by_iteration[1], expected, atol=1e-12)
    assert torch.linalg.vector_norm(forward(result.retained_corrections_by_iteration[1])) > 0


def test_affine_observation_mode_projects_learned_field_to_measurement() -> None:
    shape = (1, 1, 3)
    matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=DTYPE,
    )
    forward, adjoint, counts = _maps(matrix, shape)
    reference = torch.tensor([[[9.0, 9.0, 9.0]]], dtype=DTYPE)
    learned = torch.tensor([[[3.0, 4.0, 5.0]]], dtype=DTYPE)
    target = torch.tensor([1.0, -2.0], dtype=DTYPE)
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=DTYPE),
        snapshot_iterations=(0, 2, 4),
        target_observation=target,
    )
    assert result.target_mode == "affine_observation"
    assert torch.allclose(forward(result.fields_by_iteration[2]), target, atol=1e-12)
    assert torch.allclose(result.fields_by_iteration[2][..., 2], learned[..., 2])
    assert torch.allclose(result.fields_by_iteration[4], result.fields_by_iteration[2])
    # The assertion's extra evaluation call is outside the returned ledger.
    assert result.forward_calls == 5
    assert result.adjoint_calls == 4
    assert counts == {"forward": 6, "adjoint": 4}


def test_affine_target_shape_fails_closed_after_single_forward() -> None:
    shape = (1, 1, 2)
    matrix = torch.eye(2, dtype=DTYPE)
    forward, adjoint, counts = _maps(matrix, shape)
    with pytest.raises(ValueError, match="target_observation must match"):
        matrix_free_measurement_projection_path(
            reference_field=torch.zeros(shape, dtype=DTYPE),
            learned_field=torch.ones(shape, dtype=DTYPE),
            forward=forward,
            adjoint=adjoint,
            support=torch.ones(shape, dtype=DTYPE),
            snapshot_iterations=(0, 1),
            target_observation=torch.zeros(3, dtype=DTYPE),
        )
    assert counts == {"forward": 1, "adjoint": 0}


def test_supplied_spd_preconditioner_callable_is_applied_on_fixed_budget() -> None:
    shape = (1, 1, 3)
    matrix = torch.diag(torch.tensor([1.0, 2.0, 4.0], dtype=DTYPE))
    forward, adjoint, _ = _maps(matrix, shape)
    calls = 0

    def apply(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return value / torch.tensor([1.0, 4.0, 16.0], dtype=DTYPE)

    reference = torch.zeros(shape, dtype=DTYPE)
    learned = torch.tensor([[[1.0, -2.0, 0.5]]], dtype=DTYPE)
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=DTYPE),
        snapshot_iterations=(0, 1, 4),
        preconditioner_apply=apply,
        preconditioner_name="dense_spd_test",
    )
    assert result.preconditioner == "dense_spd_test"
    assert result.preconditioner_applications == 5
    assert calls == 5
    assert torch.allclose(result.fields_by_iteration[1], reference, atol=1e-12)
    assert torch.allclose(result.fields_by_iteration[4], reference, atol=1e-12)


def test_callable_and_diagonal_preconditioners_are_mutually_exclusive() -> None:
    shape = (1, 1, 2)
    matrix = torch.eye(2, dtype=DTYPE)
    forward, adjoint, _ = _maps(matrix, shape)
    with pytest.raises(ValueError, match="mutually exclusive"):
        matrix_free_measurement_projection_path(
            reference_field=torch.zeros(shape, dtype=DTYPE),
            learned_field=torch.ones(shape, dtype=DTYPE),
            forward=forward,
            adjoint=adjoint,
            support=torch.ones(shape, dtype=DTYPE),
            snapshot_iterations=(0, 1),
            preconditioner_diagonal=torch.ones(2, dtype=DTYPE),
            preconditioner_apply=lambda value: value,
            preconditioner_name="invalid",
        )


def test_support_masks_reference_and_correction() -> None:
    shape = (1, 2, 2)
    matrix = torch.eye(4, dtype=DTYPE)
    forward, adjoint, _ = _maps(matrix, shape)
    support = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], dtype=DTYPE)
    reference = torch.tensor([[[1.0, 2.0], [9.0, 9.0]]], dtype=DTYPE)
    learned = torch.tensor([[[3.0, 4.0], [8.0, 7.0]]], dtype=DTYPE)
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=support,
        snapshot_iterations=(0, 1),
    )
    assert torch.equal(result.fields_by_iteration[0][support == 0], torch.zeros(2, dtype=DTYPE))
    assert torch.equal(result.fields_by_iteration[1][support == 0], torch.zeros(2, dtype=DTYPE))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"snapshot_iterations": (1,)}, "must include zero"),
        ({"snapshot_iterations": (0, 1, 1)}, "duplicates"),
        ({"snapshot_iterations": (0,), "damping": -1.0}, "damping"),
        (
            {
                "snapshot_iterations": (0,),
                "preconditioner_diagonal": torch.tensor([1.0, 0.0], dtype=DTYPE),
            },
            "positive and finite",
        ),
    ],
)
def test_invalid_contracts_fail_closed(kwargs: dict, message: str) -> None:
    shape = (1, 1, 2)
    matrix = torch.eye(2, dtype=DTYPE)
    forward, adjoint, _ = _maps(matrix, shape)
    common = {
        "reference_field": torch.zeros(shape, dtype=DTYPE),
        "learned_field": torch.ones(shape, dtype=DTYPE),
        "forward": forward,
        "adjoint": adjoint,
        "support": torch.ones(shape, dtype=DTYPE),
    }
    with pytest.raises(ValueError, match=message):
        matrix_free_measurement_projection_path(**common, **kwargs)


def test_zero_visible_correction_keeps_fixed_budget_without_breakdown() -> None:
    shape = (1, 1, 3)
    matrix = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    forward, adjoint, counts = _maps(matrix, shape)
    reference = torch.zeros(shape, dtype=DTYPE)
    learned = torch.tensor([[[0.0, 2.0, -3.0]]], dtype=DTYPE)
    result = matrix_free_measurement_projection_path(
        reference_field=reference,
        learned_field=learned,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=DTYPE),
        snapshot_iterations=(0, 4),
    )
    assert torch.equal(result.fields_by_iteration[4], learned)
    assert counts == {"forward": 5, "adjoint": 4}
    assert all(row["converged"] for row in result.history)
    assert not any(row["breakdown"] for row in result.history)
