from __future__ import annotations

import pytest
import torch

from demo_t16_operator.jacru_m2_data_consistency import data_consistency_path


def _maps(matrix: torch.Tensor, shape: tuple[int, int, int]):
    def forward(field: torch.Tensor) -> torch.Tensor:
        return matrix @ field.reshape(-1)

    def adjoint(observation: torch.Tensor) -> torch.Tensor:
        return (matrix.T @ observation).reshape(shape)

    return forward, adjoint


def test_measurement_pullback_reduces_identity_residual_and_counts_calls() -> None:
    shape = (2, 2, 2)
    matrix = torch.eye(8, dtype=torch.float64)
    forward, adjoint = _maps(matrix, shape)
    initial = torch.zeros(shape, dtype=torch.float64)
    observation = torch.arange(8, dtype=torch.float64)
    path = data_consistency_path(
        initial_field=initial,
        observation=observation,
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=torch.float64),
        step_size=0.5,
        operator_norm_squared_bound=1.0,
        snapshot_steps=(0, 1, 3),
        mode="measurement_pullback",
    )
    errors = [
        torch.linalg.vector_norm(forward(path.fields_by_step[step]) - observation)
        for step in (0, 1, 3)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert path.forward_calls == 3
    assert path.adjoint_calls == 3


def test_nullspace_filter_removes_only_observable_correction() -> None:
    shape = (1, 2, 2)
    matrix = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    forward, adjoint = _maps(matrix, shape)
    base = torch.zeros(shape, dtype=torch.float64)
    initial = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], dtype=torch.float64)
    path = data_consistency_path(
        initial_field=initial,
        base_field=base,
        observation=torch.zeros(1, dtype=torch.float64),
        forward=forward,
        adjoint=adjoint,
        support=torch.ones(shape, dtype=torch.float64),
        step_size=1.0,
        operator_norm_squared_bound=1.0,
        snapshot_steps=(0, 1),
        mode="base_nullspace_filter",
    )
    result = path.fields_by_step[1]
    assert result.reshape(-1)[0].item() == pytest.approx(0.0)
    torch.testing.assert_close(result.reshape(-1)[1:], initial.reshape(-1)[1:])
    assert torch.linalg.vector_norm(forward(result - base)).item() == pytest.approx(0.0)


def test_support_is_enforced_on_every_snapshot() -> None:
    shape = (1, 2, 2)
    matrix = torch.eye(4, dtype=torch.float64)
    forward, adjoint = _maps(matrix, shape)
    support = torch.ones(shape, dtype=torch.float64)
    support[0, 0, 0] = 0.0
    path = data_consistency_path(
        initial_field=torch.ones(shape, dtype=torch.float64),
        observation=torch.ones(4, dtype=torch.float64),
        forward=forward,
        adjoint=adjoint,
        support=support,
        step_size=0.5,
        operator_norm_squared_bound=1.0,
        snapshot_steps=(0, 2),
        mode="measurement_pullback",
    )
    assert path.fields_by_step[0][0, 0, 0].item() == 0.0
    assert path.fields_by_step[2][0, 0, 0].item() == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"snapshot_steps": (1,)}, "include zero"),
        ({"snapshot_steps": (0, 0)}, "duplicates"),
        ({"step_size": 0.0}, "positive and finite"),
        ({"step_size": 2.0}, "smaller than 2"),
        ({"operator_norm_squared_bound": 0.0}, "positive and finite"),
        ({"mode": "oracle"}, "unknown"),
    ],
)
def test_invalid_contract_is_rejected(kwargs: dict[str, object], match: str) -> None:
    shape = (1, 1, 2)
    matrix = torch.eye(2, dtype=torch.float64)
    forward, adjoint = _maps(matrix, shape)
    values = {
        "initial_field": torch.zeros(shape, dtype=torch.float64),
        "observation": torch.zeros(2, dtype=torch.float64),
        "forward": forward,
        "adjoint": adjoint,
        "support": torch.ones(shape, dtype=torch.float64),
        "step_size": 0.5,
        "operator_norm_squared_bound": 1.0,
        "snapshot_steps": (0, 1),
        "mode": "measurement_pullback",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        data_consistency_path(**values)


def test_nullspace_filter_requires_a_base_field() -> None:
    shape = (1, 1, 2)
    matrix = torch.eye(2, dtype=torch.float64)
    forward, adjoint = _maps(matrix, shape)
    with pytest.raises(ValueError, match="base_field"):
        data_consistency_path(
            initial_field=torch.zeros(shape, dtype=torch.float64),
            observation=torch.zeros(2, dtype=torch.float64),
            forward=forward,
            adjoint=adjoint,
            support=torch.ones(shape, dtype=torch.float64),
            step_size=0.5,
            operator_norm_squared_bound=1.0,
            snapshot_steps=(0, 1),
            mode="base_nullspace_filter",
        )
