"""Contract tests for the generic interface-reconstruction baselines."""

from __future__ import annotations

import inspect

import pytest
import torch

from .interface_baselines import (
    cgls_baseline,
    edge_preserving_pdhg_baseline,
    robust_data_pdhg_baseline,
)


class CountingIdentity:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.adjoint_calls = 0

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return field.clone()

    def adjoint(self, measurement: torch.Tensor) -> torch.Tensor:
        self.adjoint_calls += 1
        return measurement.clone()


def _support() -> torch.Tensor:
    support = torch.ones((3, 4, 5), dtype=torch.float64)
    support[0, 0, 0] = 0.0
    return support


def _run_cgls(
    observation: torch.Tensor,
    operator: CountingIdentity,
    *,
    iterations: int = 6,
):
    return cgls_baseline(
        observation,
        forward=operator.forward,
        adjoint=operator.adjoint,
        support=_support(),
        spacing_xyz=(1.0, 1.5, 2.0),
        iterations=iterations,
    )


def _run_pdhg(
    observation: torch.Tensor,
    operator: CountingIdentity,
    *,
    iterations: int = 40,
    penalty: str = "huber",
    regularization_weight: float = 0.05,
):
    return edge_preserving_pdhg_baseline(
        observation,
        forward=operator.forward,
        adjoint=operator.adjoint,
        support=_support(),
        spacing_xyz=(1.0, 1.5, 2.0),
        iterations=iterations,
        regularization_weight=regularization_weight,
        data_norm_squared_bound=1.0,
        penalty=penalty,
        huber_delta=0.2,
    )


@pytest.mark.parametrize(
    "solver",
    [cgls_baseline, edge_preserving_pdhg_baseline, robust_data_pdhg_baseline],
)
def test_public_solver_signatures_have_no_reference_field_parameter(solver) -> None:
    names = {name.lower() for name in inspect.signature(solver).parameters}
    forbidden_fragments = ("truth", "ground_truth", "target_field", "reference_field")
    assert not any(
        fragment in name for name in names for fragment in forbidden_fragments
    )


def test_cgls_uses_exact_fixed_call_budget_and_preserves_support() -> None:
    operator = CountingIdentity()
    observation = torch.arange(60, dtype=torch.float64).reshape(3, 4, 5) / 60.0

    result = _run_cgls(observation, operator, iterations=7)

    assert result.forward_calls == result.adjoint_calls == 7
    assert operator.forward_calls == operator.adjoint_calls == 7
    assert len(result.history) == 7
    assert result.field[0, 0, 0] == 0.0
    torch.testing.assert_close(
        result.field[_support().bool()],
        observation[_support().bool()],
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("penalty", ["tv", "huber"])
def test_edge_pdhg_uses_exact_fixed_call_budget(penalty: str) -> None:
    operator = CountingIdentity()
    observation = torch.ones((3, 4, 5), dtype=torch.float64)
    observation[0, 0, 0] = 0.0

    result = _run_pdhg(
        observation,
        operator,
        iterations=23,
        penalty=penalty,
    )

    assert result.forward_calls == result.adjoint_calls == 23
    assert operator.forward_calls == operator.adjoint_calls == 23
    assert len(result.history) == 23
    assert result.field[0, 0, 0] == 0.0
    assert all(0.0 < row["step_contract"] < 1.0 for row in result.history)


@pytest.mark.parametrize("solver_name", ["cgls", "pdhg"])
def test_zero_observation_stays_zero_without_shortening_budget(solver_name: str) -> None:
    operator = CountingIdentity()
    observation = torch.zeros((3, 4, 5), dtype=torch.float64)
    iterations = 5

    if solver_name == "cgls":
        result = _run_cgls(observation, operator, iterations=iterations)
    else:
        result = _run_pdhg(observation, operator, iterations=iterations)

    torch.testing.assert_close(result.field, torch.zeros_like(result.field))
    assert result.forward_calls == result.adjoint_calls == iterations
    assert operator.forward_calls == operator.adjoint_calls == iterations


def test_cgls_converges_for_simple_identity_operator() -> None:
    operator = CountingIdentity()
    observation = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64).reshape(3, 4, 5)
    observation[0, 0, 0] = 0.0

    result = _run_cgls(observation, operator, iterations=4)

    torch.testing.assert_close(result.field, observation, atol=1e-12, rtol=1e-12)
    assert result.history[-1]["data_residual_norm"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("penalty", ["tv", "huber"])
def test_edge_pdhg_converges_for_constant_identity_problem(penalty: str) -> None:
    operator = CountingIdentity()
    observation = torch.ones((3, 4, 5), dtype=torch.float64)
    observation[0, 0, 0] = 0.0

    result = _run_pdhg(
        observation,
        operator,
        iterations=180,
        penalty=penalty,
        regularization_weight=0.05,
    )

    initial_error = torch.linalg.vector_norm(observation)
    final_error = torch.linalg.vector_norm(result.field - observation)
    assert final_error < 0.08 * initial_error
    assert result.history[-1]["primal_update_norm"] < 1e-5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iterations": 0}, "iterations"),
        ({"spacing_xyz": (1.0, 0.0, 1.0)}, "spacing_xyz"),
        ({"support": torch.full((3, 4, 5), 0.5)}, "binary"),
    ],
)
def test_cgls_rejects_invalid_inputs_before_operator_calls(kwargs, message: str) -> None:
    operator = CountingIdentity()
    arguments = {
        "forward": operator.forward,
        "adjoint": operator.adjoint,
        "support": _support(),
        "spacing_xyz": (1.0, 1.0, 1.0),
        "iterations": 3,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        cgls_baseline(torch.zeros((3, 4, 5)), **arguments)

    assert operator.forward_calls == operator.adjoint_calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iterations": 0}, "iterations"),
        ({"data_norm_squared_bound": 0.0}, "data_norm_squared_bound"),
        ({"regularization_weight": -1.0}, "regularization_weight"),
        ({"penalty": "quadratic"}, "penalty"),
        ({"step_safety": 1.0}, "step_safety"),
        ({"primal_step": 0.2}, "supplied together"),
    ],
)
def test_edge_pdhg_rejects_invalid_inputs_before_operator_calls(kwargs, message: str) -> None:
    operator = CountingIdentity()
    arguments = {
        "forward": operator.forward,
        "adjoint": operator.adjoint,
        "support": _support(),
        "spacing_xyz": (1.0, 1.0, 1.0),
        "iterations": 3,
        "regularization_weight": 0.1,
        "data_norm_squared_bound": 1.0,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        edge_preserving_pdhg_baseline(
            torch.zeros((3, 4, 5)),
            **arguments,
        )

    assert operator.forward_calls == operator.adjoint_calls == 0


def test_nonfinite_observation_is_rejected_before_operator_calls() -> None:
    operator = CountingIdentity()
    observation = torch.zeros((3, 4, 5), dtype=torch.float64)
    observation[1, 1, 1] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        _run_cgls(observation, operator)

    assert operator.forward_calls == operator.adjoint_calls == 0


def test_invalid_explicit_pdhg_step_contract_is_rejected() -> None:
    operator = CountingIdentity()

    with pytest.raises(ValueError, match="PDHG steps"):
        edge_preserving_pdhg_baseline(
            torch.zeros((3, 4, 5), dtype=torch.float64),
            forward=operator.forward,
            adjoint=operator.adjoint,
            support=_support(),
            spacing_xyz=(1.0, 1.0, 1.0),
            iterations=3,
            regularization_weight=0.1,
            data_norm_squared_bound=1.0,
            primal_step=1.0,
            data_dual_step=1.0,
            edge_dual_step=1.0,
        )

    assert operator.forward_calls == operator.adjoint_calls == 0


class RepeatedScalarOperator:
    def __init__(self, repeats: int) -> None:
        self.repeats = repeats
        self.forward_calls = 0
        self.adjoint_calls = 0

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return field.reshape(1).expand(self.repeats).clone()

    def adjoint(self, measurement: torch.Tensor) -> torch.Tensor:
        self.adjoint_calls += 1
        return torch.sum(measurement).reshape(1, 1, 1)


def test_robust_data_pdhg_resists_one_large_measurement_outlier() -> None:
    observation = torch.tensor([1.0, 1.0, 1.0, 10.0], dtype=torch.float64)
    support = torch.ones((1, 1, 1), dtype=torch.float64)
    robust_operator = RepeatedScalarOperator(4)
    least_squares_operator = RepeatedScalarOperator(4)

    robust = robust_data_pdhg_baseline(
        observation,
        forward=robust_operator.forward,
        adjoint=robust_operator.adjoint,
        support=support,
        spacing_xyz=(1.0, 1.0, 1.0),
        iterations=500,
        regularization_weight=0.0,
        data_norm_squared_bound=4.0,
        data_huber_delta=0.5,
        ridge_weight=1e-8,
    )
    least_squares = edge_preserving_pdhg_baseline(
        observation,
        forward=least_squares_operator.forward,
        adjoint=least_squares_operator.adjoint,
        support=support,
        spacing_xyz=(1.0, 1.0, 1.0),
        iterations=500,
        regularization_weight=0.0,
        data_norm_squared_bound=4.0,
    )

    robust_value = float(robust.field.item())
    least_squares_value = float(least_squares.field.item())
    assert abs(robust_value - 1.0) < abs(least_squares_value - 1.0)
    assert robust_value < 1.5
    assert least_squares_value == pytest.approx(3.25, abs=1e-6)
    assert robust.forward_calls == robust.adjoint_calls == 500
    assert 0.0 < robust.history[-1]["data_dual_saturation_fraction"] <= 1.0


def test_robust_data_pdhg_accepts_warm_start_and_spatial_edge_weights() -> None:
    operator = CountingIdentity()
    observation = torch.ones((3, 4, 5), dtype=torch.float64)
    observation[0, 0, 0] = 0.0
    initial = 0.25 * observation
    edge_weights = torch.ones_like(observation)
    edge_weights[:, :, 2:] = 0.2
    result = robust_data_pdhg_baseline(
        observation,
        forward=operator.forward,
        adjoint=operator.adjoint,
        support=_support(),
        spacing_xyz=(1.0, 1.5, 2.0),
        iterations=12,
        regularization_weight=0.05,
        data_norm_squared_bound=1.0,
        data_huber_delta=0.5,
        initial_field=initial,
        edge_weight_map=edge_weights,
    )
    assert result.forward_calls == result.adjoint_calls == 12
    assert result.field[0, 0, 0] == 0.0
    assert torch.linalg.vector_norm(result.field - observation) < torch.linalg.vector_norm(
        initial - observation
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"data_huber_delta": 0.0}, "data_huber_delta"),
        ({"edge_huber_delta": 0.0}, "edge_huber_delta"),
        ({"ridge_weight": -1.0}, "ridge_weight"),
        ({"edge_penalty": "quadratic"}, "edge_penalty"),
        ({"edge_weight_map": torch.zeros((3, 4, 5))}, "edge_weight_map"),
        ({"initial_field": torch.zeros((2, 2, 2))}, "initial_field"),
    ],
)
def test_robust_data_pdhg_rejects_invalid_inputs_before_calls(kwargs, message) -> None:
    operator = CountingIdentity()
    arguments = {
        "forward": operator.forward,
        "adjoint": operator.adjoint,
        "support": _support(),
        "spacing_xyz": (1.0, 1.0, 1.0),
        "iterations": 3,
        "regularization_weight": 0.1,
        "data_norm_squared_bound": 1.0,
        "data_huber_delta": 0.5,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        robust_data_pdhg_baseline(torch.zeros((3, 4, 5)), **arguments)
    assert operator.forward_calls == operator.adjoint_calls == 0


def test_callback_shape_violation_is_reported() -> None:
    observation = torch.zeros((3, 4, 5), dtype=torch.float64)

    with pytest.raises(ValueError, match="returned shape"):
        cgls_baseline(
            observation,
            forward=lambda field: field,
            adjoint=lambda _: torch.zeros((2, 2, 2), dtype=torch.float64),
            support=_support(),
            spacing_xyz=(1.0, 1.0, 1.0),
            iterations=2,
        )
