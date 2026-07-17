from __future__ import annotations

import inspect

import torch

from .phase_interface_bost import (
    PhaseInterfaceConfig,
    PhaseInterfaceField,
    optimize_phase_interface_bost,
    phase_interface_objective,
)


def _identity(values: torch.Tensor) -> torch.Tensor:
    return values


def _config(**overrides: object) -> PhaseInterfaceConfig:
    values: dict[str, object] = {
        "grid_shape": (5, 5, 5),
        "max_interfaces": 2,
        "epsilon": 0.7,
        "optimization_steps": 8,
        "learning_rate": 2e-2,
        "seed": 712,
        "initial_gate_logit": -1.5,
        "deployment_gate_threshold": 0.5,
        "minimum_interface_rms": 1e-6,
        "background_smoothness_weight": 2e-3,
        "phase_smoothness_weight": 2e-4,
        "eikonal_weight": 2e-3,
        "gate_sparsity_weight": 1e-2,
        "amplitude_weight": 1e-4,
    }
    values.update(overrides)
    return PhaseInterfaceConfig(**values)


def test_declared_phase_interface_parameterization_is_exact() -> None:
    model = PhaseInterfaceField(_config(optimization_steps=1), batch_size=1, dtype=torch.float64)
    with torch.no_grad():
        model.background.fill_(0.25)
        model.gate_logits.copy_(torch.tensor([[0.2, -0.4]], dtype=torch.float64))
        model.amplitudes.copy_(torch.tensor([[0.7, -0.3]], dtype=torch.float64))
        model.phase_fields[0, 0].copy_(
            torch.linspace(-1.0, 1.0, 125, dtype=torch.float64).reshape(5, 5, 5)
        )
        model.phase_fields[0, 1].copy_(
            torch.linspace(0.8, -0.6, 125, dtype=torch.float64).reshape(5, 5, 5)
        )
    gates = torch.sigmoid(model.gate_logits)
    expected = model.background + torch.sum(
        gates[:, :, None, None, None]
        * model.amplitudes[:, :, None, None, None]
        * torch.tanh(model.phase_fields / model.config.epsilon),
        dim=1,
        keepdim=True,
    )
    torch.testing.assert_close(model(), expected, rtol=0.0, atol=0.0)


def test_no_interface_observations_close_false_positive_interface_gate() -> None:
    axis = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    smooth = torch.exp(
        -0.5 * ((xx / 0.75).square() + (yy / 0.85).square() + (zz / 0.9).square())
    )[None, None]
    observation = torch.cat((torch.zeros_like(smooth), smooth), dim=0)
    result = optimize_phase_interface_bost(
        observation,
        forward=_identity,
        adjoint=_identity,
        config=_config(optimization_steps=12),
    )
    assert result.adjoint_evaluations == 1
    assert result.forward_evaluations == 12
    assert result.optimization_steps == 12
    assert not torch.any(result.active_interface_mask)
    assert torch.count_nonzero(result.amplitudes[0]) == 0
    assert torch.count_nonzero(result.prediction[0]) == 0
    torch.testing.assert_close(result.prediction, result.background, rtol=0.0, atol=0.0)


def test_all_regularized_parameters_receive_gradients() -> None:
    generator = torch.Generator().manual_seed(99)
    observation = torch.randn((1, 1, 5, 5, 5), generator=generator, dtype=torch.float64)
    model = PhaseInterfaceField(_config(optimization_steps=1), batch_size=1, dtype=torch.float64)
    with torch.no_grad():
        model.amplitudes.copy_(torch.tensor([[0.35, -0.22]], dtype=torch.float64))
        model.gate_logits.copy_(torch.tensor([[0.15, -0.1]], dtype=torch.float64))
    objective = phase_interface_objective(model, observation, forward=_identity)
    objective.total.backward()
    for parameter in (
        model.background,
        model.gate_logits,
        model.amplitudes,
        model.phase_fields,
    ):
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))
        assert torch.linalg.vector_norm(parameter.grad) > 0.0
    assert objective.eikonal > 0.0
    assert objective.phase_smoothness > 0.0
    assert objective.background_smoothness == 0.0
    assert objective.gate_sparsity > 0.0


def test_inverse_entry_points_do_not_accept_evaluation_target() -> None:
    for entry_point in (
        optimize_phase_interface_bost,
        phase_interface_objective,
        PhaseInterfaceField.forward,
    ):
        assert "truth" not in inspect.signature(entry_point).parameters


def test_fixed_seed_replays_the_same_fixed_budget_run() -> None:
    axis = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    observation = (0.4 * torch.tanh((xx + 0.12 * yy - 0.05 * zz) / 0.25))[None, None]
    initial = torch.zeros_like(observation)
    config = _config(optimization_steps=9, seed=2027)
    first = optimize_phase_interface_bost(
        observation,
        forward=_identity,
        config=config,
        initial_background=initial,
    )
    second = optimize_phase_interface_bost(
        observation,
        forward=_identity,
        config=config,
        initial_background=initial,
    )
    assert first.forward_evaluations == second.forward_evaluations == 9
    assert first.adjoint_evaluations == second.adjoint_evaluations == 0
    assert first.history == second.history
    for left, right in (
        (first.prediction, second.prediction),
        (first.soft_prediction, second.soft_prediction),
        (first.background, second.background),
        (first.phase_fields, second.phase_fields),
        (first.amplitudes, second.amplitudes),
        (first.gate_probabilities, second.gate_probabilities),
        (first.interface_rms, second.interface_rms),
    ):
        assert torch.equal(left, right)
    assert torch.equal(first.active_interface_mask, second.active_interface_mask)


def test_zero_interface_mode_is_background_only() -> None:
    config = _config(max_interfaces=0, optimization_steps=1)
    model = PhaseInterfaceField(config, batch_size=2)
    assert model.phase_fields.shape == (2, 0, 5, 5, 5)
    assert model.gate_probabilities().shape == (2, 0)
    torch.testing.assert_close(model(), model.background, rtol=0.0, atol=0.0)
