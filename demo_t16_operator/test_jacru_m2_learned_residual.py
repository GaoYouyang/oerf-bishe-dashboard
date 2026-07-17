"""Contract tests for the JACRU M2 CGLS-plus-residual component."""

from __future__ import annotations

from dataclasses import fields
import inspect

import pytest
import torch

from .jacru_m2_learned_residual import (
    JACRUM2Batch,
    JACRUM2LearnedResidual,
    prepare_jacru_m2_batch,
)
from .jacru_synthetic_fixture import (
    JACRUSyntheticFixtureConfig,
    build_jacru_synthetic_case,
)


@pytest.fixture(scope="module")
def prepared_case() -> tuple[JACRUM2Batch, dict[str, int]]:
    config = JACRUSyntheticFixtureConfig(
        grid_shape=(12, 12, 12),
        detector_shape=(3, 3),
        samples_per_ray=6,
        enable_noise=False,
        enable_camera_bias=False,
    )
    case = build_jacru_synthetic_case(
        family="single_interface",
        split="train",
        base_seed=1729,
        config=config,
    )
    case.inference.operator.reset_call_counts()
    batch = prepare_jacru_m2_batch(
        case.inference,
        cgls_iterations=3,
        model_dtype=torch.float32,
    )
    return batch, case.inference.operator.call_report()


def _active_model() -> JACRUM2LearnedResidual:
    torch.manual_seed(3107)
    model = JACRUM2LearnedResidual()
    with torch.no_grad():
        model.residual_head.weight.fill_(0.125)
        model.residual_head.bias.fill_(0.05)
    return model


def test_zero_initialization_is_exactly_the_cgls_base(
    prepared_case: tuple[JACRUM2Batch, dict[str, int]],
) -> None:
    batch, physical_calls = prepared_case
    torch.manual_seed(19)
    model = JACRUM2LearnedResidual()

    output, gate = model(**batch.model_kwargs(), return_gate=True)
    correction, correction_gate = model.residual(**batch.model_kwargs())

    assert torch.count_nonzero(batch.base_field) > 0
    assert torch.equal(output, batch.base_field)
    assert torch.count_nonzero(correction) == 0
    assert torch.equal(gate, correction_gate)
    assert physical_calls == {"forward_calls": 4, "adjoint_calls": 4}
    assert model.trainable_parameter_count < 20_000


def test_preparation_applies_one_support_to_every_physical_stage() -> None:
    config = JACRUSyntheticFixtureConfig(
        grid_shape=(12, 12, 12),
        detector_shape=(3, 3),
        samples_per_ray=6,
        enable_noise=False,
        enable_camera_bias=False,
    )
    case = build_jacru_synthetic_case(
        family="smooth_no_interface",
        split="train",
        base_seed=1733,
        config=config,
    )
    support = torch.ones(config.grid_shape, dtype=torch.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    support[:, :, [0, -1]] = 0.0

    batch = prepare_jacru_m2_batch(
        case.inference,
        support=support,
        cgls_iterations=2,
        model_dtype=torch.float32,
    )

    assert torch.equal(case.inference.operator.support, support)
    assert torch.count_nonzero(batch.base_field[batch.support == 0.0]) == 0
    expanded_support = batch.support[:, None]
    assert torch.count_nonzero(
        batch.adjoint_lifted_data_residual * (1.0 - expanded_support)
    ) == 0


def test_camera_set_is_invariant_to_joint_view_permutation(
    prepared_case: tuple[JACRUM2Batch, dict[str, int]],
) -> None:
    batch, _ = prepared_case
    model = _active_model().eval()
    camera_mask = batch.camera_mask.clone()
    camera_mask[:, 1] = 0.0
    arguments = batch.model_kwargs()
    arguments["camera_mask"] = camera_mask

    with torch.no_grad():
        expected, expected_gate = model(**arguments, return_gate=True)
        permutation = torch.tensor([2, 0, 1])
        permuted = dict(arguments)
        permuted["adjoint_lifted_data_residual"] = arguments[
            "adjoint_lifted_data_residual"
        ][:, permutation]
        permuted["camera_pose_features"] = arguments["camera_pose_features"][
            :, permutation
        ]
        permuted["camera_mask"] = arguments["camera_mask"][:, permutation]
        actual, actual_gate = model(**permuted, return_gate=True)

    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(actual_gate, expected_gate, atol=2e-7, rtol=2e-6)


def test_residual_is_support_limited_gated_and_bounded(
    prepared_case: tuple[JACRUM2Batch, dict[str, int]],
) -> None:
    batch, _ = prepared_case
    model = _active_model().eval()
    support = batch.support.clone()
    support[:, :, 0] = 0.0
    support[:, :, :, 0] = 0.0
    arguments = batch.model_kwargs()
    arguments["support"] = support

    with torch.no_grad():
        output, gate = model(**arguments, return_gate=True)
    residual = output - batch.base_field

    assert gate.shape == (batch.base_field.shape[0], 1, 1, 1, 1)
    assert bool(torch.all((gate >= 0.0) & (gate <= 1.0)))
    assert torch.count_nonzero(residual.masked_select(support == 0.0)) == 0
    assert torch.equal(
        output.masked_select(support == 0.0),
        batch.base_field.masked_select(support == 0.0),
    )
    bound = model.maximum_residual_magnitude * gate.expand_as(residual)
    assert bool(torch.all(residual.abs() <= bound + 1e-7))


def test_zero_initialized_model_has_a_valid_learning_gradient(
    prepared_case: tuple[JACRUM2Batch, dict[str, int]],
) -> None:
    batch, _ = prepared_case
    torch.manual_seed(23)
    model = JACRUM2LearnedResidual()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    arguments = batch.model_kwargs()
    base_field = batch.base_field.detach().clone().requires_grad_(True)
    arguments["base_field"] = base_field

    output = model(**arguments)
    output.mean().backward()

    assert base_field.grad is not None
    assert bool(torch.all(torch.isfinite(base_field.grad)))
    assert model.residual_head.bias.grad is not None
    assert abs(float(model.residual_head.bias.grad.item())) > 0.0
    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(
        bool(torch.all(torch.isfinite(gradient)))
        for gradient in parameter_gradients
    )

    optimizer.step()
    with torch.no_grad():
        updated = model(**arguments)
    assert not torch.equal(updated, base_field)


def test_public_api_has_no_truth_or_evaluation_parameter(
    prepared_case: tuple[JACRUM2Batch, dict[str, int]],
) -> None:
    batch, _ = prepared_case
    forbidden = ("truth", "label", "reference", "evaluation", "target_field")
    callables = (
        JACRUM2LearnedResidual.__init__,
        JACRUM2LearnedResidual.forward,
        JACRUM2LearnedResidual.residual,
        prepare_jacru_m2_batch,
    )
    for callable_object in callables:
        names = tuple(inspect.signature(callable_object).parameters)
        assert not any(
            fragment in name.lower() for name in names for fragment in forbidden
        )
    batch_names = tuple(field.name for field in fields(JACRUM2Batch))
    assert not any(
        fragment in name.lower() for name in batch_names for fragment in forbidden
    )

    model = JACRUM2LearnedResidual()
    with pytest.raises(TypeError, match="truth"):
        model(
            **batch.model_kwargs(),
            truth=torch.zeros_like(batch.base_field),
        )


def test_forward_is_deterministic_without_stochastic_layers(
    prepared_case: tuple[JACRUM2Batch, dict[str, int]],
) -> None:
    batch, _ = prepared_case
    model = _active_model().train()

    first, first_gate = model(**batch.model_kwargs(), return_gate=True)
    second, second_gate = model(**batch.model_kwargs(), return_gate=True)

    assert torch.equal(first, second)
    assert torch.equal(first_gate, second_gate)


_TRAINING_DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    _TRAINING_DEVICES.append("mps")


@pytest.mark.parametrize("device_name", _TRAINING_DEVICES)
def test_twelve_cubed_training_step_runs_on_cpu_and_available_mps(
    device_name: str,
) -> None:
    torch.manual_seed(101)
    batch = 2
    views = 3
    shape = (12, 12, 12)
    base_field = torch.randn((batch, 1, *shape), dtype=torch.float32)
    support = torch.ones_like(base_field)
    support[:, :, 0] = 0.0
    lifted = torch.randn((batch, views, 1, *shape), dtype=torch.float32)
    poses = torch.randn((batch, views, 4), dtype=torch.float32)
    camera_mask = torch.tensor(
        ((1.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
        dtype=torch.float32,
    )
    device = torch.device(device_name)
    model = JACRUM2LearnedResidual().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    output, gate = model(
        base_field.to(device),
        support.to(device),
        lifted.to(device),
        poses.to(device),
        camera_mask.to(device),
        return_gate=True,
    )
    loss = output.square().mean() + 0.01 * gate.square().mean()
    loss.backward()
    optimizer.step()

    assert output.shape == base_field.shape
    assert bool(torch.isfinite(loss.detach()).cpu())
    assert model.trainable_parameter_count < 20_000
