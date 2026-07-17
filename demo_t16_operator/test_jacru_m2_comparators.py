"""Contract tests for the conventional JACRU M2 neural comparators."""

from __future__ import annotations

import builtins
import importlib.util
import inspect

import pytest
import torch

from . import jacru_m2_comparators as comparators
from .jacru_m2_comparators import (
    FixedGridDeepONetResidualComparator,
    NeuralOpFNOResidualComparator,
    PooledCNN3DResidualComparator,
    parameter_count,
)
from .jacru_m2_learned_residual import JACRUM2Batch, JACRUM2LearnedResidual


_MODEL_NAMES = ("pooled_cnn", "fixed_grid_deeponet", "official_fno")


def _make_model(name: str):
    if name == "pooled_cnn":
        return PooledCNN3DResidualComparator(
            pose_channels=3,
            hidden_channels=6,
        )
    if name == "fixed_grid_deeponet":
        return FixedGridDeepONetResidualComparator(
            branch_hidden=16,
            trunk_hidden=12,
            rank=8,
        )
    if name == "official_fno":
        if importlib.util.find_spec("neuralop") is None:
            pytest.skip("official neuraloperator package is not installed")
        return NeuralOpFNOResidualComparator(
            pose_channels=2,
            hidden_channels=4,
            n_modes=(3, 3, 3),
            n_layers=1,
        )
    raise AssertionError(f"unknown comparator test case: {name}")


@pytest.fixture(scope="module")
def m2_batch() -> JACRUM2Batch:
    generator = torch.Generator().manual_seed(2901)
    batch_size, view_count = 2, 4
    shape = (12, 12, 12)
    base_field = 0.05 * torch.randn(
        (batch_size, 1, *shape),
        generator=generator,
    )
    support = torch.ones_like(base_field)
    support[:, :, 0] = 0.0
    support[:, :, :, 0] = 0.0
    support[:, :, :, :, -1] = 0.0
    lifted = torch.randn(
        (batch_size, view_count, 1, *shape),
        generator=generator,
    )
    poses = torch.randn(
        (batch_size, view_count, 4),
        generator=generator,
    )
    camera_mask = torch.tensor(
        ((1.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 0.0)),
    )
    return JACRUM2Batch(
        base_field=base_field,
        support=support,
        adjoint_lifted_data_residual=lifted,
        camera_pose_features=poses,
        camera_mask=camera_mask,
    )


def _first_sample(batch: JACRUM2Batch) -> dict[str, torch.Tensor]:
    return {name: value[:1].clone() for name, value in batch.model_kwargs().items()}


def _activate_residual_head(model) -> None:
    with torch.no_grad():
        model.residual_head.weight.fill_(0.05)
        model.residual_head.bias.fill_(0.10)


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_zero_initialization_falls_back_exactly_to_cgls_base(
    model_name: str,
    m2_batch: JACRUM2Batch,
) -> None:
    torch.manual_seed(401)
    model = _make_model(model_name)

    output, gate = model(**m2_batch.model_kwargs(), return_gate=True)
    correction, residual_gate = model.residual(**m2_batch.model_kwargs())

    assert torch.equal(output, m2_batch.base_field)
    assert torch.count_nonzero(correction) == 0
    assert torch.equal(gate, torch.ones_like(gate))
    assert torch.equal(gate, residual_gate)
    assert model.maximum_residual_magnitude == 0.25
    assert parameter_count(model) == model.parameter_count
    assert parameter_count(model) == model.trainable_parameter_count
    assert 0 < parameter_count(model) < 250_000


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_active_residual_is_support_limited_and_bounded(
    model_name: str,
    m2_batch: JACRUM2Batch,
) -> None:
    torch.manual_seed(409)
    model = _make_model(model_name).eval()
    _activate_residual_head(model)

    with torch.no_grad():
        output, gate = model(**m2_batch.model_kwargs(), return_gate=True)
    residual = output - m2_batch.base_field
    outside = m2_batch.support == 0.0

    assert gate.shape == (m2_batch.base_field.shape[0], 1, 1, 1, 1)
    assert torch.count_nonzero(residual.masked_select(outside)) == 0
    assert torch.equal(
        output.masked_select(outside),
        m2_batch.base_field.masked_select(outside),
    )
    assert torch.count_nonzero(residual.masked_select(~outside)) > 0
    assert bool(
        torch.all(residual.abs() <= model.maximum_residual_magnitude + 1e-7)
    )


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_zero_initialized_head_has_a_finite_learning_gradient(
    model_name: str,
    m2_batch: JACRUM2Batch,
) -> None:
    torch.manual_seed(419)
    model = _make_model(model_name)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    arguments = _first_sample(m2_batch)
    base_field = arguments["base_field"].requires_grad_(True)
    arguments["base_field"] = base_field

    output = model(**arguments)
    output.mean().backward()

    assert base_field.grad is not None
    assert bool(torch.all(torch.isfinite(base_field.grad)))
    assert model.residual_head.bias.grad is not None
    assert abs(float(model.residual_head.bias.grad.item())) > 0.0
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.all(torch.isfinite(gradient))) for gradient in gradients)

    optimizer.step()
    with torch.no_grad():
        updated = model(**arguments)
    assert not torch.equal(updated, base_field)


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_forward_is_deterministic_and_view_permutation_invariant(
    model_name: str,
    m2_batch: JACRUM2Batch,
) -> None:
    torch.manual_seed(431)
    model = _make_model(model_name).train()
    _activate_residual_head(model)
    arguments = _first_sample(m2_batch)

    first, first_gate = model(**arguments, return_gate=True)
    second, second_gate = model(**arguments, return_gate=True)
    permutation = torch.tensor((2, 0, 3, 1))
    permuted = dict(arguments)
    permuted["adjoint_lifted_data_residual"] = arguments[
        "adjoint_lifted_data_residual"
    ][:, permutation]
    permuted["camera_pose_features"] = arguments["camera_pose_features"][
        :, permutation
    ]
    permuted["camera_mask"] = arguments["camera_mask"][:, permutation]
    permuted_output, permuted_gate = model(**permuted, return_gate=True)

    assert torch.equal(first, second)
    assert torch.equal(first_gate, second_gate)
    torch.testing.assert_close(permuted_output, first, atol=2e-7, rtol=2e-6)
    assert torch.equal(permuted_gate, first_gate)


def test_public_forward_contract_matches_m2_and_exposes_no_truth_api(
    m2_batch: JACRUM2Batch,
) -> None:
    classes = (
        PooledCNN3DResidualComparator,
        FixedGridDeepONetResidualComparator,
        NeuralOpFNOResidualComparator,
    )
    expected_forward = inspect.signature(JACRUM2LearnedResidual.forward)
    expected_residual = inspect.signature(JACRUM2LearnedResidual.residual)
    forbidden = ("truth", "label", "family", "reference", "target")

    for model_class in classes:
        assert inspect.signature(model_class.forward) == expected_forward
        assert inspect.signature(model_class.residual) == expected_residual
        for callable_object in (
            model_class.__init__,
            model_class.forward,
            model_class.residual,
        ):
            names = tuple(inspect.signature(callable_object).parameters)
            assert not any(
                fragment in name.lower()
                for name in names
                for fragment in forbidden
            )

    model = PooledCNN3DResidualComparator()
    with pytest.raises(TypeError, match="truth"):
        model(
            **m2_batch.model_kwargs(),
            truth=torch.zeros_like(m2_batch.base_field),
        )


def test_fixed_grid_deeponet_rejects_a_different_grid(
    m2_batch: JACRUM2Batch,
) -> None:
    model = FixedGridDeepONetResidualComparator()
    arguments = {
        name: value[..., :10, :10, :10]
        if name in {"base_field", "support", "adjoint_lifted_data_residual"}
        else value
        for name, value in m2_batch.model_kwargs().items()
    }
    with pytest.raises(ValueError, match="requires grid_shape"):
        model(**arguments)


def test_fno_missing_dependency_error_is_actionable(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "neuralop.models":
            raise ModuleNotFoundError("blocked for dependency-error test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="official.*neuraloperator.*install"):
        NeuralOpFNOResidualComparator()


def test_official_fno_smoke_when_neuralop_is_available(
    m2_batch: JACRUM2Batch,
) -> None:
    pytest.importorskip("neuralop")
    torch.manual_seed(443)
    model = NeuralOpFNOResidualComparator(
        pose_channels=2,
        hidden_channels=4,
        n_modes=(3, 3, 3),
        n_layers=1,
    )
    arguments = _first_sample(m2_batch)

    output = model(**arguments)
    output.square().mean().backward()

    assert output.shape == arguments["base_field"].shape
    assert torch.equal(output, arguments["base_field"])
    assert model.backbone.__class__.__module__.startswith("neuralop.")
    assert model.residual_head.bias.grad is not None
    assert bool(torch.all(torch.isfinite(model.residual_head.bias.grad)))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available on this host",
)
@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_twelve_cubed_training_step_runs_on_available_mac_mps(
    model_name: str,
    m2_batch: JACRUM2Batch,
) -> None:
    torch.manual_seed(449)
    device = torch.device("mps")
    model = _make_model(model_name).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    arguments = {
        name: value[:1].to(device)
        for name, value in m2_batch.model_kwargs().items()
    }

    output = model(**arguments)
    loss = output.square().mean()
    loss.backward()
    optimizer.step()
    torch.mps.synchronize()

    assert output.shape == (1, 1, 12, 12, 12)
    assert bool(torch.isfinite(loss.detach()).cpu())
    assert model.residual_head.bias.grad is not None
    assert bool(torch.all(torch.isfinite(model.residual_head.bias.grad)).cpu())


def test_parameter_count_rejects_non_modules() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        parameter_count(object())
