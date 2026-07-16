from __future__ import annotations

import inspect

import pytest
import torch

from .model import (
    GeometryConditionedResidualInverseOperator,
    adjoint_fisher_statistics,
    source_data_consistency_step,
)


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    batch, views, measurements, voxels = 2, 3, 5, 18
    source_operator = torch.randn(
        batch, views, measurements, voxels, generator=generator
    ) * 0.15
    target_operator = torch.randn(
        batch, measurements, voxels, generator=generator
    ) * 0.15
    source_residual = torch.randn(
        batch, views, measurements, generator=generator
    ) * 0.04
    return {
        "source_operator": source_operator,
        "target_operator": target_operator,
        "source_residual": source_residual,
        "source_sigma": torch.full((batch, views), 0.05),
        "target_sigma": torch.full((batch,), 0.055),
        "base_field": torch.randn(batch, voxels, generator=generator) * 0.1,
        "analytic_correction": torch.randn(batch, voxels, generator=generator)
        * 0.02,
        "support": torch.ones(voxels),
    }


def _model(*, use_target_geometry: bool = True):
    return GeometryConditionedResidualInverseOperator(
        (2, 3, 3),
        hidden_channels=8,
        residual_blocks=1,
        ridge_lambda=0.7,
        data_consistency_step=0.15,
        use_target_geometry=use_target_geometry,
    )


def test_predictor_signature_cannot_accept_target_observation_or_truth() -> None:
    names = set(inspect.signature(_model().forward).parameters)
    forbidden = {
        "target_observation",
        "clean_observation",
        "truth_field",
        "true_sigma",
        "true_radius",
    }
    assert not names & forbidden
    assert forbidden.isdisjoint(_model().predictor_argument_names)


def test_zero_initialized_model_equals_declared_analytic_dc_baseline() -> None:
    batch = _batch()
    model = _model().eval()
    with torch.no_grad():
        output = model(**batch)
        _, fisher, _ = adjoint_fisher_statistics(
            batch["source_operator"],
            batch["target_operator"],
            batch["source_residual"],
            batch["source_sigma"],
            batch["target_sigma"],
        )
        expected = source_data_consistency_step(
            batch["analytic_correction"],
            batch["source_operator"],
            batch["source_residual"],
            batch["source_sigma"],
            fisher,
            batch["support"],
            ridge_lambda=model.ridge_lambda,
            step_fraction=model.data_consistency_step,
        )
    torch.testing.assert_close(output.learned_increment, torch.zeros_like(expected))
    torch.testing.assert_close(output.correction, expected)


def test_source_camera_permutation_is_exactly_invariant() -> None:
    batch = _batch()
    model = _model().eval()
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        first = model(**batch).correction
        changed = dict(batch)
        changed["source_operator"] = batch["source_operator"][:, permutation]
        changed["source_residual"] = batch["source_residual"][:, permutation]
        changed["source_sigma"] = batch["source_sigma"][:, permutation]
        second = model(**changed).correction
    torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)


def test_zero_source_residual_and_zero_analytic_correction_give_zero() -> None:
    batch = _batch()
    batch["source_residual"] = torch.zeros_like(batch["source_residual"])
    batch["analytic_correction"] = torch.zeros_like(batch["analytic_correction"])
    model = _model().eval()
    with torch.no_grad():
        output = model(**batch)
    torch.testing.assert_close(output.correction, torch.zeros_like(output.correction))
    torch.testing.assert_close(
        output.target_residual_prediction,
        torch.zeros_like(output.target_residual_prediction),
    )


def test_target_query_changes_correction_only_when_geometry_is_enabled() -> None:
    batch = _batch()
    changed = dict(batch)
    changed["target_operator"] = 1.7 * batch["target_operator"]
    geometry = _model(use_target_geometry=True).eval()
    control = _model(use_target_geometry=False).eval()
    with torch.no_grad():
        geometry.head.weight.fill_(0.04)
        control.load_state_dict(geometry.state_dict())
        geometry_first = geometry(**batch).correction
        geometry_second = geometry(**changed).correction
        control_first = control(**batch).correction
        control_second = control(**changed).correction
    assert torch.max(torch.abs(geometry_first - geometry_second)) > 1e-7
    torch.testing.assert_close(control_first, control_second, rtol=1e-6, atol=1e-7)


def test_shuffled_conditioning_keeps_true_target_physics_decoder() -> None:
    batch = _batch()
    model = _model(use_target_geometry=True).eval()
    shuffled = batch["target_operator"].flip(0)
    with torch.no_grad():
        model.head.weight.fill_(0.04)
        regular = model(**batch)
        changed = model(**batch, conditioning_target_operator=shuffled)
    assert torch.max(torch.abs(regular.correction - changed.correction)) > 1e-7
    torch.testing.assert_close(
        changed.target_residual_prediction,
        torch.einsum("bmp,bp->bm", batch["target_operator"], changed.correction),
    )


def test_parameter_matched_control_and_gradient_flow() -> None:
    batch = _batch()
    geometry = _model(use_target_geometry=True)
    control = _model(use_target_geometry=False)
    assert sum(value.numel() for value in geometry.parameters()) == sum(
        value.numel() for value in control.parameters()
    )
    output = geometry(**batch)
    loss = output.target_residual_prediction.square().mean()
    loss.backward()
    assert geometry.head.weight.grad is not None
    assert torch.linalg.vector_norm(geometry.head.weight.grad) > 0.0


def test_shape_contract_rejects_mismatched_target_operator() -> None:
    batch = _batch()
    batch["target_operator"] = batch["target_operator"][:, :-1]
    with pytest.raises(ValueError, match="target_operator"):
        _model()(**batch)
