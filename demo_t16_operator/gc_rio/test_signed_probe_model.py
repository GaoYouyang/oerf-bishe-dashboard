from __future__ import annotations

import inspect

import pytest
import torch

from .model import adjoint_fisher_statistics, source_data_consistency_step
from .signed_probe_model import (
    SignedProbeResidualInverseOperator,
    detector_probe_bank,
    signed_operator_probe_maps,
)


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(29)
    batch, views, measurements, voxels = 2, 3, 6, 18
    return {
        "source_operator": torch.randn(batch, views, measurements, voxels, generator=generator) * 0.12,
        "target_operator": torch.randn(batch, measurements, voxels, generator=generator) * 0.12,
        "source_residual": torch.randn(batch, views, measurements, generator=generator) * 0.03,
        "source_sigma": torch.full((batch, views), 0.05),
        "target_sigma": torch.full((batch,), 0.06),
        "base_field": torch.randn(batch, voxels, generator=generator) * 0.1,
        "analytic_correction": torch.randn(batch, voxels, generator=generator) * 0.02,
        "support": torch.ones(batch, voxels),
    }


def _model(*, use_target_geometry: bool = True):
    return SignedProbeResidualInverseOperator(
        (2, 3, 3),
        hidden_channels=8,
        residual_blocks=1,
        ridge_lambda=0.7,
        data_consistency_step=0.15,
        use_target_geometry=use_target_geometry,
    )


def test_signature_has_no_target_observation_or_truth() -> None:
    names = set(inspect.signature(_model().forward).parameters)
    assert not names & {"target_observation", "truth_field", "clean_observation", "true_sigma"}


def test_signed_probe_maps_preserve_operator_sign() -> None:
    generator = torch.Generator().manual_seed(5)
    operator = torch.randn(2, 6, 18, generator=generator)
    probes = detector_probe_bank(2, 3)
    positive = signed_operator_probe_maps(operator, torch.ones(2), probes)
    negative = signed_operator_probe_maps(-operator, torch.ones(2), probes)
    torch.testing.assert_close(negative, -positive)


def test_zero_init_equals_analytic_data_consistency_baseline() -> None:
    batch = _batch()
    model = _model().eval()
    with torch.no_grad():
        output = model(**batch)
        _, source_fisher, _ = adjoint_fisher_statistics(
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
            source_fisher,
            batch["support"],
            ridge_lambda=model.ridge_lambda,
            step_fraction=model.data_consistency_step,
        )
    torch.testing.assert_close(output.correction, expected)


def test_source_camera_permutation_is_invariant() -> None:
    batch = _batch()
    permutation = torch.tensor([2, 0, 1])
    changed = dict(batch)
    changed["source_operator"] = batch["source_operator"][:, permutation]
    changed["source_residual"] = batch["source_residual"][:, permutation]
    changed["source_sigma"] = batch["source_sigma"][:, permutation]
    model = _model().eval()
    with torch.no_grad():
        model.head.weight.fill_(0.03)
        first = model(**batch).correction
        second = model(**changed).correction
    torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)


def test_query_swap_changes_geometry_model_but_not_matched_control() -> None:
    batch = _batch()
    query = batch["target_operator"].flip(0)
    geometry = _model(use_target_geometry=True).eval()
    control = _model(use_target_geometry=False).eval()
    with torch.no_grad():
        geometry.head.weight.fill_(0.03)
        control.load_state_dict(geometry.state_dict())
        geometry_first = geometry(**batch).correction
        geometry_second = geometry(
            **batch, conditioning_target_operator=query
        ).correction
        control_first = control(**batch).correction
        control_second = control(
            **batch, conditioning_target_operator=query
        ).correction
    assert torch.max(torch.abs(geometry_first - geometry_second)) > 1e-7
    torch.testing.assert_close(control_first, control_second, rtol=1e-6, atol=1e-7)


def test_parameter_matched_control_and_true_target_decoder() -> None:
    batch = _batch()
    geometry = _model(use_target_geometry=True).eval()
    control = _model(use_target_geometry=False).eval()
    assert sum(item.numel() for item in geometry.parameters()) == sum(
        item.numel() for item in control.parameters()
    )
    with torch.no_grad():
        geometry.head.weight.fill_(0.03)
        output = geometry(
            **batch, conditioning_target_operator=batch["target_operator"].flip(0)
        )
    torch.testing.assert_close(
        output.target_residual_prediction,
        torch.einsum("bmp,bp->bm", batch["target_operator"], output.correction),
    )


def test_measurement_grid_mismatch_is_rejected() -> None:
    batch = _batch()
    batch["source_operator"] = batch["source_operator"][:, :, :-1]
    batch["source_residual"] = batch["source_residual"][:, :, :-1]
    batch["target_operator"] = batch["target_operator"][:, :-1]
    with pytest.raises(ValueError, match="measurement count"):
        _model()(**batch)
