from __future__ import annotations

import inspect

import torch

from .shared_field_model import (
    SharedFieldResidualInverseOperator,
    source_adjoint_fisher,
    source_krylov_stack,
)


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(37)
    return {
        "source_operator": torch.randn(2, 3, 6, 18, generator=generator) * 0.12,
        "target_operator": torch.randn(2, 6, 18, generator=generator) * 0.12,
        "source_residual": torch.randn(2, 3, 6, generator=generator) * 0.03,
        "source_sigma": torch.full((2, 3), 0.05),
        "target_sigma": torch.full((2,), 0.06),
        "base_field": torch.randn(2, 18, generator=generator) * 0.15,
        "analytic_correction": torch.randn(2, 18, generator=generator) * 0.01,
        "support": torch.ones(2, 18),
    }


def _model(*, use_krylov_features: bool = True):
    return SharedFieldResidualInverseOperator(
        (2, 3, 3),
        hidden_channels=8,
        residual_blocks=1,
        ridge_lambda=0.8,
        data_consistency_step=0.0,
        use_krylov_features=use_krylov_features,
    )


def test_signature_rejects_target_observation_and_truth() -> None:
    names = set(inspect.signature(_model().forward).parameters)
    assert not names & {"target_observation", "truth_field", "clean_observation", "true_sigma"}


def test_zero_init_is_exact_zero_correction_baseline() -> None:
    output = _model().eval()(**_batch())
    torch.testing.assert_close(output.correction, torch.zeros_like(output.correction))
    torch.testing.assert_close(
        output.target_residual_prediction,
        torch.zeros_like(output.target_residual_prediction),
    )


def test_krylov_stack_uses_signed_operator_residual_pair() -> None:
    batch = _batch()
    first, _, _ = source_krylov_stack(
        batch["source_operator"],
        batch["source_residual"],
        batch["source_sigma"],
    )
    second, _, _ = source_krylov_stack(
        -batch["source_operator"],
        batch["source_residual"],
        batch["source_sigma"],
    )
    torch.testing.assert_close(second, -first)


def test_source_camera_permutation_is_invariant() -> None:
    batch = _batch()
    changed = dict(batch)
    permutation = torch.tensor([2, 0, 1])
    changed["source_operator"] = batch["source_operator"][:, permutation]
    changed["source_residual"] = batch["source_residual"][:, permutation]
    changed["source_sigma"] = batch["source_sigma"][:, permutation]
    model = _model().eval()
    with torch.no_grad():
        model.head.weight.fill_(0.03)
        first = model(**batch).correction
        second = model(**changed).correction
    torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)


def test_target_query_does_not_change_shared_field_or_true_decoder() -> None:
    batch = _batch()
    model = _model().eval()
    with torch.no_grad():
        model.head.weight.fill_(0.03)
        first = model(**batch)
        second = model(
            **batch,
            conditioning_target_operator=batch["target_operator"].flip(0),
        )
    torch.testing.assert_close(first.correction, second.correction)
    torch.testing.assert_close(
        second.target_residual_prediction,
        torch.einsum("bmp,bp->bm", batch["target_operator"], second.correction),
    )


def test_matched_krylov_control_has_same_parameter_count() -> None:
    candidate = _model(use_krylov_features=True)
    control = _model(use_krylov_features=False)
    assert sum(item.numel() for item in candidate.parameters()) == sum(
        item.numel() for item in control.parameters()
    )


def test_adjoint_only_skips_discarded_krylov_operator_calls() -> None:
    counter: dict[str, int] = {}
    _model(use_krylov_features=False).eval()(
        **_batch(), operator_call_counter=counter
    )
    assert counter == {"source_adjoint": 2}


def test_precomputed_statistics_are_reusable_without_prediction_drift() -> None:
    batch = _batch()
    model = _model(use_krylov_features=False).eval()
    with torch.no_grad():
        model.head.weight.fill_(0.03)
        direct = model(**batch).correction
        statistics = source_adjoint_fisher(
            batch["source_operator"],
            batch["source_residual"],
            batch["source_sigma"],
        )
        counter: dict[str, int] = {}
        cached = model(
            **batch,
            precomputed_source_statistics=statistics,
            operator_call_counter=counter,
        ).correction
    torch.testing.assert_close(cached, direct)
    assert counter == {}


def test_krylov_call_counter_matches_executed_einsums() -> None:
    batch = _batch()
    counter: dict[str, int] = {}
    source_krylov_stack(
        batch["source_operator"],
        batch["source_residual"],
        batch["source_sigma"],
        operator_call_counter=counter,
    )
    assert counter == {"source_adjoint": 8, "source_forward": 6}
