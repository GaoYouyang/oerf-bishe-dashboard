from __future__ import annotations

import pytest
import torch

from .psu_b0_view_decomposed_features import (
    view_adjoint_conflict_features,
)


def _feature_lookup(
    fields: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    values, names = view_adjoint_conflict_features(
        fields,
        view_mask=mask,
    )
    return {
        name: float(values[0, index])
        for index, name in enumerate(names)
    }


def test_identical_views_have_full_coherence() -> None:
    base = torch.arange(64, dtype=torch.float64).reshape(1, 1, 1, 4, 4, 4)
    fields = base.repeat(1, 3, 1, 1, 1, 1)
    result = _feature_lookup(
        fields,
        torch.ones((1, 3), dtype=torch.float64),
    )

    assert result["view_pooled_to_sum_norm_ratio"] == pytest.approx(1.0)
    assert result["view_coherent_energy_ratio"] == pytest.approx(1.0)
    assert result["view_pair_cosine_mean"] == pytest.approx(1.0)
    assert result["view_negative_pair_fraction"] == pytest.approx(0.0)


def test_opposing_views_expose_cancellation_and_negative_pair() -> None:
    base = torch.arange(64, dtype=torch.float64).reshape(1, 1, 1, 4, 4, 4)
    fields = torch.cat((base, -base), dim=1)
    result = _feature_lookup(
        fields,
        torch.ones((1, 2), dtype=torch.float64),
    )

    assert result["view_pooled_to_sum_norm_ratio"] == pytest.approx(0.0)
    assert result["view_coherent_energy_ratio"] == pytest.approx(0.0)
    assert result["view_pair_cosine_mean"] == pytest.approx(-1.0)
    assert result["view_negative_pair_fraction"] == pytest.approx(1.0)


def test_features_are_permutation_invariant_and_ignore_inactive_fields() -> None:
    generator = torch.Generator().manual_seed(824)
    fields = torch.randn(
        (2, 4, 1, 4, 5, 6),
        generator=generator,
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    reference, names = view_adjoint_conflict_features(
        fields,
        view_mask=mask,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, permuted_names = view_adjoint_conflict_features(
        fields[:, permutation],
        view_mask=mask[:, permutation],
    )
    changed = torch.where(
        mask[:, :, None, None, None, None] > 0.5,
        fields,
        torch.full_like(fields, 1e9),
    )
    inactive_changed, _ = view_adjoint_conflict_features(
        changed,
        view_mask=mask,
    )

    assert names == permuted_names
    assert torch.allclose(reference, permuted, atol=1e-12, rtol=1e-12)
    assert torch.allclose(reference, inactive_changed, atol=1e-12, rtol=1e-12)


def test_feature_validation_rejects_misaligned_inputs() -> None:
    fields = torch.zeros((1, 2, 1, 4, 4, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="align"):
        view_adjoint_conflict_features(
            fields,
            view_mask=torch.ones((1, 3), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="at least one active"):
        view_adjoint_conflict_features(
            fields,
            view_mask=torch.zeros((1, 2), dtype=torch.float64),
        )
