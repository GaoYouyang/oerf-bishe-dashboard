from __future__ import annotations

import torch

from demo_t16_operator.psu_b0_initial_normal_features import (
    initial_normal_spectral_features,
)
from demo_t16_operator.psu_b0_morphology_spectral_experts import (
    ObservableMorphologyExpertFactory,
    materialize_log_expert_mixture,
)


def _expert_logs() -> torch.Tensor:
    first = torch.zeros((4, 4, 3))
    second = torch.linspace(-0.4, 0.4, 48).reshape(4, 4, 3)
    second = second - second.mean()
    return torch.stack((first, second), dim=0)


def test_log_space_mixture_is_positive_and_normalized() -> None:
    materialized = materialize_log_expert_mixture(
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
        expert_log_gains=_expert_logs(),
        baseline_expert_index=0,
        temperature=1.0,
        confidence_threshold=-1.0,
        maximum_blend=0.7,
    )
    assert torch.all(materialized.gain > 0.0)
    geometric = torch.exp(
        torch.mean(torch.log(materialized.gain), dim=(1, 2, 3))
    )
    assert torch.allclose(geometric, torch.ones_like(geometric), atol=1e-6)


def test_high_confidence_threshold_is_exact_baseline() -> None:
    materialized = materialize_log_expert_mixture(
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
        expert_log_gains=_expert_logs(),
        baseline_expert_index=0,
        temperature=1.0,
        confidence_threshold=10.0,
        maximum_blend=1.0,
    )
    assert torch.equal(materialized.gain, torch.ones_like(materialized.gain))


def test_baseline_top_score_forces_exact_fallback() -> None:
    materialized = materialize_log_expert_mixture(
        torch.tensor([[3.0, 1.0], [2.0, -4.0]]),
        expert_log_gains=_expert_logs(),
        baseline_expert_index=0,
        temperature=0.5,
        confidence_threshold=-1.0,
        maximum_blend=1.0,
    )
    assert torch.equal(materialized.gain, torch.ones_like(materialized.gain))


def test_factory_matches_manual_ridge_scores() -> None:
    normal = torch.randn(3, 1, 8, 8, 8)
    features, _ = initial_normal_spectral_features(normal)
    mean = torch.zeros(features.shape[1])
    scale = torch.ones(features.shape[1])
    weights = torch.randn(features.shape[1] + 1, 2) * 0.03
    factory = ObservableMorphologyExpertFactory(
        expert_log_gains=_expert_logs(),
        expert_candidate_ids=("base", "expert"),
        baseline_expert_index=0,
        feature_mean=mean,
        feature_scale=scale,
        ridge_weights=weights,
        temperature=0.7,
        confidence_threshold=0.1,
        maximum_blend=0.5,
    )
    scores, extracted = factory.selector_scores(normal)
    expected = torch.cat(
        (torch.ones((len(normal), 1)), features),
        dim=1,
    ) @ weights
    assert torch.allclose(extracted, features)
    assert torch.allclose(scores, expected, atol=1e-6, rtol=1e-6)
    materialized = factory(normal)
    assert torch.all(materialized.gain > 0.0)
