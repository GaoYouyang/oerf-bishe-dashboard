from __future__ import annotations

import torch

from demo_t16_operator.psu_b0_initial_normal_features import (
    initial_normal_spectral_features,
)
from demo_t16_operator.psu_b0_risk_quantile_experts import (
    MultiObjectiveRiskSingleExpertFactory,
    RiskQuantileSingleExpertFactory,
    materialize_single_expert_path,
    multiobjective_risk_actions,
    risk_quantile_actions,
)


def _expert_logs() -> torch.Tensor:
    baseline = torch.zeros((4, 4, 3))
    first = torch.linspace(-0.5, 0.5, 48).reshape(4, 4, 3)
    second = torch.flip(first, dims=(0,))
    return torch.stack(
        (
            baseline,
            first - first.mean(),
            second - second.mean(),
        ),
        dim=0,
    )


def test_risk_actions_reject_harmful_and_low_quantile_rows() -> None:
    actions, accepted, harm = risk_quantile_actions(
        mean_scores=torch.tensor(
            [[0.0, 3.0, 1.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0]]
        ),
        lower_scores=torch.tensor(
            [[0.0, 1.5, 0.2], [0.0, 0.1, 0.2], [2.0, 1.0, 0.0]]
        ),
        harm_logits=torch.tensor(
            [[-20.0, -3.0, 0.0], [-20.0, 3.0, -3.0], [-20.0, 0.0, 0.0]]
        ),
        baseline_expert_index=0,
        route_mode="mean_quantile_harm",
        minimum_score=0.5,
        maximum_harm_probability=0.25,
    )
    assert actions.tolist() == [1, 0, 0]
    assert accepted.tolist() == [True, False, False]
    assert harm.shape == (3,)


def test_exact_fallback_and_positive_normalized_path() -> None:
    materialized = materialize_single_expert_path(
        torch.tensor([0, 1, 2]),
        expert_log_gains=_expert_logs(),
        baseline_expert_index=0,
        interpolation_fraction=0.5,
    )
    assert torch.equal(
        materialized.gain[0],
        torch.ones_like(materialized.gain[0]),
    )
    assert torch.all(materialized.gain > 0.0)
    geometric = torch.exp(
        torch.mean(torch.log(materialized.gain), dim=(1, 2, 3))
    )
    assert torch.allclose(geometric, torch.ones_like(geometric), atol=1e-6)


def test_factory_scores_match_manual_linear_heads() -> None:
    normal = torch.randn(3, 1, 8, 8, 8)
    features, _ = initial_normal_spectral_features(normal)
    width = features.shape[1] + 1
    mean_weights = torch.randn(width, 3) * 0.02
    lower_weights = torch.randn(width, 3) * 0.02
    harm_weights = torch.randn(width, 3) * 0.02
    factory = RiskQuantileSingleExpertFactory(
        expert_log_gains=_expert_logs(),
        expert_candidate_ids=("base", "first", "second"),
        baseline_expert_index=0,
        feature_mean=torch.zeros(features.shape[1]),
        feature_scale=torch.ones(features.shape[1]),
        mean_weights=mean_weights,
        lower_weights=lower_weights,
        harm_weights=harm_weights,
        route_mode="quantile_harm",
        minimum_score=0.0,
        maximum_harm_probability=0.5,
        interpolation_fraction=0.25,
    )
    mean, lower, harm, extracted = factory.risk_scores(normal)
    design = torch.cat((torch.ones((len(normal), 1)), features), dim=1)
    assert torch.allclose(extracted, features)
    assert torch.allclose(mean, design @ mean_weights, atol=1e-6)
    assert torch.allclose(lower, design @ lower_weights, atol=1e-6)
    assert torch.allclose(harm, design @ harm_weights, atol=1e-6)
    assert torch.all(factory(normal).gain > 0.0)


def test_multiobjective_route_vetoes_front_risk() -> None:
    actions, accepted, _, front_harm = multiobjective_risk_actions(
        field_mean_scores=torch.tensor([[0.0, 3.0], [0.0, 3.0]]),
        field_lower_scores=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        field_harm_logits=torch.tensor([[-20.0, -3.0], [-20.0, -3.0]]),
        front_lower_scores=torch.tensor([[0.0, 0.01], [0.0, -0.03]]),
        front_harm_logits=torch.tensor([[-20.0, -3.0], [-20.0, 3.0]]),
        baseline_expert_index=0,
        minimum_field_score=1.0,
        maximum_field_harm_probability=0.25,
        minimum_front_lower_delta=-0.01,
        maximum_front_harm_probability=0.25,
    )
    assert actions.tolist() == [1, 0]
    assert accepted.tolist() == [True, False]
    assert front_harm[0] < front_harm[1]


def test_multiobjective_factory_produces_positive_direction() -> None:
    normal = torch.randn(2, 1, 8, 8, 8)
    features, _ = initial_normal_spectral_features(normal)
    width = features.shape[1] + 1
    weights = [torch.randn(width, 3) * 0.01 for _ in range(5)]
    factory = MultiObjectiveRiskSingleExpertFactory(
        expert_log_gains=_expert_logs(),
        expert_candidate_ids=("base", "first", "second"),
        baseline_expert_index=0,
        feature_mean=torch.zeros(features.shape[1]),
        feature_scale=torch.ones(features.shape[1]),
        field_mean_weights=weights[0],
        field_lower_weights=weights[1],
        field_harm_weights=weights[2],
        front_lower_weights=weights[3],
        front_harm_weights=weights[4],
        minimum_field_score=0.0,
        maximum_field_harm_probability=0.5,
        minimum_front_lower_delta=-0.01,
        maximum_front_harm_probability=0.5,
        interpolation_fraction=0.5,
    )
    assert torch.all(factory(normal).gain > 0.0)
