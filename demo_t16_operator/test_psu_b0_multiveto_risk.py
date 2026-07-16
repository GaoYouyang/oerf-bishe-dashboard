from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .psu_b0_multiveto_risk import (
    ObservableMultiVetoDirection,
    balanced_view_masks,
    canonical_observable_risk_features,
    observable_stress_scores,
)
from .psu_b0_residual_risk import (
    RISK_FEATURE_NAMES,
    observable_risk_features,
)


class _Direction(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(
        self,
        gradient: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = len(gradient)
        return self.scale * gradient, {
            "gain_minimum": torch.full((batch,), 0.5, dtype=gradient.dtype),
            "gain_maximum": torch.full((batch,), 2.0, dtype=gradient.dtype),
            "gain_geometric_mean": torch.ones(batch, dtype=gradient.dtype),
            "controller_coefficients": torch.zeros(
                (batch, 3),
                dtype=gradient.dtype,
            ),
        }


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(731)
    gradient = torch.randn((2, 1, 8, 8, 8), generator=generator)
    residual = torch.randn((2, 12, 2), generator=generator)
    sigma = torch.tensor([[0.2, 0.3, 0.4], [0.4, 0.2, 0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    return gradient, residual, sigma, mask


def test_canonical_features_equal_manual_support_projection() -> None:
    gradient, residual, sigma, mask = _fixture()
    candidate = 1.4 * gradient
    fallback = 0.8 * gradient
    support = torch.ones((8, 8, 8))
    support[0] = 0.0
    diagnostics = {
        "gain_minimum": torch.tensor([0.4, 0.5]),
        "gain_maximum": torch.tensor([2.0, 2.5]),
        "controller_coefficients": torch.ones((2, 7)),
    }
    canonical = canonical_observable_risk_features(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=4,
        candidate_direction=candidate,
        fallback_direction=fallback,
        candidate_diagnostics=diagnostics,
        support=support,
    )
    manual = observable_risk_features(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=4,
        candidate_direction=candidate * support[None, None],
        fallback_direction=fallback * support[None, None],
        candidate_diagnostics=diagnostics,
    )
    assert torch.equal(canonical, manual)


def test_balanced_masks_report_unavoidable_full_view_reuse() -> None:
    plan = balanced_view_masks(
        count=16,
        view_count=9,
        active_view_counts=(6, 7, 8, 9),
        seed=99,
    )
    counts = torch.sum(plan.masks, dim=1).to(torch.int64)
    assert sorted(counts.tolist()) == [6] * 4 + [7] * 4 + [8] * 4 + [9] * 4
    assert plan.requested_count_by_active_views == {6: 4, 7: 4, 8: 4, 9: 4}
    assert plan.unique_pattern_count_by_active_views[9] == 1
    assert plan.pattern_reuse_count_by_active_views[9] == 3
    repeated = balanced_view_masks(
        count=16,
        view_count=9,
        active_view_counts=(6, 7, 8, 9),
        seed=99,
    )
    assert torch.equal(plan.masks, repeated.masks)


def test_stress_scores_follow_declared_observable_signs() -> None:
    values = np.zeros((1, len(RISK_FEATURE_NAMES)), dtype=np.float64)
    values[0, RISK_FEATURE_NAMES.index("direction_relative_correction")] = 2.0
    values[0, RISK_FEATURE_NAMES.index("candidate_log_gain_span")] = 2.0
    values[0, RISK_FEATURE_NAMES.index("gradient_spectral_centroid")] = -2.0
    values[
        0,
        RISK_FEATURE_NAMES.index("gradient_high_frequency_fraction"),
    ] = -2.0
    values[
        0,
        RISK_FEATURE_NAMES.index("white_component_correlation_abs"),
    ] = 3.0
    spectral, camera = observable_stress_scores(values)
    assert spectral.tolist() == [2.0]
    assert camera[0] == 0.75


def test_multiveto_direction_holds_the_first_stage_decision() -> None:
    gradient, residual, sigma, mask = _fixture()
    width = len(RISK_FEATURE_NAMES)
    direction = ObservableMultiVetoDirection(
        candidate=_Direction(1.4),
        fallback=_Direction(0.7),
        support=torch.ones((8, 8, 8)),
        stages=4,
        feature_mean=np.zeros(width),
        feature_scale=np.ones(width),
        coefficients=np.zeros(width),
        intercept=2.0,
        overprediction_quantile=1.0,
        distance_threshold=1e9,
        minimum_lower_gain_percent=0.0,
        spectral_stress_threshold=1e9,
        camera_stress_threshold=1e9,
        six_view_extra_margin_percent=0.0,
        minimum_active_views=3,
        maximum_active_views=3,
    )
    first, diagnostics = direction(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=4,
        stage_fraction=0.25,
    )
    assert diagnostics["multiveto_trust"].tolist() == [0.0, 1.0]
    assert torch.allclose(first[0], 0.7 * gradient[0])
    assert torch.allclose(first[1], 1.4 * gradient[1])
    later, later_diagnostics = direction(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=torch.ones_like(mask),
        rays_per_view=4,
        stage_fraction=0.5,
    )
    assert later_diagnostics["multiveto_trust"].tolist() == [0.0, 1.0]
    assert torch.allclose(later, first)
