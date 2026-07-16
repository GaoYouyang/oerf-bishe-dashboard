from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .psu_b0_residual_risk import (
    RISK_FEATURE_NAMES,
    CalibratedResidualRiskDirection,
    fit_ridge_risk_model,
    observable_risk_features,
    one_sided_conformal_quantile,
)


class _Direction(nn.Module):
    def __init__(self, scale: float, gain_minimum: float, gain_maximum: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.gain_minimum = float(gain_minimum)
        self.gain_maximum = float(gain_maximum)

    def forward(
        self,
        gradient: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = len(gradient)
        coefficients = torch.full(
            (batch, 7),
            0.1 * self.scale,
            dtype=gradient.dtype,
            device=gradient.device,
        )
        return self.scale * gradient, {
            "gain_minimum": torch.full(
                (batch,),
                self.gain_minimum,
                dtype=gradient.dtype,
                device=gradient.device,
            ),
            "gain_maximum": torch.full(
                (batch,),
                self.gain_maximum,
                dtype=gradient.dtype,
                device=gradient.device,
            ),
            "gain_geometric_mean": torch.ones(
                batch,
                dtype=gradient.dtype,
                device=gradient.device,
            ),
            "controller_coefficients": coefficients,
        }


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    gradient = torch.randn((2, 1, 8, 8, 8), generator=generator)
    residual = torch.randn((2, 12, 2), generator=generator)
    sigma = torch.tensor([[0.2, 0.3, 0.4], [0.4, 0.2, 0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    return gradient, residual, sigma, mask


def test_observable_features_are_truth_free_finite_and_schema_aligned() -> None:
    gradient, residual, sigma, mask = _fixture()
    candidate = 1.3 * gradient
    fallback = 0.8 * gradient
    diagnostics = {
        "gain_minimum": torch.tensor([0.4, 0.5]),
        "gain_maximum": torch.tensor([2.0, 2.5]),
        "controller_coefficients": torch.ones((2, 7)),
    }
    features = observable_risk_features(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=4,
        candidate_direction=candidate,
        fallback_direction=fallback,
        candidate_diagnostics=diagnostics,
    )
    assert features.shape == (2, len(RISK_FEATURE_NAMES))
    assert torch.all(torch.isfinite(features))
    assert torch.allclose(features[:, 0], torch.tensor([2 / 3, 1.0]))


def test_ridge_selection_and_conformal_quantile_are_deterministic() -> None:
    rng = np.random.default_rng(52)
    train = rng.normal(size=(80, len(RISK_FEATURE_NAMES)))
    validation = rng.normal(size=(24, len(RISK_FEATURE_NAMES)))
    weights = rng.normal(size=len(RISK_FEATURE_NAMES))
    train_y = train @ weights + 0.05 * rng.normal(size=80)
    validation_y = validation @ weights + 0.05 * rng.normal(size=24)
    fit = fit_ridge_risk_model(
        train,
        train_y,
        validation,
        validation_y,
        ridge_grid=(1e-4, 1e-2, 1.0, 100.0),
    )
    assert fit.validation_rmse < 0.2
    predicted = fit.predict(validation)
    distance = fit.distance(validation)
    assert distance.shape == (len(validation),)
    assert np.all(np.isfinite(distance))
    quantile = one_sided_conformal_quantile(predicted, validation_y, alpha=0.1)
    errors = np.sort(predicted - validation_y)
    rank = min(int(np.ceil((len(errors) + 1) * 0.9)), len(errors))
    assert quantile == errors[rank - 1]


def test_calibrated_direction_holds_one_decision_for_all_stages() -> None:
    gradient, residual, sigma, mask = _fixture()
    width = len(RISK_FEATURE_NAMES)
    gate = CalibratedResidualRiskDirection(
        candidate=_Direction(1.4, 0.5, 2.0),
        fallback=_Direction(0.7, 1.0, 1.0),
        stages=4,
        feature_mean=np.zeros(width),
        feature_scale=np.ones(width),
        coefficients=np.zeros(width),
        intercept=2.0,
        overprediction_quantile=1.0,
        distance_threshold=1e6,
        minimum_lower_gain_percent=0.0,
        minimum_active_views=3,
        maximum_active_views=3,
    )
    first, diagnostics = gate(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=4,
        stage_fraction=0.25,
    )
    assert diagnostics["residual_risk_trust"].tolist() == [0.0, 1.0]
    assert torch.allclose(first[0], 0.7 * gradient[0])
    assert torch.allclose(first[1], 1.4 * gradient[1])
    changed_mask = torch.ones_like(mask)
    later, later_diagnostics = gate(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=changed_mask,
        rays_per_view=4,
        stage_fraction=0.5,
    )
    assert later_diagnostics["residual_risk_trust"].tolist() == [0.0, 1.0]
    assert torch.allclose(later[0], 0.7 * gradient[0])
    assert torch.allclose(later[1], 1.4 * gradient[1])
