"""Tests for call-matched positive spectral PSU reconstruction."""

from __future__ import annotations

import numpy as np
import torch

from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)
from .psu_b0_spectral_preconditioner import (
    ActiveViewSupportEnvelopeDirection,
    FixedSobolevDirection,
    IdentityDirection,
    PositiveSpectralDirection,
    exact_line_search_reconstruction,
    normalized_field_loss,
    weighted_cgls_reconstruction,
)


def _operator() -> PSUB0VoxelGradientOperator:
    rng = np.random.default_rng(4)
    points = rng.uniform(-0.85, 0.85, size=(18, 4, 3))
    stencil = build_trilinear_stencil(
        points,
        grid_shape=(8, 8, 8),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    projection_u = rng.normal(size=(18, 3))
    projection_v = rng.normal(size=(18, 3))
    return PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        line_length=np.ones(18),
        system_constant=np.ones(18),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )


def test_positive_spectral_direction_starts_as_identity() -> None:
    model = PositiveSpectralDirection(
        (8, 8, 8),
        view_count=3,
        hidden=8,
    ).to(torch.float64)
    gradient = torch.randn(2, 1, 8, 8, 8, dtype=torch.float64)
    residual = torch.randn(2, 18, 2, dtype=torch.float64)
    sigma = torch.full((2, 3), 0.2, dtype=torch.float64)
    mask = torch.ones(2, 3, dtype=torch.float64)
    direction, diagnostics = model(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=6,
        stage_fraction=0.25,
    )
    assert torch.allclose(direction, gradient, atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        diagnostics["gain_geometric_mean"],
        torch.ones(2, dtype=torch.float64),
        atol=1e-11,
    )


def test_positive_spectral_direction_can_start_from_sobolev_reference() -> None:
    model = PositiveSpectralDirection(
        (8, 8, 8),
        view_count=3,
        hidden=8,
        base_sobolev_strength=2.0,
    ).to(torch.float64)
    gradient = torch.randn(2, 1, 8, 8, 8, dtype=torch.float64)
    residual = torch.randn(2, 18, 2, dtype=torch.float64)
    sigma = torch.full((2, 3), 0.2, dtype=torch.float64)
    mask = torch.ones(2, 3, dtype=torch.float64)
    direction, diagnostics = model(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=6,
        stage_fraction=0.25,
    )
    assert not torch.allclose(direction, gradient)
    assert torch.all(diagnostics["gain_minimum"] > 0.0)
    assert torch.allclose(
        diagnostics["gain_geometric_mean"],
        torch.ones(2, dtype=torch.float64),
        atol=1e-10,
    )


def test_active_view_envelope_is_exact_candidate_or_fallback() -> None:
    candidate = PositiveSpectralDirection(
        (8, 8, 8),
        view_count=3,
        hidden=8,
        base_sobolev_strength=1.0,
    ).to(torch.float64)
    with torch.no_grad():
        candidate.controller[-1].bias.fill_(0.4)
    fallback = FixedSobolevDirection(
        (8, 8, 8),
        strength=2.0,
    ).to(torch.float64)
    envelope = ActiveViewSupportEnvelopeDirection(
        candidate=candidate,
        fallback=fallback,
        minimum_active_views=3,
        maximum_active_views=3,
    )
    gradient = torch.randn(2, 1, 8, 8, 8, dtype=torch.float64)
    residual = torch.randn(2, 18, 2, dtype=torch.float64)
    sigma = torch.full((2, 3), 0.2, dtype=torch.float64)
    mask = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    actual, diagnostics = envelope(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=6,
        stage_fraction=0.25,
    )
    expected_candidate, _ = candidate(
        gradient,
        residual_uv=residual,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=6,
        stage_fraction=0.25,
    )
    expected_fallback, _ = fallback(gradient)
    assert torch.equal(actual[0], expected_candidate[0])
    assert torch.equal(actual[1], expected_fallback[1])
    assert diagnostics["support_envelope_trust"].tolist() == [1.0, 0.0]


def test_exact_line_search_is_monotone_and_call_matched() -> None:
    operator = _operator()
    truth = torch.randn(2, 1, 8, 8, 8, dtype=torch.float64)
    observation = operator(truth).detach()
    operator.reset_call_counts()
    result = exact_line_search_reconstruction(
        operator,
        observation,
        sigma_by_view=torch.full((2, 3), 0.15, dtype=torch.float64),
        view_mask=torch.ones(2, 3, dtype=torch.float64),
        rays_per_view=6,
        stages=3,
        direction=IdentityDirection(),
    )
    assert result.forward_calls == 3
    assert result.adjoint_calls == 3
    assert operator.call_report() == {"forward_calls": 3, "adjoint_calls": 3}
    for row in result.history:
        assert torch.all(
            row["relative_objective_after"]
            <= row["relative_objective_before"] + 1e-12
        )


def test_weighted_cgls_and_normalized_loss_are_finite() -> None:
    operator = _operator()
    truth = torch.randn(2, 1, 8, 8, 8, dtype=torch.float64)
    observation = operator(truth).detach()
    operator.reset_call_counts()
    result = weighted_cgls_reconstruction(
        operator,
        observation,
        sigma_by_view=torch.full((2, 3), 0.15, dtype=torch.float64),
        view_mask=torch.ones(2, 3, dtype=torch.float64),
        rays_per_view=6,
        stages=2,
    )
    loss = normalized_field_loss(result.volume, truth)
    assert result.forward_calls == 2
    assert result.adjoint_calls == 3
    assert operator.call_report() == {"forward_calls": 2, "adjoint_calls": 3}
    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))
