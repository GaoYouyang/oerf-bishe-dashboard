"""Tests for geometry-conditioned fixed-SPD PSU PCGLS."""

from __future__ import annotations

import numpy as np
import torch

from .psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
)
from .psu_b0_conditioned_pcgls import (
    GeometryConditionedSPDPreconditioner,
    view_geometry_features_from_operator,
)
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)
from .psu_b0_spectral_preconditioner import normalized_field_loss


def _operator() -> PSUB0VoxelGradientOperator:
    rng = np.random.default_rng(3101)
    rays = 21
    points = rng.uniform(-0.85, 0.85, size=(rays, 5, 3))
    stencil = build_trilinear_stencil(
        points,
        grid_shape=(8, 8, 8),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    return PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=rng.normal(size=(rays, 3)),
        projection_v_xyz=rng.normal(size=(rays, 3)),
        line_length=np.linspace(0.8, 1.2, rays),
        system_constant=np.ones(rays),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )


def _inputs(
    operator: PSUB0VoxelGradientOperator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3102)
    truth = torch.randn(
        (2, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    observation = operator(truth).detach()
    sigma = torch.tensor(
        [[0.12, 0.18, 0.24], [0.20, 0.14, 0.17]],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    return truth, observation, sigma, mask


def test_view_geometry_features_are_finite_and_view_resolved() -> None:
    features = view_geometry_features_from_operator(
        _operator(),
        rays_per_view=7,
    )
    assert features.shape == (3, 14)
    assert torch.all(torch.isfinite(features))
    assert not torch.equal(features[0], features[1])


def test_zero_initialized_model_materializes_static_sobolev() -> None:
    operator = _operator()
    _, observation, sigma, mask = _inputs(operator)
    model = GeometryConditionedSPDPreconditioner(
        operator.grid_shape,
        view_geometry_features=view_geometry_features_from_operator(
            operator,
            rays_per_view=7,
        ),
        hidden=8,
        base_sobolev_strength=4.0,
        base_epsilon=0.05,
    ).to(torch.float64)
    materialized = model.materialize(
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=7,
    )
    gradient = torch.randn(
        (2, 1, *operator.grid_shape),
        dtype=torch.float64,
    )
    actual_first, diagnostics_first = materialized(
        gradient,
        stage_fraction=0.25,
    )
    actual_last, diagnostics_last = materialized(
        gradient,
        residual_uv=torch.randn_like(observation),
        stage_fraction=1.0,
    )
    expected, _ = GeneralizedSobolevDirection(
        operator.grid_shape,
        strength=4.0,
        epsilon=0.05,
    ).to(torch.float64)(gradient)
    assert torch.allclose(actual_first, expected, atol=1e-11, rtol=1e-11)
    assert torch.equal(actual_first, actual_last)
    assert torch.all(diagnostics_first["gain_minimum"] > 0.0)
    assert torch.allclose(
        diagnostics_first["gain_geometric_mean"],
        torch.ones(2, dtype=torch.float64),
        atol=1e-10,
    )
    assert diagnostics_last["fixed_within_solve"].tolist() == [1.0, 1.0]


def test_zero_initialized_conditioned_pcgls_matches_static_pcgls() -> None:
    operator = _operator()
    _, observation, sigma, mask = _inputs(operator)
    geometry = view_geometry_features_from_operator(
        operator,
        rays_per_view=7,
    )
    model = GeometryConditionedSPDPreconditioner(
        operator.grid_shape,
        view_geometry_features=geometry,
        hidden=8,
    ).to(torch.float64)
    materialized = model.materialize(
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=7,
    )
    operator.reset_call_counts()
    actual = preconditioned_cgls_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=7,
        stages=4,
        preconditioner=materialized,
    )
    assert operator.call_report() == {"forward_calls": 4, "adjoint_calls": 4}
    expected = preconditioned_cgls_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=7,
        stages=4,
        preconditioner=GeneralizedSobolevDirection(
            operator.grid_shape,
            strength=4.0,
            epsilon=0.05,
        ).to(torch.float64),
    )
    assert torch.allclose(actual.volume, expected.volume, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        actual.residual_uv,
        expected.residual_uv,
        atol=1e-10,
        rtol=1e-10,
    )


def test_conditioned_pcgls_backpropagates_to_controller() -> None:
    operator = _operator()
    truth, observation, sigma, mask = _inputs(operator)
    model = GeometryConditionedSPDPreconditioner(
        operator.grid_shape,
        view_geometry_features=view_geometry_features_from_operator(
            operator,
            rays_per_view=7,
        ),
        hidden=8,
        maximum_log_correction=0.4,
    ).to(torch.float64)
    materialized = model.materialize(
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=7,
    )
    result = preconditioned_cgls_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=7,
        stages=2,
        preconditioner=materialized,
    )
    loss = normalized_field_loss(result.volume, truth).mean()
    loss.backward()
    final = model.controller[-1]
    assert final.weight.grad is not None
    assert final.bias.grad is not None
    assert torch.all(torch.isfinite(final.weight.grad))
    assert torch.linalg.vector_norm(final.weight.grad) > 0.0


def test_nonzero_coefficients_remain_positive_and_normalized() -> None:
    operator = _operator()
    _, observation, sigma, mask = _inputs(operator)
    model = GeometryConditionedSPDPreconditioner(
        operator.grid_shape,
        view_geometry_features=view_geometry_features_from_operator(
            operator,
            rays_per_view=7,
        ),
        hidden=8,
        maximum_log_correction=0.3,
    ).to(torch.float64)
    with torch.no_grad():
        model.controller[-1].bias.copy_(
            torch.linspace(
                -0.5,
                0.5,
                len(model.frequency_basis),
                dtype=torch.float64,
            )
        )
    materialized = model.materialize(
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=7,
    )
    assert torch.all(materialized.gain > 0.0)
    geometric = torch.exp(
        torch.mean(torch.log(materialized.gain), dim=(1, 2, 3))
    )
    assert torch.allclose(
        geometric,
        torch.ones_like(geometric),
        atol=1e-10,
    )
    assert torch.amax(torch.abs(materialized.log_correction)) <= 0.6 + 1e-12
