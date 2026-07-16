"""Tests for call-accounted PSU B0 classical baselines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    ScheduledSobolevDirection,
    preconditioned_cgls_reconstruction,
    quadratic_tikhonov_reconstruction,
)
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)
from .psu_b0_spectral_preconditioner import (
    FixedSobolevDirection,
    IdentityDirection,
    exact_line_search_reconstruction,
    weighted_cgls_reconstruction,
)


def _operator() -> PSUB0VoxelGradientOperator:
    rng = np.random.default_rng(2701)
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
        line_length=np.ones(rays),
        system_constant=np.ones(rays),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )


@pytest.mark.parametrize("regularizer", ["identity", "h1"])
def test_quadratic_tikhonov_is_monotone_and_call_matched(
    regularizer: str,
) -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(2702)
    truth = torch.randn(
        (3, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    observation = operator(truth).detach()
    operator.reset_call_counts()
    result = quadratic_tikhonov_reconstruction(
        operator,
        observation,
        sigma_by_view=torch.full((3, 3), 0.2, dtype=torch.float64),
        view_mask=torch.ones((3, 3), dtype=torch.float64),
        rays_per_view=7,
        stages=4,
        regularization_lambda=0.03,
        regularizer=regularizer,
    )
    assert result.forward_calls == 4
    assert result.adjoint_calls == 4
    assert operator.call_report() == {"forward_calls": 4, "adjoint_calls": 4}
    for row in result.history:
        assert torch.all(
            row["relative_total_objective_after"]
            <= row["relative_total_objective_before"] + 1e-11
        )


def test_zero_lambda_matches_unpreconditioned_exact_line_search() -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(2703)
    truth = torch.randn(
        (2, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    observation = operator(truth).detach()
    kwargs = {
        "sigma_by_view": torch.full((2, 3), 0.15, dtype=torch.float64),
        "view_mask": torch.ones((2, 3), dtype=torch.float64),
        "rays_per_view": 7,
        "stages": 3,
    }
    expected = exact_line_search_reconstruction(
        operator,
        observation,
        direction=IdentityDirection(),
        **kwargs,
    )
    actual = quadratic_tikhonov_reconstruction(
        operator,
        observation,
        regularization_lambda=0.0,
        regularizer="h1",
        **kwargs,
    )
    assert torch.allclose(actual.volume, expected.volume, atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        actual.residual_uv,
        expected.residual_uv,
        atol=1e-11,
        rtol=1e-11,
    )


def test_generalized_isotropic_sobolev_matches_frozen_reference() -> None:
    gradient = torch.randn((2, 1, 8, 8, 8), dtype=torch.float64)
    expected, _ = FixedSobolevDirection(
        (8, 8, 8),
        strength=4.0,
    ).to(torch.float64)(gradient)
    actual, diagnostics = GeneralizedSobolevDirection(
        (8, 8, 8),
        strength=4.0,
        epsilon=0.05,
        axis_weights_xyz=(1.0, 1.0, 1.0),
    ).to(torch.float64)(gradient)
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)
    assert diagnostics["sobolev_strength"].tolist() == [4.0, 4.0]


def test_scheduled_sobolev_uses_declared_stage() -> None:
    gradient = torch.randn((1, 1, 8, 8, 8), dtype=torch.float64)
    schedule = ScheduledSobolevDirection(
        (8, 8, 8),
        strengths=(2.0, 4.0, 6.0, 8.0),
    ).to(torch.float64)
    first, first_diagnostics = schedule(gradient, stage_fraction=0.25)
    last, last_diagnostics = schedule(gradient, stage_fraction=1.0)
    expected_first, _ = GeneralizedSobolevDirection(
        (8, 8, 8),
        strength=2.0,
    ).to(torch.float64)(gradient)
    expected_last, _ = GeneralizedSobolevDirection(
        (8, 8, 8),
        strength=8.0,
    ).to(torch.float64)(gradient)
    assert torch.allclose(first, expected_first)
    assert torch.allclose(last, expected_last)
    assert first_diagnostics["sobolev_schedule_index"].item() == 0
    assert last_diagnostics["sobolev_schedule_index"].item() == 3


def test_identity_preconditioned_cgls_matches_cgls() -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(2704)
    truth = torch.randn(
        (2, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    observation = operator(truth).detach()
    kwargs = {
        "sigma_by_view": torch.full((2, 3), 0.15, dtype=torch.float64),
        "view_mask": torch.ones((2, 3), dtype=torch.float64),
        "rays_per_view": 7,
        "stages": 3,
    }
    expected = weighted_cgls_reconstruction(operator, observation, **kwargs)
    operator.reset_call_counts()
    actual = preconditioned_cgls_reconstruction(
        operator,
        observation,
        preconditioner=IdentityDirection(),
        **kwargs,
    )
    assert actual.forward_calls == 3
    assert actual.adjoint_calls == 3
    assert operator.call_report() == {"forward_calls": 3, "adjoint_calls": 3}
    assert torch.allclose(actual.volume, expected.volume, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        actual.residual_uv,
        expected.residual_uv,
        atol=1e-10,
        rtol=1e-10,
    )


def test_initial_normal_factory_shares_the_first_adjoint() -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(2705)
    truth = torch.randn(
        (2, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    observation = operator(truth).detach()
    kwargs = {
        "sigma_by_view": torch.full((2, 3), 0.15, dtype=torch.float64),
        "view_mask": torch.ones((2, 3), dtype=torch.float64),
        "rays_per_view": 7,
        "stages": 3,
    }
    expected = preconditioned_cgls_reconstruction(
        operator,
        observation,
        preconditioner=IdentityDirection(),
        **kwargs,
    )
    captured = []

    def factory(initial_normal: torch.Tensor, **_: object) -> IdentityDirection:
        captured.append(initial_normal.detach().clone())
        return IdentityDirection()

    operator.reset_call_counts()
    actual = preconditioned_cgls_reconstruction(
        operator,
        observation,
        preconditioner_factory=factory,
        **kwargs,
    )
    assert len(captured) == 1
    assert operator.call_report() == {"forward_calls": 3, "adjoint_calls": 3}
    assert torch.allclose(actual.volume, expected.volume, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        actual.residual_uv,
        expected.residual_uv,
        atol=1e-10,
        rtol=1e-10,
    )


def test_invalid_regularizer_and_lambda_are_rejected() -> None:
    operator = _operator()
    observation = torch.zeros((1, 21, 2), dtype=torch.float64)
    kwargs = {
        "sigma_by_view": torch.ones((1, 3), dtype=torch.float64),
        "view_mask": torch.ones((1, 3), dtype=torch.float64),
        "rays_per_view": 7,
        "stages": 1,
    }
    with pytest.raises(ValueError, match="nonnegative"):
        quadratic_tikhonov_reconstruction(
            operator,
            observation,
            regularization_lambda=-1.0,
            **kwargs,
        )
    with pytest.raises(ValueError, match="regularizer"):
        quadratic_tikhonov_reconstruction(
            operator,
            observation,
            regularization_lambda=1.0,
            regularizer="tv",  # type: ignore[arg-type]
            **kwargs,
        )
