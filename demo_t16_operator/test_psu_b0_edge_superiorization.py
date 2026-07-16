"""Tests for the edge-preserving PSU B0 superiorization primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
)
from .psu_b0_edge_superiorization import (
    edge_penalty_and_gradient,
    nonascending_edge_perturbation,
    superiorized_pcgls_reconstruction,
)
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)


def _operator() -> PSUB0VoxelGradientOperator:
    rng = np.random.default_rng(2901)
    rays = 24
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


@pytest.mark.parametrize("penalty", ["tv", "huber"])
def test_edge_penalty_gradient_matches_directional_difference(
    penalty: str,
) -> None:
    generator = torch.Generator().manual_seed(2902)
    volume = torch.randn(
        (2, 1, 8, 8, 8),
        generator=generator,
        dtype=torch.float64,
    )
    direction = torch.randn(
        volume.shape,
        generator=generator,
        dtype=torch.float64,
    )
    spacing = (0.25, 0.25, 0.25)
    _, gradient = edge_penalty_and_gradient(
        volume,
        spacing_xyz=spacing,
        penalty=penalty,
        smoothing=1e-2,
        huber_delta=0.4,
    )
    epsilon = 1e-6
    plus, _ = edge_penalty_and_gradient(
        volume + epsilon * direction,
        spacing_xyz=spacing,
        penalty=penalty,
        smoothing=1e-2,
        huber_delta=0.4,
    )
    minus, _ = edge_penalty_and_gradient(
        volume - epsilon * direction,
        spacing_xyz=spacing,
        penalty=penalty,
        smoothing=1e-2,
        huber_delta=0.4,
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytic = torch.sum(
        gradient * direction,
        dim=(1, 2, 3, 4),
    )
    assert torch.allclose(
        analytic,
        finite_difference,
        atol=2e-5,
        rtol=2e-6,
    )


@pytest.mark.parametrize("penalty", ["tv", "huber"])
def test_nonascending_perturbation_preserves_support(
    penalty: str,
) -> None:
    generator = torch.Generator().manual_seed(2903)
    volume = torch.randn(
        (3, 1, 8, 8, 8),
        generator=generator,
        dtype=torch.float64,
    )
    support = torch.ones((8, 8, 8), dtype=torch.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    support[:, :, [0, -1]] = 0.0
    result = nonascending_edge_perturbation(
        volume,
        support=support,
        spacing_xyz=(0.25, 0.25, 0.25),
        penalty=penalty,
        exponent=torch.zeros(3, dtype=torch.int64),
        inner_steps=3,
        initial_step=0.2,
        decay=0.7,
        smoothing=1e-2,
        huber_delta=0.4,
    )
    assert torch.all(
        result.penalty_after <= result.penalty_before + 1e-10
    )
    assert torch.all(result.perturbation_norm > 0.0)
    assert torch.all(result.exponent >= 3)
    assert torch.count_nonzero(
        result.volume * (1.0 - support)[None, None]
    ) == 0


def test_zero_perturbation_matches_fixed_pcgls_volume() -> None:
    generator = torch.Generator().manual_seed(2904)
    truth = torch.randn(
        (2, 1, 8, 8, 8),
        generator=generator,
        dtype=torch.float64,
    )
    kwargs = {
        "sigma_by_view": torch.full(
            (2, 3),
            0.2,
            dtype=torch.float64,
        ),
        "view_mask": torch.ones((2, 3), dtype=torch.float64),
        "rays_per_view": 8,
        "stages": 4,
    }
    reference_operator = _operator()
    observation = reference_operator(truth).detach()
    reference = preconditioned_cgls_reconstruction(
        reference_operator,
        observation,
        preconditioner=GeneralizedSobolevDirection(
            reference_operator.grid_shape,
            strength=1.0,
        ).to(torch.float64),
        **kwargs,
    )
    operator = _operator()
    operator.reset_call_counts()
    result = superiorized_pcgls_reconstruction(
        operator,
        observation,
        preconditioner=GeneralizedSobolevDirection(
            operator.grid_shape,
            strength=1.0,
        ).to(torch.float64),
        penalty="tv",
        perturbation_steps=2,
        perturbation_initial_step=0.0,
        perturbation_decay=0.7,
        **kwargs,
    )
    assert result.forward_calls == 7
    assert result.adjoint_calls == 4
    assert operator.call_report() == {
        "forward_calls": 7,
        "adjoint_calls": 4,
    }
    assert torch.allclose(
        result.volume,
        reference.volume,
        atol=2e-8,
        rtol=2e-8,
    )
    assert torch.allclose(
        result.residual_uv,
        reference.residual_uv,
        atol=2e-8,
        rtol=2e-8,
    )


def test_superiorization_records_nonascending_inner_steps() -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(2905)
    truth = torch.randn(
        (2, 1, 8, 8, 8),
        generator=generator,
        dtype=torch.float64,
    )
    observation = operator(truth).detach()
    operator.reset_call_counts()
    result = superiorized_pcgls_reconstruction(
        operator,
        observation,
        sigma_by_view=torch.full(
            (2, 3),
            0.2,
            dtype=torch.float64,
        ),
        view_mask=torch.ones((2, 3), dtype=torch.float64),
        rays_per_view=8,
        stages=4,
        preconditioner=GeneralizedSobolevDirection(
            operator.grid_shape,
            strength=1.0,
        ).to(torch.float64),
        penalty="huber",
        perturbation_steps=2,
        perturbation_initial_step=0.05,
        perturbation_decay=0.7,
        smoothing=1e-2,
        huber_delta=0.4,
    )
    assert operator.call_report() == {
        "forward_calls": 7,
        "adjoint_calls": 4,
    }
    for row in result.history[1:]:
        assert torch.all(
            row["edge_penalty_after_perturbation"]
            <= row["edge_penalty_before"] + 1e-10
        )
        assert torch.all(row["perturbation_norm"] > 0.0)
