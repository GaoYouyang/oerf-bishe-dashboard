"""Tests for covariance-weighted one-pair PDHG reconstruction."""

from __future__ import annotations

from dataclasses import replace
import itertools
from typing import get_type_hints

import numpy as np
import pytest
import torch

from . import psu_b0_primal_dual as primal_dual_module
from .psu_b0_primal_dual import (
    BlockOperatorNormEstimate,
    ForwardNeumannRegularizationOperator,
    PrimalDualReconstruction,
    estimate_block_operator_norms,
    gradient_operator_norm_squared_bound,
    isotropic_edge_penalty,
    primal_dual_reconstruction,
    proximal_edge_conjugate,
    regularization_gradient,
    regularization_gradient_adjoint,
    regularization_site_mask,
)
from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)


_PDHG_SMOKE_CANDIDATES = tuple(
    (
        f"pdhg_{penalty}_a{alpha_slug}_k{formal_iterations}",
        penalty,
        alpha,
        formal_iterations,
    )
    for penalty, (alpha_slug, alpha), formal_iterations in itertools.product(
        ("tv", "huber"),
        (
            ("1of256", 1.0 / 256.0),
            ("1of64", 1.0 / 64.0),
            ("1of16", 1.0 / 16.0),
            ("1of4", 1.0 / 4.0),
        ),
        (4, 8, 16, 32),
    )
)


def _prewhitened_views(
    operator: torch.nn.Module,
    *,
    batch_size: int,
    rays_per_view: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    view_count, remainder = divmod(int(operator.ray_count), rays_per_view)
    if remainder:
        raise ValueError("rays_per_view must divide the operator ray count")
    values = torch.ones(
        (batch_size, view_count),
        dtype=operator.support.dtype,
        device=operator.support.device,
    )
    return values, values.clone()


def _bound_norm_estimate(
    operator: torch.nn.Module,
    *,
    batch_size: int,
    power_iterations: int = 4,
    safety_factor: float = 1.25,
    measurement_scale: float = 1.0,
    seed: int = 3100,
) -> BlockOperatorNormEstimate:
    sigma, mask = _prewhitened_views(operator, batch_size=batch_size)
    return estimate_block_operator_norms(
        operator,
        batch_size=batch_size,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        power_iterations=power_iterations,
        safety_factor=safety_factor,
        measurement_scale=measurement_scale,
        seed=seed,
    )


def _balanced_steps(
    norm_estimate: BlockOperatorNormEstimate,
    *,
    contract: float = 0.8,
) -> tuple[float, float, float]:
    root_data = norm_estimate.data_norm_squared_upper**0.5
    root_edge = norm_estimate.gradient_norm_squared_upper**0.5
    return (
        contract / (root_data + root_edge),
        1.0 / root_data,
        1.0 / root_edge,
    )


def _operator(
    dtype: torch.dtype = torch.float64,
) -> PSUB0VoxelGradientOperator:
    rng = np.random.default_rng(3001)
    rays = 24
    stencil = build_trilinear_stencil(
        rng.uniform(-0.85, 0.85, size=(rays, 5, 3)),
        grid_shape=(8, 8, 8),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=dtype,
    )
    operator = PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=rng.normal(size=(rays, 3)),
        projection_v_xyz=rng.normal(size=(rays, 3)),
        line_length=np.ones(rays),
        system_constant=np.ones(rays),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=dtype,
    )
    support = torch.ones_like(operator.support)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    support[:, :, [0, -1]] = 0.0
    operator.support.copy_(support)
    return operator


class _MaskedIdentityOperator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grid_shape = (4, 4, 4)
        self.spacing_xyz = (1.0, 1.0, 1.0)
        self.ray_count = 32
        support = torch.ones(self.grid_shape, dtype=torch.float64)
        support[[0, -1], :, :] = 0.0
        support[:, [0, -1], :] = 0.0
        support[:, :, [0, -1]] = 0.0
        self.register_buffer("support", support)
        self.forward_calls = 0
        self.adjoint_calls = 0

    def reset_call_counts(self) -> None:
        self.forward_calls = 0
        self.adjoint_calls = 0

    def call_report(self) -> dict[str, int]:
        return {
            "forward_calls": self.forward_calls,
            "adjoint_calls": self.adjoint_calls,
        }

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return (volume[:, 0] * self.support).reshape(len(volume), 32, 2)

    def adjoint(self, residual_uv: torch.Tensor) -> torch.Tensor:
        self.adjoint_calls += 1
        return residual_uv.reshape(len(residual_uv), 1, 4, 4, 4) * self.support


class _SwitchableDoubleCallOperator(_MaskedIdentityOperator):
    def __init__(self) -> None:
        super().__init__()
        self.double_call: str | None = None

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        output = super().forward(volume)
        if self.double_call == "forward":
            super().forward(volume)
        return output

    def adjoint(self, residual_uv: torch.Tensor) -> torch.Tensor:
        output = super().adjoint(residual_uv)
        if self.double_call == "adjoint":
            super().adjoint(residual_uv)
        return output


class _SwitchableDoubleCallRegularizationOperator(
    ForwardNeumannRegularizationOperator
):
    def __init__(self, spacing_xyz: tuple[float, float, float]) -> None:
        super().__init__(spacing_xyz)
        self.double_call: str | None = None

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        output = super().__call__(volume)
        if self.double_call == "gradient":
            super().__call__(volume)
        return output

    def adjoint(self, gradient: torch.Tensor) -> torch.Tensor:
        output = super().adjoint(gradient)
        if self.double_call == "gradient_adjoint":
            super().adjoint(gradient)
        return output


class _TracingMaskedIdentityOperator(_MaskedIdentityOperator):
    def __init__(self) -> None:
        super().__init__()
        self.forward_inputs: list[torch.Tensor] = []
        self.forward_outputs: list[torch.Tensor] = []

    def clear_trace(self) -> None:
        self.forward_inputs.clear()
        self.forward_outputs.clear()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        output = super().forward(volume)
        self.forward_inputs.append(volume.detach().clone())
        self.forward_outputs.append(output.detach().clone())
        return output


@pytest.mark.parametrize("penalty", ["tv", "huber"])
def test_edge_conjugate_prox_respects_dual_ball(penalty: str) -> None:
    generator = torch.Generator().manual_seed(3002)
    dual = 5.0 * torch.randn(
        (2, 3, 5, 5, 5),
        generator=generator,
        dtype=torch.float64,
    )
    output = proximal_edge_conjugate(
        dual,
        regularization_weight=0.7,
        dual_step=0.3,
        penalty=penalty,
        huber_delta=0.4,
    )
    norms = torch.linalg.vector_norm(output, dim=1)
    assert float(torch.max(norms)) <= 0.7 + 1e-12
    if penalty == "huber":
        unsaturated = 0.02 * dual
        output = proximal_edge_conjugate(
            unsaturated,
            regularization_weight=0.7,
            dual_step=0.3,
            penalty="huber",
            huber_delta=0.4,
        )
        tv_output = proximal_edge_conjugate(
            unsaturated,
            regularization_weight=0.7,
            dual_step=0.3,
            penalty="tv",
            huber_delta=0.4,
        )
        assert torch.linalg.vector_norm(output) < torch.linalg.vector_norm(
            tv_output
        )


@pytest.mark.parametrize("penalty", ["tv", "huber"])
def test_edge_conjugate_prox_matches_independent_two_regime_oracle(
    penalty: str,
) -> None:
    weight = 0.7
    dual_step = 0.3
    huber_delta = 0.4
    direction = torch.tensor((3.0 / 5.0, 4.0 / 5.0, 0.0), dtype=torch.float64)
    input_magnitudes = torch.tensor((0.2, 5.0), dtype=torch.float64)
    dual = (
        direction[:, None] * input_magnitudes[None]
    ).reshape(1, 3, 1, 1, 2)

    if penalty == "tv":
        expected_magnitudes = torch.tensor((0.2, weight), dtype=torch.float64)
    else:
        unconstrained = 0.2 / (1.0 + dual_step * huber_delta / weight)
        expected_magnitudes = torch.tensor(
            (unconstrained, weight),
            dtype=torch.float64,
        )
    expected = (
        direction[:, None] * expected_magnitudes[None]
    ).reshape_as(dual)

    actual = proximal_edge_conjugate(
        dual,
        regularization_weight=weight,
        dual_step=dual_step,
        penalty=penalty,
        huber_delta=huber_delta,
    )

    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    actual_magnitudes = torch.linalg.vector_norm(actual, dim=1).flatten()
    assert actual_magnitudes[0] < weight
    assert actual_magnitudes[1] == pytest.approx(weight)


def test_zero_weight_forces_zero_edge_dual() -> None:
    dual = torch.ones((1, 3, 4, 4, 4), dtype=torch.float64)
    output = proximal_edge_conjugate(
        dual,
        regularization_weight=0.0,
        dual_step=0.4,
        penalty="tv",
    )
    assert torch.count_nonzero(output) == 0


def test_exact_tv_and_huber_penalties_have_expected_order() -> None:
    volume = torch.zeros((1, 1, 5, 5, 5), dtype=torch.float64)
    volume[:, :, 2:, :, :] = 1.0
    tv = isotropic_edge_penalty(
        volume,
        spacing_xyz=(0.5, 0.5, 0.5),
        penalty="tv",
        huber_delta=0.4,
    )
    huber = isotropic_edge_penalty(
        volume,
        spacing_xyz=(0.5, 0.5, 0.5),
        penalty="huber",
        huber_delta=0.4,
    )
    assert torch.all(tv > 0.0)
    assert torch.all(huber > 0.0)
    assert torch.all(huber < tv)


def test_declared_gradient_norm_bound_dominates_random_rayleigh() -> None:
    generator = torch.Generator().manual_seed(3007)
    spacing = (0.3, 0.4, 0.5)
    volume = torch.randn(
        (16, 1, 5, 6, 7),
        generator=generator,
        dtype=torch.float64,
    )
    gradient = regularization_gradient(
        volume[:, 0],
        spacing_xyz=spacing,
    )
    ratios = torch.sum(gradient.square(), dim=(1, 2, 3, 4)) / torch.sum(
        volume.square(), dim=(1, 2, 3, 4)
    )
    assert float(torch.max(ratios)) < gradient_operator_norm_squared_bound(
        spacing
    )


def test_forward_neumann_gradient_has_exact_adjoint() -> None:
    generator = torch.Generator().manual_seed(3008)
    volume = torch.randn(
        (3, 5, 6, 7),
        generator=generator,
        dtype=torch.float64,
    )
    dual = torch.randn(
        (3, 3, 5, 6, 7),
        generator=generator,
        dtype=torch.float64,
    )
    spacing = (0.3, 0.4, 0.5)
    lhs = torch.sum(
        regularization_gradient(volume, spacing_xyz=spacing) * dual
    )
    rhs = torch.sum(
        volume
        * regularization_gradient_adjoint(dual, spacing_xyz=spacing)
    )
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_forward_neumann_gradient_detects_checkerboard() -> None:
    indices = torch.arange(8, dtype=torch.float64)
    checkerboard = (
        (-1.0) ** (
            indices[:, None, None]
            + indices[None, :, None]
            + indices[None, None, :]
        )
    )[None]
    gradient = regularization_gradient(
        checkerboard,
        spacing_xyz=(1.0, 1.0, 1.0),
    )
    assert float(torch.mean(gradient.square())) > 2.0


def test_regularization_site_mask_has_explicit_single_voxel_example() -> None:
    support = torch.zeros((3, 3, 3), dtype=torch.bool)
    support[1, 1, 1] = True
    expected = torch.tensor(
        (
            ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
            ((0, 1, 0), (1, 1, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        ),
        dtype=torch.bool,
    )

    sites = regularization_site_mask(support)
    gradient = regularization_gradient(
        support.to(torch.float64)[None],
        spacing_xyz=(1.0, 1.0, 1.0),
    )

    assert torch.equal(sites, expected)
    assert torch.equal(torch.any(gradient != 0.0, dim=1)[0], expected)


def test_norm_estimator_records_only_declared_physical_pairs() -> None:
    operator = _operator()
    sigma, mask = _prewhitened_views(operator, batch_size=2)
    operator.reset_call_counts()
    estimate = estimate_block_operator_norms(
        operator,
        batch_size=2,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        power_iterations=6,
        safety_factor=1.3,
        seed=3003,
    )
    assert estimate.forward_calls == 6
    assert estimate.adjoint_calls == 6
    assert operator.call_report() == {
        "forward_calls": 6,
        "adjoint_calls": 6,
    }
    assert estimate.data_norm_squared_by_sample.shape == (2,)
    assert estimate.data_norm_squared_upper > 0.0
    assert estimate.gradient_norm_squared_upper > 0.0
    assert estimate.operator_identity == id(operator)
    assert estimate.batch_size == 2
    assert estimate.measurement_scale == 1.0
    assert estimate.grid_shape == operator.grid_shape
    assert estimate.spacing_xyz == operator.spacing_xyz
    assert estimate.operator_state_token


def test_scaled_identity_norm_estimate_matches_dense_oracle() -> None:
    operator = _MaskedIdentityOperator()
    estimate = estimate_block_operator_norms(
        operator,
        batch_size=2,
        sigma_by_view=torch.ones((2, 4), dtype=torch.float64),
        view_mask=torch.ones((2, 4), dtype=torch.float64),
        rays_per_view=8,
        power_iterations=4,
        safety_factor=1.2,
        measurement_scale=0.25,
        seed=3009,
    )
    expected = torch.full((2,), 1.2 * 0.25**2, dtype=torch.float64)
    assert torch.allclose(
        estimate.data_norm_squared_by_sample,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_matrix_free_data_norm_probe_matches_tiny_dense_svd() -> None:
    operator = _operator()
    voxel_count = int(np.prod(operator.grid_shape))
    basis = torch.eye(voxel_count, dtype=torch.float64).reshape(
        voxel_count,
        1,
        *operator.grid_shape,
    )
    dense = operator._forward(basis).reshape(voxel_count, -1).T
    dense_weighted = 0.1 * dense
    exact_squared = float(torch.linalg.svdvals(dense_weighted)[0].square())
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    operator.reset_call_counts()
    estimate = estimate_block_operator_norms(
        operator,
        batch_size=1,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        power_iterations=30,
        safety_factor=1.2,
        measurement_scale=0.1,
        seed=3013,
    )
    raw_power = estimate.data_norm_squared_upper / 1.2
    assert raw_power <= exact_squared * (1.0 + 1e-10)
    assert raw_power >= exact_squared * 0.99
    assert estimate.data_norm_squared_upper > exact_squared


def test_zero_regularization_converges_on_scaled_identity() -> None:
    operator = _MaskedIdentityOperator()
    generator = torch.Generator().manual_seed(3010)
    truth = torch.randn(
        (2, 1, 4, 4, 4),
        generator=generator,
        dtype=torch.float64,
    ) * operator.support
    observation = operator(truth).detach()
    norm = _bound_norm_estimate(
        operator,
        batch_size=2,
        safety_factor=1.1,
        measurement_scale=0.25,
        seed=3010,
    )
    operator.reset_call_counts()
    data_bound = norm.data_norm_squared_upper
    sigma_data = 1.0 / data_bound**0.5
    tau = 0.9 / data_bound**0.5
    result = primal_dual_reconstruction(
        operator,
        observation / 2.0,
        sigma_by_view=torch.ones((2, 4), dtype=torch.float64),
        view_mask=torch.ones((2, 4), dtype=torch.float64),
        rays_per_view=8,
        iterations=120,
        regularization_weight=0.0,
        penalty="tv",
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=0.1,
        norm_estimate=norm,
        measurement_scale=0.25,
    )
    expected = 2.0 * truth
    relative = torch.linalg.vector_norm(
        result.volume - expected
    ) / torch.linalg.vector_norm(expected)
    assert float(relative) < 1e-6
    assert operator.call_report() == {
        "forward_calls": 120,
        "adjoint_calls": 120,
    }


def test_one_iteration_matches_manual_zero_regularization_update() -> None:
    operator = _operator()
    generator = torch.Generator().manual_seed(3004)
    truth = torch.randn(
        (2, 1, 8, 8, 8),
        generator=generator,
        dtype=torch.float64,
    )
    observation = operator(truth).detach()
    sigma, mask = _prewhitened_views(operator, batch_size=2)
    norm = _bound_norm_estimate(
        operator,
        batch_size=2,
        power_iterations=8,
        safety_factor=1.5,
        seed=3004,
    )
    tau = 0.01
    dual_step = 0.2
    expected_dual = -dual_step * observation / (1.0 + dual_step)
    expected = (
        -tau * operator._adjoint(expected_dual)[:, None]
        * operator.support
    )
    operator.reset_call_counts()
    result = primal_dual_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        iterations=1,
        regularization_weight=0.0,
        penalty="tv",
        primal_step=tau,
        data_dual_step=dual_step,
        edge_dual_step=0.2,
        norm_estimate=norm,
    )
    assert torch.allclose(result.volume, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result.data_dual, expected_dual, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(result.edge_dual) == 0
    assert operator.call_report() == {
        "forward_calls": 1,
        "adjoint_calls": 1,
    }


def test_pdhg_preserves_support_and_exact_call_budget() -> None:
    operator = _operator()
    support = torch.ones_like(operator.support)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    support[:, :, [0, -1]] = 0.0
    operator.support.copy_(support)
    generator = torch.Generator().manual_seed(3005)
    truth = torch.randn(
        (2, 1, 8, 8, 8),
        generator=generator,
        dtype=torch.float64,
    ) * support
    observation = operator(truth).detach()
    sigma, mask = _prewhitened_views(operator, batch_size=2)
    operator.reset_call_counts()
    norm = estimate_block_operator_norms(
        operator,
        batch_size=2,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        power_iterations=8,
        safety_factor=1.5,
        seed=3006,
    )
    root_data = norm.data_norm_squared_upper**0.5
    root_edge = norm.gradient_norm_squared_upper**0.5
    sigma_data = 1.0 / root_data
    sigma_edge = 1.0 / root_edge
    tau = 0.8 / (root_data + root_edge)
    operator.reset_call_counts()
    result = primal_dual_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        iterations=7,
        regularization_weight=0.02,
        penalty="huber",
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm,
        huber_delta=0.4,
        checkpoint_iterations=[3, 7],
    )
    assert result.forward_calls == 7
    assert result.adjoint_calls == 7
    assert operator.call_report() == {
        "forward_calls": 7,
        "adjoint_calls": 7,
    }
    assert result.step_contract_value == pytest.approx(0.8)
    assert torch.count_nonzero(
        result.volume * (1.0 - support)[None, None]
    ) == 0
    assert len(result.history) == 7
    assert set(result.checkpoint_volumes) == {3, 7}
    operator.reset_call_counts()
    prefix = primal_dual_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        iterations=3,
        regularization_weight=0.02,
        penalty="huber",
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm,
        huber_delta=0.4,
    )
    assert torch.allclose(
        result.checkpoint_volumes[3],
        prefix.volume,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.all(torch.isfinite(result.volume))


def test_invalid_step_contract_is_rejected_before_operator_calls() -> None:
    operator = _operator()
    observation = torch.zeros((1, 24, 2), dtype=torch.float64)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3014)
    operator.reset_call_counts()
    with pytest.raises(ValueError, match="step contract"):
        primal_dual_reconstruction(
            operator,
            observation,
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=8,
            iterations=2,
            regularization_weight=0.1,
            penalty="tv",
            primal_step=1.0,
            data_dual_step=1.0,
            edge_dual_step=1.0,
            norm_estimate=norm,
        )
    assert operator.call_report() == {
        "forward_calls": 0,
        "adjoint_calls": 0,
    }


def test_non_dirichlet_support_is_rejected_before_operator_calls() -> None:
    operator = _operator()
    observation = torch.zeros((1, 24, 2), dtype=torch.float64)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3015)
    operator.support.fill_(1.0)
    operator.reset_call_counts()
    with pytest.raises(ValueError, match="fix the BOS gauge"):
        primal_dual_reconstruction(
            operator,
            observation,
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=8,
            iterations=2,
            regularization_weight=0.1,
            penalty="tv",
            primal_step=0.1,
            data_dual_step=0.1,
            edge_dual_step=0.1,
            norm_estimate=norm,
        )
    assert operator.call_report() == {
        "forward_calls": 0,
        "adjoint_calls": 0,
    }


@pytest.mark.parametrize("entrypoint", ["estimate", "solve"])
@pytest.mark.parametrize(
    ("invalid_field", "message"),
    [
        ("sigma", "sigma_by_view must be one"),
        ("mask", "all-view prewhitened operator"),
    ],
)
def test_non_prewhitened_inputs_are_rejected_without_physical_calls(
    entrypoint: str,
    invalid_field: str,
    message: str,
) -> None:
    operator = _MaskedIdentityOperator()
    observation = torch.zeros((1, 32, 2), dtype=torch.float64)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    if entrypoint == "solve":
        norm = _bound_norm_estimate(operator, batch_size=1, seed=3020)
        tau, sigma_data, sigma_edge = _balanced_steps(norm)
    if invalid_field == "sigma":
        sigma.fill_(0.5)
    else:
        mask[0, 0] = 0.0
    operator.reset_call_counts()

    with pytest.raises(ValueError, match=message):
        if entrypoint == "estimate":
            estimate_block_operator_norms(
                operator,
                batch_size=1,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=8,
                power_iterations=2,
            )
        else:
            primal_dual_reconstruction(
                operator,
                observation,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=8,
                iterations=2,
                regularization_weight=0.1,
                penalty="tv",
                primal_step=tau,
                data_dual_step=sigma_data,
                edge_dual_step=sigma_edge,
                norm_estimate=norm,
            )

    assert operator.call_report() == {
        "forward_calls": 0,
        "adjoint_calls": 0,
    }


def test_norm_estimate_from_different_operator_is_rejected_without_calls() -> None:
    estimated_operator = _MaskedIdentityOperator()
    solve_operator = _MaskedIdentityOperator()
    norm = _bound_norm_estimate(estimated_operator, batch_size=1, seed=3021)
    sigma, mask = _prewhitened_views(solve_operator, batch_size=1)
    tau, sigma_data, sigma_edge = _balanced_steps(norm)
    solve_operator.reset_call_counts()

    with pytest.raises(ValueError, match="different operator instance"):
        primal_dual_reconstruction(
            solve_operator,
            torch.zeros((1, 32, 2), dtype=torch.float64),
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=8,
            iterations=2,
            regularization_weight=0.1,
            penalty="tv",
            primal_step=tau,
            data_dual_step=sigma_data,
            edge_dual_step=sigma_edge,
            norm_estimate=norm,
        )

    assert solve_operator.call_report() == {
        "forward_calls": 0,
        "adjoint_calls": 0,
    }


def test_tampered_norm_scalar_is_rejected_without_calls() -> None:
    operator = _MaskedIdentityOperator()
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3250)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    tampered = replace(
        norm,
        data_norm_squared_upper=0.5 * norm.data_norm_squared_upper,
    )
    operator.reset_call_counts()

    with pytest.raises(ValueError, match="internally inconsistent"):
        primal_dual_reconstruction(
            operator,
            torch.zeros((1, operator.ray_count, 2), dtype=torch.float64),
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=8,
            iterations=1,
            regularization_weight=0.0,
            penalty="tv",
            primal_step=0.01,
            data_dual_step=0.01,
            edge_dual_step=0.01,
            norm_estimate=tampered,
        )

    assert operator.call_report() == {
        "forward_calls": 0,
        "adjoint_calls": 0,
    }


def test_in_place_support_change_invalidates_bound_norm_without_calls() -> None:
    operator = _MaskedIdentityOperator()
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3022)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    tau, sigma_data, sigma_edge = _balanced_steps(norm)
    old_version = operator.support._version
    changed_support = operator.support.clone()
    changed_support[1, 1, 1] = 0.0
    operator.support.copy_(changed_support)
    assert operator.support._version > old_version
    operator.reset_call_counts()

    with pytest.raises(ValueError, match="operator state changed"):
        primal_dual_reconstruction(
            operator,
            torch.zeros((1, 32, 2), dtype=torch.float64),
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=8,
            iterations=2,
            regularization_weight=0.1,
            penalty="tv",
            primal_step=tau,
            data_dual_step=sigma_data,
            edge_dual_step=sigma_edge,
            norm_estimate=norm,
        )

    assert operator.call_report() == {
        "forward_calls": 0,
        "adjoint_calls": 0,
    }


@pytest.mark.parametrize("double_call", ["forward", "adjoint"])
def test_actual_call_report_delta_detects_double_call_operator(
    double_call: str,
) -> None:
    operator = _SwitchableDoubleCallOperator()
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3023)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    tau, sigma_data, sigma_edge = _balanced_steps(norm)
    operator.double_call = double_call
    operator.reset_call_counts()

    with pytest.raises(RuntimeError, match="PDHG call contract violated"):
        primal_dual_reconstruction(
            operator,
            torch.zeros((1, 32, 2), dtype=torch.float64),
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=8,
            iterations=2,
            regularization_weight=0.1,
            penalty="tv",
            primal_step=tau,
            data_dual_step=sigma_data,
            edge_dual_step=sigma_edge,
            norm_estimate=norm,
        )

    expected = {
        "forward_calls": 4 if double_call == "forward" else 2,
        "adjoint_calls": 4 if double_call == "adjoint" else 2,
    }
    assert operator.call_report() == expected


def test_injected_regularization_operator_reports_actual_solve_delta() -> None:
    operator = _MaskedIdentityOperator()
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3024)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    tau, sigma_data, sigma_edge = _balanced_steps(norm)
    regularizer = ForwardNeumannRegularizationOperator(operator.spacing_xyz)
    regularizer(
        torch.zeros((1, *operator.grid_shape), dtype=torch.float64)
    )
    regularizer.adjoint(
        torch.zeros((1, 3, *operator.grid_shape), dtype=torch.float64)
    )
    operator.reset_call_counts()

    result = primal_dual_reconstruction(
        operator,
        torch.zeros((1, 32, 2), dtype=torch.float64),
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        iterations=3,
        regularization_weight=0.1,
        penalty="tv",
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm,
        regularization_operator=regularizer,
    )

    assert result.gradient_calls == 3
    assert result.gradient_adjoint_calls == 3
    assert regularizer.call_report() == {
        "gradient_calls": 4,
        "gradient_adjoint_calls": 4,
    }


@pytest.mark.parametrize("double_call", ["gradient", "gradient_adjoint"])
def test_actual_regularization_report_delta_fails_closed_on_double_call(
    double_call: str,
) -> None:
    operator = _MaskedIdentityOperator()
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3025)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    tau, sigma_data, sigma_edge = _balanced_steps(norm)
    regularizer = _SwitchableDoubleCallRegularizationOperator(
        operator.spacing_xyz
    )
    regularizer.double_call = double_call
    operator.reset_call_counts()

    with pytest.raises(RuntimeError, match="PDHG call contract violated"):
        primal_dual_reconstruction(
            operator,
            torch.zeros((1, 32, 2), dtype=torch.float64),
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=8,
            iterations=2,
            regularization_weight=0.1,
            penalty="tv",
            primal_step=tau,
            data_dual_step=sigma_data,
            edge_dual_step=sigma_edge,
            norm_estimate=norm,
            regularization_operator=regularizer,
        )

    assert regularizer.call_report() == {
        "gradient_calls": 4 if double_call == "gradient" else 2,
        "gradient_adjoint_calls": (
            4 if double_call == "gradient_adjoint" else 2
        ),
    }


def test_primal_dual_report_retains_runner_call_fields() -> None:
    runner_fields = {
        "forward_calls",
        "adjoint_calls",
        "gradient_calls",
        "gradient_adjoint_calls",
    }

    assert runner_fields <= set(PrimalDualReconstruction.__dataclass_fields__)
    report_types = get_type_hints(PrimalDualReconstruction)
    for field_name in runner_fields:
        assert report_types[field_name] is int


@pytest.mark.parametrize(
    ("candidate_id", "penalty", "alpha", "formal_iterations"),
    _PDHG_SMOKE_CANDIDATES,
    ids=[candidate[0] for candidate in _PDHG_SMOKE_CANDIDATES],
)
def test_each_of_32_candidates_runs_three_finite_dual_feasible_steps(
    candidate_id: str,
    penalty: str,
    alpha: float,
    formal_iterations: int,
) -> None:
    assert len(_PDHG_SMOKE_CANDIDATES) == 32
    assert len({candidate[0] for candidate in _PDHG_SMOKE_CANDIDATES}) == 32
    assert candidate_id.endswith(f"_k{formal_iterations}")
    operator = _MaskedIdentityOperator()
    generator = torch.Generator().manual_seed(
        3200 + formal_iterations + int(round(256.0 * alpha))
    )
    truth = torch.randn(
        (1, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    ) * operator.support
    observation = operator(truth).detach()
    norm = _bound_norm_estimate(
        operator,
        batch_size=1,
        power_iterations=2,
        safety_factor=1.5,
        seed=formal_iterations,
    )
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    tau, sigma_data, sigma_edge = _balanced_steps(norm)
    operator.reset_call_counts()

    result = primal_dual_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        iterations=3,
        regularization_weight=alpha,
        penalty=penalty,
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm,
        huber_delta=0.4,
    )

    assert result.forward_calls == result.adjoint_calls == 3
    assert result.gradient_calls == result.gradient_adjoint_calls == 3
    assert operator.call_report() == {
        "forward_calls": 3,
        "adjoint_calls": 3,
    }
    for tensor in (result.volume, result.data_dual, result.edge_dual):
        assert torch.all(torch.isfinite(tensor))
    for row in result.history:
        assert all(torch.all(torch.isfinite(value)) for value in row.values())
    edge_norm = torch.linalg.vector_norm(result.edge_dual, dim=1)
    assert float(torch.max(edge_norm)) <= alpha + 1e-12


def test_history_reuses_same_extrapolated_point_with_one_d_and_dt_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterations = 5
    penalty = "huber"
    alpha = 0.05
    huber_delta = 0.3
    operator = _TracingMaskedIdentityOperator()
    generator = torch.Generator().manual_seed(3301)
    truth = torch.randn(
        (1, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    ) * operator.support
    observation = operator(truth).detach()
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3302)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    tau, sigma_data, sigma_edge = _balanced_steps(norm)
    operator.clear_trace()
    operator.reset_call_counts()

    original_gradient = primal_dual_module.regularization_gradient
    original_adjoint = primal_dual_module.regularization_gradient_adjoint
    gradient_inputs: list[torch.Tensor] = []
    gradient_outputs: list[torch.Tensor] = []
    gradient_adjoint_inputs: list[torch.Tensor] = []

    def tracked_gradient(volume: torch.Tensor, *, spacing_xyz):
        output = original_gradient(volume, spacing_xyz=spacing_xyz)
        gradient_inputs.append(volume.detach().clone())
        gradient_outputs.append(output.detach().clone())
        return output

    def tracked_gradient_adjoint(gradient: torch.Tensor, *, spacing_xyz):
        gradient_adjoint_inputs.append(gradient.detach().clone())
        return original_adjoint(gradient, spacing_xyz=spacing_xyz)

    monkeypatch.setattr(
        primal_dual_module,
        "regularization_gradient",
        tracked_gradient,
    )
    monkeypatch.setattr(
        primal_dual_module,
        "regularization_gradient_adjoint",
        tracked_gradient_adjoint,
    )

    result = primal_dual_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        iterations=iterations,
        regularization_weight=alpha,
        penalty=penalty,
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm,
        huber_delta=huber_delta,
    )

    assert len(operator.forward_inputs) == iterations
    assert len(gradient_inputs) == iterations
    assert len(gradient_adjoint_inputs) == iterations
    assert result.gradient_calls == iterations
    assert result.gradient_adjoint_calls == iterations
    assert operator.call_report() == {
        "forward_calls": iterations,
        "adjoint_calls": iterations,
    }
    initial_objective = torch.sum(observation.square(), dim=(1, 2)).clamp_min(
        1e-20
    )
    for index, row in enumerate(result.history):
        assert torch.equal(
            operator.forward_inputs[index][:, 0],
            gradient_inputs[index],
        )
        residual = operator.forward_outputs[index] - observation
        magnitude = torch.linalg.vector_norm(gradient_outputs[index], dim=1)
        edge_value = torch.sum(
            torch.where(
                magnitude <= huber_delta,
                0.5 * magnitude.square() / huber_delta,
                magnitude - 0.5 * huber_delta,
            ),
            dim=(1, 2, 3),
        )
        data_value = torch.sum(residual.square(), dim=(1, 2))
        assert torch.allclose(
            row["extrapolated_relative_data_objective"],
            data_value / initial_objective,
            atol=1e-12,
            rtol=1e-12,
        )
        assert torch.allclose(
            row["extrapolated_edge_penalty"],
            edge_value,
            atol=1e-12,
            rtol=1e-12,
        )
        assert torch.allclose(
            row["extrapolated_total_objective"],
            0.5 * data_value + alpha * edge_value,
            atol=1e-12,
            rtol=1e-12,
        )


def test_normalized_fixed_point_residual_matches_declared_metric() -> None:
    operator = _MaskedIdentityOperator()
    generator = torch.Generator().manual_seed(3351)
    initial_volume = torch.randn(
        (1, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    ) * operator.support
    initial_data_dual = torch.randn(
        (1, operator.ray_count, 2),
        generator=generator,
        dtype=torch.float64,
    )
    observation = torch.randn(
        (1, operator.ray_count, 2),
        generator=generator,
        dtype=torch.float64,
    )
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3352)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    tau, sigma_data, sigma_edge = _balanced_steps(norm)

    result = primal_dual_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        iterations=1,
        regularization_weight=0.03,
        penalty="tv",
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm,
        initial_volume=initial_volume,
        initial_data_dual=initial_data_dual,
    )

    previous_squared = (
        torch.sum(initial_volume.flatten(1).square(), dim=1) / tau
        + torch.sum(initial_data_dual.flatten(1).square(), dim=1) / sigma_data
    )
    next_squared = (
        torch.sum(result.volume.flatten(1).square(), dim=1) / tau
        + torch.sum(result.data_dual.flatten(1).square(), dim=1) / sigma_data
        + torch.sum(result.edge_dual.flatten(1).square(), dim=1) / sigma_edge
    )
    increment_squared = (
        torch.sum(
            (result.volume - initial_volume).flatten(1).square(),
            dim=1,
        )
        / tau
        + torch.sum(
            (result.data_dual - initial_data_dual).flatten(1).square(),
            dim=1,
        )
        / sigma_data
        + torch.sum(result.edge_dual.flatten(1).square(), dim=1) / sigma_edge
    )
    expected = torch.sqrt(increment_squared) / torch.maximum(
        torch.maximum(torch.sqrt(previous_squared), torch.sqrt(next_squared)),
        torch.full_like(previous_squared, 1e-12),
    )

    assert torch.allclose(
        result.history[0]["normalized_joint_fixed_point_residual"],
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_zero_alpha_checkpoints_match_independent_manual_recurrence() -> None:
    operator = _MaskedIdentityOperator()
    generator = torch.Generator().manual_seed(3401)
    truth = torch.randn(
        (1, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    ) * operator.support
    observation = operator(truth).detach()
    initial_volume = 0.2 * torch.randn(
        truth.shape,
        generator=generator,
        dtype=torch.float64,
    ) * operator.support
    initial_data_dual = 0.1 * torch.randn(
        observation.shape,
        generator=generator,
        dtype=torch.float64,
    )
    norm = _bound_norm_estimate(operator, batch_size=1, seed=3402)
    sigma, mask = _prewhitened_views(operator, batch_size=1)
    sigma_data = 0.4
    sigma_edge = 0.3
    tau = 0.6 / (sigma_data * norm.data_norm_squared_upper)
    theta = 0.6
    checkpoints = (1, 2, 4)
    operator.reset_call_counts()

    result = primal_dual_reconstruction(
        operator,
        observation,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        iterations=4,
        regularization_weight=0.0,
        penalty="tv",
        primal_step=tau,
        data_dual_step=sigma_data,
        edge_dual_step=sigma_edge,
        norm_estimate=norm,
        extrapolation=theta,
        initial_volume=initial_volume,
        initial_data_dual=initial_data_dual,
        checkpoint_iterations=checkpoints,
    )

    support = operator.support[None, None]
    current = initial_volume.clone() * support
    extrapolated = current.clone()
    data_dual = initial_data_dual.clone()
    manual_checkpoints: dict[int, torch.Tensor] = {}
    for iteration in range(1, 5):
        projected = (extrapolated[:, 0] * operator.support).reshape_as(
            observation
        )
        data_dual = (
            data_dual + sigma_data * (projected - observation)
        ) / (1.0 + sigma_data)
        data_normal = data_dual.reshape(
            1,
            1,
            *operator.grid_shape,
        ) * support
        next_volume = (current - tau * data_normal) * support
        extrapolated = (
            next_volume + theta * (next_volume - current)
        ) * support
        current = next_volume
        if iteration in checkpoints:
            manual_checkpoints[iteration] = current.clone()

    assert set(result.checkpoint_volumes) == set(checkpoints)
    for checkpoint in checkpoints:
        assert torch.allclose(
            result.checkpoint_volumes[checkpoint],
            manual_checkpoints[checkpoint],
            atol=1e-12,
            rtol=1e-12,
        )
    assert torch.allclose(result.volume, current, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result.data_dual, data_dual, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(result.edge_dual) == 0
    assert operator.call_report() == {
        "forward_calls": 4,
        "adjoint_calls": 4,
    }


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_cpu_float32_and_mps_float32_match_on_fixed_short_run() -> None:
    generator = torch.Generator().manual_seed(3011)
    truth = torch.randn(
        (2, 1, 8, 8, 8),
        generator=generator,
        dtype=torch.float32,
    )
    cpu_operator = _operator(torch.float32)
    truth = truth * cpu_operator.support
    observation = cpu_operator(truth).detach()
    sigma, mask = _prewhitened_views(cpu_operator, batch_size=2)
    cpu_operator.reset_call_counts()
    cpu_estimate = estimate_block_operator_norms(
        cpu_operator,
        batch_size=2,
        sigma_by_view=sigma,
        view_mask=mask,
        rays_per_view=8,
        power_iterations=8,
        safety_factor=1.5,
        seed=3012,
    )
    mps_operator = _operator(torch.float32).to("mps")
    mps_sigma, mps_mask = _prewhitened_views(mps_operator, batch_size=2)
    mps_estimate = estimate_block_operator_norms(
        mps_operator,
        batch_size=2,
        sigma_by_view=mps_sigma,
        view_mask=mps_mask,
        rays_per_view=8,
        power_iterations=8,
        safety_factor=1.5,
        seed=3012,
    )
    root_data = max(
        cpu_estimate.data_norm_squared_upper,
        mps_estimate.data_norm_squared_upper,
    ) ** 0.5
    root_edge = max(
        cpu_estimate.gradient_norm_squared_upper,
        mps_estimate.gradient_norm_squared_upper,
    ) ** 0.5
    kwargs = {
        "sigma_by_view": sigma,
        "view_mask": mask,
        "rays_per_view": 8,
        "iterations": 5,
        "regularization_weight": 0.02,
        "penalty": "huber",
        "primal_step": 0.8 / (root_data + root_edge),
        "data_dual_step": 1.0 / root_data,
        "edge_dual_step": 1.0 / root_edge,
        "huber_delta": 0.4,
    }
    cpu_result = primal_dual_reconstruction(
        cpu_operator,
        observation,
        norm_estimate=cpu_estimate,
        **kwargs,
    )
    mps_result = primal_dual_reconstruction(
        mps_operator,
        observation.to("mps"),
        norm_estimate=mps_estimate,
        **{
            **kwargs,
            "sigma_by_view": sigma.to("mps"),
            "view_mask": mask.to("mps"),
        },
    )
    relative = torch.linalg.vector_norm(
        cpu_result.volume - mps_result.volume.cpu()
    ) / torch.linalg.vector_norm(cpu_result.volume).clamp_min(1e-12)
    assert float(relative) < 5e-4
