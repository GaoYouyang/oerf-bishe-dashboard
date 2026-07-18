from __future__ import annotations

import itertools

import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import (
    SyntheticRayRig,
    automatic_spatial_gradient,
    central_difference_spatial_gradient,
    evaluate_automatic_discrete_pair,
    evaluate_automatic_projected,
    evaluate_discrete_projected,
    joint_state_geometry,
    optimal_two_level_allocation,
    sample_joint_pupil_path_sobol,
    smoothstep_grid_field,
    trace_sample_variance,
    two_level_efficiency,
    two_level_mean,
)


def _linear_grid(size: int = 7) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return 0.7 * xx - 0.4 * yy + 0.2 * zz


def test_smoothstep_grid_reproduces_vertices_and_remains_differentiable() -> None:
    values = _linear_grid()
    points = torch.tensor(
        [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    actual = smoothstep_grid_field(values, points)
    expected = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    gradient = automatic_spatial_gradient(values, points[1:2], create_graph=False)
    assert gradient.shape == (1, 3)
    assert torch.all(torch.isfinite(gradient))


def test_central_difference_converges_to_automatic_gradient_inside_one_cell() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn((7, 7, 7), generator=generator, dtype=torch.float64)
    points = torch.tensor(
        [[-0.41, -0.17, 0.22], [0.14, 0.38, -0.46]], dtype=torch.float64
    )
    automatic = automatic_spatial_gradient(values, points, create_graph=False)
    discrete = central_difference_spatial_gradient(values, points, step=1e-5)
    assert torch.allclose(automatic, discrete, atol=2e-6, rtol=2e-6)


def test_joint_geometry_uses_common_state_and_declared_bend() -> None:
    states = sample_joint_pupil_path_sobol(16, seed=13)
    rig = SyntheticRayRig(
        rig_id="boundary",
        view_angle_degrees=35.0,
        detector_u=0.08,
        detector_z=-0.04,
        aperture_radius=0.07,
        bend=0.025,
    )
    low_points, low_u, low_v = joint_state_geometry(
        states, rig, high_geometry=False
    )
    high_points, high_u, high_v = joint_state_geometry(
        states, rig, high_geometry=True
    )
    assert low_points.shape == high_points.shape == (16, 3)
    assert torch.max(torch.abs(low_points)) < 1.0
    assert torch.max(torch.abs(high_points)) < 1.0
    assert not torch.allclose(low_points, high_points)
    assert torch.allclose(low_u, high_u)
    assert torch.allclose(low_v, high_v)


def test_two_level_estimator_is_exact_in_expectation_on_finite_population() -> None:
    low_population = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    high_population = torch.tensor([[2.0], [5.0]], dtype=torch.float64)
    estimates = []
    for low_index, residual_index in itertools.product(range(2), repeat=2):
        estimates.append(
            two_level_mean(
                low_population[low_index : low_index + 1],
                high_population[residual_index : residual_index + 1],
                low_population[residual_index : residual_index + 1],
            )
        )
    expected = torch.mean(high_population, dim=0)
    assert torch.allclose(torch.mean(torch.stack(estimates), dim=0), expected)


def test_two_level_parameter_gradient_is_unbiased_before_nonlinear_loss() -> None:
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    state = torch.tensor([1.0, 2.0], dtype=torch.float64)
    low_population = theta * state[:, None]
    high_population = (2.0 * theta * state + state.square())[:, None]
    gradients = []
    for low_index, residual_index in itertools.product(range(2), repeat=2):
        estimate = two_level_mean(
            low_population[low_index : low_index + 1],
            high_population[residual_index : residual_index + 1],
            low_population[residual_index : residual_index + 1],
        )
        gradients.append(torch.autograd.grad(estimate.sum(), theta, retain_graph=True)[0])
    exact = torch.autograd.grad(torch.mean(high_population), theta)[0]
    assert torch.allclose(torch.mean(torch.stack(gradients)), exact)


def test_squared_forward_estimate_has_variance_bias() -> None:
    population = torch.tensor([[-1.0], [3.0]], dtype=torch.float64)
    exact_square = torch.mean(population).square()
    sampled_square = torch.mean(population.square())
    assert sampled_square > exact_square
    assert float(sampled_square - exact_square) == pytest.approx(
        trace_sample_variance(population) * (len(population) - 1) / len(population)
    )


def test_efficiency_and_integer_allocation_account_for_residual_cost() -> None:
    low = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    high = low + torch.tensor([[0.0], [0.1], [-0.1], [0.0]], dtype=torch.float64)
    efficiency = two_level_efficiency(
        high,
        low,
        high_cost=6.0,
        low_cost=2.0,
        residual_cost=8.0,
    )
    assert efficiency.residual_trace_variance < efficiency.high_trace_variance
    assert efficiency.predicted_efficiency_gain > 1.0
    n_low, n_residual, consumed = optimal_two_level_allocation(
        total_cost=200.0,
        low_variance=efficiency.low_trace_variance,
        residual_variance=efficiency.residual_trace_variance,
        low_cost=2.0,
        residual_cost=8.0,
    )
    assert n_low >= 2 and n_residual >= 2
    assert consumed <= 200.0
    assert consumed + 2.0 > 200.0 or consumed + 8.0 > 200.0


def test_projected_pair_is_finite_and_supports_parameter_vjp() -> None:
    values = _linear_grid().clone().requires_grad_(True)
    states = sample_joint_pupil_path_sobol(12, seed=29)
    rig = SyntheticRayRig(
        rig_id="medium",
        view_angle_degrees=70.0,
        detector_u=-0.06,
        detector_z=0.03,
        aperture_radius=0.05,
        bend=0.015,
    )
    low, high = evaluate_automatic_discrete_pair(
        values,
        states,
        rig,
        difference_step=1e-3,
        create_graph=True,
    )
    assert low.shape == high.shape == (12, 2)
    assert torch.all(torch.isfinite(low)) and torch.all(torch.isfinite(high))
    estimate = two_level_mean(low[:8], high[8:], low[8:])
    gradient = torch.autograd.grad(estimate.square().sum(), values)[0]
    assert gradient.shape == values.shape
    assert torch.all(torch.isfinite(gradient))


def test_separate_projected_evaluators_match_the_paired_interface() -> None:
    values = _linear_grid()
    states = sample_joint_pupil_path_sobol(16, seed=31)
    rig = SyntheticRayRig(
        rig_id="timing",
        view_angle_degrees=42.0,
        detector_u=0.02,
        detector_z=-0.03,
        aperture_radius=0.04,
        bend=0.01,
    )
    low_pair, high_pair = evaluate_automatic_discrete_pair(
        values,
        states,
        rig,
        difference_step=1e-3,
        create_graph=False,
    )
    low = evaluate_automatic_projected(
        values,
        states,
        rig,
        create_graph=False,
    )
    high = evaluate_discrete_projected(
        values,
        states,
        rig,
        difference_step=1e-3,
    )
    assert torch.equal(low_pair, low)
    assert torch.equal(high_pair, high)


@pytest.mark.parametrize("count", [0, -2])
def test_invalid_sample_count_is_rejected(count: int) -> None:
    with pytest.raises(ValueError):
        sample_joint_pupil_path_sobol(count, seed=1)


def test_allocation_rejects_an_unfunded_minimum() -> None:
    with pytest.raises(ValueError):
        optimal_two_level_allocation(
            total_cost=5.0,
            low_variance=1.0,
            residual_variance=0.1,
            low_cost=2.0,
            residual_cost=8.0,
            minimum_count=2,
        )


def test_integer_allocation_is_exact_on_a_continuous_rounding_counterexample() -> None:
    low_count, residual_count, consumed = optimal_two_level_allocation(
        total_cost=100.0,
        low_variance=0.1,
        residual_variance=0.1,
        low_cost=0.1,
        residual_cost=1.0,
    )
    objective = 0.1 / low_count + 0.1 / residual_count
    brute_force = min(
        (
            0.1 / candidate_low + 0.1 / candidate_residual,
            candidate_low,
            candidate_residual,
        )
        for candidate_low in range(2, 981)
        for candidate_residual in range(2, 99)
        if 0.1 * candidate_low + candidate_residual <= 100.0 + 1e-12
    )
    assert objective == pytest.approx(brute_force[0])
    assert consumed <= 100.0
