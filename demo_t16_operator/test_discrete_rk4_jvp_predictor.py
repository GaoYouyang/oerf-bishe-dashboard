from __future__ import annotations

import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import SyntheticRayRig
from demo_t16_operator.discrete_rk4_jvp_predictor import (
    DiscreteRK4JVPDomainError,
    _bend_parameterized_discrete_map,
    predict_discrete_rk4_jvp_residual,
)
from demo_t16_operator.field_dependent_ray import (
    path_integrated_deflection,
    sample_pupil_sobol,
    straight_ray_deflection,
    trace_field_dependent_rays,
)


def _rig(**overrides: float | str) -> SyntheticRayRig:
    values: dict[str, float | str] = {
        "rig_id": "discrete-rk4-jvp-test",
        "view_angle_degrees": 18.0,
        "detector_u": 0.037,
        "detector_z": -0.029,
        "aperture_radius": 0.018,
        "path_half_length": 0.61,
        "cone_u": 0.014,
        "cone_z": 0.011,
        "bend": 0.0,
    }
    values.update(overrides)
    return SyntheticRayRig(**values)


def _smooth_grid(size: int = 13) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    core = -torch.exp(
        -(
            (xx / 0.39) ** 2
            + (yy / 0.31) ** 2
            + (zz / 0.36) ** 2
        )
    )
    shoulder = 0.22 * torch.exp(
        -(
            ((xx - 0.19) / 0.25) ** 2
            + ((yy + 0.12) / 0.28) ** 2
            + ((zz - 0.08) / 0.24) ** 2
        )
    )
    return core + shoulder


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(candidate.detach() - reference.detach())
        / torch.linalg.vector_norm(reference.detach()).clamp_min(1e-30)
    )


def _exact_high(
    values: torch.Tensor,
    states: torch.Tensor,
    *,
    scale: float,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trace = trace_field_dependent_rays(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=2e-4,
        refractivity_scale=scale,
        step_count=steps,
        create_graph=False,
    )
    output = path_integrated_deflection(
        values,
        trace,
        gradient_mode="central",
        difference_step=2e-4,
        refractivity_scale=scale,
        create_graph=False,
        detach_path=False,
    )
    return output, trace.positions, trace.directions


def test_constant_index_has_zero_tangent_and_risk() -> None:
    values = torch.full((9, 9, 9), 0.27, dtype=torch.float64)
    result = predict_discrete_rk4_jvp_residual(
        values,
        sample_pupil_sobol(5, seed=201),
        _rig(),
        difference_step=2e-4,
        refractivity_scale=2e-3,
        step_count=12,
    )
    assert torch.max(torch.abs(result.base_output_uv)) < 1e-14
    assert torch.max(torch.abs(result.residual_prediction_uv)) < 1e-14
    assert torch.max(torch.abs(result.delta_positions)) < 1e-14
    assert torch.max(torch.abs(result.delta_directions)) < 1e-14
    assert torch.max(torch.abs(result.risk_norm)) < 1e-14
    assert torch.all(result.valid_mask)
    assert result.failure_reasons == ("ok",) * 5


def test_shapes_detachment_and_query_accounting_are_explicit() -> None:
    result = predict_discrete_rk4_jvp_residual(
        _smooth_grid(),
        sample_pupil_sobol(6, seed=203),
        _rig(),
        difference_step=2e-4,
        refractivity_scale=8e-4,
        step_count=14,
    )
    assert result.base_output_uv.shape == (6, 2)
    assert result.residual_prediction_uv.shape == (6, 2)
    assert result.candidate_output_uv.shape == (6, 2)
    assert result.base_positions.shape == (6, 15, 3)
    assert result.delta_positions.shape == (6, 15, 3)
    assert result.risk_norm.shape == (6,)
    assert result.query_accounting["logical_scalar_grid_point_queries"] == 35 * 6 * 14
    assert result.query_accounting["interpolation_dispatches"] == 35 * 14
    assert result.query_accounting["exact_high_evaluations"] == 0
    assert result.query_accounting["reverse_mode_vjp_evaluations"] == 0
    assert result.stop_gradient_applied
    for tensor in (
        result.base_output_uv,
        result.residual_prediction_uv,
        result.delta_positions,
        result.delta_directions,
        result.risk_norm,
    ):
        assert tensor.dtype == torch.float64
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_zero_bend_primal_matches_straight_medium_output() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(7, seed=205)
    result = predict_discrete_rk4_jvp_residual(
        values,
        states,
        _rig(),
        difference_step=2e-4,
        refractivity_scale=7e-4,
        step_count=18,
    )
    medium = straight_ray_deflection(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=2e-4,
        refractivity_scale=7e-4,
        step_count=18,
        create_graph=False,
    )
    assert torch.allclose(result.base_output_uv, medium, atol=2e-15, rtol=2e-12)


def test_unit_bend_map_matches_existing_high_discretization() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(5, seed=207)
    scale = 9e-4
    steps = 16
    bend = torch.tensor(1.0, dtype=torch.float64)
    output, positions, directions = _bend_parameterized_discrete_map(
        values,
        states,
        _rig(),
        bend,
        difference_step=2e-4,
        refractivity_scale=scale,
        step_count=steps,
        refractive_index_floor=0.500001,
    )
    expected, expected_positions, expected_directions = _exact_high(
        values,
        states,
        scale=scale,
        steps=steps,
    )
    assert torch.allclose(output, expected, atol=2e-15, rtol=2e-12)
    assert torch.allclose(positions, expected_positions, atol=2e-15, rtol=2e-12)
    assert torch.allclose(directions, expected_directions, atol=2e-15, rtol=2e-12)


def test_jvp_matches_centered_bend_finite_difference() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(5, seed=211)
    scale = 8e-4
    steps = 18
    result = predict_discrete_rk4_jvp_residual(
        values,
        states,
        _rig(),
        difference_step=2e-4,
        refractivity_scale=scale,
        step_count=steps,
    )
    epsilon = 2e-4
    plus = _bend_parameterized_discrete_map(
        values,
        states,
        _rig(),
        torch.tensor(epsilon, dtype=torch.float64),
        difference_step=2e-4,
        refractivity_scale=scale,
        step_count=steps,
        refractive_index_floor=0.500001,
    )[0]
    minus = _bend_parameterized_discrete_map(
        values,
        states,
        _rig(),
        torch.tensor(-epsilon, dtype=torch.float64),
        difference_step=2e-4,
        refractivity_scale=scale,
        step_count=steps,
        refractive_index_floor=0.500001,
    )[0]
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert _relative_l2(result.residual_prediction_uv, finite_difference) < 1e-6


def test_discrete_jvp_is_accurate_for_weak_curved_ray_residual() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(16, seed=213)
    scale = 8e-4
    steps = 32
    result = predict_discrete_rk4_jvp_residual(
        values,
        states,
        _rig(),
        difference_step=2e-4,
        refractivity_scale=scale,
        step_count=steps,
    )
    high, _, _ = _exact_high(values, states, scale=scale, steps=steps)
    exact_residual = high - result.base_output_uv
    assert _relative_l2(result.residual_prediction_uv, exact_residual) < 0.01
    assert _relative_l2(result.candidate_output_uv, high) < _relative_l2(
        result.base_output_uv,
        high,
    )


def test_direction_tangent_includes_normalization_derivative() -> None:
    result = predict_discrete_rk4_jvp_residual(
        _smooth_grid(),
        sample_pupil_sobol(8, seed=215),
        _rig(),
        difference_step=2e-4,
        refractivity_scale=1e-3,
        step_count=20,
    )
    dot = torch.sum(result.base_directions * result.delta_directions, dim=-1)
    assert torch.max(torch.abs(dot)) < 1e-12
    assert torch.max(result.direction_tangent_orthogonality_error) < 1e-12
    assert torch.all(result.valid_mask)


def test_result_is_deterministic() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(6, seed=217)
    kwargs = {
        "difference_step": 2e-4,
        "refractivity_scale": 9e-4,
        "step_count": 16,
    }
    left = predict_discrete_rk4_jvp_residual(values, states, _rig(), **kwargs)
    right = predict_discrete_rk4_jvp_residual(values, states, _rig(), **kwargs)
    assert torch.equal(left.base_output_uv, right.base_output_uv)
    assert torch.equal(left.residual_prediction_uv, right.residual_prediction_uv)
    assert torch.equal(left.delta_positions, right.delta_positions)
    assert left.failure_reasons == right.failure_reasons


def test_linearization_radius_excess_fails_closed_per_ray() -> None:
    result = predict_discrete_rk4_jvp_residual(
        _smooth_grid(),
        sample_pupil_sobol(4, seed=219),
        _rig(),
        difference_step=2e-4,
        refractivity_scale=2e-3,
        step_count=16,
        max_position_perturbation=1e-14,
        max_direction_perturbation=1e-14,
    )
    assert not torch.any(result.valid_mask)
    assert torch.all(torch.isinf(result.risk_norm))
    assert all("linearization_radius" in reason for reason in result.failure_reasons)


def test_rig_outside_stencil_domain_fails_closed() -> None:
    with pytest.raises(DiscreteRK4JVPDomainError):
        predict_discrete_rk4_jvp_residual(
            _smooth_grid(),
            sample_pupil_sobol(2, seed=223),
            _rig(detector_u=0.92, path_half_length=0.94),
            difference_step=2e-4,
            refractivity_scale=5e-4,
            step_count=8,
            domain_margin=0.02,
        )


def test_nonpositive_refractive_index_fails_closed() -> None:
    with pytest.raises(DiscreteRK4JVPDomainError):
        predict_discrete_rk4_jvp_residual(
            torch.full((7, 7, 7), -100.0, dtype=torch.float64),
            sample_pupil_sobol(2, seed=227),
            _rig(),
            difference_step=2e-4,
            refractivity_scale=0.02,
            step_count=8,
        )


@pytest.mark.parametrize(
    ("values", "states", "kwargs", "error"),
    [
        (
            torch.zeros((7, 7, 7), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float64),
            {},
            ValueError,
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float32),
            {},
            ValueError,
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.tensor([[0.2, 1.2]], dtype=torch.float64),
            {},
            ValueError,
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            {"step_count": 1},
            ValueError,
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            {"step_count": 8.5},
            TypeError,
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            {"difference_step": 0.0},
            ValueError,
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            {"refractivity_scale": float("nan")},
            ValueError,
        ),
    ],
)
def test_invalid_inputs_are_rejected(
    values: torch.Tensor,
    states: torch.Tensor,
    kwargs: dict[str, float | int],
    error: type[Exception],
) -> None:
    parameters: dict[str, float | int] = {
        "difference_step": 2e-4,
        "refractivity_scale": 5e-4,
        "step_count": 8,
    }
    parameters.update(kwargs)
    with pytest.raises(error):
        predict_discrete_rk4_jvp_residual(
            values,
            states,
            _rig(),
            **parameters,
        )
