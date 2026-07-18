from __future__ import annotations

import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import SyntheticRayRig
from demo_t16_operator.discrete_rk4_jvp_predictor import (
    predict_discrete_rk4_jvp_residual,
)
from demo_t16_operator.field_dependent_ray import (
    path_integrated_deflection,
    sample_pupil_sobol,
    straight_ray_deflection,
    trace_field_dependent_rays,
)
from demo_t16_operator.operator_consistent_homotopy_predictor import (
    OperatorConsistentHomotopyDomainError,
    predict_operator_consistent_homotopy_residual,
)


def _rig(**overrides: float | str) -> SyntheticRayRig:
    values: dict[str, float | str] = {
        "rig_id": "operator-consistent-homotopy-test",
        "view_angle_degrees": 31.0,
        "detector_u": 0.041,
        "detector_z": -0.033,
        "aperture_radius": 0.032,
        "path_half_length": 0.62,
        "cone_u": 0.031,
        "cone_z": 0.024,
        "bend": 0.0,
    }
    values.update(overrides)
    return SyntheticRayRig(**values)


def _smooth_grid(size: int = 13) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    plume = -torch.exp(
        -(
            (xx / 0.38) ** 2
            + (yy / 0.29) ** 2
            + (zz / 0.34) ** 2
        )
    )
    lobe = 0.18 * torch.exp(
        -(
            ((xx - 0.21) / 0.23) ** 2
            + ((yy + 0.11) / 0.26) ** 2
            + ((zz - 0.09) / 0.22) ** 2
        )
    )
    return plume + lobe


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(candidate.detach() - reference.detach())
        / torch.linalg.vector_norm(reference.detach()).clamp_min(1e-30)
    )


def _high_output(
    values: torch.Tensor,
    states: torch.Tensor,
    *,
    scale: float,
    steps: int,
) -> torch.Tensor:
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
    return path_integrated_deflection(
        values,
        trace,
        gradient_mode="central",
        difference_step=2e-4,
        refractivity_scale=scale,
        create_graph=False,
        detach_path=False,
    )


def test_matches_full_forward_mode_discrete_jvp_at_float64_precision() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(9, seed=301)
    kwargs = {
        "difference_step": 2e-4,
        "refractivity_scale": 1.7e-3,
        "step_count": 24,
    }
    analytic = predict_operator_consistent_homotopy_residual(
        values,
        states,
        _rig(),
        **kwargs,
    )
    automatic = predict_discrete_rk4_jvp_residual(
        values,
        states,
        _rig(),
        **kwargs,
    )
    assert _relative_l2(
        analytic.residual_prediction_uv,
        automatic.residual_prediction_uv,
    ) < 2e-11
    assert _relative_l2(analytic.delta_positions, automatic.delta_positions) < 2e-11
    assert _relative_l2(analytic.delta_directions, automatic.delta_directions) < 2e-11
    assert torch.allclose(
        analytic.base_output_uv,
        automatic.base_output_uv,
        atol=2e-15,
        rtol=2e-12,
    )


def test_constant_index_has_only_float64_roundoff() -> None:
    result = predict_operator_consistent_homotopy_residual(
        torch.full((9, 9, 9), 0.27, dtype=torch.float64),
        sample_pupil_sobol(5, seed=303),
        _rig(),
        difference_step=2e-4,
        refractivity_scale=2e-3,
        step_count=12,
    )
    assert torch.max(torch.abs(result.base_output_uv)) < 1e-14
    assert torch.max(torch.abs(result.residual_prediction_uv)) < 1e-14
    assert torch.max(torch.abs(result.delta_positions)) < 1e-14
    assert torch.max(torch.abs(result.delta_directions)) < 1e-14
    assert torch.all(result.valid_mask)


def test_base_output_matches_central_straight_route() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(7, seed=305)
    result = predict_operator_consistent_homotopy_residual(
        values,
        states,
        _rig(),
        difference_step=2e-4,
        refractivity_scale=8e-4,
        step_count=18,
    )
    medium = straight_ray_deflection(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=2e-4,
        refractivity_scale=8e-4,
        step_count=18,
        create_graph=False,
    )
    assert torch.allclose(result.base_output_uv, medium, atol=2e-15, rtol=2e-12)


def test_candidate_improves_weak_curved_ray_output() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(16, seed=307)
    scale = 9e-4
    steps = 32
    result = predict_operator_consistent_homotopy_residual(
        values,
        states,
        _rig(),
        difference_step=2e-4,
        refractivity_scale=scale,
        step_count=steps,
    )
    high = _high_output(values, states, scale=scale, steps=steps)
    exact_residual = high - result.base_output_uv
    assert _relative_l2(result.residual_prediction_uv, exact_residual) < 0.01
    assert _relative_l2(result.candidate_output_uv, high) < 0.02 * _relative_l2(
        result.base_output_uv,
        high,
    )


def test_shapes_detachment_and_query_contract() -> None:
    ray_count = 6
    step_count = 14
    result = predict_operator_consistent_homotopy_residual(
        _smooth_grid(),
        sample_pupil_sobol(ray_count, seed=311),
        _rig(),
        difference_step=2e-4,
        refractivity_scale=8e-4,
        step_count=step_count,
    )
    assert result.base_output_uv.shape == (ray_count, 2)
    assert result.residual_prediction_uv.shape == (ray_count, 2)
    assert result.base_positions.shape == (ray_count, step_count + 1, 3)
    assert result.forcing.shape == (ray_count, 2 * step_count + 1, 3)
    assert result.output_position_jacobian.shape == (
        ray_count,
        2 * step_count + 1,
        3,
        3,
    )
    assert result.coefficient_sample_count == 2 * step_count + 1
    accounting = result.query_accounting
    assert accounting["logical_scalar_grid_point_queries"] == (
        7 * ray_count * (2 * step_count + 1)
    )
    assert accounting["batched_interpolation_dispatches"] == 7
    assert accounting["coordinate_reverse_sweeps"] == 4
    assert accounting["forward_mode_bend_jvp_evaluations"] == 0
    assert accounting["exact_high_evaluations"] == 0
    assert result.stop_gradient_applied
    for tensor in (
        result.base_output_uv,
        result.residual_prediction_uv,
        result.delta_positions,
        result.delta_directions,
        result.output_position_jacobian,
        result.risk_norm,
    ):
        assert tensor.dtype == torch.float64
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_direction_tangent_is_orthogonal_after_every_rk4_step() -> None:
    result = predict_operator_consistent_homotopy_residual(
        _smooth_grid(),
        sample_pupil_sobol(8, seed=313),
        _rig(),
        difference_step=2e-4,
        refractivity_scale=1.5e-3,
        step_count=20,
    )
    dot = torch.sum(result.base_directions * result.delta_directions, dim=-1)
    assert torch.max(torch.abs(dot)) < 1e-12
    assert torch.max(result.direction_tangent_orthogonality_error) < 1e-12
    assert torch.all(result.valid_mask)


def test_result_is_deterministic() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(6, seed=317)
    kwargs = {
        "difference_step": 2e-4,
        "refractivity_scale": 1e-3,
        "step_count": 16,
    }
    left = predict_operator_consistent_homotopy_residual(
        values,
        states,
        _rig(),
        **kwargs,
    )
    right = predict_operator_consistent_homotopy_residual(
        values,
        states,
        _rig(),
        **kwargs,
    )
    assert torch.equal(left.base_output_uv, right.base_output_uv)
    assert torch.equal(left.residual_prediction_uv, right.residual_prediction_uv)
    assert torch.equal(left.output_position_jacobian, right.output_position_jacobian)
    assert left.failure_reasons == right.failure_reasons


def test_linearization_radius_excess_fails_closed_per_ray() -> None:
    result = predict_operator_consistent_homotopy_residual(
        _smooth_grid(),
        sample_pupil_sobol(4, seed=319),
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
    with pytest.raises(OperatorConsistentHomotopyDomainError):
        predict_operator_consistent_homotopy_residual(
            _smooth_grid(),
            sample_pupil_sobol(2, seed=323),
            _rig(detector_u=0.92, path_half_length=0.94),
            difference_step=2e-4,
            refractivity_scale=5e-4,
            step_count=8,
            domain_margin=0.02,
        )


def test_nonpositive_refractive_index_fails_closed() -> None:
    with pytest.raises(OperatorConsistentHomotopyDomainError):
        predict_operator_consistent_homotopy_residual(
            torch.full((7, 7, 7), -100.0, dtype=torch.float64),
            sample_pupil_sobol(2, seed=327),
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
        predict_operator_consistent_homotopy_residual(
            values,
            states,
            _rig(),
            **parameters,
        )
