from __future__ import annotations

import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import (
    SyntheticRayRig,
    smoothstep_grid_field,
)
from demo_t16_operator.field_dependent_ray import (
    path_integrated_deflection,
    sample_pupil_sobol,
    straight_ray_deflection,
    trace_field_dependent_rays,
)
from demo_t16_operator.trajectory_variational_predictor import (
    TrajectoryPredictorDomainError,
    integrate_affine_variational_rk4,
    linearize_straight_medium_path,
    predict_trajectory_variational_residual,
)


def _rig(**overrides: float | str) -> SyntheticRayRig:
    values: dict[str, float | str] = {
        "rig_id": "trajectory-variational-test",
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


def _nonlinear_high_minus_medium(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    scale: float,
    step_count: int,
    gradient_mode: str,
    difference_step: float = 2e-4,
) -> torch.Tensor:
    trace = trace_field_dependent_rays(
        values,
        states,
        rig,
        gradient_mode=gradient_mode,
        difference_step=difference_step,
        refractivity_scale=scale,
        step_count=step_count,
        create_graph=False,
    )
    high = path_integrated_deflection(
        values,
        trace,
        gradient_mode=gradient_mode,
        difference_step=difference_step,
        refractivity_scale=scale,
        create_graph=False,
        detach_path=False,
    )
    medium = straight_ray_deflection(
        values,
        states,
        rig,
        gradient_mode=gradient_mode,
        difference_step=difference_step,
        refractivity_scale=scale,
        step_count=step_count,
        create_graph=False,
    )
    return high - medium


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate = candidate.detach()
    reference = reference.detach()
    return float(
        torch.linalg.vector_norm(candidate - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    )


def test_affine_variational_rk4_matches_constant_forcing_solution() -> None:
    ray_count = 2
    step_count = 10
    step_size = 0.07
    direction = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    tangent_force = torch.tensor(
        [[0.0, 0.12, -0.07], [0.08, -0.05, 0.0]],
        dtype=torch.float64,
    )
    forcing = tangent_force[:, None, :].expand(-1, 2 * step_count + 1, -1).clone()
    zeros = torch.zeros(
        (ray_count, 2 * step_count + 1, 3, 3),
        dtype=torch.float64,
    )
    delta_r, delta_d = integrate_affine_variational_rk4(
        forcing,
        zeros,
        zeros,
        direction,
        step_size=step_size,
    )
    length = step_count * step_size
    assert torch.allclose(delta_d[:, -1], length * tangent_force, atol=2e-15, rtol=2e-15)
    assert torch.allclose(
        delta_r[:, -1],
        0.5 * length**2 * tangent_force,
        atol=2e-15,
        rtol=2e-15,
    )
    assert torch.max(torch.abs(torch.sum(delta_d * direction[:, None], dim=-1))) < 2e-15


def test_constant_index_has_zero_state_prediction_and_risk() -> None:
    values = torch.full((9, 9, 9), 0.27, dtype=torch.float64)
    result = predict_trajectory_variational_residual(
        values,
        sample_pupil_sobol(7, seed=101),
        _rig(),
        refractivity_scale=2e-3,
        step_count=12,
    )
    assert torch.count_nonzero(result.delta_positions) == 0
    assert torch.count_nonzero(result.delta_directions) == 0
    assert torch.count_nonzero(result.residual_prediction_uv) == 0
    assert torch.count_nonzero(result.risk_norm) == 0
    assert torch.all(result.valid_mask)
    assert result.failure_reasons == ("ok",) * 7


def test_batched_float64_shapes_and_risk_definition_are_explicit() -> None:
    result = predict_trajectory_variational_residual(
        _smooth_grid(),
        sample_pupil_sobol(6, seed=103),
        _rig(),
        refractivity_scale=8e-4,
        step_count=14,
    )
    assert result.residual_prediction_xyz.shape == (6, 3)
    assert result.residual_prediction_uv.shape == (6, 2)
    assert result.delta_positions.shape == (6, 15, 3)
    assert result.delta_directions.shape == (6, 15, 3)
    assert result.risk_norm.shape == (6,)
    assert result.coefficient_sample_count == 29
    assert result.residual_prediction_uv.dtype == torch.float64
    assert result.risk_norm.dtype == torch.float64
    assert torch.all(result.valid_mask)
    assert torch.allclose(
        result.risk_norm,
        torch.linalg.vector_norm(result.residual_prediction_uv, dim=-1),
        atol=0.0,
        rtol=0.0,
    )
    assert "H-M" in result.risk_definition


def test_field_and_route_are_always_stop_gradient() -> None:
    values = _smooth_grid().requires_grad_(True)
    states = sample_pupil_sobol(3, seed=107).requires_grad_(True)
    linearization = linearize_straight_medium_path(
        values,
        states,
        _rig(),
        refractivity_scale=6e-4,
        step_count=8,
    )
    prediction = predict_trajectory_variational_residual(
        values,
        states,
        _rig(),
        refractivity_scale=6e-4,
        step_count=8,
    )
    assert linearization.stop_gradient_applied
    assert prediction.stop_gradient_applied
    for tensor in (
        linearization.forcing,
        linearization.position_jacobian,
        linearization.direction_jacobian,
        prediction.residual_prediction_uv,
        prediction.risk_norm,
        prediction.delta_positions,
        prediction.delta_directions,
    ):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_spatial_gradient_and_hessian_match_centered_finite_differences() -> None:
    values = _smooth_grid(size=11)
    scale = 7e-4
    linearization = linearize_straight_medium_path(
        values,
        sample_pupil_sobol(3, seed=109),
        _rig(),
        refractivity_scale=scale,
        step_count=10,
    )
    points = linearization.positions.reshape(-1, 3)
    sizes_xyz = torch.tensor(
        [values.shape[2] - 1, values.shape[1] - 1, values.shape[0] - 1],
        dtype=torch.float64,
    )
    fractions = torch.remainder(0.5 * (points + 1.0) * sizes_xyz, 1.0)
    away_from_cell_knots = torch.all((fractions > 0.12) & (fractions < 0.88), dim=1)
    selected = torch.nonzero(away_from_cell_knots, as_tuple=False).flatten()[:8]
    assert selected.numel() >= 4
    points = points.index_select(0, selected)
    expected_gradient = linearization.gradient_n.reshape(-1, 3).index_select(0, selected)
    expected_hessian = linearization.hessian_n.reshape(-1, 3, 3).index_select(0, selected)
    epsilon = 2e-5

    gradient_columns = []
    hessian_columns = []
    for axis in range(3):
        offset = torch.zeros_like(points)
        offset[:, axis] = epsilon
        right_value = smoothstep_grid_field(values, points + offset)
        left_value = smoothstep_grid_field(values, points - offset)
        gradient_columns.append(scale * (right_value - left_value) / (2.0 * epsilon))

        right_points = (points + offset).detach().requires_grad_(True)
        left_points = (points - offset).detach().requires_grad_(True)
        right_field = smoothstep_grid_field(values, right_points)
        left_field = smoothstep_grid_field(values, left_points)
        right_gradient = torch.autograd.grad(right_field.sum(), right_points)[0]
        left_gradient = torch.autograd.grad(left_field.sum(), left_points)[0]
        hessian_columns.append(
            scale * (right_gradient - left_gradient) / (2.0 * epsilon)
        )
    centered_gradient = torch.stack(gradient_columns, dim=-1)
    centered_hessian = torch.stack(hessian_columns, dim=-1)
    assert _relative_l2(centered_gradient, expected_gradient) < 2e-7
    assert _relative_l2(centered_hessian, expected_hessian) < 3e-6
    assert torch.max(torch.abs(expected_hessian - expected_hessian.transpose(-1, -2))) < 2e-13


def test_position_and_tangent_direction_jacobians_match_rhs_finite_difference() -> None:
    values = _smooth_grid(size=11)
    scale = 8e-4
    linearization = linearize_straight_medium_path(
        values,
        sample_pupil_sobol(2, seed=111),
        _rig(),
        refractivity_scale=scale,
        step_count=10,
    )
    points = linearization.positions[0]
    sizes_xyz = torch.tensor(
        [values.shape[2] - 1, values.shape[1] - 1, values.shape[0] - 1],
        dtype=torch.float64,
    )
    fractions = torch.remainder(0.5 * (points + 1.0) * sizes_xyz, 1.0)
    candidates = torch.nonzero(
        torch.all((fractions > 0.12) & (fractions < 0.88), dim=1),
        as_tuple=False,
    ).flatten()
    assert candidates.numel() > 0
    sample_index = int(candidates[len(candidates) // 2])
    point = points[sample_index]
    direction = linearization.direction[0]
    projector = torch.eye(3, dtype=torch.float64) - torch.outer(direction, direction)

    def ray_rhs(query_point: torch.Tensor, query_direction: torch.Tensor) -> torch.Tensor:
        coordinate = query_point[None].detach().requires_grad_(True)
        field = smoothstep_grid_field(values, coordinate)
        gradient = torch.autograd.grad(field.sum(), coordinate)[0][0]
        unit_direction = query_direction / torch.linalg.vector_norm(query_direction)
        gradient_n = scale * gradient
        index = 1.0 + scale * field[0]
        return (
            gradient_n - unit_direction * torch.dot(unit_direction, gradient_n)
        ) / index

    position_direction = torch.tensor([0.31, -0.47, 0.23], dtype=torch.float64)
    position_direction /= torch.linalg.vector_norm(position_direction)
    raw_direction = torch.tensor([-0.29, 0.41, 0.37], dtype=torch.float64)
    tangent_direction = projector @ raw_direction
    tangent_direction /= torch.linalg.vector_norm(tangent_direction)
    epsilon = 2e-6

    finite_position = (
        ray_rhs(point + epsilon * position_direction, direction)
        - ray_rhs(point - epsilon * position_direction, direction)
    ) / (2.0 * epsilon)
    finite_direction = projector @ (
        ray_rhs(point, direction + epsilon * tangent_direction)
        - ray_rhs(point, direction - epsilon * tangent_direction)
    ) / (2.0 * epsilon)
    predicted_position = (
        linearization.position_jacobian[0, sample_index] @ position_direction
    )
    predicted_direction = (
        linearization.direction_jacobian[0, sample_index] @ tangent_direction
    )
    assert _relative_l2(predicted_position, finite_position) < 2e-7
    assert _relative_l2(predicted_direction, finite_direction) < 2e-7


def test_weak_affine_like_field_has_quadratic_trajectory_residual_scaling() -> None:
    axis = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64)
    _, yy, _ = torch.meshgrid(axis, axis, axis, indexing="ij")
    values = yy.clone()
    states = torch.tensor([[0.17, 0.29], [0.63, 0.81]], dtype=torch.float64)
    rig = _rig(
        view_angle_degrees=0.0,
        detector_u=0.043,
        detector_z=0.017,
        aperture_radius=0.0,
        cone_u=0.0,
        cone_z=0.0,
    )
    coarse_scale = 8e-4
    coarse = predict_trajectory_variational_residual(
        values,
        states,
        rig,
        refractivity_scale=coarse_scale,
        step_count=24,
    )
    fine = predict_trajectory_variational_residual(
        values,
        states,
        rig,
        refractivity_scale=0.5 * coarse_scale,
        step_count=24,
    )
    ratio = coarse.risk_norm / fine.risk_norm
    assert torch.all(coarse.valid_mask & fine.valid_mask)
    assert torch.all((ratio > 3.96) & (ratio < 4.04))
    assert torch.all(coarse.residual_prediction_uv[:, 0] < 0.0)
    assert torch.max(torch.abs(coarse.residual_prediction_uv[:, 1])) < 1e-16


@pytest.mark.parametrize("gradient_mode", ["automatic", "central"])
def test_directional_prediction_matches_weak_nonlinear_h_minus_m(
    gradient_mode: str,
) -> None:
    values = _smooth_grid(size=15)
    states = sample_pupil_sobol(5, seed=113)
    rig = _rig()
    scale = 1e-3
    step_count = 48
    prediction = predict_trajectory_variational_residual(
        values,
        states,
        rig,
        refractivity_scale=scale,
        step_count=step_count,
    )
    reference = _nonlinear_high_minus_medium(
        values,
        states,
        rig,
        scale=scale,
        step_count=step_count,
        gradient_mode=gradient_mode,
    )
    dot = torch.sum(prediction.residual_prediction_uv * reference)
    cosine = dot / (
        torch.linalg.vector_norm(prediction.residual_prediction_uv)
        * torch.linalg.vector_norm(reference)
    ).clamp_min(1e-30)
    assert torch.all(prediction.valid_mask)
    assert float(cosine) > 0.99999
    assert _relative_l2(prediction.residual_prediction_uv, reference) < 0.005


def test_predictor_refines_with_more_straight_path_steps() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(4, seed=127)
    outputs = []
    for steps in (8, 16, 32, 64):
        outputs.append(
            predict_trajectory_variational_residual(
                values,
                states,
                _rig(),
                refractivity_scale=8e-4,
                step_count=steps,
            ).residual_prediction_uv
        )
    assert _relative_l2(outputs[2], outputs[3]) < _relative_l2(outputs[0], outputs[3])
    assert _relative_l2(outputs[2], outputs[3]) < 0.05


def test_linearization_radius_excess_sets_infinite_risk() -> None:
    result = predict_trajectory_variational_residual(
        _smooth_grid(),
        sample_pupil_sobol(4, seed=131),
        _rig(),
        refractivity_scale=2e-3,
        step_count=16,
        max_position_perturbation=1e-14,
        max_direction_perturbation=1e-14,
    )
    assert not torch.any(result.valid_mask)
    assert torch.all(torch.isinf(result.risk_norm))
    assert all("linearization_radius" in reason for reason in result.failure_reasons)


def test_rig_outside_domain_fails_closed() -> None:
    with pytest.raises(TrajectoryPredictorDomainError):
        predict_trajectory_variational_residual(
            _smooth_grid(),
            sample_pupil_sobol(2, seed=137),
            _rig(detector_u=0.92, path_half_length=0.94),
            refractivity_scale=5e-4,
            step_count=8,
            domain_margin=0.02,
        )


def test_nonpositive_refractive_index_fails_closed() -> None:
    with pytest.raises(TrajectoryPredictorDomainError):
        predict_trajectory_variational_residual(
            torch.full((7, 7, 7), -100.0, dtype=torch.float64),
            sample_pupil_sobol(2, seed=139),
            _rig(),
            refractivity_scale=0.02,
            step_count=8,
            refractive_index_floor=0.5,
        )


@pytest.mark.parametrize(
    ("values", "states", "kwargs"),
    [
        (
            torch.zeros((7, 7, 7), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float64),
            {},
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float32),
            {},
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.tensor([[0.2, 1.2]], dtype=torch.float64),
            {},
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            {"step_count": 1},
        ),
        (
            torch.zeros((7, 7, 7), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            {"refractivity_scale": float("nan")},
        ),
    ],
)
def test_invalid_predictor_inputs_are_rejected(
    values: torch.Tensor,
    states: torch.Tensor,
    kwargs: dict[str, float | int],
) -> None:
    parameters: dict[str, float | int] = {
        "refractivity_scale": 5e-4,
        "step_count": 8,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        predict_trajectory_variational_residual(
            values,
            states,
            _rig(),
            **parameters,
        )


def test_invalid_low_level_direction_and_coefficients_are_rejected() -> None:
    forcing = torch.zeros((2, 9, 3), dtype=torch.float64)
    matrices = torch.zeros((2, 9, 3, 3), dtype=torch.float64)
    bad_direction = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="unit length"):
        integrate_affine_variational_rk4(
            forcing,
            matrices,
            matrices,
            bad_direction,
            step_size=0.1,
        )
    with pytest.raises(ValueError, match="shape"):
        integrate_affine_variational_rk4(
            forcing,
            matrices[:, :-1],
            matrices,
            torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
            step_size=0.1,
        )
