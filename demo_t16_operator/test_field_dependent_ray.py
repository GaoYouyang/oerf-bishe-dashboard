from __future__ import annotations

import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import SyntheticRayRig
from demo_t16_operator.field_dependent_ray import (
    RayDomainError,
    exit_direction_deflection,
    path_integrated_deflection,
    path_topology_diagnostics,
    ray_momentum_balance,
    relative_l2,
    sample_pupil_sobol,
    straight_ray_deflection,
    trace_field_dependent_rays,
)


def _grid_from_function(size: int = 17) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return -torch.exp(-((xx / 0.42) ** 2 + (yy / 0.34) ** 2 + (zz / 0.38) ** 2))


def _rig(**overrides: float | str) -> SyntheticRayRig:
    values: dict[str, float | str] = {
        "rig_id": "ray-test",
        "view_angle_degrees": 27.0,
        "detector_u": 0.04,
        "detector_z": -0.03,
        "aperture_radius": 0.025,
        "path_half_length": 0.62,
        "cone_u": 0.02,
        "cone_z": 0.015,
        "bend": 0.0,
    }
    values.update(overrides)
    return SyntheticRayRig(**values)


def test_constant_index_keeps_rays_straight_and_deflection_zero() -> None:
    values = torch.zeros((9, 9, 9), dtype=torch.float64)
    states = sample_pupil_sobol(8, seed=11)
    trace = trace_field_dependent_rays(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=1e-3,
        refractivity_scale=3e-4,
        step_count=8,
        create_graph=False,
    )
    expected = trace.positions[:, :1] + (
        torch.arange(9, dtype=values.dtype)[None, :, None]
        * trace.step_size
        * trace.directions[:, :1]
    )
    assert torch.allclose(trace.positions, expected, atol=2e-14, rtol=2e-14)
    assert torch.max(torch.abs(exit_direction_deflection(trace))) < 1e-14
    assert trace.maximum_direction_norm_error < 2e-14


def test_nonuniform_index_bends_rays_and_rk4_refines() -> None:
    values = _grid_from_function()
    states = sample_pupil_sobol(6, seed=19)
    outputs = []
    for steps in (8, 16, 32, 64):
        trace = trace_field_dependent_rays(
            values,
            states,
            _rig(),
            gradient_mode="central",
            difference_step=2e-3,
            refractivity_scale=3e-3,
            step_count=steps,
            create_graph=False,
        )
        outputs.append(exit_direction_deflection(trace))
    assert torch.linalg.vector_norm(outputs[-1]) > 1e-6
    assert relative_l2(outputs[2], outputs[3]) < relative_l2(outputs[0], outputs[3])
    assert relative_l2(outputs[2], outputs[3]) < 0.03


def test_path_integral_matches_exit_direction_change_at_refined_step_count() -> None:
    values = _grid_from_function()
    states = sample_pupil_sobol(5, seed=23)
    trace = trace_field_dependent_rays(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=1e-3,
        refractivity_scale=1e-3,
        step_count=48,
        create_graph=True,
    )
    integrated = path_integrated_deflection(
        values,
        trace,
        gradient_mode="central",
        difference_step=1e-3,
        refractivity_scale=1e-3,
        create_graph=True,
        detach_path=False,
    )
    assert relative_l2(integrated, exit_direction_deflection(trace)) < 0.03


def test_endpoint_momentum_balance_matches_integrated_index_gradient() -> None:
    values = _grid_from_function()
    states = sample_pupil_sobol(5, seed=25)
    trace = trace_field_dependent_rays(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=1e-3,
        refractivity_scale=1e-3,
        step_count=96,
        create_graph=False,
    )
    endpoint, integrated = ray_momentum_balance(
        values,
        trace,
        gradient_mode="central",
        difference_step=1e-3,
        refractivity_scale=1e-3,
        create_graph=False,
    )
    assert relative_l2(integrated, endpoint) < 0.02


def test_full_trajectory_jvp_matches_fixed_state_finite_difference() -> None:
    values = _grid_from_function()
    states = sample_pupil_sobol(3, seed=29)
    generator = torch.Generator().manual_seed(31)
    direction = torch.randn(values.shape, generator=generator, dtype=values.dtype)
    direction = direction / torch.linalg.vector_norm(direction)
    epsilon = 2e-5

    def forward(grid: torch.Tensor) -> torch.Tensor:
        trace = trace_field_dependent_rays(
            grid,
            states,
            _rig(),
            gradient_mode="central",
            difference_step=2e-3,
            refractivity_scale=4e-3,
            step_count=10,
            create_graph=True,
        )
        return path_integrated_deflection(
            grid,
            trace,
            gradient_mode="central",
            difference_step=2e-3,
            refractivity_scale=4e-3,
            create_graph=True,
            detach_path=False,
        ).mean(dim=0)

    variable = values.detach().clone().requires_grad_(True)
    output = forward(variable)
    jvp = torch.stack(
        [
            torch.sum(
                torch.autograd.grad(output[index], variable, retain_graph=True)[0]
                * direction
            )
            for index in range(len(output))
        ]
    )
    finite_difference = (forward(values + epsilon * direction) - forward(values - epsilon * direction)) / (
        2.0 * epsilon
    )
    assert relative_l2(jvp, finite_difference) < 3e-5

    cotangent = torch.tensor([0.6, -0.8], dtype=values.dtype)
    vjp = torch.autograd.grad(torch.sum(output * cotangent), variable)[0]
    lhs = torch.sum(vjp * direction)
    rhs = torch.sum(cotangent * jvp)
    assert float(torch.abs(lhs - rhs)) < 2e-12


def test_frozen_path_vjp_is_a_distinct_direct_effect_control() -> None:
    values = _grid_from_function().requires_grad_(True)
    states = sample_pupil_sobol(4, seed=37)
    trace = trace_field_dependent_rays(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=2e-3,
        refractivity_scale=3e-2,
        step_count=12,
        create_graph=True,
    )
    full = path_integrated_deflection(
        values,
        trace,
        gradient_mode="central",
        difference_step=2e-3,
        refractivity_scale=3e-2,
        create_graph=True,
        detach_path=False,
    ).sum()
    frozen = path_integrated_deflection(
        values,
        trace,
        gradient_mode="central",
        difference_step=2e-3,
        refractivity_scale=3e-2,
        create_graph=True,
        detach_path=True,
    ).sum()
    full_vjp = torch.autograd.grad(full, values, retain_graph=True)[0]
    frozen_vjp = torch.autograd.grad(frozen, values)[0]
    difference = torch.linalg.vector_norm(full_vjp - frozen_vjp)
    assert difference > 1e-7
    assert difference / torch.linalg.vector_norm(full_vjp) > 1e-4


def test_straight_automatic_route_supports_parameter_vjp() -> None:
    values = _grid_from_function().requires_grad_(True)
    output = straight_ray_deflection(
        values,
        sample_pupil_sobol(4, seed=41),
        _rig(),
        gradient_mode="automatic",
        difference_step=1e-3,
        refractivity_scale=3e-4,
        step_count=8,
        create_graph=True,
    )
    gradient = torch.autograd.grad(output.square().sum(), values)[0]
    assert output.shape == (4, 2)
    assert torch.all(torch.isfinite(gradient))
    assert torch.linalg.vector_norm(gradient) > 0.0


def test_topology_diagnostic_reports_support_crossings_without_frustum_escape() -> None:
    values = _grid_from_function()
    trace = trace_field_dependent_rays(
        values,
        sample_pupil_sobol(5, seed=43),
        _rig(),
        gradient_mode="central",
        difference_step=1e-3,
        refractivity_scale=3e-3,
        step_count=24,
        create_graph=False,
    )
    diagnostic = path_topology_diagnostics(
        values,
        trace,
        support_threshold=0.1,
        frustum_half_width_u=0.02,
        frustum_half_width_v=0.02,
    )
    assert len(diagnostic.support_crossings_per_ray) == 5
    assert any(count > 0 for count in diagnostic.support_crossings_per_ray)
    assert not any(diagnostic.frustum_violations_per_ray)
    assert diagnostic.minimum_domain_margin > 0.1


def test_trace_fails_closed_when_rig_leaves_stencil_domain() -> None:
    values = _grid_from_function()
    with pytest.raises((RayDomainError, ValueError)):
        trace_field_dependent_rays(
            values,
            sample_pupil_sobol(2, seed=47),
            _rig(detector_u=0.88, path_half_length=0.9),
            gradient_mode="central",
            difference_step=0.02,
            refractivity_scale=3e-4,
            step_count=8,
            create_graph=False,
        )


@pytest.mark.parametrize("mode", ["bad", "", "CENTRAL"])
def test_invalid_gradient_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError):
        trace_field_dependent_rays(
            _grid_from_function(),
            sample_pupil_sobol(2, seed=53),
            _rig(),
            gradient_mode=mode,
            difference_step=1e-3,
            refractivity_scale=3e-4,
            step_count=8,
            create_graph=False,
        )
