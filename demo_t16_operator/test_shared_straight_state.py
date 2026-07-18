from __future__ import annotations

import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import (
    SyntheticRayRig,
    central_difference_spatial_gradient,
    smoothstep_grid_field,
)
from demo_t16_operator.field_dependent_ray import (
    RayDomainError,
    initial_pupil_rays,
    sample_pupil_sobol,
    straight_ray_deflection,
)
import demo_t16_operator.shared_straight_state as shared
from demo_t16_operator.shared_straight_state import (
    StraightPathState,
    build_straight_path_state,
)


def _grid(size: int = 13) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return (
        -0.19 * torch.exp(-((xx / 0.43) ** 2 + (yy / 0.31) ** 2 + (zz / 0.37) ** 2))
        + 0.018 * torch.sin(2.3 * xx + 0.4 * yy) * torch.cos(1.7 * zz)
    )


def _rig(**overrides: float | str) -> SyntheticRayRig:
    parameters: dict[str, float | str] = {
        "rig_id": "shared-straight-test",
        "view_angle_degrees": 29.0,
        "detector_u": 0.035,
        "detector_z": -0.025,
        "aperture_radius": 0.03,
        "path_half_length": 0.64,
        "cone_u": 0.018,
        "cone_z": 0.012,
        "bend": 0.0,
    }
    parameters.update(overrides)
    return SyntheticRayRig(**parameters)


def _build(
    values: torch.Tensor | None = None,
    states: torch.Tensor | None = None,
    **overrides: object,
) -> StraightPathState:
    arguments: dict[str, object] = {
        "values_zyx": _grid() if values is None else values,
        "pupil_states": sample_pupil_sobol(5, seed=811) if states is None else states,
        "rig": _rig(),
        "difference_step": 2e-3,
        "refractivity_scale": 4e-3,
        "step_count": 12,
        "frustum_half_width_u": 0.03,
        "frustum_half_width_v": 0.02,
    }
    arguments.update(overrides)
    return build_straight_path_state(**arguments)  # type: ignore[arg-type]


def test_shared_state_matches_existing_medium_route_and_primitives() -> None:
    values = _grid()
    states = sample_pupil_sobol(7, seed=823)
    state = _build(values, states, step_count=15)
    expected_output = straight_ray_deflection(
        values,
        states,
        _rig(),
        gradient_mode="central",
        difference_step=2e-3,
        refractivity_scale=4e-3,
        step_count=15,
        create_graph=True,
    )
    expected_scalar = smoothstep_grid_field(values, state.positions.reshape(-1, 3)).reshape(
        7, 15
    )
    expected_gradient = central_difference_spatial_gradient(
        values,
        state.positions.reshape(-1, 3),
        step=2e-3,
    ).reshape(7, 15, 3)

    assert torch.allclose(state.projected_outputs, expected_output, atol=2e-15, rtol=2e-13)
    assert torch.allclose(state.scalar_values, expected_scalar, atol=0.0, rtol=0.0)
    assert torch.allclose(
        state.central_difference_gradients,
        expected_gradient,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        state.projected_outputs,
        torch.sum(state.projected_integrands, dim=1) * state.step_size,
    )


def test_positions_cells_and_raw_margins_have_declared_geometry() -> None:
    state = _build(step_count=9)
    start, direction, _, _ = initial_pupil_rays(
        sample_pupil_sobol(5, seed=811),
        _rig(),
    )
    distance = (
        torch.arange(9, dtype=torch.float64) + 0.5
    ) * state.step_size
    expected_positions = start[:, None, :] + distance[None, :, None] * direction[:, None, :]
    assert torch.allclose(state.positions, expected_positions, atol=0.0, rtol=0.0)
    assert state.cell_ids.shape == (5, 9, 3)
    assert state.cell_ids.dtype == torch.long
    assert torch.all(state.cell_ids >= 0)
    assert torch.all(state.cell_ids <= 11)
    assert torch.allclose(
        state.domain_margins,
        1.0 - torch.amax(torch.abs(state.positions), dim=-1),
    )
    assert torch.allclose(
        state.stencil_domain_margins,
        state.domain_margins - state.difference_step,
    )
    assert torch.all(state.frustum_margins[..., 0] == 0.03)
    assert torch.all(state.frustum_margins[..., 1] == 0.02)
    assert torch.equal(
        state.minimum_frustum_margin_per_ray,
        torch.full((5,), 0.02, dtype=torch.float64),
    )


def test_cell_ids_use_xyz_order_for_anisotropic_zyx_grid() -> None:
    values = torch.zeros((7, 9, 11), dtype=torch.float64)
    states = torch.tensor([[0.3, 0.4]], dtype=torch.float64)
    state = _build(values, states, step_count=4)
    sizes_xyz = torch.tensor([10.0, 8.0, 6.0], dtype=torch.float64)
    maximum_xyz = torch.tensor([9, 7, 5], dtype=torch.long)
    expected = torch.floor(0.5 * (state.positions + 1.0) * sizes_xyz).to(torch.long)
    expected = torch.minimum(
        torch.maximum(expected, torch.zeros_like(expected)),
        maximum_xyz,
    )
    assert state.grid_shape_zyx == (7, 9, 11)
    assert torch.equal(state.cell_ids, expected)


def test_builder_uses_one_vectorized_interpolation_call_and_accounts_every_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = shared.smoothstep_grid_field

    def counted(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        calls.append(len(points))
        return original(values, points)

    monkeypatch.setattr(shared, "smoothstep_grid_field", counted)
    state = _build(step_count=11)
    accounting = state.query_accounting
    midpoint_count = 5 * 11
    assert calls == [7 * midpoint_count]
    assert accounting.ray_count == 5
    assert accounting.step_count == 11
    assert accounting.midpoint_count == midpoint_count
    assert accounting.scalar_value_point_queries == midpoint_count
    assert accounting.central_difference_point_queries == 6 * midpoint_count
    assert accounting.total_field_point_queries == 7 * midpoint_count
    assert accounting.total_query_count == 7 * midpoint_count
    assert accounting.field_queries_per_midpoint == 7
    assert accounting.vectorized_interpolation_calls == 1
    assert accounting.projected_output_additional_point_queries == 0
    assert accounting.cell_and_margin_additional_point_queries == 0
    assert accounting.as_dict()["query_unit"] == "scalar_grid_evaluation_at_one_coordinate"


def test_float64_vjp_and_jvp_match_existing_medium_and_finite_difference() -> None:
    values = _grid()
    states = torch.tensor(
        [[0.17, 0.23], [0.38, 0.61], [0.72, 0.44]],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(829)
    tangent = torch.randn(values.shape, generator=generator, dtype=torch.float64)
    tangent = tangent / torch.linalg.vector_norm(tangent)

    def shared_forward(grid: torch.Tensor) -> torch.Tensor:
        return _build(grid, states, step_count=10).projected_outputs

    def existing_forward(grid: torch.Tensor) -> torch.Tensor:
        return straight_ray_deflection(
            grid,
            states,
            _rig(),
            gradient_mode="central",
            difference_step=2e-3,
            refractivity_scale=4e-3,
            step_count=10,
            create_graph=True,
        )

    variable = values.detach().clone().requires_grad_(True)
    output = shared_forward(variable)
    cotangent = torch.tensor(
        [[0.4, -0.2], [-0.1, 0.7], [0.3, 0.5]],
        dtype=torch.float64,
    )
    vjp = torch.autograd.grad(torch.sum(output * cotangent), variable)[0]
    _, jvp = torch.autograd.functional.jvp(shared_forward, values, tangent)
    _, existing_jvp = torch.autograd.functional.jvp(existing_forward, values, tangent)
    epsilon = 2e-5
    finite_difference = (
        shared_forward(values + epsilon * tangent)
        - shared_forward(values - epsilon * tangent)
    ) / (2.0 * epsilon)

    assert output.dtype == torch.float64
    assert output.requires_grad
    assert torch.all(torch.isfinite(vjp))
    assert torch.linalg.vector_norm(vjp) > 0.0
    assert torch.allclose(jvp, existing_jvp, atol=2e-14, rtol=2e-11)
    assert torch.allclose(jvp, finite_difference, atol=2e-10, rtol=2e-6)
    assert torch.allclose(
        torch.sum(vjp * tangent),
        torch.sum(cotangent * jvp),
        atol=2e-12,
        rtol=2e-11,
    )


def test_projected_output_is_differentiable_with_respect_to_pupil_state() -> None:
    states = torch.tensor(
        [[0.19, 0.27], [0.43, 0.58], [0.76, 0.81]],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = _build(states=states, step_count=10).projected_outputs
    gradient = torch.autograd.grad(output.square().sum(), states)[0]
    assert output.dtype == torch.float64
    assert output.requires_grad
    assert torch.all(torch.isfinite(gradient))
    assert torch.linalg.vector_norm(gradient) > 0.0


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"difference_step": 0.0}, ValueError),
        ({"difference_step": 0.25}, ValueError),
        ({"refractivity_scale": 0.0}, ValueError),
        ({"step_count": 1}, ValueError),
        ({"step_count": 2.5}, TypeError),
        ({"frustum_half_width_u": 0.0}, ValueError),
        ({"frustum_half_width_v": float("nan")}, ValueError),
    ],
)
def test_invalid_scalar_parameters_fail_closed(
    override: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _build(**override)


def test_invalid_grid_and_pupil_inputs_fail_closed() -> None:
    with pytest.raises(TypeError, match="float64"):
        _build(values=_grid().float())
    with pytest.raises(TypeError, match="float64"):
        _build(states=sample_pupil_sobol(3, seed=839).float())
    with pytest.raises(ValueError, match="shape"):
        _build(values=torch.ones((5, 5), dtype=torch.float64))
    invalid_grid = _grid()
    invalid_grid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        _build(values=invalid_grid)
    invalid_states = sample_pupil_sobol(3, seed=853)
    invalid_states[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _build(states=invalid_states)


def test_stencil_escape_and_invalid_refractive_index_fail_closed() -> None:
    with pytest.raises((RayDomainError, ValueError)):
        _build(
            rig=_rig(detector_u=0.86, path_half_length=0.88),
            difference_step=0.04,
        )
    with pytest.raises(RayDomainError, match="refractive index"):
        _build(
            values=torch.full((7, 7, 7), -1.0, dtype=torch.float64),
            refractivity_scale=0.6,
        )


def test_classmethod_build_matches_free_function() -> None:
    values = _grid()
    states = sample_pupil_sobol(4, seed=857)
    keyword = {
        "difference_step": 2e-3,
        "refractivity_scale": 4e-3,
        "step_count": 8,
        "frustum_half_width_u": 0.03,
        "frustum_half_width_v": 0.02,
    }
    direct = build_straight_path_state(values, states, _rig(), **keyword)
    classmethod = StraightPathState.build(values, states, _rig(), **keyword)
    assert torch.equal(direct.projected_outputs, classmethod.projected_outputs)
    assert direct.query_accounting == classmethod.query_accounting
