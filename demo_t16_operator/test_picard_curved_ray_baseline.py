from __future__ import annotations

import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import SyntheticRayRig
from demo_t16_operator.field_dependent_ray import (
    path_integrated_deflection,
    sample_pupil_sobol,
    trace_field_dependent_rays,
)
import demo_t16_operator.picard_curved_ray_baseline as picard
from demo_t16_operator.picard_curved_ray_baseline import (
    PicardRayDomainError,
    trace_picard_curved_rays,
)


def _rig(**overrides: float | str) -> SyntheticRayRig:
    values: dict[str, float | str] = {
        "rig_id": "picard-test",
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


def _smooth_grid(size: int = 17) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return -torch.exp(
        -(
            (xx / 0.42) ** 2
            + (yy / 0.34) ** 2
            + (zz / 0.38) ** 2
        )
    )


def _run(
    values: torch.Tensor | None = None,
    states: torch.Tensor | None = None,
    **overrides: float | int | SyntheticRayRig,
) -> picard.PicardCurvedRayResult:
    parameters: dict[str, float | int | SyntheticRayRig] = {
        "rig": _rig(),
        "difference_step": 1e-3,
        "refractivity_scale": 3e-3,
        "step_count": 32,
        "sweep_count": 2,
    }
    parameters.update(overrides)
    rig = parameters.pop("rig")
    assert isinstance(rig, SyntheticRayRig)
    return trace_picard_curved_rays(
        _smooth_grid() if values is None else values,
        sample_pupil_sobol(6, seed=701) if states is None else states,
        rig,
        **parameters,
    )


def _exact_high(
    values: torch.Tensor,
    states: torch.Tensor,
    rig: SyntheticRayRig,
    *,
    scale: float,
    step_count: int,
) -> torch.Tensor:
    """Test-only oracle; it is never supplied to the Picard implementation."""

    trace = trace_field_dependent_rays(
        values,
        states,
        rig,
        gradient_mode="central",
        difference_step=1e-3,
        refractivity_scale=scale,
        step_count=step_count,
        create_graph=False,
    )
    return path_integrated_deflection(
        values,
        trace,
        gradient_mode="central",
        difference_step=1e-3,
        refractivity_scale=scale,
        create_graph=False,
        detach_path=False,
    )


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(candidate - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    )


@pytest.mark.parametrize("sweep_count", [1, 2])
def test_zero_field_keeps_the_full_trajectory_straight_and_output_zero(
    sweep_count: int,
) -> None:
    ray_count = 5
    step_count = 12
    values = torch.zeros((9, 9, 9), dtype=torch.float64)
    result = _run(
        values=values,
        states=sample_pupil_sobol(ray_count, seed=703),
        step_count=step_count,
        sweep_count=sweep_count,
    )

    assert torch.count_nonzero(result.detector_plane_deflection) == 0
    assert torch.count_nonzero(result.exit_direction_deflection) == 0
    assert torch.allclose(
        result.position_history,
        result.position_history[:1].expand_as(result.position_history),
        atol=2e-15,
        rtol=2e-15,
    )
    assert torch.equal(
        result.direction_history,
        result.direction_history[:1].expand_as(result.direction_history),
    )
    assert torch.count_nonzero(result.curvature_history) == 0
    assert torch.all(result.valid_mask)
    assert result.failure_reasons == ("ok",) * ray_count
    assert result.detector_plane_deflection.dtype == torch.float64
    assert result.position_history.shape == (
        sweep_count + 1,
        ray_count,
        step_count + 1,
        3,
    )


def test_one_and_two_sweeps_preserve_the_declared_frozen_history_contract() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(6, seed=709)
    one = _run(
        values=values,
        states=states,
        refractivity_scale=3e-2,
        sweep_count=1,
    )
    two = _run(
        values=values,
        states=states,
        refractivity_scale=3e-2,
        sweep_count=2,
    )

    assert torch.equal(two.position_history[0], one.position_history[0])
    assert torch.equal(two.direction_history[0], one.direction_history[0])
    assert torch.equal(two.position_history[1], one.positions)
    assert torch.equal(two.direction_history[1], one.directions)
    assert torch.equal(two.curvature_history[0], one.curvature_history[0])
    assert torch.equal(two.curvature_history[1], one.output_curvature)
    assert torch.equal(
        two.refractive_index_history[1],
        one.output_refractive_index,
    )
    assert not torch.equal(two.detector_plane_deflection, one.detector_plane_deflection)
    assert torch.all(
        two.maximum_position_change_per_sweep[1]
        < two.maximum_position_change_per_sweep[0]
    )
    assert torch.all(
        two.maximum_direction_change_per_sweep[1]
        < two.maximum_direction_change_per_sweep[0]
    )
    assert two.query_accounting.total_field_point_queries == (
        3 * one.query_accounting.total_field_point_queries // 2
    )
    assert two.query_accounting.exact_high_calls == 0
    assert "frozen-midpoint" in two.update_scheme


def test_two_sweeps_are_close_to_weak_field_exact_high_without_oracle_input() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(6, seed=719)
    rig = _rig()
    scale = 1e-3
    step_count = 48
    one = _run(
        values=values,
        states=states,
        rig=rig,
        refractivity_scale=scale,
        step_count=step_count,
        sweep_count=1,
    )
    two = _run(
        values=values,
        states=states,
        rig=rig,
        refractivity_scale=scale,
        step_count=step_count,
        sweep_count=2,
    )
    reference = _exact_high(
        values,
        states,
        rig,
        scale=scale,
        step_count=step_count,
    )
    one_error = _relative_l2(one.detector_plane_deflection, reference)
    two_error = _relative_l2(two.detector_plane_deflection, reference)

    assert one_error < 1e-4
    assert two_error < 1e-4
    assert two_error <= 1.05 * one_error
    assert two.query_accounting.exact_high_calls == 0
    assert two.query_accounting.output_additional_field_point_queries == (
        7 * len(states) * step_count
    )


def test_second_sweep_materially_improves_a_stronger_curved_field_case() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(6, seed=727)
    rig = _rig()
    scale = 3e-2
    step_count = 32
    reference = _exact_high(
        values,
        states,
        rig,
        scale=scale,
        step_count=step_count,
    )
    one = _run(
        values=values,
        states=states,
        rig=rig,
        refractivity_scale=scale,
        step_count=step_count,
        sweep_count=1,
    )
    two = _run(
        values=values,
        states=states,
        rig=rig,
        refractivity_scale=scale,
        step_count=step_count,
        sweep_count=2,
    )
    one_error = _relative_l2(one.detector_plane_deflection, reference)
    two_error = _relative_l2(two.detector_plane_deflection, reference)

    assert one_error < 1e-3
    assert two_error < 1e-3
    assert two_error < 0.7 * one_error


def test_query_accounting_matches_one_batched_seven_point_bundle_per_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = picard.smoothstep_grid_field

    def counted(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        calls.append(len(points))
        return original(values, points)

    monkeypatch.setattr(picard, "smoothstep_grid_field", counted)
    result = _run(
        states=sample_pupil_sobol(4, seed=733),
        step_count=11,
        sweep_count=2,
    )
    accounting = result.query_accounting
    midpoint_count = 4 * 11

    assert calls == [7 * midpoint_count, 7 * midpoint_count, 7 * midpoint_count]
    assert accounting.midpoint_curvature_evaluations_per_sweep == midpoint_count
    assert accounting.scalar_value_point_queries_per_sweep == midpoint_count
    assert accounting.central_difference_point_queries_per_sweep == 6 * midpoint_count
    assert accounting.total_field_point_queries_per_sweep == 7 * midpoint_count
    assert accounting.total_field_point_queries == 21 * midpoint_count
    assert accounting.vectorized_interpolation_calls == 3
    assert accounting.direction_updates == 2
    assert accounting.position_updates == 2
    assert accounting.output_additional_field_point_queries == 7 * midpoint_count
    assert accounting.as_dict()["query_unit"] == (
        "scalar_grid_evaluation_at_one_coordinate"
    )


def test_domain_and_refractive_index_fail_closed() -> None:
    with pytest.raises(PicardRayDomainError, match="initial calibrated straight ray"):
        _run(rig=_rig(detector_u=0.88, path_half_length=0.9))
    with pytest.raises(PicardRayDomainError, match="refractive index violates floor"):
        _run(
            values=torch.full((7, 7, 7), -100.0, dtype=torch.float64),
            refractivity_scale=0.02,
        )


def test_nonfinite_field_evaluation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite_samples(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (len(points),),
            torch.nan,
            dtype=torch.float64,
            device=points.device,
        )

    monkeypatch.setattr(picard, "smoothstep_grid_field", nonfinite_samples)
    with pytest.raises(PicardRayDomainError, match="field samples became non-finite"):
        _run()


def test_repeated_runs_are_bitwise_deterministic() -> None:
    values = _smooth_grid()
    states = sample_pupil_sobol(5, seed=739)
    first = _run(values=values, states=states)
    second = _run(values=values, states=states)

    for name in (
        "detector_plane_deflection",
        "exit_direction_deflection",
        "positions",
        "directions",
        "position_history",
        "direction_history",
        "curvature_history",
        "refractive_index_history",
        "output_curvature",
        "output_refractive_index",
        "minimum_domain_margin_per_ray",
        "minimum_stencil_margin_per_ray",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert first.query_accounting == second.query_accounting
    assert first.failure_reasons == second.failure_reasons


def test_invalid_grid_and_pupil_inputs_are_rejected() -> None:
    with pytest.raises(TypeError, match="values_zyx must use torch.float64"):
        _run(values=_smooth_grid().float())
    with pytest.raises(TypeError, match="pupil_states must use torch.float64"):
        _run(states=sample_pupil_sobol(3, seed=743).float())
    with pytest.raises(ValueError, match="values_zyx must have shape"):
        _run(values=torch.zeros((7, 7), dtype=torch.float64))
    invalid_grid = _smooth_grid()
    invalid_grid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="values_zyx must be finite"):
        _run(values=invalid_grid)
    invalid_states = sample_pupil_sobol(3, seed=751)
    invalid_states[0, 1] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]\^2"):
        _run(states=invalid_states)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"difference_step": 0.0}, ValueError),
        ({"difference_step": 0.25}, ValueError),
        ({"refractivity_scale": float("nan")}, ValueError),
        ({"step_count": 1}, ValueError),
        ({"step_count": 8.0}, TypeError),
        ({"sweep_count": 0}, ValueError),
        ({"sweep_count": 2.0}, TypeError),
        ({"domain_margin": -1e-3}, ValueError),
        ({"domain_margin": 1.0}, ValueError),
        ({"refractive_index_floor": 0.0}, ValueError),
    ],
)
def test_invalid_scalar_inputs_are_rejected(
    override: dict[str, float | int],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _run(**override)
