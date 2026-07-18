from __future__ import annotations

import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import SyntheticRayRig
from demo_t16_operator.cancellation_aware_residual import (
    evaluate_paired_residual,
    neumaier_sum,
    pairwise_sum,
)
from demo_t16_operator.field_dependent_ray import (
    path_integrated_deflection,
    relative_l2,
    sample_pupil_sobol,
)
from demo_t16_operator.shared_straight_state import build_straight_path_state


def _radial_grid(size: int = 17) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return -torch.exp(-((xx / 0.39) ** 2 + (yy / 0.39) ** 2 + (zz / 0.39) ** 2))


def _asymmetric_grid(size: int = 17) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return (
        -0.8 * torch.exp(-(((xx - 0.08) / 0.42) ** 2 + ((yy + 0.05) / 0.31) ** 2 + (zz / 0.36) ** 2))
        + 0.03 * xx
        - 0.02 * yy
    )


def _rig(**overrides: float | str) -> SyntheticRayRig:
    parameters: dict[str, float | str] = {
        "rig_id": "n5-paired-test",
        "view_angle_degrees": 27.0,
        "detector_u": 0.04,
        "detector_z": -0.03,
        "aperture_radius": 0.025,
        "path_half_length": 0.62,
        "cone_u": 0.02,
        "cone_z": 0.015,
        "bend": 0.0,
    }
    parameters.update(overrides)
    return SyntheticRayRig(**parameters)


def _evaluate(
    values: torch.Tensor,
    *,
    states: torch.Tensor | None = None,
    rig: SyntheticRayRig | None = None,
    scale: float = 3e-3,
    steps: int = 32,
):
    return evaluate_paired_residual(
        values,
        sample_pupil_sobol(8, seed=1701) if states is None else states,
        _rig() if rig is None else rig,
        difference_step=2e-3,
        refractivity_scale=scale,
        step_count=steps,
    )


def test_neumaier_recovers_a_small_term_lost_by_sequential_sum() -> None:
    values = torch.tensor([1e16, 1.0, -1e16], dtype=torch.float64)
    sequential = values[0] + values[1] + values[2]
    assert sequential == 0.0
    assert neumaier_sum(values, dim=0) == 1.0
    assert float(pairwise_sum(values, dim=0)) in {0.0, 1.0}


def test_constant_index_returns_machine_zero_for_every_ordering() -> None:
    result = _evaluate(torch.zeros((9, 9, 9), dtype=torch.float64), steps=16)
    for value in (
        result.curved_output_naive,
        result.straight_output_naive,
        result.raw_separate_subtraction,
        result.paired_naive,
        result.paired_pairwise,
        result.paired_neumaier,
        result.separate_neumaier_subtraction,
    ):
        assert torch.count_nonzero(value) == 0
    assert result.query_accounting.total_field_point_queries == 42 * 8 * 16


def test_paired_ordering_matches_the_frozen_n4_discrete_observable() -> None:
    values = _asymmetric_grid()
    states = sample_pupil_sobol(9, seed=1723)
    rig = _rig()
    result = _evaluate(values, states=states, rig=rig, steps=31)
    expected_curved = path_integrated_deflection(
        values,
        result.trace,
        gradient_mode="central",
        difference_step=2e-3,
        refractivity_scale=3e-3,
        create_graph=False,
        detach_path=False,
    )
    expected_straight = build_straight_path_state(
        values,
        states,
        rig,
        difference_step=2e-3,
        refractivity_scale=3e-3,
        step_count=31,
        frustum_half_width_u=0.05,
        frustum_half_width_v=0.05,
    ).projected_outputs
    expected = expected_curved - expected_straight

    assert torch.allclose(result.curved_output_naive, expected_curved, atol=3e-16, rtol=2e-13)
    assert torch.allclose(result.straight_output_naive, expected_straight, atol=3e-16, rtol=2e-13)
    assert torch.allclose(result.raw_separate_subtraction, expected, atol=5e-16, rtol=3e-13)
    assert torch.allclose(result.paired_neumaier, expected, atol=5e-16, rtol=3e-12)


def test_weak_field_residual_has_the_expected_second_order_finite_difference() -> None:
    values = _asymmetric_grid()
    states = sample_pupil_sobol(6, seed=1741)
    alpha = 4e-4
    epsilon = 1e-4
    lower = _evaluate(values, states=states, scale=alpha - epsilon, steps=48).paired_neumaier
    center = _evaluate(values, states=states, scale=alpha, steps=48).paired_neumaier
    upper = _evaluate(values, states=states, scale=alpha + epsilon, steps=48).paired_neumaier
    finite_difference = (upper - lower) / (2.0 * epsilon)
    quadratic_prediction = 2.0 * center / alpha
    assert relative_l2(finite_difference, quadratic_prediction) < 0.03


def test_radial_field_is_equivariant_under_a_ninety_degree_view_rotation() -> None:
    values = _radial_grid()
    states = sample_pupil_sobol(10, seed=1753)
    first = _evaluate(
        values,
        states=states,
        rig=_rig(view_angle_degrees=0.0, detector_z=0.0),
        steps=36,
    ).paired_neumaier
    rotated = _evaluate(
        values,
        states=states,
        rig=_rig(view_angle_degrees=90.0, detector_z=0.0),
        steps=36,
    ).paired_neumaier
    assert relative_l2(first, rotated) < 2e-10


@pytest.mark.parametrize("bad_steps", [True, 1, 2.5])
def test_invalid_step_count_fails_closed(bad_steps: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_paired_residual(
            _radial_grid(),
            sample_pupil_sobol(2, seed=1777),
            _rig(),
            difference_step=2e-3,
            refractivity_scale=3e-3,
            step_count=bad_steps,  # type: ignore[arg-type]
        )
