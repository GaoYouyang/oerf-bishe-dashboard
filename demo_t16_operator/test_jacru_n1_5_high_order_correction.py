from __future__ import annotations

import torch

from demo_t16_operator.interface_baselines import cgls_baseline
from demo_t16_operator.jacru_n1_5_high_order_correction import (
    HighOrderTeacherMaps,
    fourth_order_difference_matrix,
    warm_start_cgls,
)
from demo_t16_operator.jacru_synthetic_fixture import (
    JACRUSyntheticFixtureConfig,
    build_jacru_synthetic_case,
)


def _case():
    config = JACRUSyntheticFixtureConfig(
        detector_shape=(5, 5),
        samples_per_ray=16,
        enable_noise=False,
        enable_camera_bias=False,
    )
    return build_jacru_synthetic_case(
        family="single_interface", split="train", base_seed=1101, config=config
    )


def test_fourth_order_matrix_is_exact_for_quartic_interior() -> None:
    coordinate = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64)
    spacing = float(coordinate[1] - coordinate[0])
    matrix = fourth_order_difference_matrix(12, spacing)
    derivative = matrix @ coordinate.pow(4)
    assert torch.allclose(derivative[2:-2], 4.0 * coordinate[2:-2].pow(3), atol=1e-10)


def test_high_order_maps_pass_adjoint_identity() -> None:
    case = _case()
    maps = HighOrderTeacherMaps(case.inference.operator)
    generator = torch.Generator().manual_seed(71)
    field = torch.randn(case.inference.operator.grid_shape, generator=generator, dtype=torch.float64)
    observation = torch.randn(
        (case.inference.operator.ray_count, 2), generator=generator, dtype=torch.float64
    )
    left = torch.sum(maps.forward(field) * observation)
    right = torch.sum(field * maps.adjoint(observation))
    assert torch.allclose(left, right, atol=1e-10, rtol=1e-10)
    assert maps.call_report() == {"forward_calls": 1, "adjoint_calls": 1}


def test_warm_start_cgls_uses_exact_registered_budget() -> None:
    case = _case()
    operator = case.inference.operator
    observation = case.evaluation.clean_observations_uv[0]

    def forward(field: torch.Tensor) -> torch.Tensor:
        return operator(field[None, None])[0]

    def adjoint(residual: torch.Tensor) -> torch.Tensor:
        return operator.adjoint(residual[None])[0, 0]

    support = operator.support
    operator.reset_call_counts()
    warm = cgls_baseline(
        observation,
        forward=forward,
        adjoint=adjoint,
        support=support,
        spacing_xyz=operator.spacing_xyz,
        iterations=3,
    )
    initial_projection = operator(warm.field[None, None])[0]
    operator.reset_call_counts()
    result = warm_start_cgls(
        observation,
        forward=forward,
        adjoint=adjoint,
        support=support,
        initial_field=warm.field,
        initial_projection=initial_projection,
        iterations=4,
    )
    assert result.forward_calls == 4
    assert result.adjoint_calls == 4
    assert operator.call_report() == {"forward_calls": 4, "adjoint_calls": 4}
    assert len(result.history) == 4


def test_high_order_correction_reuses_supplied_low_projection() -> None:
    case = _case()
    operator = case.inference.operator
    maps = HighOrderTeacherMaps(operator)
    field = case.evaluation.truth_volume[0, 0]
    operator.reset_call_counts()
    low = operator(field[None, None])[0]
    operator.reset_call_counts()
    correction = maps.correction(field, low_projection=low)
    assert correction.shape == low.shape
    assert operator.call_report() == {"forward_calls": 0, "adjoint_calls": 0}
    assert maps.call_report()["forward_calls"] == 1
