from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import demo_t16_operator.analytic_bost_phantoms as analytic_module
from demo_t16_operator.analytic_bost_phantoms import (
    ANALYTIC_FAMILIES,
    analytic_phantom_grid,
    evaluate_analytic_phantom,
    make_analytic_phantom,
    render_analytic_bost,
)


@pytest.mark.parametrize("family", ANALYTIC_FAMILIES)
def test_analytic_gradient_matches_central_difference(family: str) -> None:
    spec = make_analytic_phantom(family=family, seed=217)
    rng = np.random.default_rng(301)
    points = torch.as_tensor(rng.uniform(-0.72, 0.72, size=(11, 3)), dtype=torch.float64)
    evaluation = evaluate_analytic_phantom(spec, points)
    step = 1e-6
    numerical = []
    for axis in range(3):
        offset = torch.zeros_like(points)
        offset[:, axis] = step
        plus = evaluate_analytic_phantom(spec, points + offset).field
        minus = evaluate_analytic_phantom(spec, points - offset).field
        numerical.append((plus - minus) / (2.0 * step))
    numerical_gradient = torch.stack(numerical, dim=-1)
    assert torch.allclose(
        evaluation.gradient_xyz,
        numerical_gradient,
        rtol=2e-5,
        atol=2e-6,
    )


def test_spec_generation_is_deterministic_and_seed_sensitive() -> None:
    first = make_analytic_phantom(family="wrinkled_density_interface", seed=19)
    second = make_analytic_phantom(family="wrinkled_density_interface", seed=19)
    third = make_analytic_phantom(family="wrinkled_density_interface", seed=20)
    assert first == second
    assert first != third


def test_window_zeroes_field_and_gradient_on_domain_boundary() -> None:
    spec = make_analytic_phantom(family="oblique_compression_sheet", seed=41)
    points = torch.tensor(
        [[-1.0, 0.2, -0.1], [1.0, -0.3, 0.4], [0.3, -1.0, 0.2]],
        dtype=torch.float64,
    )
    evaluation = evaluate_analytic_phantom(spec, points)
    assert torch.count_nonzero(evaluation.field) == 0
    assert torch.count_nonzero(evaluation.gradient_xyz) == 0


@pytest.mark.parametrize(
    ("family", "level_count"),
    (
        ("smooth_plume", 0),
        ("wrinkled_density_interface", 1),
        ("oblique_compression_sheet", 1),
        ("shock_expansion_pair", 2),
    ),
)
def test_grid_evaluation_exposes_declared_level_sets(
    family: str,
    level_count: int,
) -> None:
    spec = make_analytic_phantom(family=family, seed=53)
    evaluation = analytic_phantom_grid(spec, grid_shape=(9, 10, 11))
    assert evaluation.field.shape == (9, 10, 11)
    assert evaluation.gradient_xyz.shape == (9, 10, 11, 3)
    assert evaluation.level_sets.shape == (9, 10, 11, level_count)
    assert evaluation.level_set_gradients_xyz.shape == (9, 10, 11, level_count, 3)
    assert torch.all(torch.isfinite(evaluation.field))


def test_renderer_matches_manual_analytic_gradient_sum() -> None:
    spec = make_analytic_phantom(family="smooth_plume", seed=71)
    points = torch.tensor(
        [
            [[-0.4, 0.1, 0.2], [0.0, 0.1, 0.2], [0.4, 0.1, 0.2]],
            [[0.2, -0.4, -0.2], [0.2, 0.0, -0.2], [0.2, 0.4, -0.2]],
        ],
        dtype=torch.float64,
    )
    projection_u = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    projection_v = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    length = torch.tensor([2.0, 3.0], dtype=torch.float64)
    constant = torch.tensor([0.5, 1.25], dtype=torch.float64)
    rendered = render_analytic_bost(
        spec,
        sample_points_xyz=points,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        line_length=length,
        system_constant=constant,
    )
    gradient = evaluate_analytic_phantom(spec, points).gradient_xyz
    manual_u = torch.sum(gradient * projection_u[:, None, :], dim=(1, 2))
    manual_v = torch.sum(gradient * projection_v[:, None, :], dim=(1, 2))
    manual = torch.stack((manual_u, manual_v), dim=-1)
    manual = manual * (length * constant / points.shape[1])[:, None]
    assert torch.allclose(rendered, manual)


def test_invalid_samples_keep_the_fixed_denominator() -> None:
    spec = make_analytic_phantom(family="smooth_plume", seed=83)
    points = torch.tensor(
        [[[-0.2, 0.1, 0.0], [0.3, -0.1, 0.2]]],
        dtype=torch.float64,
    )
    projection_u = [[1.0, 0.0, 0.0]]
    projection_v = [[0.0, 1.0, 0.0]]
    valid = torch.tensor([[True, False]])
    rendered = render_analytic_bost(
        spec,
        sample_points_xyz=points,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        line_length=[2.0],
        system_constant=[1.0],
        sample_valid=valid,
    )
    first_gradient = evaluate_analytic_phantom(spec, points[:, :1]).gradient_xyz[0, 0]
    expected = torch.stack((first_gradient[0], first_gradient[1]))
    assert torch.allclose(rendered[0], expected)


def test_renderer_module_does_not_import_the_voxel_inverse_chain() -> None:
    source = inspect.getsource(analytic_module)
    assert "PSUB0VoxelGradientOperator" not in source
    assert "finite_difference_gradient" not in source
