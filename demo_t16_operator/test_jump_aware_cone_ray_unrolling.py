from __future__ import annotations

import inspect

import pytest
import torch

from demo_t16_operator.jump_aware_cone_ray_unrolling import (
    JumpAwareConfig,
    JumpAwareConeRayField,
    jump_aware_objective,
    optimize_jump_aware_cone_ray,
    voxel_operator_gradient_forward,
)
from demo_t16_operator.psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
    finite_difference_gradient,
)


def _identity_gradient_forward(gradient: torch.Tensor) -> torch.Tensor:
    return gradient.flatten(2).transpose(1, 2)


def _uv_gradient_forward(gradient: torch.Tensor) -> torch.Tensor:
    return gradient[:, :2].flatten(2).transpose(1, 2)


def _small_config(**overrides: object) -> JumpAwareConfig:
    values: dict[str, object] = {
        "grid_shape": (4, 4, 4),
        "spacing_xyz": (0.5, 0.5, 0.5),
        "outer_steps": 2,
        "field_updates_per_outer": 1,
        "interface_updates_per_outer": 1,
        "bias_updates_per_outer": 0,
        "field_learning_rate": 1e-2,
        "interface_learning_rate": 1e-2,
        "seed": 13,
    }
    values.update(overrides)
    return JumpAwareConfig(**values)


def test_discrete_gradient_split_closes_exactly() -> None:
    model = JumpAwareConeRayField(
        _small_config(),
        batch_size=2,
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.upstream.copy_(torch.randn_like(model.upstream))
        model.jump_field.copy_(torch.randn_like(model.jump_field))
        model.phase_field.copy_(torch.randn_like(model.phase_field))
    split = model.gradient_split(hard=False)
    torch.testing.assert_close(
        split.total,
        split.smooth_side + split.jump,
        rtol=0.0,
        atol=5e-16,
    )
    assert float(split.closure_rms.detach()) < 5e-16


def test_gradient_factor_forward_matches_scalar_operator() -> None:
    points = torch.tensor(
        [
            [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
        ],
        dtype=torch.float64,
    )
    stencil = build_trilinear_stencil(
        points,
        grid_shape=(4, 4, 4),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    operator = PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        projection_v_xyz=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        line_length=torch.tensor([1.2, 0.8]),
        system_constant=torch.tensor([0.7, 1.1]),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    volume = torch.randn((2, 1, 4, 4, 4), dtype=torch.float64)
    gradient = finite_difference_gradient(
        volume[:, 0],
        spacing_xyz=operator.spacing_xyz,
    )
    expected = operator(volume)
    actual = voxel_operator_gradient_forward(operator, gradient)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_camera_bias_is_zero_mean_and_group_bound() -> None:
    model = JumpAwareConeRayField(
        _small_config(),
        batch_size=1,
        camera_group_count=3,
    )
    assert model.camera_bias is not None
    with torch.no_grad():
        model.camera_bias.copy_(
            torch.tensor([[[1.0, -2.0], [0.0, 1.0], [2.0, 4.0]]])
        )
    centered = model.centered_camera_bias()
    assert centered is not None
    torch.testing.assert_close(centered.mean(dim=1), torch.zeros((1, 2)))
    selected = model.bias_for_rays(torch.tensor([2, 0, 1, 2]))
    assert selected.shape == (1, 4, 2)
    with pytest.raises(ValueError, match="outside"):
        model.bias_for_rays(torch.tensor([3]))


def test_fixed_block_budget_and_truth_free_signature() -> None:
    config = _small_config(outer_steps=3)
    observed = torch.zeros((1, 64, 3), dtype=torch.float64)
    result = optimize_jump_aware_cone_ray(
        observed,
        gradient_forward=_identity_gradient_forward,
        config=config,
    )
    assert result.optimization_forward_evaluations == 6
    assert result.implicit_data_vjp_evaluations == 6
    assert result.reporting_forward_evaluations == 2
    assert result.total_forward_evaluations == 8
    assert len(result.history) == 6
    assert {row.block for row in result.history} == {"field", "interface"}
    parameters = inspect.signature(optimize_jump_aware_cone_ray).parameters
    assert "truth" not in parameters
    assert "target" not in parameters
    assert "level_set_truth" not in parameters


def test_objective_and_result_are_finite_for_zero_observation() -> None:
    config = _small_config()
    model = JumpAwareConeRayField(config, batch_size=1, dtype=torch.float64)
    observed = torch.zeros((1, 64, 3), dtype=torch.float64)
    objective = jump_aware_objective(
        model,
        observed,
        gradient_forward=_identity_gradient_forward,
    )
    assert torch.isfinite(objective.total)
    result = optimize_jump_aware_cone_ray(
        observed,
        gradient_forward=_identity_gradient_forward,
        config=config,
    )
    assert torch.all(torch.isfinite(result.soft_volume))
    assert torch.all(torch.isfinite(result.hard_volume))
    assert float(result.soft_gradient_split.closure_rms) == 0.0


def test_seeded_runs_are_deterministic() -> None:
    config = _small_config(outer_steps=1)
    observed = torch.randn((1, 64, 3), generator=torch.Generator().manual_seed(7))
    first = optimize_jump_aware_cone_ray(
        observed,
        gradient_forward=_identity_gradient_forward,
        config=config,
    )
    second = optimize_jump_aware_cone_ray(
        observed,
        gradient_forward=_identity_gradient_forward,
        config=config,
    )
    torch.testing.assert_close(first.soft_volume, second.soft_volume)
    torch.testing.assert_close(first.gate_probability, second.gate_probability)


def test_random_plane_initialization_is_seeded_without_fixed_x_alignment() -> None:
    first = JumpAwareConeRayField(
        _small_config(initial_phase_mode="random_plane", seed=31),
        batch_size=1,
        dtype=torch.float64,
    )
    repeat = JumpAwareConeRayField(
        _small_config(initial_phase_mode="random_plane", seed=31),
        batch_size=1,
        dtype=torch.float64,
    )
    other = JumpAwareConeRayField(
        _small_config(initial_phase_mode="random_plane", seed=37),
        batch_size=1,
        dtype=torch.float64,
    )
    fixed = JumpAwareConeRayField(
        _small_config(initial_phase_mode="fixed_x", seed=31),
        batch_size=1,
        dtype=torch.float64,
    )
    torch.testing.assert_close(first.phase_field, repeat.phase_field)
    assert not torch.allclose(first.phase_field, other.phase_field)
    assert not torch.allclose(first.phase_field, fixed.phase_field)


def test_camera_bias_requires_groups_and_adds_one_block_per_outer() -> None:
    config = _small_config(bias_updates_per_outer=1, outer_steps=2)
    observed = torch.zeros((1, 64, 2))
    groups = torch.arange(64) % 2
    result = optimize_jump_aware_cone_ray(
        observed,
        gradient_forward=_uv_gradient_forward,
        config=config,
        ray_group_index=groups,
        camera_group_count=2,
    )
    assert result.optimization_forward_evaluations == 6
    assert [row.block for row in result.history].count("bias") == 2
    with pytest.raises(ValueError, match="ray_group_index"):
        optimize_jump_aware_cone_ray(
            observed,
            gradient_forward=_uv_gradient_forward,
            config=config,
            camera_group_count=2,
        )


@pytest.mark.parametrize(
    "config, message",
    [
        (JumpAwareConfig(grid_shape=(2, 4, 4)), "grid_shape"),
        (JumpAwareConfig(grid_shape=(4, 4, 4), epsilon=0.0), "positive"),
        (
            JumpAwareConfig(grid_shape=(4, 4, 4), deployment_gate_threshold=1.0),
            "deployment_gate_threshold",
        ),
        (
            JumpAwareConfig(grid_shape=(4, 4, 4), bias_updates_per_outer=-1),
            "bias_updates_per_outer",
        ),
        (
            JumpAwareConfig(grid_shape=(4, 4, 4), initial_phase_mode="oracle"),
            "initial_phase_mode",
        ),
    ],
)
def test_invalid_configs_fail(config: JumpAwareConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config.validated()


def test_nonfinite_observation_fails() -> None:
    observation = torch.zeros((1, 64, 3))
    observation[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        optimize_jump_aware_cone_ray(
            observation,
            gradient_forward=_identity_gradient_forward,
            config=_small_config(),
        )
