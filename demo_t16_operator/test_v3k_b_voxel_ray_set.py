from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from demo_t16_operator.counterfactual_geometry import (
    CounterfactualInputFactory,
    build_pair_schedule,
    ray_angle_pairing_derangement_rows,
    ray_set_components_for_pairs,
)
from demo_t16_operator.own_algorithm_models import VoxelRaySetResidualOperator
from demo_t16_operator.test_v3k_a_counterfactual_supervision import (
    synthetic_geometry_data,
)


ROOT = Path(__file__).resolve().parent


def synthetic_inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20261571)
    x = torch.randn(4, 42, 2, 4, 4, generator=generator)
    x[:, 1] = 1.0
    x[:, 2] = 6.0 / 9.0
    angles = torch.deg2rad(torch.arange(0, 180, 20, dtype=torch.float32))
    masks = torch.tensor(
        [
            [1, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    for camera in range(9):
        x[:, 3 + camera] = masks[:, camera, None, None, None]
        x[:, 15 + camera] *= masks[:, camera, None, None, None]
        x[:, 24 + camera] = (
            masks[:, camera] * torch.sin(angles[camera])
        )[:, None, None, None]
        x[:, 33 + camera] = (
            masks[:, camera] * torch.cos(angles[camera])
        )[:, None, None, None]
    return x


def make_model() -> VoxelRaySetResidualOperator:
    return VoxelRaySetResidualOperator(
        base_operator=nn.Conv3d(42, 1, kernel_size=1),
        view_count=9,
        mask_channel_start=3,
        ray_channel_start=15,
        angle_sin_channel_start=24,
        angle_cos_channel_start=33,
        coordinate_channels=(12, 13, 14),
        token_hidden=18,
        latent_features=10,
        adapter_hidden=8,
        spectral_modes=(2, 3, 3),
        freeze_base=True,
    )


def trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def test_v3k_b_config_preregisters_mechanism_only() -> None:
    config = json.loads(
        (ROOT / "configs" / "v3k_b_voxel_ray_set_pilot.json").read_text()
    )
    assert config["training_arm"] == "m4_counterfactual"
    assert config["methods"] == [
        "pooled_static",
        "geometry_only",
        "shuffled_pairing",
        "correct_ray_set",
    ]
    assert len(config["model_seeds"]) == 3
    assert config["mechanism_gate"]["minimum_mean_gain_pct"] == 0.25
    assert config["claims_boundary"]["superiority_tested"] is False
    assert config["claims_boundary"]["blind_final_opened"] is False


def test_zero_initialization_freezes_base_and_matches_base_output() -> None:
    torch.manual_seed(9)
    model = make_model().eval()
    x = synthetic_inputs()
    with torch.no_grad():
        expected = model.base_operator(x)
        actual = model(x)
    assert torch.equal(expected, actual)
    assert all(not parameter.requires_grad for parameter in model.base_operator.parameters())
    assert 700 <= trainable_parameters(model) <= 1100


def test_joint_camera_permutation_is_exactly_invariant() -> None:
    torch.manual_seed(10)
    model = make_model().eval()
    with torch.no_grad():
        model.head.weight.fill_(0.2)
        model.head.bias.fill_(0.03)
    x = synthetic_inputs()
    components = model.components(x)
    permutation = torch.tensor([7, 1, 5, 0, 8, 3, 2, 6, 4])
    permuted = tuple(value[:, permutation] for value in components)
    with torch.no_grad():
        original, _ = model.correction(x, acquisition_components=components)
        changed, _ = model.correction(x, acquisition_components=permuted)
    assert torch.allclose(original, changed, atol=1e-6, rtol=1e-6)


def test_inactive_camera_values_cannot_change_the_correction() -> None:
    torch.manual_seed(11)
    model = make_model().eval()
    with torch.no_grad():
        model.head.weight.fill_(0.15)
    x = synthetic_inputs()
    masks, angle_sin, angle_cos, rays = model.components(x)
    corrupted = rays.clone()
    corrupted += (1.0 - masks[:, :, None, None, None]) * 1000.0
    with torch.no_grad():
        reference, _ = model.correction(
            x, acquisition_components=(masks, angle_sin, angle_cos, rays)
        )
        changed, _ = model.correction(
            x, acquisition_components=(masks, angle_sin, angle_cos, corrupted)
        )
    assert torch.allclose(reference, changed, atol=1e-6, rtol=1e-6)


def test_ray_angle_pairing_shuffle_is_not_a_joint_permutation() -> None:
    torch.manual_seed(13)
    model = make_model().eval()
    with torch.no_grad():
        model.head.weight.fill_(0.2)
        model.head.bias.fill_(0.01)
    x = synthetic_inputs()
    masks, angle_sin, angle_cos, rays = model.components(x)
    shuffled_sin = torch.zeros_like(angle_sin)
    shuffled_cos = torch.zeros_like(angle_cos)
    for batch_index in range(len(x)):
        active = torch.nonzero(
            masks[batch_index] > 0.5, as_tuple=False
        ).flatten()
        donors = torch.roll(active, shifts=-1)
        shuffled_sin[batch_index, active] = angle_sin[batch_index, donors]
        shuffled_cos[batch_index, active] = angle_cos[batch_index, donors]
    with torch.no_grad():
        correct, _ = model.correction(
            x, acquisition_components=(masks, angle_sin, angle_cos, rays)
        )
        shuffled, _ = model.correction(
            x,
            acquisition_components=(masks, shuffled_sin, shuffled_cos, rays),
        )
    assert not torch.allclose(correct, shuffled, atol=1e-7, rtol=1e-7)


def test_gradients_reach_local_token_encoder_without_unfreezing_base() -> None:
    torch.manual_seed(12)
    model = make_model().train()
    with torch.no_grad():
        model.head.weight.fill_(0.05)
    x = synthetic_inputs()
    target = torch.randn_like(model.base_operator(x))
    loss = torch.mean((model(x) - target) ** 2)
    loss.backward()
    token_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.token_encoder.parameters()
        if parameter.grad is not None
    )
    assert token_grad > 0.0
    assert all(parameter.grad is None for parameter in model.base_operator.parameters())


def test_counterfactual_ray_sets_are_consistent_and_audit_safe() -> None:
    data = synthetic_geometry_data()
    factory = CounterfactualInputFactory(data, ridge_relative=1e-6)
    pairs = build_pair_schedule(
        data,
        source_split="train",
        geometry_partition="train",
        arm="m4_counterfactual",
        repeats_per_source=4,
        assignment_seed=20261371,
        counterfactual_stride=5,
    )
    correct = ray_set_components_for_pairs(factory, pairs)
    wrong = ray_set_components_for_pairs(
        factory, pairs, shuffle_angle_pairing=True
    )
    assert correct[3].shape == (64, 9, 2, 4, 4)
    assert np.all(correct[0].sum(axis=1) == 6)
    assert np.all(wrong[0].sum(axis=1) == 6)
    assert np.all(correct[0][:, 3] == 0.0)
    assert np.all(correct[3][:, 3] == 0.0)
    assert np.all(wrong[3][:, 3] == 0.0)
    assert np.array_equal(correct[0], wrong[0])
    assert np.array_equal(correct[3], wrong[3])
    assert np.all(
        np.sort(correct[1], axis=1) == np.sort(wrong[1], axis=1)
    )
    assert np.all(np.any(correct[1] != wrong[1], axis=1))
    rows = ray_angle_pairing_derangement_rows(factory)
    assert len(rows) == 28 * 6
    assert all(not bool(row["fixed_point"]) for row in rows)
