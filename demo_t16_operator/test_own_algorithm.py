from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from demo_t16_operator.data import load_npz
from demo_t16_operator.direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
from demo_t16_operator.own_algorithm_data import append_ray_view_channels
from demo_t16_operator.models import make_model
from demo_t16_operator.own_algorithm_models import (
    GridDeepONetResidual,
    RaySetResidualOperator,
    ZeroInitializedRaySetAdapter,
)
from demo_t16_operator.run_direct_operator_pilot import tune_classical_baselines


ROOT = Path(__file__).resolve().parent


def pipeline(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    direct = prepare_direct_operator_data(base, [4, 6, 8], 4, 3)
    selected, champions, _ = tune_classical_baselines(
        direct, [4, 6, 8], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    )
    assert set(champions.values()) == {"ridge"}
    return append_ray_view_channels(replace_lift_with_ridge(direct, selected))


def prepared() -> dict[str, np.ndarray]:
    return pipeline(load_npz(ROOT / "results" / "t16_direct_operator_fields_v1_dataset.npz"))


def test_ray_channels_are_masked_and_audit_zero() -> None:
    data = prepared()
    start = int(data["ray_view_channel_start"])
    channels = data["inputs"][:, start : start + int(data["ray_view_channel_count"])]
    assert np.max(np.abs(channels[:, 3])) == 0.0
    inactive = data["view_mask"] < 0.5
    assert np.max(np.abs(channels[inactive])) == 0.0


def test_audit_observation_mutation_cannot_change_ray_inputs() -> None:
    base = load_npz(ROOT / "results" / "t16_direct_operator_fields_v1_dataset.npz")
    changed = copy.deepcopy(base)
    changed["clean_observation"][:, :, 3] += 1e5
    original_inputs = pipeline(base)
    changed_inputs = pipeline(changed)
    assert np.max(original_inputs["view_mask"][:, 3]) == 0.0
    np.testing.assert_allclose(original_inputs["inputs"], changed_inputs["inputs"], atol=0.0, rtol=0.0)


def test_deeponet_and_ray_set_shapes() -> None:
    data = prepared()
    names = [str(value) for value in data["input_channel_names"].tolist()]
    view_start = int(data["ray_view_channel_start"])
    view_count = int(data["ray_view_channel_count"])
    mask_start = names.index("camera_0_active")
    angle_sin_start = int(data["ray_angle_sin_channel_start"])
    angle_cos_start = int(data["ray_angle_cos_channel_start"])
    coords = tuple(names.index(axis) for axis in ("z", "y", "x"))
    x = torch.from_numpy(data["inputs"][:2])
    deeponet = GridDeepONetResidual(
        view_start, view_count, mask_start, angle_sin_start, angle_cos_start, coords
    )
    ray_set = RaySetResidualOperator(
        view_count, int(data["inputs"].shape[1]), view_start, mask_start, angle_sin_start, angle_cos_start, coords,
        view_features=6, hidden_channels=12, n_modes=(4, 6, 6), n_layers=3,
    )
    assert deeponet(x).shape == (2, 1, 8, 16, 16)
    assert ray_set(x).shape == (2, 1, 8, 16, 16)
    assert sum(parameter.numel() for parameter in ray_set.parameters()) == 45973
    fno = make_model(
        "fno",
        {"hidden_channels": 12, "n_modes": [4, 6, 6], "n_layers": 3},
        int(data["inputs"].shape[1]),
        residual=True,
    )
    assert fno(x).shape == (2, 1, 8, 16, 16)
    assert len(names) == data["inputs"].shape[1]


def test_ray_set_attention_is_permutation_invariant_when_geometry_moves_together() -> None:
    data = prepared()
    names = [str(value) for value in data["input_channel_names"].tolist()]
    view_start = int(data["ray_view_channel_start"])
    view_count = int(data["ray_view_channel_count"])
    mask_start = names.index("camera_0_active")
    angle_sin_start = int(data["ray_angle_sin_channel_start"])
    angle_cos_start = int(data["ray_angle_cos_channel_start"])
    coords = tuple(names.index(axis) for axis in ("z", "y", "x"))
    model = RaySetResidualOperator(
        view_count, int(data["inputs"].shape[1]), view_start, mask_start, angle_sin_start, angle_cos_start, coords,
        view_features=6, hidden_channels=12, n_modes=(4, 6, 6), n_layers=3,
    ).eval()
    x = torch.from_numpy(data["inputs"][:1])
    permutation = torch.tensor([8, 7, 6, 5, 4, 3, 2, 1, 0])
    permuted = x.clone()
    permuted[:, mask_start : mask_start + view_count] = x[:, mask_start : mask_start + view_count][:, permutation]
    permuted[:, view_start : view_start + view_count] = x[:, view_start : view_start + view_count][:, permutation]
    permuted[:, angle_sin_start : angle_sin_start + view_count] = x[:, angle_sin_start : angle_sin_start + view_count][:, permutation]
    permuted[:, angle_cos_start : angle_cos_start + view_count] = x[:, angle_cos_start : angle_cos_start + view_count][:, permutation]
    permuted_model = copy.deepcopy(model)
    with torch.no_grad():
        original = model.attention(x)
        changed = permuted_model.attention(permuted)
    for original_value, changed_value in zip(original, changed):
        torch.testing.assert_close(original_value, changed_value, atol=1e-5, rtol=1e-5)


def test_zero_initialized_adapter_starts_exactly_at_frozen_fno() -> None:
    data = prepared()
    names = [str(value) for value in data["input_channel_names"].tolist()]
    view_start = int(data["ray_view_channel_start"])
    view_count = int(data["ray_view_channel_count"])
    mask_start = names.index("camera_0_active")
    angle_sin_start = int(data["ray_angle_sin_channel_start"])
    angle_cos_start = int(data["ray_angle_cos_channel_start"])
    coords = tuple(names.index(axis) for axis in ("z", "y", "x"))
    base = make_model(
        "fno",
        {"hidden_channels": 12, "n_modes": [4, 6, 6], "n_layers": 3},
        int(data["inputs"].shape[1]),
        residual=True,
    ).eval()
    adapter = ZeroInitializedRaySetAdapter(
        base,
        view_count,
        view_start,
        mask_start,
        angle_sin_start,
        angle_cos_start,
        coords,
    ).eval()
    x = torch.from_numpy(data["inputs"][:2])
    with torch.no_grad():
        expected = base(x)
        actual = adapter(x)
        correction, gate = adapter.correction(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(correction) == 0
    assert torch.all(gate > 0.0)
    assert torch.all(gate < adapter.maximum_correction_scale)
    assert all(not parameter.requires_grad for parameter in adapter.base_operator.parameters())


def test_zero_initialized_adapter_first_step_preserves_frozen_fno_weights() -> None:
    data = prepared()
    names = [str(value) for value in data["input_channel_names"].tolist()]
    view_start = int(data["ray_view_channel_start"])
    view_count = int(data["ray_view_channel_count"])
    mask_start = names.index("camera_0_active")
    coords = tuple(names.index(axis) for axis in ("z", "y", "x"))
    base = make_model(
        "fno",
        {"hidden_channels": 12, "n_modes": [4, 6, 6], "n_layers": 3},
        int(data["inputs"].shape[1]),
        residual=True,
    )
    adapter = ZeroInitializedRaySetAdapter(
        base,
        view_count,
        view_start,
        mask_start,
        int(data["ray_angle_sin_channel_start"]),
        int(data["ray_angle_cos_channel_start"]),
        coords,
    )
    frozen_state = {
        name: value.detach().clone()
        for name, value in adapter.base_operator.state_dict().items()
        if torch.is_tensor(value)
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    adapter.train()
    assert adapter.training is True
    assert adapter.base_operator.training is False
    x = torch.from_numpy(data["inputs"][:1])
    target = torch.from_numpy(data["field"][:1, None])
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((adapter(x) - target) ** 2)
    loss.backward()
    final_head = adapter.adapter[-1]
    assert final_head.weight.grad is not None
    assert torch.count_nonzero(final_head.weight.grad) > 0
    optimizer.step()
    assert torch.count_nonzero(final_head.weight.detach()) > 0
    for name, value in adapter.base_operator.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, frozen_state[name], atol=0.0, rtol=0.0)
