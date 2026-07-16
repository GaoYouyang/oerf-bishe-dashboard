from __future__ import annotations

import json

import numpy as np
import torch

from demo_t16_operator.bost_physics import build_forward_matrix, deflection_sinogram, forward_volume, make_phantom
from demo_t16_operator.data import generate_dataset, load_npz, split_indices
from demo_t16_operator.models import count_parameters, make_dual_branch_model, make_model
from demo_t16_operator.run_dual_branch_query import closed_form_support_weight
from demo_t16_operator.run_fair_camera_budget import (
    controlled_support_mask,
    mask_for_index,
    numeric_query_null_update,
    select_query_index,
)
from demo_t16_operator.run_nullspace_identifiability_audit import (
    nullspace_decomposition,
    project_to_nullspace,
    support_matrix,
)
from demo_t16_operator.run_query_calibrated_nullspace import (
    clipped_line_search_alpha,
    informative_query_mask,
)
from demo_t16_operator.run_support_nullspace_corrector import (
    TorchSupportNullProjector,
    bounded_correction,
    corrector_input,
)


def test_forward_matrix_matches_direct_projection() -> None:
    n = 8
    depth = 3
    angles = np.linspace(0.0, 180.0, 5, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(17)
    volume = make_phantom("gaussian", n, depth, rng)
    operator = build_forward_matrix(n, angles)
    matrix_result = forward_volume(volume, operator)
    direct_result = np.stack([deflection_sinogram(volume[z], angles).T for z in range(depth)])
    np.testing.assert_allclose(matrix_result, direct_result, rtol=2e-6, atol=2e-6)


def test_dataset_split_and_family_holdout(tmp_path) -> None:
    config = {
        "name": "test",
        "seed": 9,
        "grid_size": 8,
        "depth": 4,
        "max_views": 5,
        "splits": {
            "train": {"count": 8, "families": ["gaussian", "flame"], "views": [2, 3], "noise": [0.01, 0.02]},
            "val": {"count": 2, "families": ["gaussian"], "views": [3], "noise": [0.01]},
            "test_family_ood": {"count": 2, "families": ["thin_front"], "views": [3], "noise": [0.01]},
        },
    }
    path = generate_dataset(config, tmp_path / "tiny.npz")
    data = load_npz(path)
    indices = split_indices(data)
    assert data["inputs"].shape == (12, 7, 4, 8, 8)
    assert set(data["sample_seed"].tolist()) == set(np.unique(data["sample_seed"]).tolist())
    assert set(data["family_id"][indices["train"]].tolist()) == {0, 1}
    assert set(data["family_id"][indices["test_family_ood"]].tolist()) == {2}
    train_conditions = set(
        zip(
            data["family_id"][indices["train"]].tolist(),
            data["view_count"][indices["train"]].tolist(),
            data["noise_level"][indices["train"]].round(3).tolist(),
        )
    )
    assert len(train_conditions) == 8
    assert data["calibration"].shape == (2,)
    assert int(data["schema_version"]) == 2
    assert json.loads(str(data["config_json"]))["name"] == "test"


def test_3d_model_shapes() -> None:
    x = torch.randn(2, 7, 8, 16, 16)
    unet = make_model("unet", {"base_channels": 4}, in_channels=7)
    fno = make_model(
        "fno",
        {"hidden_channels": 8, "n_modes": [4, 6, 6], "n_layers": 2},
        in_channels=7,
    )
    assert unet(x).shape == (2, 1, 8, 16, 16)
    assert fno(x).shape == (2, 1, 8, 16, 16)

    absolute = make_model(
        "fno",
        {"hidden_channels": 8, "n_modes": [4, 6, 6], "n_layers": 2},
        in_channels=7,
        residual=False,
    )
    matched_unet = make_model("unet", {"base_channels": 6}, in_channels=7)
    matched_fno = make_model(
        "fno",
        {"hidden_channels": 12, "n_modes": [4, 6, 6], "n_layers": 3},
        in_channels=7,
    )
    assert absolute(x).shape == (2, 1, 8, 16, 16)
    assert matched_unet(x).shape == (2, 1, 8, 16, 16)
    ratio = count_parameters(matched_unet) / count_parameters(matched_fno)
    assert 0.8 < ratio < 1.25


def test_reliability_gate_contracts() -> None:
    x = torch.zeros(2, 7, 4, 8, 8)
    x[0, 0] = 2.0
    x[1, 0] = 2.0
    x[0, 2] = 5.0 / 9.0
    x[1, 2] = 3.0 / 9.0
    model = make_model(
        "unet",
        {"base_channels": 4},
        in_channels=7,
        gate_config={"type": "fixed_view", "view_channel": 2, "reference_fraction": 5.0 / 9.0},
    )
    for parameter in model.backbone.parameters():
        parameter.data.zero_()
    output = model(x)
    torch.testing.assert_close(output[0], torch.full_like(output[0], 2.0))
    torch.testing.assert_close(output[1], torch.full_like(output[1], 1.2))

    learned = make_model(
        "unet",
        {"base_channels": 4},
        in_channels=7,
        gate_config={"type": "learned", "feature_channels": [2, 3], "hidden_channels": 4, "max_scale": 1.25},
    )
    alpha = learned.gate_values(x)
    torch.testing.assert_close(alpha, torch.ones_like(alpha), rtol=1e-5, atol=1e-5)


def test_dual_branch_operator_contract() -> None:
    x = torch.zeros(2, 7, 8, 16, 16)
    x[:, 0] = 0.4
    model = make_dual_branch_model(
        {"hidden_channels": 4, "n_modes": [2, 3, 3], "n_layers": 2},
        in_channels=7,
        router_features=6,
        router_hidden=4,
    )
    residual, absolute = model.experts(x)
    assert residual.shape == absolute.shape == (2, 1, 8, 16, 16)
    features = torch.zeros(2, 6)
    weights = model.route(features)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
    mixture = model.combine(residual, absolute, weights)
    torch.testing.assert_close(mixture, 0.5 * (residual + absolute))

    independent = make_dual_branch_model(
        {"hidden_channels": 4, "n_modes": [2, 3, 3], "n_layers": 2},
        in_channels=7,
        router_features=6,
        router_hidden=4,
        expert_sharing="independent",
    )
    independent_residual, independent_absolute = independent.experts(x)
    assert independent_residual.shape == independent_absolute.shape == (2, 1, 8, 16, 16)
    assert count_parameters(independent) > count_parameters(model)
    independent_weight = independent.route(features)
    torch.testing.assert_close(independent_weight, torch.full_like(independent_weight, 0.5))


def test_closed_form_support_weight() -> None:
    absolute = torch.zeros(2, 1, 3, 4)
    residual = torch.ones_like(absolute)
    observed = torch.stack(
        [torch.full((1, 3, 4), 0.25), torch.full((1, 3, 4), 1.5)],
        dim=0,
    )
    support_mask = torch.ones(2, 3)
    weight = closed_form_support_weight(residual, absolute, observed, support_mask)
    assert weight.shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(weight[0], torch.full_like(weight[0], 0.25))
    torch.testing.assert_close(weight[1], torch.ones_like(weight[1]))


def test_support_nullspace_projection_contract() -> None:
    n = 8
    angles = np.linspace(0.0, 180.0, 5, endpoint=False, dtype=np.float32)
    operator = build_forward_matrix(n, angles)
    mask = np.asarray([1, 0, 1, 0, 1], dtype=np.float32)
    decomposition = nullspace_decomposition(operator, mask)
    matrix = support_matrix(operator, mask)
    basis = np.asarray(decomposition["null_basis"])
    assert int(decomposition["rank"]) + int(decomposition["nullity"]) == n * n
    assert np.linalg.norm(matrix @ basis.T) < 1e-8

    rng = np.random.default_rng(41)
    volume = rng.normal(size=(3, n, n))
    projected = project_to_nullspace(volume, basis)
    support_projection = projected.reshape(3, -1) @ matrix.T
    assert np.linalg.norm(support_projection) < 1e-8
    assert np.linalg.norm(projected) > 0.0


def test_torch_support_nullspace_and_correction_cap_contracts() -> None:
    n = 8
    angles = np.linspace(0.0, 180.0, 5, endpoint=False, dtype=np.float32)
    operator = build_forward_matrix(n, angles)
    masks = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 1]], dtype=torch.float32)
    projector = TorchSupportNullProjector(operator)
    raw = torch.randn(2, 1, 3, n, n)
    projected = projector.project(raw, masks)
    for index in range(2):
        matrix = support_matrix(operator, masks[index].numpy())
        measured = projected[index, 0].double().reshape(3, -1) @ torch.from_numpy(matrix.T)
        assert float(torch.linalg.vector_norm(measured)) < 2e-6

    base = torch.ones_like(raw)
    capped, scale = bounded_correction(10.0 * raw, base, cap_ratio=0.25)
    relative = torch.linalg.vector_norm(capped.flatten(start_dim=1), dim=1) / torch.linalg.vector_norm(
        base.flatten(start_dim=1), dim=1
    )
    assert torch.all(relative <= 0.250001)
    assert torch.all((0.0 < scale) & (scale <= 1.0))

    x = torch.randn(2, 7, 3, n, n)
    combined = corrector_input(x, base, base + 0.1, base - 0.1)
    assert combined.shape == (2, 9, 3, n, n)


def test_query_line_search_recovers_known_amplitude() -> None:
    direction = np.zeros((2, 3, 4), dtype=np.float64)
    direction[:, 1, :] = 2.0
    residual = 0.35 * direction
    mask = np.asarray([0.0, 1.0, 0.0])
    alpha = clipped_line_search_alpha(direction, residual, mask)
    assert abs(alpha - 0.35) < 1e-12


def test_informative_query_selects_largest_correction_projection() -> None:
    direction = np.zeros((2, 4, 3), dtype=np.float64)
    direction[:, 1, :] = 0.5
    direction[:, 3, :] = 2.0
    query = np.asarray([0.0, 1.0, 0.0, 1.0])
    selected_mask, selected = informative_query_mask(direction, query)
    assert selected == 3
    assert selected_mask.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_equal_camera_budget_masks_and_query_strategies() -> None:
    max_views = 9
    angles = np.linspace(0.0, 180.0, max_views, endpoint=False)
    support = controlled_support_mask(
        max_views,
        support_count=5,
        fixed_query_index=4,
        audit_query_index=3,
    )
    assert int(np.sum(support)) == 5
    assert support[4] == 0.0
    assert support[3] == 0.0
    direction = np.zeros((2, max_views, 3), dtype=np.float64)
    direction[:, 7] = 4.0
    strategies = {
        strategy: select_query_index(
            strategy,
            support,
            angles,
            fixed_query_index=4,
            audit_query_index=3,
            sample_seed=71,
            total_budget=6,
            direction_projection=direction,
            random_seed=17,
        )
        for strategy in ("fixed", "random", "max_gap", "adaptive_energy")
    }
    assert strategies["fixed"] == 4
    assert strategies["adaptive_energy"] == 7
    assert all(support[index] == 0.0 for index in strategies.values())
    for index in strategies.values():
        query = mask_for_index(max_views, index)
        audit = mask_for_index(max_views, 3)
        assert np.all((support + query) * audit == 0.0)


def test_numeric_query_null_update_preserves_support_and_reduces_query_residual() -> None:
    n = 8
    depth = 3
    angles = np.linspace(0.0, 180.0, 7, endpoint=False, dtype=np.float32)
    operator = build_forward_matrix(n, angles)
    support = np.asarray([1, 0, 1, 0, 1, 0, 0], dtype=np.float64)
    query = np.asarray([0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
    rng = np.random.default_rng(91)
    target = make_phantom("gaussian", n, depth, rng)
    observed = forward_volume(target, operator)
    base = np.zeros_like(target)
    projector = TorchSupportNullProjector(operator)
    correction = numeric_query_null_update(
        base,
        observed,
        support,
        query,
        operator,
        projector,
        ridge_relative=1e-6,
        cap_ratio=1e12,
    )
    projected = forward_volume(correction, operator)
    assert np.linalg.norm(projected * support[None, :, None]) < 1e-6
    initial_query = np.linalg.norm(observed * query[None, :, None])
    updated_query = np.linalg.norm((observed - projected) * query[None, :, None])
    assert updated_query < initial_query
