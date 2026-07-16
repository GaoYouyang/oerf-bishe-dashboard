"""Unit tests for the camera-budget-matched direct inverse-operator pilot."""

from __future__ import annotations

import copy

import numpy as np

from .bost_physics import forward_volume
from .data import generate_dataset, load_npz
from .direct_operator_data import (
    prepare_direct_operator_data,
    reconstruction_mask,
    ridge_reconstruct,
)
from .run_direct_operator_pilot import collapse_model_seeds, subset_budget_data


def tiny_base_data(tmp_path):
    config = {
        "name": "tiny_direct_operator_test",
        "seed": 771,
        "grid_size": 8,
        "depth": 4,
        "max_views": 9,
        "splits": {
            "train": {"count": 4, "families": ["gaussian"], "views": [5], "noise": [0.02]},
            "val": {"count": 2, "families": ["flame"], "views": [5], "noise": [0.02]},
            "test_iid": {"count": 2, "families": ["gaussian"], "views": [5], "noise": [0.10]},
        },
    }
    path = tmp_path / "tiny.npz"
    generate_dataset(config, path, force=True)
    return load_npz(path)


def test_mask_cardinality_and_audit_disjoint():
    expected = {
        4: {0, 4, 5, 8},
        6: {0, 2, 4, 5, 6, 8},
        8: {0, 1, 2, 4, 5, 6, 7, 8},
    }
    for budget, indices in expected.items():
        mask = reconstruction_mask(9, budget, 4, 3)
        assert set(np.flatnonzero(mask > 0.5)) == indices
        assert int(mask.sum()) == budget
        assert mask[3] == 0.0
        assert mask[4] == 1.0


def test_budget_data_encodes_geometry_and_keeps_field_variants_in_split(tmp_path):
    data = prepare_direct_operator_data(tiny_base_data(tmp_path), [4, 6, 8], 4, 3)
    assert data["inputs"].shape[1] == 15
    assert data["input_channel_names"].tolist()[3:12] == [
        f"camera_{index}_active" for index in range(9)
    ]
    for source_index in np.unique(data["source_index"]):
        selected = data["source_index"] == source_index
        assert set(data["total_budget"][selected].tolist()) == {4, 6, 8}
        assert len(set(data["split_id"][selected].tolist())) == 1
    assert np.all(data["inputs"][:, 3:12, 0, 0, 0] == data["view_mask"])


def test_audit_camera_and_test_truth_cannot_change_test_input(tmp_path):
    base = tiny_base_data(tmp_path)
    original = prepare_direct_operator_data(base, [4, 6, 8], 4, 3)
    test_source = int(np.flatnonzero(base["split_id"] == 2)[0])
    changed = copy.deepcopy(base)
    changed["clean_observation"][test_source, :, 3, :] += 100.0
    changed["field"][test_source] *= 7.0
    mutated = prepare_direct_operator_data(changed, [4, 6, 8], 4, 3)
    selected = original["source_index"] == test_source
    np.testing.assert_array_equal(original["inputs"][selected], mutated["inputs"][selected])


def test_preparation_is_deterministic_and_per_budget_subsets_are_rectangular(tmp_path):
    base = tiny_base_data(tmp_path)
    first = prepare_direct_operator_data(base, [4, 6, 8], 4, 3)
    second = prepare_direct_operator_data(base, [4, 6, 8], 4, 3)
    np.testing.assert_array_equal(first["observation"], second["observation"])
    np.testing.assert_array_equal(first["inputs"], second["inputs"])
    for budget in (4, 6, 8):
        subset = subset_budget_data(first, budget)
        assert len(subset["field"]) == len(base["field"])
        assert np.all(subset["total_budget"] == budget)
        assert np.array_equal(subset["split_names"], first["split_names"])


def test_ridge_reconstruction_reduces_selected_view_residual(tmp_path):
    base = tiny_base_data(tmp_path)
    data = prepare_direct_operator_data(base, [4], 4, 3)
    index = 0
    prediction = ridge_reconstruct(
        data["observation"][index],
        data["forward_matrix"],
        data["view_mask"][index],
        1e-5,
        data["support"],
    )
    projected = forward_volume(prediction, data["forward_matrix"])
    mask = data["view_mask"][index][None, :, None]
    observed = data["observation"][index]
    ridge_error = np.linalg.norm((projected - observed) * mask)
    zero_error = np.linalg.norm(observed * mask)
    assert ridge_error < zero_error


def test_model_seeds_collapse_inside_field():
    rows = []
    for seed, value in [(1, 2.0), (2, 4.0), (3, 6.0)]:
        row = {
            "model_seed": seed,
            "source_index": 9,
            "sample_seed": 99,
            "source_split": "test_iid",
            "family_id": 0,
            "noise_level": 0.02,
            "total_budget": 4,
            "method": "unet",
            "classical_champion": "ridge",
            "ridge_relative": 1e-5,
        }
        row.update({metric: value for metric in [
            "field_rel_l2",
            "gradient_rel_l2",
            "observed_reprojection_rel_l2",
            "audit_reprojection_rel_l2",
            "improvement_vs_classical_pct",
        ]})
        rows.append(row)
    collapsed = collapse_model_seeds(rows)
    assert len(collapsed) == 1
    assert collapsed[0]["model_seed_count"] == 3
    assert collapsed[0]["field_rel_l2"] == 4.0
