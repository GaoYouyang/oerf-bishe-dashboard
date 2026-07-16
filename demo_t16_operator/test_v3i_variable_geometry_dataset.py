from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from demo_t16_operator.data import generate_dataset, load_npz
from demo_t16_operator.variable_geometry import (
    assign_geometry_partitions,
    balanced_source_geometry_assignment,
    build_geometry_manifest,
    build_variable_geometry_operator_data,
    reference_mask_id,
)


ROOT = Path(__file__).resolve().parent


def small_base_data(tmp_path: Path) -> dict[str, np.ndarray]:
    config = {
        "name": "v3i_test",
        "seed": 9401,
        "grid_size": 8,
        "depth": 4,
        "max_views": 9,
        "splits": {
            "train": {"count": 32, "families": ["gaussian"], "views": [6], "noise": [0.02]},
            "val": {"count": 8, "families": ["flame"], "views": [6], "noise": [0.02]},
            "test_iid": {"count": 8, "families": ["gaussian"], "views": [6], "noise": [0.02]},
            "test_noise_ood": {"count": 8, "families": ["flame"], "views": [6], "noise": [0.08]},
            "test_family_ood": {"count": 8, "families": ["thin_front"], "views": [6], "noise": [0.02]},
            "test_joint_ood": {"count": 8, "families": ["thin_front"], "views": [6], "noise": [0.08]}
        }
    }
    path = tmp_path / "base.npz"
    generate_dataset(config, path)
    return load_npz(path)


def manifest(data: dict[str, np.ndarray]):
    rows, masks = build_geometry_manifest(
        data["angles"], data["forward_matrix"], 6, 3, 180.0
    )
    rows = assign_geometry_partitions(
        rows,
        reference_mask_id([1, 0, 1, 0, 1, 1, 1, 0, 1]),
        20261201,
        {"train": 16, "validation": 4, "geometry_ood": 4, "stress": 4},
    )
    return rows, masks


def split_map() -> dict[str, str]:
    return {
        "train": "train",
        "val": "validation",
        "test_iid": "geometry_ood",
        "test_noise_ood": "geometry_ood",
        "test_family_ood": "stress",
        "test_joint_ood": "stress",
    }


def test_balanced_assignment_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    data = small_base_data(tmp_path)
    rows, _ = manifest(data)
    first = balanced_source_geometry_assignment(data, rows, split_map(), 20261221)
    second = balanced_source_geometry_assignment(data, rows, split_map(), 20261221)
    assert first == second
    assert [row["source_index"] for row in first] == list(range(len(data["field"])))
    train = [row for row in first if row["source_split"] == "train"]
    val = [row for row in first if row["source_split"] == "val"]
    assert set(row["geometry_id"] for row in train).isdisjoint(
        row["geometry_id"] for row in val
    )
    assert set(Counter(row["geometry_id"] for row in train).values()) == {2}
    assert set(Counter(row["geometry_id"] for row in val).values()) == {2}


def test_variable_dataset_has_one_source_one_geometry_and_no_audit_leak(tmp_path: Path) -> None:
    data = small_base_data(tmp_path)
    rows, masks = manifest(data)
    packed, assignments = build_variable_geometry_operator_data(
        data, rows, masks, split_map(), 20261221, 1e-6, 6, 3
    )
    assert len(assignments) == len(data["field"])
    assert packed["inputs"].shape == (72, 42, 4, 8, 8)
    assert np.array_equal(packed["source_index"], np.arange(72))
    assert np.all(packed["view_mask"].sum(axis=1) == 6)
    assert np.all(packed["view_mask"][:, 3] == 0)
    assert len(set(packed["geometry_id"].tolist())) == 28
    assert bool(packed["shared_full_view_noise"])
    names = packed["input_channel_names"].tolist()
    assert names[3] == "camera_0_active"
    assert int(packed["ray_view_channel_start"]) == 15
    assert int(packed["ray_angle_sin_channel_start"]) == 24
    assert int(packed["ray_angle_cos_channel_start"]) == 33


def test_v3i_config_keeps_dataset_private_and_claims_closed() -> None:
    config = json.loads(
        (ROOT / "configs" / "v3i_variable_geometry_dataset.json").read_text()
    )
    assert config["claims_boundary"]["private_dataset_npz_public"] is False
    assert config["claims_boundary"]["functional_training_completed"] is False
    assert config["claims_boundary"]["superiority_tested"] is False
    assert config["claims_boundary"]["blind_final_opened"] is False
