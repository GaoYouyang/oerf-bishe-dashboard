from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from demo_t16_operator.counterfactual_geometry import (
    CounterfactualGeometryDataset,
    CounterfactualInputFactory,
    build_pair_schedule,
    descriptor_components_for_pairs,
    geometry_derangement_map,
    schedule_balance,
)
from demo_t16_operator.variable_geometry import geometry_id


ROOT = Path(__file__).resolve().parent


def synthetic_geometry_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261471)
    view_count = 9
    audit = 3
    masks = []
    for active in itertools.combinations(
        [index for index in range(view_count) if index != audit], 6
    ):
        mask = np.zeros(view_count, dtype=np.float32)
        mask[list(active)] = 1.0
        masks.append(mask)
    assert len(masks) == 28
    partitions = np.asarray(
        ["train"] * 16
        + ["validation"] * 4
        + ["geometry_ood"] * 4
        + ["stress"] * 4
    )
    split_names = np.asarray(
        [
            "train",
            "val",
            "test_iid",
            "test_noise_ood",
            "test_family_ood",
            "test_joint_ood",
        ]
    )
    split_id = np.asarray([0] * 16 + [1] * 4 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 2)
    depth = 2
    height = width = detector = 4
    names = [
        "validation_tuned_ridge_lift",
        "support",
        "view_fraction",
        *[f"camera_{index}_active" for index in range(view_count)],
        "z",
        "y",
        "x",
        *[f"ray_backprojection_camera_{index}" for index in range(view_count)],
        *[f"camera_{index}_sin_active" for index in range(view_count)],
        *[f"camera_{index}_cos_active" for index in range(view_count)],
    ]
    inputs = np.zeros((28, len(names), depth, height, width), dtype=np.float32)
    coordinates = np.meshgrid(
        np.linspace(-1.0, 1.0, depth, dtype=np.float32),
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    for axis, grid in zip(("z", "y", "x"), coordinates):
        inputs[:, names.index(axis)] = grid
    observation = rng.normal(size=(28, depth, view_count, detector)).astype(np.float32)
    return {
        "inputs": inputs,
        "input_channel_names": np.asarray(names),
        "field": np.abs(rng.normal(size=(28, depth, height, width))).astype(np.float32),
        "observation": observation,
        "clean_observation": observation.copy(),
        "forward_matrix": rng.normal(
            size=(view_count, detector, height * width)
        ).astype(np.float32),
        "support": np.ones((depth, height, width), dtype=np.float32),
        "angles": np.linspace(0.0, 160.0, view_count, dtype=np.float32),
        "view_mask": np.asarray(masks),
        "geometry_id": np.asarray([geometry_id(mask) for mask in masks]),
        "geometry_partition": partitions,
        "split_names": split_names,
        "split_id": split_id,
        "source_index": np.arange(28, dtype=np.int64),
        "sample_seed": np.arange(1000, 1028, dtype=np.int64),
        "family_id": np.arange(28, dtype=np.int64) % 4,
        "noise_level": np.full(28, 0.01, dtype=np.float64),
        "total_budget": np.full(28, 6, dtype=np.int64),
        "audit_query_index": np.asarray(audit, dtype=np.int64),
        "ray_view_channel_start": np.asarray(15, dtype=np.int64),
        "ray_view_channel_count": np.asarray(9, dtype=np.int64),
        "ray_angle_sin_channel_start": np.asarray(24, dtype=np.int64),
        "ray_angle_cos_channel_start": np.asarray(33, dtype=np.int64),
        "ray_view_scales": np.ones(9, dtype=np.float32),
    }


def test_v3k_config_locks_equal_exposure_and_claims() -> None:
    config = json.loads(
        (ROOT / "configs" / "v3k_a_counterfactual_supervision.json").read_text()
    )
    assert config["training_arms"] == ["m1_repeat", "m4_counterfactual"]
    assert config["pair_design"]["repeats_per_source"] == 4
    assert config["methods"] == [
        "static",
        "k_cardinality",
        "shuffled_geometry",
        "correct_geometry",
    ]
    assert len(config["model_seeds"]) == 3
    assert config["training"]["epochs"] == 24
    assert config["mechanism_gate"]["minimum_mean_gain_pct"] == 0.25
    assert config["claims_boundary"]["superiority_tested"] is False
    assert config["claims_boundary"]["blind_final_opened"] is False


def test_equal_exposure_schedules_isolate_layout_diversity() -> None:
    data = synthetic_geometry_data()
    kwargs = {
        "data": data,
        "source_split": "train",
        "geometry_partition": "train",
        "repeats_per_source": 4,
        "assignment_seed": 20261371,
        "counterfactual_stride": 5,
    }
    m1 = build_pair_schedule(arm="m1_repeat", **kwargs)
    m4 = build_pair_schedule(arm="m4_counterfactual", **kwargs)
    m1_balance = schedule_balance(m1)
    m4_balance = schedule_balance(m4)
    assert m1_balance["row_count"] == m4_balance["row_count"] == 64
    assert m1_balance["source_count"] == m4_balance["source_count"] == 16
    assert m1_balance["minimum_rows_per_geometry"] == 4
    assert m1_balance["maximum_rows_per_geometry"] == 4
    assert m4_balance["minimum_rows_per_geometry"] == 4
    assert m4_balance["maximum_rows_per_geometry"] == 4
    assert m1_balance["minimum_unique_geometries_per_source"] == 1
    assert m1_balance["maximum_unique_geometries_per_source"] == 1
    assert m4_balance["minimum_unique_geometries_per_source"] == 4
    assert m4_balance["maximum_unique_geometries_per_source"] == 4
    assert {row["source_index"] for row in m1} == {row["source_index"] for row in m4}


def test_lazy_inputs_are_geometry_consistent_and_audit_safe() -> None:
    data = synthetic_geometry_data()
    factory = CounterfactualInputFactory(data, ridge_relative=1e-6)
    kwargs = {
        "data": data,
        "source_split": "train",
        "geometry_partition": "train",
        "repeats_per_source": 4,
        "assignment_seed": 20261371,
        "counterfactual_stride": 5,
    }
    m1 = CounterfactualGeometryDataset(
        factory, build_pair_schedule(arm="m1_repeat", **kwargs)
    )
    m4 = CounterfactualGeometryDataset(
        factory, build_pair_schedule(arm="m4_counterfactual", **kwargs)
    )
    first_source = int(m1.pairs[0]["source_index"])
    repeated = [
        m1[index]["x"].numpy()
        for index, row in enumerate(m1.pairs)
        if int(row["source_index"]) == first_source
    ]
    assert len(repeated) == 4
    assert all(np.array_equal(repeated[0], value) for value in repeated[1:])
    distinct_ids = {
        str(row["geometry_id"])
        for row in m4.pairs
        if int(row["source_index"]) == int(m4.pairs[0]["source_index"])
    }
    assert len(distinct_ids) == 4
    sample = m4[0]
    mask = sample["view_mask"].numpy()
    x = sample["x"].numpy()
    assert x.shape == (42, 2, 4, 4)
    assert mask[3] == 0.0
    assert np.all(x[3 + 3] == 0.0)
    assert np.all(x[15 + 3] == 0.0)
    assert np.all(x[24 + 3] == 0.0)
    assert np.all(x[33 + 3] == 0.0)
    assert np.array_equal(x[3:12, 0, 0, 0], mask)
    mapping, rows = geometry_derangement_map(factory.catalog)
    correct = descriptor_components_for_pairs(factory, m4.pairs)
    wrong = descriptor_components_for_pairs(factory, m4.pairs, mapping)
    assert len(rows) == 28
    assert all(not bool(row["fixed_point"]) for row in rows)
    assert np.all(correct[0].sum(axis=1) == 6)
    assert np.all(wrong[0].sum(axis=1) == 6)
    assert np.all(np.any(correct[0] != wrong[0], axis=1))
