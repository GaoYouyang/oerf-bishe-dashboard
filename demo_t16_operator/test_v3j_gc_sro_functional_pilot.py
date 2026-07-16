from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from demo_t16_operator.run_v3j_gc_sro_functional_pilot import geometry_derangement


ROOT = Path(__file__).resolve().parent


def test_v3j_config_is_matched_and_keeps_claims_closed() -> None:
    config = json.loads(
        (ROOT / "configs" / "v3j_gc_sro_functional_pilot.json").read_text()
    )
    assert config["methods"] == [
        "static",
        "k_cardinality",
        "shuffled_geometry",
        "correct_geometry",
    ]
    assert len(config["model_seeds"]) == 3
    assert config["training"]["epochs"] == 24
    assert config["functional_gate"]["required_positive_seed_count"] == 3
    assert config["claims_boundary"]["matched_variable_geometry_fno_trained"] is False
    assert config["claims_boundary"]["superiority_tested"] is False
    assert config["claims_boundary"]["blind_final_opened"] is False


def test_geometry_derangement_is_partition_local_and_fixed_point_free() -> None:
    identifiers = np.asarray(
        ["g_1010", "g_0110", "g_1100", "g_0011", "g_0101", "g_1001"]
    )
    partitions = np.asarray(["train", "train", "train", "validation", "validation", "validation"])
    data = {
        "geometry_id": identifiers,
        "geometry_partition": partitions,
        "angles": np.asarray([0.0, 45.0, 90.0, 135.0], dtype=np.float32),
    }
    rows, components = geometry_derangement(data)
    partition = {str(identifier): str(value) for identifier, value in zip(identifiers, partitions)}
    assert len(rows) == 6
    assert all(row["correct_geometry_id"] != row["wrong_geometry_id"] for row in rows)
    assert all(
        partition[str(row["correct_geometry_id"])]
        == partition[str(row["wrong_geometry_id"])]
        for row in rows
    )
    assert components[0].shape == (6, 4)
    assert np.all(components[0].sum(axis=1) == 2)
    assert np.allclose(components[1] ** 2 + components[2] ** 2, components[0])
