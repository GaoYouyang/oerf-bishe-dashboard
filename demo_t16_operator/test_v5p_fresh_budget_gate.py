from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from .run_v5p_fresh_budget_gate import (
    CONFIG_PATH,
    _target_prediction,
    fresh_dataset_config,
    validate_preregistered_config,
)


def test_v5p_preregistration_freezes_non_dominating_source_call_budget() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_preregistered_config(config)
    candidate = config["candidate"]
    baseline = config["primary_baseline"]
    assert candidate["source_forward_calls_per_field"] <= baseline[
        "source_forward_calls_per_field"
    ]
    assert candidate["source_adjoint_calls_per_field"] <= baseline[
        "source_adjoint_calls_per_field"
    ]
    assert len(config["rigs"]) * len(config["families"]) == 18


def test_v5p_dataset_config_constructs_validation_only() -> None:
    preregistration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base = {
        "seed": 1,
        "fields_per_family": 1,
        "rigs": [],
        "splits": {},
        "camera_sigma": [0.1] * 7,
    }
    config = fresh_dataset_config(base, preregistration)
    assert config["splits"]["train"]["families"] == []
    assert config["splits"]["design_lock"]["families"] == []
    assert config["splits"]["validation"]["families"] == preregistration[
        "families"
    ]
    assert {rig["split"] for rig in config["rigs"]} == {"validation"}


def test_target_prediction_firewall_does_not_read_target_label() -> None:
    class GuardedRow(dict):
        def __getitem__(self, key):
            if key == "target_residual_label":
                raise AssertionError("target label crossed the prediction firewall")
            return super().__getitem__(key)

    row = GuardedRow(
        row_index=0,
        field_uid="field-0",
        target_operator=np.eye(2, dtype=np.float32),
    )
    bundle = SimpleNamespace(rows=[row])
    indices, prediction, _ = _target_prediction(
        bundle, {"field-0": np.asarray([0.2, -0.1], dtype=np.float32)}
    )
    assert indices == [0]
    np.testing.assert_allclose(prediction, [[0.2, -0.1]])
