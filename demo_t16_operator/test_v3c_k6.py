from __future__ import annotations

import torch
import pytest

from demo_t16_operator.run_v3c_k6_dev2_pilot import (
    build_pairwise_rows,
    maximum_state_drift,
    state_sha256,
    training_config,
)


def test_state_hash_and_drift_detect_parameter_change() -> None:
    model = torch.nn.Linear(3, 2)
    reference = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    digest = state_sha256(reference)
    assert state_sha256(reference) == digest
    assert maximum_state_drift(model, reference) == 0.0
    with torch.no_grad():
        model.weight[0, 0] += 0.25
    assert state_sha256({name: value.detach().clone() for name, value in model.state_dict().items()}) != digest
    assert maximum_state_drift(model, reference) == pytest.approx(0.25)


def test_adaptation_training_config_locks_extra_epochs() -> None:
    source = {
        "training": {
            "device": "auto",
            "epochs": 24,
            "learning_rate": 0.002,
            "early_stop_patience": 6,
        }
    }
    configured = training_config(
        source,
        seed=17,
        epochs=12,
        device="mps",
        learning_rate=0.001,
        fixed_epochs=True,
    )
    assert configured["seed"] == 17
    assert configured["training"]["device"] == "mps"
    assert configured["training"]["epochs"] == 12
    assert configured["training"]["early_stop_patience"] == 12
    assert configured["training"]["learning_rate"] == 0.001
    assert source["training"]["epochs"] == 24


def test_pairwise_rows_use_continued_fno_as_primary_comparator() -> None:
    rows = []
    for method, field, audit in (
        ("base_fno", 0.40, 0.50),
        ("continued_fno", 0.35, 0.45),
        ("zero_init_adapter", 0.30, 0.40),
    ):
        rows.append(
            {
                "source_index": 9,
                "sample_seed": 101,
                "source_split": "test_iid",
                "family_id": 0,
                "noise_level": 0.02,
                "method": method,
                "field_rel_l2": field,
                "audit_reprojection_rel_l2": audit,
                "model_seed_count": 3,
            }
        )
    pairwise = build_pairwise_rows(rows)
    primary = next(row for row in pairwise if row["comparison"] == "adapter_vs_continued_fno")
    continuation = next(
        row for row in pairwise if row["comparison"] == "continued_fno_vs_base_fno"
    )
    adapter_base = next(
        row for row in pairwise if row["comparison"] == "adapter_vs_base_fno"
    )
    assert primary["candidate"] == "zero_init_adapter"
    assert primary["comparator"] == "continued_fno"
    assert abs(float(primary["field_superiority_pct"]) - 100.0 / 7.0) < 1e-9
    assert abs(float(adapter_base["field_superiority_pct"]) - 25.0) < 1e-9
    assert abs(float(continuation["field_superiority_pct"]) - 12.5) < 1e-9
