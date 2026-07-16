from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from site_tools.validate_psu_heldout_camera_protocol import (
    SUMMARY_SCHEMA,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "demo_t16_operator" / "configs" / "psu_heldout_camera_protocol_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_protocol_partitions_all_70_views_without_leakage() -> None:
    summary = validate_protocol(_config())

    assert summary["schema_version"] == SUMMARY_SCHEMA
    assert summary["dataset"]["view_count"] == 70
    assert summary["checks"]["all_70_views_covered_exactly_once"]
    assert summary["checks"]["released_9_view_support_reproduced"]
    counts = {split["id"]: split["view_count"] for split in summary["partition"]}
    assert counts == {
        "support_reconstruction": 9,
        "development_rotation_40": 7,
        "audit_rotation_same_cameras": 18,
        "audit_camera_same_runs": 12,
        "audit_joint_camera_rotation": 24,
    }
    assert summary["claim_boundary"]["algorithm_superiority"].startswith("LOCKED")


def test_rejects_overlap_between_development_and_final_audit() -> None:
    config = copy.deepcopy(_config())
    primary = next(
        split
        for split in config["splits"]
        if split["id"] == "audit_rotation_same_cameras"
    )
    primary["rotation_degrees"].remove(10)
    primary["rotation_degrees"].append(40)
    with pytest.raises(ValueError, match="overlaps"):
        validate_protocol(config)


def test_rejects_missing_view_or_pixel_independence_claim() -> None:
    config = copy.deepcopy(_config())
    joint = next(
        split
        for split in config["splits"]
        if split["id"] == "audit_joint_camera_rotation"
    )
    joint["camera_ids"].remove(7)
    joint["camera_ids"].append(8)
    with pytest.raises(ValueError, match="partition all 70"):
        validate_protocol(config)

    config = _config()
    config["dataset"]["pixel_independence_assumption"] = True
    with pytest.raises(ValueError, match="explicitly rejected"):
        validate_protocol(config)


def test_rejects_audit_access_for_model_selection() -> None:
    config = _config()
    config["data_access_firewall"]["audit_may_influence_model_selection"] = True
    with pytest.raises(ValueError, match="must be false"):
        validate_protocol(config)
