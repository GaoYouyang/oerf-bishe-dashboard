#!/usr/bin/env python3
"""Validate and summarize the frozen PSU 70-view held-out camera protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "psu-heldout-camera-preregistration-1.0"
STATUS = "FROZEN_BEFORE_ARCHIVE_04_05_PAYLOAD_SCORING"
SUMMARY_SCHEMA = "psu-heldout-camera-protocol-public-summary-1.0"
EXPECTED_SUPPORT_CAMERAS = (2, 3, 4)
EXPECTED_SUPPORT_ROTATIONS = (0, 50, 90)
EXPECTED_PRIMARY_AUDIT_ROTATIONS = (10, 20, 30, 60, 70, 80)
EXPECTED_SPLIT_COUNTS = {
    "support_reconstruction": 9,
    "development_rotation_40": 7,
    "audit_rotation_same_cameras": 18,
    "audit_camera_same_runs": 12,
    "audit_joint_camera_rotation": 24,
}


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _integers(value: Any, location: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{location} must be an integer array")
    if len(set(value)) != len(value):
        raise ValueError(f"{location} must not contain duplicates")
    return value


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid held-out protocol: {path}") from exc
    return dict(_mapping(value, str(path)))


def _pairs(split: Mapping[str, Any], location: str) -> set[tuple[int, int]]:
    cameras = _integers(split.get("camera_ids"), f"{location}.camera_ids")
    rotations = _integers(
        split.get("rotation_degrees"), f"{location}.rotation_degrees"
    )
    return {(rotation, camera) for rotation in rotations for camera in cameras}


def validate_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unsupported held-out protocol schema")
    if config.get("registration_status") != STATUS:
        raise ValueError("held-out protocol is not frozen before payload scoring")

    dataset = _mapping(config.get("dataset"), "dataset")
    cameras = _integers(dataset.get("camera_ids"), "dataset.camera_ids")
    rotations = _integers(
        dataset.get("rotation_degrees"), "dataset.rotation_degrees"
    )
    if cameras != list(range(1, 8)):
        raise ValueError("dataset cameras must be exactly 1 through 7")
    if rotations != list(range(0, 100, 10)):
        raise ValueError("dataset rotations must be exactly 0 through 90 by 10")
    universe = {(rotation, camera) for rotation in rotations for camera in cameras}
    if int(dataset.get("view_count", -1)) != len(universe):
        raise ValueError("dataset view_count does not match the camera-rotation grid")
    if dataset.get("independent_experimental_unit") != "MODEL_ROTATION_RUN_BLOCK":
        raise ValueError("independent experimental unit must be the rotation run block")
    if dataset.get("pixel_independence_assumption") is not False:
        raise ValueError("pixel independence must be explicitly rejected")

    splits_raw = config.get("splits")
    if not isinstance(splits_raw, list):
        raise ValueError("splits must be an array")
    splits = [_mapping(value, f"splits[{index}]") for index, value in enumerate(splits_raw)]
    ids = [str(split.get("id", "")) for split in splits]
    if set(ids) != set(EXPECTED_SPLIT_COUNTS) or len(ids) != len(EXPECTED_SPLIT_COUNTS):
        raise ValueError("split ids must match the frozen five-way protocol")

    owner: dict[tuple[int, int], str] = {}
    split_pairs: dict[str, set[tuple[int, int]]] = {}
    for index, split in enumerate(splits):
        split_id = str(split["id"])
        pairs = _pairs(split, f"splits[{index}]")
        expected_count = EXPECTED_SPLIT_COUNTS[split_id]
        if int(split.get("expected_view_count", -1)) != expected_count:
            raise ValueError(f"{split_id} expected_view_count must be {expected_count}")
        if len(pairs) != expected_count:
            raise ValueError(f"{split_id} Cartesian product must contain {expected_count} views")
        for pair in pairs:
            if pair in owner:
                raise ValueError(
                    f"view {pair} overlaps {owner[pair]} and {split_id}"
                )
            owner[pair] = split_id
        split_pairs[split_id] = pairs

    missing = sorted(universe - set(owner))
    extra = sorted(set(owner) - universe)
    if missing or extra:
        raise ValueError(
            f"the five splits must partition all 70 views; missing={missing}, extra={extra}"
        )

    support = split_pairs["support_reconstruction"]
    expected_support = {
        (rotation, camera)
        for rotation in EXPECTED_SUPPORT_ROTATIONS
        for camera in EXPECTED_SUPPORT_CAMERAS
    }
    if support != expected_support:
        raise ValueError("support split does not reproduce the released nine-view set")

    development = split_pairs["development_rotation_40"]
    if {rotation for rotation, _ in development} != {40}:
        raise ValueError("development split must be the complete 40-degree run")

    primary = split_pairs["audit_rotation_same_cameras"]
    if {rotation for rotation, _ in primary} != set(
        EXPECTED_PRIMARY_AUDIT_ROTATIONS
    ):
        raise ValueError("primary audit rotations do not match the frozen six blocks")
    if {camera for _, camera in primary} != set(EXPECTED_SUPPORT_CAMERAS):
        raise ValueError("primary audit must use the same cameras as support")

    firewall = _mapping(config.get("data_access_firewall"), "data_access_firewall")
    for key in (
        "audit_may_influence_model_selection",
        "audit_may_influence_stopping",
        "audit_may_influence_parameter_selection",
    ):
        if firewall.get(key) is not False:
            raise ValueError(f"{key} must be false")
    primary_endpoint = _mapping(config.get("primary_endpoint"), "primary_endpoint")
    if primary_endpoint.get("name") != "rotation_block_active_vector_relative_l2":
        raise ValueError("unexpected primary endpoint")
    success = _mapping(config.get("success_gate"), "success_gate")
    if "1/64" not in str(success.get("primary_sign_gate", "")):
        raise ValueError("primary sign gate must declare the exact six-block probability")

    split_summary = []
    for split in splits:
        split_id = str(split["id"])
        pairs = split_pairs[split_id]
        split_summary.append(
            {
                "id": split_id,
                "role": str(split.get("role", "")),
                "access_stage": str(split.get("access_stage", "")),
                "camera_ids": sorted({camera for _, camera in pairs}),
                "rotation_degrees": sorted({rotation for rotation, _ in pairs}),
                "view_count": len(pairs),
            }
        )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "PSU_70_VIEW_PARTITION_VALIDATED_AUDIT_PAYLOAD_SEALED",
        "dataset": {
            "name": dataset["name"],
            "doi": dataset["doi"],
            "paper": dataset["paper"],
            "camera_count": len(cameras),
            "rotation_run_count": len(rotations),
            "view_count": len(universe),
            "independent_experimental_unit": dataset[
                "independent_experimental_unit"
            ],
            "pixel_independence_assumption": False,
        },
        "partition": split_summary,
        "checks": {
            "all_70_views_covered_exactly_once": True,
            "released_9_view_support_reproduced": True,
            "development_is_complete_rotation_40_run": True,
            "primary_audit_has_six_unseen_rotation_blocks": True,
            "primary_audit_uses_support_camera_ids": True,
            "final_audit_forbidden_from_selection_and_stopping": True,
        },
        "primary_endpoint": dict(primary_endpoint),
        "success_gate": dict(success),
        "claim_boundary": {
            "field_l2_available": False,
            "pixel_bootstrap_permitted": False,
            "held_out_reprojection_is_unique_3d_truth": False,
            "experimental_superiority_without_flowoff_repeatability": False,
            "algorithm_superiority": "LOCKED_UNTIL_SINGLE_FINAL_AUDIT",
        },
        "reporting_rules": list(config["reporting_rules"]),
    }


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = validate_protocol(_load(args.config))
    if args.output is not None:
        write_summary(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
