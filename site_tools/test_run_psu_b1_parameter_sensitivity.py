from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from site_tools.run_psu_b1_parameter_sensitivity import (
    audit_parameter_sensitivity_view,
    load_frozen_config,
    run_all_view_parameter_sensitivity,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_view(root: Path, view_id: int) -> Path:
    view_dir = root / f"view_{view_id:02d}"
    bundle = view_dir / "bundle"
    setup = view_dir / "setup"
    masks = view_dir / "corrected_masks"
    bundle.mkdir(parents=True)
    setup.mkdir(parents=True)
    masks.mkdir(parents=True)

    origin = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-2.0, 0.25, 0.0],
            [-2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    direction = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(bundle / "c.npy", origin, allow_pickle=False)
    np.save(bundle / "v.npy", direction, allow_pickle=False)
    np.save(setup / "cam_data.npy", np.zeros((3, 18), dtype=np.float32))
    np.save(setup / "geometry_flags.npy", np.zeros(3, dtype=np.uint8))
    np.save(
        masks / "amask_all_zero_based.npy",
        np.array([0, 1], dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        masks / "imask_all_zero_based.npy",
        np.array([2], dtype=np.int64),
        allow_pickle=False,
    )
    _write_json(
        bundle / "view_bundle_manifest.json",
        {
            "status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
            "view": {"view_id_zero_based": view_id},
        },
    )
    _write_json(
        setup / "streamed_setup_manifest.json",
        {"status": "STREAMED_SETUP_DIAGNOSTIC"},
    )
    _write_json(
        masks / "corrected_view_masks_manifest.json",
        {"status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"},
    )
    return view_dir


def _make_config(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "psu-b1-parameter-sensitivity-preregistration-1.0",
            "registration_status": "FROZEN_BEFORE_REAL_SCORING",
            "frozen_at_utc": "2026-07-16T00:00:00Z",
            "baseline_variant_id": "reference",
            "outer_box": {
                "minimum_m": [-1.0, -1.0, -1.0],
                "maximum_m": [1.0, 1.0, 1.0],
            },
            "variants": [
                {
                    "id": "reference",
                    "family": "reference",
                    "role": "test reference",
                    "cone_vertex_m": [0.0, 0.0, 0.0],
                    "cone_axis": [1.0, 0.0, 0.0],
                    "cone_angle_degrees": 45.0,
                },
                {
                    "id": "axis_sign_flip",
                    "family": "axis_semantics",
                    "role": "test sign flip",
                    "cone_vertex_m": [0.0, 0.0, 0.0],
                    "cone_axis": [-1.0, 0.0, 0.0],
                    "cone_angle_degrees": 45.0,
                },
            ],
            "interpretation_rules": ["no selection"],
            "selection_policy": "NO_PARAMETER_SELECTION_FROM_TEST",
        },
    )
    return path


def test_axis_flip_detects_spatial_support_change_even_when_lengths_match(
    tmp_path: Path,
) -> None:
    view_dir = _make_view(tmp_path / "audit", 0)
    config_path = _make_config(tmp_path / "config.json")
    config = load_frozen_config(config_path)
    report = audit_parameter_sensitivity_view(
        view_dir=view_dir,
        config=config,
        config_path=config_path,
        chunk_rows=2,
    )

    variants = {variant["id"]: variant for variant in report["variants"]}
    reference = variants["reference"]["scopes"]["active"]
    flipped = variants["axis_sign_flip"]["scopes"]["active"]
    assert report["status"].endswith("MECHANICAL_PASS_SELECTION_LOCKED")
    assert reference["changed_interval_count"] == 0
    assert reference["ray_support_length_iou"] == pytest.approx(1.0)
    assert flipped["candidate_length_sum_m"] == pytest.approx(
        flipped["baseline_length_sum_m"]
    )
    assert flipped["hit_disagreement_count"] == 0
    assert flipped["changed_interval_count"] == 2
    assert flipped["ray_support_length_iou"] == pytest.approx(0.0)
    assert flipped["maximum_abs_endpoint_delta_m"] >= 1.0


def test_config_rejects_unfrozen_or_selectable_design(tmp_path: Path) -> None:
    path = _make_config(tmp_path / "config.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["registration_status"] = "DRAFT"
    _write_json(path, value)
    with pytest.raises(ValueError, match="frozen"):
        load_frozen_config(path)

    value["registration_status"] = "FROZEN_BEFORE_REAL_SCORING"
    value["selection_policy"] = "PICK_BEST_RESULT"
    _write_json(path, value)
    with pytest.raises(ValueError, match="lock"):
        load_frozen_config(path)


def test_all_view_runner_writes_private_report_and_metrics(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    _make_view(audit_root, 0)
    config_path = _make_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    csv_output = tmp_path / "metrics.csv"
    report = run_all_view_parameter_sensitivity(
        audit_root=audit_root,
        config_path=config_path,
        output_path=output,
        csv_output_path=csv_output,
        view_count=1,
        chunk_rows=2,
    )

    assert report["status"].endswith("PHYSICAL_SELECTION_LOCKED")
    assert report["decision"]["selected_variant_id"] is None
    assert output.exists()
    assert csv_output.exists()
    assert not list(tmp_path.glob(".*.partial"))
    with csv_output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 3 * 2
    assert {row["level"] for row in rows} == {"view", "pooled"}
