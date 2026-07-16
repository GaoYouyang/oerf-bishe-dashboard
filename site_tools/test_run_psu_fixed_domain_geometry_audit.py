from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from site_tools.run_psu_fixed_domain_geometry_audit import (
    aggregate_fixed_domain_views,
    audit_fixed_domain_view,
    run_all_view_fixed_domain_audit,
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
            [-2.0, 0.5, 0.0],
            [-2.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    direction = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    cam_data = np.zeros((5, 18), dtype=np.float32)
    cam_data[:, 0] = [2.0, 2.0, 0.0, 2.0, 0.5]
    geometry_flags = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
    np.save(bundle / "c.npy", origin, allow_pickle=False)
    np.save(bundle / "v.npy", direction, allow_pickle=False)
    np.save(setup / "cam_data.npy", cam_data, allow_pickle=False)
    np.save(setup / "geometry_flags.npy", geometry_flags, allow_pickle=False)
    np.save(
        masks / "amask_all_zero_based.npy",
        np.array([0, 1, 4], dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        masks / "imask_all_zero_based.npy",
        np.array([2, 3], dtype=np.int64),
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
        {"status": "STREAMED_SETUP_DIAGNOSTIC_NO_GO"},
    )
    _write_json(
        masks / "corrected_view_masks_manifest.json",
        {"status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"},
    )
    return view_dir


def test_real_view_style_audit_separates_b0_and_b1(tmp_path: Path) -> None:
    view_dir = _make_view(tmp_path, 0)
    report = audit_fixed_domain_view(
        view_dir=view_dir,
        chunk_rows=2,
        outer_minimum=(-1.0, -1.0, -1.0),
        outer_maximum=(1.0, 1.0, 1.0),
        cone_vertex=(0.0, 0.0, 0.0),
        cone_axis=(1.0, 0.0, 0.0),
        cone_angle_degrees=45.0,
    )

    assert report["status"] == "B0_B1_FIXED_DOMAIN_ANALYTIC_CONTRACT_PASS_B2_REQUIRED"
    assert report["counts"]["ray_count"] == 5
    assert report["counts"]["b0_hit_count"] == 4
    assert report["counts"]["b1_hit_count"] == 3
    assert report["counts"]["b1_removed_from_b0_count"] == 1
    assert report["counts"]["b1_hit_without_b0_hit_count"] == 0
    assert report["counts"]["b1_length_exceeds_b0_count"] == 0
    assert report["path_length"]["b0_length_sum_m"] == pytest.approx(6.5)
    assert report["path_length"]["b1_length_sum_m"] == pytest.approx(2.5)
    assert report["mask_conditioned"]["amask_all"]["b1_hit_count"] == 2
    assert report["mask_conditioned"]["imask_all"]["b1_hit_count"] == 1
    assert report["decision"]["training_ready"] == "NO"
    assert not report["decision"]["finite_aperture_sample_support_audited"]


def test_aggregate_requires_ordered_contiguous_views(tmp_path: Path) -> None:
    record = audit_fixed_domain_view(
        view_dir=_make_view(tmp_path, 1),
        chunk_rows=2,
        outer_minimum=(-1.0, -1.0, -1.0),
        outer_maximum=(1.0, 1.0, 1.0),
        cone_vertex=(0.0, 0.0, 0.0),
        cone_axis=(1.0, 0.0, 0.0),
        cone_angle_degrees=45.0,
    )
    with pytest.raises(ValueError, match="ordered contiguous"):
        aggregate_fixed_domain_views([record])


def test_all_view_writer_emits_atomic_json_and_csv(tmp_path: Path) -> None:
    _make_view(tmp_path / "audit", 0)
    output = tmp_path / "report.json"
    csv_output = tmp_path / "metrics.csv"
    report = run_all_view_fixed_domain_audit(
        audit_root=tmp_path / "audit",
        output_path=output,
        csv_output_path=csv_output,
        view_count=1,
        chunk_rows=2,
    )

    assert output.exists()
    assert csv_output.exists()
    assert not list(tmp_path.glob(".*.partial"))
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == report["status"]
    with csv_output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["view_id_zero_based"] == "0"


def test_unsorted_mask_is_rejected(tmp_path: Path) -> None:
    view_dir = _make_view(tmp_path, 0)
    np.save(
        view_dir / "corrected_masks" / "amask_all_zero_based.npy",
        np.array([1, 0], dtype=np.int64),
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        audit_fixed_domain_view(view_dir=view_dir)
