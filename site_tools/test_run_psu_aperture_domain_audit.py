from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from site_tools.run_psu_aperture_domain_audit import (
    aggregate_aperture_domain_views,
    audit_aperture_domain_view,
    run_all_view_aperture_domain_audit,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_view(root: Path, view_id: int) -> Path:
    view = root / f"view_{view_id:02d}"
    bundle = view / "bundle"
    setup = view / "setup"
    masks = view / "corrected_masks"
    bundle.mkdir(parents=True)
    setup.mkdir(parents=True)
    masks.mkdir(parents=True)
    rows = 3
    np.save(
        bundle / "c.npy",
        np.array([[-2.0, 0.0, 0.0], [-2.0, 0.8, 0.0], [-2.0, 2.0, 0.0]], dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        bundle / "v.npy",
        np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (rows, 1)),
        allow_pickle=False,
    )
    np.save(
        bundle / "Rxvecs.npy",
        np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (rows, 1)),
        allow_pickle=False,
    )
    np.save(
        bundle / "Ryvecs.npy",
        np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (rows, 1)),
        allow_pickle=False,
    )
    np.save(
        bundle / "Rapvec.npy",
        np.array([[2.0], [0.0], [0.0]], dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        bundle / "Dfvec.npy",
        np.full((rows, 1), 10.0, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(setup / "cam_data.npy", np.zeros((rows, 18), dtype=np.float32))
    np.save(setup / "geometry_flags.npy", np.zeros(rows, dtype=np.uint8))
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
        {"status": "STREAMED_SETUP_MECHANICAL_CONTRACT_PASS"},
    )
    _write_json(
        masks / "corrected_view_masks_manifest.json",
        {"status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"},
    )
    return view


def test_view_audit_preserves_fixed_denominator_and_mask_context(tmp_path: Path) -> None:
    report = audit_aperture_domain_view(
        view_dir=_make_view(tmp_path, 0),
        sample_count=8,
        chunk_rows=2,
        outer_minimum=(-1.0, -1.0, -1.0),
        outer_maximum=(1.0, 1.0, 1.0),
        cone_vertex=(0.0, 0.0, 0.0),
        cone_axis=(1.0, 0.0, 0.0),
        cone_angle_degrees=45.0,
    )

    assert report["status"] == "B2_DETERMINISTIC_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED"
    assert report["domains"]["B0"]["centerline_hit_count"] == 2
    assert report["domains"]["B1"]["centerline_hit_count"] == 2
    assert report["domains"]["B0"]["eligible_sample_count"] == 16
    assert report["domains"]["B0"]["in_domain_sample_count"] < 16
    assert report["domains"]["B0"]["any_sample_out_of_domain_ray_count"] == 1
    assert report["mask_conditioned"]["amask_all"]["B0"]["ray_count"] == 2
    assert report["mask_conditioned"]["imask_all"]["B0"]["centerline_hit_count"] == 0
    assert report["configuration"]["normalization_policy"] == (
        "FIXED_ORIGINAL_SAMPLE_COUNT_NO_SURVIVOR_RENORMALIZATION"
    )
    assert report["decision"]["training_ready"] == "NO"
    assert not report["decision"]["continuous_aperture_containment_proved"]


def test_aggregate_reconciles_histograms(tmp_path: Path) -> None:
    record = audit_aperture_domain_view(
        view_dir=_make_view(tmp_path, 0), sample_count=4, chunk_rows=2
    )
    aggregate = aggregate_aperture_domain_views([record])
    assert aggregate["execution_status"] == "COMPLETE"
    assert aggregate["sample_count_per_centerline_hit"] == 4
    for domain in ("B0", "B1"):
        histogram = aggregate["aggregate"]["domains"][domain][
            "retained_sample_count_histogram"
        ]
        assert sum(histogram) == aggregate["aggregate"]["domains"][domain][
            "centerline_hit_count"
        ]


def test_all_view_runner_is_deterministic_and_atomic(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    _make_view(audit_root, 0)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = run_all_view_aperture_domain_audit(
        audit_root=audit_root,
        output_path=first_path,
        view_count=1,
        sample_count=4,
        chunk_rows=2,
    )
    second = run_all_view_aperture_domain_audit(
        audit_root=audit_root,
        output_path=second_path,
        view_count=1,
        sample_count=4,
        chunk_rows=2,
    )
    first["views"][0].pop("runtime_observation")
    second["views"][0].pop("runtime_observation")
    assert first == second
    assert first_path.exists() and second_path.exists()
    assert not list(tmp_path.glob(".*.partial"))
