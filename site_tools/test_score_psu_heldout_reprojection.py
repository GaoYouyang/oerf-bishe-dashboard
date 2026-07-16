from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from site_tools.score_psu_heldout_reprojection import (
    REPORT_SCHEMA,
    score_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "demo_t16_operator" / "configs" / "psu_heldout_camera_protocol_v1.json"
)
ROTATIONS = (10, 20, 30, 60, 70, 80)
CAMERAS = (2, 3, 4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(
    tmp_path: Path,
    *,
    repeatability_floor: float | None = 0.02,
    calibration_passed: bool | None = True,
) -> Path:
    view_dir = tmp_path / "views"
    view_dir.mkdir(parents=True)
    records = []
    for rotation in ROTATIONS:
        for camera in CAMERAS:
            measured = np.column_stack(
                (
                    np.linspace(0.5, 1.5, 16),
                    np.linspace(-0.8, 0.8, 16),
                )
            )
            baseline = measured + np.array([0.16, -0.12])
            candidate = measured + np.array([0.06, -0.04])
            baseline[-4:] = np.array([0.03, -0.02])
            candidate[-4:] = np.array([0.01, -0.005])
            active = np.zeros(16, dtype=bool)
            active[:12] = True
            ambient = np.zeros(16, dtype=bool)
            ambient[-4:] = True
            front_band = np.zeros(16, dtype=bool)
            front_band[:2] = True
            path = view_dir / f"r{rotation:03d}_c{camera:02d}.npz"
            np.savez_compressed(
                path,
                measured_uv_px=measured,
                candidate_uv_px=candidate,
                baseline_uv_px=baseline,
                active_mask=active,
                ambient_mask=ambient,
                front_band_mask=front_band,
            )
            records.append(
                {
                    "rotation_degrees": rotation,
                    "camera_id": camera,
                    "npz": str(path.relative_to(tmp_path)),
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schema_version": "psu-heldout-reprojection-bundle-1.0",
        "registration_status": "FROZEN_BEFORE_SINGLE_FINAL_AUDIT",
        "protocol_sha256": _sha256(PROTOCOL),
        "split_id": "audit_rotation_same_cameras",
        "candidate": {
            "id": "fixture_candidate",
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
        },
        "baseline": {
            "id": "fixture_baseline",
            "checkpoint_sha256": "c" * 64,
            "config_sha256": "d" * 64,
        },
        "development_repeatability_floor_px": repeatability_floor,
        "calibration_perturbation_gate_passed": calibration_passed,
        "views": records,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_scores_six_independent_rotation_blocks_and_passes_fixture(
    tmp_path: Path,
) -> None:
    manifest = _write_bundle(tmp_path)
    report = score_bundle(PROTOCOL, manifest)

    assert report["schema_version"] == REPORT_SCHEMA
    assert report["status"] == "PRIMARY_HELDOUT_GATE_PASS_IMAGE_SPACE_ONLY"
    assert report["aggregate"]["candidate_lower_block_count"] == 6
    assert report["aggregate"]["one_sided_exact_sign_pvalue"] == pytest.approx(
        1 / 64
    )
    assert report["gates"]["all_image_space_gates_pass"]
    assert report["claim"]["field_l2_available"] is False


def test_missing_repeatability_or_calibration_keeps_claim_locked(
    tmp_path: Path,
) -> None:
    manifest = _write_bundle(
        tmp_path,
        repeatability_floor=None,
        calibration_passed=None,
    )
    report = score_bundle(PROTOCOL, manifest)

    assert report["aggregate"]["candidate_lower_block_count"] == 6
    assert not report["gates"]["all_image_space_gates_pass"]
    assert "FLOW_OFF_REPEATABILITY_FLOOR_MISSING" in report["claim"]["lock_reasons"]
    assert "CALIBRATION_PERTURBATION_GATE_NOT_PASSED" in report["claim"][
        "lock_reasons"
    ]


def test_rejects_view_hash_drift_duplicate_or_path_escape(tmp_path: Path) -> None:
    manifest_path = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["views"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        score_bundle(PROTOCOL, manifest_path)

    manifest_path = _write_bundle(tmp_path / "duplicate")
    manifest = json.loads(manifest_path.read_text())
    manifest["views"][1] = copy.deepcopy(manifest["views"][0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        score_bundle(PROTOCOL, manifest_path)

    manifest_path = _write_bundle(tmp_path / "escape")
    manifest = json.loads(manifest_path.read_text())
    manifest["views"][0]["npz"] = "../outside.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        score_bundle(PROTOCOL, manifest_path)
