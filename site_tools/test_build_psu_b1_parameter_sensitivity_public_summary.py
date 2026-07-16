from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from site_tools.build_psu_b1_parameter_sensitivity_public_summary import (
    PUBLIC_SCHEMA,
    build_public_summary,
)


VARIANTS = [
    ("released_reference", "reference", [-1.0, -0.1, 0.0], 25.0),
    ("axis_sign_flip", "axis_semantics", [1.0, 0.1, 0.0], 25.0),
    ("angle_minus_10deg", "angle", [-1.0, -0.1, 0.0], 15.0),
    ("angle_minus_5deg", "angle", [-1.0, -0.1, 0.0], 20.0),
    ("angle_plus_5deg", "angle", [-1.0, -0.1, 0.0], 30.0),
    ("angle_plus_10deg", "angle", [-1.0, -0.1, 0.0], 35.0),
    ("vertex_axis_plus_5mm", "vertex", [-1.0, -0.1, 0.0], 25.0),
    ("vertex_axis_minus_5mm", "vertex", [-1.0, -0.1, 0.0], 25.0),
    ("vertex_xy_normal_plus_5mm", "vertex", [-1.0, -0.1, 0.0], 25.0),
    ("vertex_xy_normal_minus_5mm", "vertex", [-1.0, -0.1, 0.0], 25.0),
    ("vertex_z_plus_5mm", "vertex", [-1.0, -0.1, 0.0], 25.0),
    ("vertex_z_minus_5mm", "vertex", [-1.0, -0.1, 0.0], 25.0),
]


def _config() -> dict:
    return {
        "schema_version": "psu-b1-parameter-sensitivity-preregistration-1.0",
        "registration_status": "FROZEN_BEFORE_REAL_SCORING",
        "frozen_at_utc": "2026-07-16T00:00:00Z",
        "baseline_variant_id": "released_reference",
        "selection_policy": "NO_PARAMETER_SELECTION_FROM_FIXTURE",
        "interpretation_rules": ["no selection"],
        "variants": [
            {
                "id": variant_id,
                "family": family,
                "role": "fixture",
                "cone_vertex_m": [0.06, 0.015, 0.0],
                "cone_axis": axis,
                "cone_angle_degrees": angle,
            }
            for variant_id, family, axis, angle in VARIANTS
        ],
    }


def _scope(index: int, *, baseline: bool = False) -> dict:
    rows = 1000
    hits = 900 if baseline else max(0, 900 - index * 10)
    gained = 0
    lost = 0 if baseline else index * 10
    return {
        "ray_count": rows,
        "candidate_hit_count": hits,
        "candidate_hit_fraction": hits / rows,
        "gained_hit_count": gained,
        "lost_hit_count": lost,
        "hit_disagreement_count": gained + lost,
        "hit_disagreement_fraction": (gained + lost) / rows,
        "changed_interval_count": 0 if baseline else hits + lost,
        "changed_interval_fraction": 0.0 if baseline else (hits + lost) / rows,
        "ray_support_length_iou": 1.0 if baseline else max(0.0, 0.95 - index * 0.04),
        "candidate_path_fraction_of_b0": max(0.0, 0.2 - index * 0.01),
        "candidate_path_relative_signed_change_from_baseline": -index * 0.01,
    }


def _variant(index: int) -> dict:
    variant_id, family, axis, angle = VARIANTS[index]
    norm = sum(value * value for value in axis) ** 0.5
    scope = _scope(index, baseline=index == 0)
    return {
        "id": variant_id,
        "family": family,
        "role": "fixture",
        "configuration": {
            "cone_vertex_m": [0.06, 0.015, 0.0],
            "cone_axis_normalized": [value / norm for value in axis],
            "cone_angle_degrees": angle,
        },
        "mechanical_invariants_pass": True,
        "scopes": {
            "all": copy.deepcopy(scope),
            "active": copy.deepcopy(scope),
            "inactive": copy.deepcopy(scope),
        },
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    variants = [_variant(index) for index in range(len(VARIANTS))]
    report = {
        "schema_version": "psu-b1-parameter-sensitivity-all-view-audit-1.0",
        "status": "B1_PARAMETER_SENSITIVITY_QUANTIFIED_PHYSICAL_SELECTION_LOCKED",
        "scientific_verdict": "MECHANICAL_PASS_PARAMETER_DEPENDENCE_QUANTIFIED_PHYSICAL_SELECTION_REQUIRED",
        "source": {
            "frozen_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()
        },
        "aggregate": {"variants": variants},
        "views": [
            {
                "view_id_zero_based": view_id,
                "variants": copy.deepcopy(variants),
            }
            for view_id in range(9)
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, config_path


def test_builds_public_summary_without_selecting_parameters(tmp_path: Path) -> None:
    report_path, config_path = _write_fixture(tmp_path)
    summary = build_public_summary(report_path, config_path)

    assert summary["schema_version"] == PUBLIC_SCHEMA
    assert summary["variant_count"] == 12
    assert summary["view_count"] == 9
    assert summary["decision"]["selected_variant_id"] is None
    assert summary["decision"]["training_ready"] == "NO"
    assert summary["headline_metrics"]["axis_sign_flip_active_lost_hit_count"] == 10


def test_public_summary_contains_no_private_provenance(tmp_path: Path) -> None:
    report_path, config_path = _write_fixture(tmp_path)
    serialized = json.dumps(
        build_public_summary(report_path, config_path), sort_keys=True
    ).lower()
    for forbidden in (
        "private_library",
        "bundle_manifest",
        "setup_manifest",
        "mask_manifest",
        "runtime_observation",
        "source_path",
        "audit_implementation_sha256",
    ):
        assert forbidden not in serialized


def test_rejects_config_drift_or_failed_variant(tmp_path: Path) -> None:
    report_path, config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["variants"][2]["cone_angle_degrees"] = 16.0
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        build_public_summary(report_path, config_path)

    report_path, config_path = _write_fixture(tmp_path / "second")
    report = json.loads(report_path.read_text())
    report["aggregate"]["variants"][3]["mechanical_invariants_pass"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="failed mechanical"):
        build_public_summary(report_path, config_path)
