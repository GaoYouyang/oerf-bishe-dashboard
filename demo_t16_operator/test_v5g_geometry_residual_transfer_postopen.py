from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from demo_t16_operator.run_v5g_geometry_residual_transfer_postopen import (
    attach_audit_outcomes,
    circular_angle_distance,
    methods_have_identical_sample_rows,
    prediction_hash,
    predictor_summaries,
)


def test_circular_angle_distance_uses_projection_period() -> None:
    assert circular_angle_distance(179.0, 2.0, 180.0) == pytest.approx(3.0)
    assert circular_angle_distance(10.0, 170.0, 180.0) == pytest.approx(20.0)


def test_attach_audit_outcomes_preserves_pre_audit_prediction_hash() -> None:
    rows = [
        {
            "rig_id": "a",
            "block_id": "a:0",
            "sample_index": 0,
            "radius_changed_from_metadata": True,
            "p": 2.0,
        }
    ]
    before = prediction_hash(rows)
    attached = attach_audit_outcomes(rows, {("a:0", 0): -3.0})
    assert prediction_hash(rows) == before
    assert "actual_audit_gain_percent" not in rows[0]
    assert attached[0]["actual_audit_gain_percent"] == -3.0


def test_predictor_summary_reports_rig_equal_weight_and_changed_only() -> None:
    rows = [
        {
            "rig_id": "a",
            "block_id": "a:0",
            "sample_index": 0,
            "radius_changed_from_metadata": True,
            "prediction": 2.0,
            "actual_audit_gain_percent": 1.0,
        },
        {
            "rig_id": "a",
            "block_id": "a:0",
            "sample_index": 1,
            "radius_changed_from_metadata": True,
            "prediction": 4.0,
            "actual_audit_gain_percent": 3.0,
        },
        {
            "rig_id": "b",
            "block_id": "b:0",
            "sample_index": 0,
            "radius_changed_from_metadata": True,
            "prediction": -2.0,
            "actual_audit_gain_percent": -4.0,
        },
        {
            "rig_id": "b",
            "block_id": "b:1",
            "sample_index": 1,
            "radius_changed_from_metadata": False,
            "prediction": 99.0,
            "actual_audit_gain_percent": -99.0,
        },
    ]
    summaries, rig_rows = predictor_summaries(rows, ("prediction",))
    summary = summaries[0]
    assert summary["changed_sample_count"] == 3
    assert summary["independent_rig_count"] == 2
    assert summary["block_sign_agreement_fraction"] == pytest.approx(1.0)
    assert summary["rig_sign_agreement_fraction"] == pytest.approx(1.0)
    assert summary["equal_weight_rig_mean_absolute_error_percent"] == pytest.approx(
        1.5
    )
    assert len(rig_rows) == 2


def test_method_identity_ignores_only_the_method_label() -> None:
    rows = [
        {
            "block_id": "a:0",
            "sample_index": "0",
            "reconstruction_method": method,
            "metric": "1.25",
        }
        for method in ("gcv", "upre")
    ]
    assert methods_have_identical_sample_rows(rows, "gcv", "upre")
    rows[1]["metric"] = "1.26"
    assert not methods_have_identical_sample_rows(rows, "gcv", "upre")


def test_attach_audit_outcomes_rejects_key_mismatch() -> None:
    rows = [
        {
            "rig_id": "a",
            "block_id": "a:0",
            "sample_index": 0,
            "radius_changed_from_metadata": True,
        }
    ]
    with pytest.raises(ValueError, match="keys disagree"):
        attach_audit_outcomes(rows, {("other", 0): 1.0})


def test_published_v5g_artifact_exposes_oracle_and_dependency_provenance() -> None:
    root = Path(__file__).resolve().parent
    result = root / "results" / "v5g_geometry_residual_transfer_postopen"
    report = json.loads((result / "report.json").read_text(encoding="utf-8"))
    assert report["noise_std_is_synthetic_clean_inner_rms_oracle"] is True
    assert report["source_outer_observation_used_for_prediction"] is True
    assert report["target_observation_used_for_prediction"] is False
    expected_hashes = {
        "runner",
        "geometry_module",
        "rig_shared_profile",
        "block_builder",
        "v5f_runner",
        "dual_regularization",
        "nested_crossview",
        "finite_aperture_forward",
        "reaction_field_noise_generator",
    }
    assert expected_hashes <= set(report["source_hashes"])
    runner_hash = hashlib.sha256(
        (root / "run_v5g_geometry_residual_transfer_postopen.py").read_bytes()
    ).hexdigest()
    assert report["source_hashes"]["runner"] == runner_hash
    with (result / "prediction_rows.csv").open(newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    assert first["prediction_uses_source_observation"] == "True"
    assert first["prediction_uses_target_observation"] == "False"
    assert first["oracle_sigma_truth_derived"] == "True"
