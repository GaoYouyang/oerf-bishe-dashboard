from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import site_tools.run_cancellation_aware_metric_surrogate_smoke as runner_module
from site_tools.run_cancellation_aware_metric_surrogate_smoke import (
    CONFIG_SCHEMA_VERSION,
    EVIDENCE_SCOPE,
    METHODS,
    REPORT_SCHEMA_VERSION,
    STATUS,
    load_config,
    run_smoke,
)


def _config() -> dict[str, object]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "seeds": {"geometry": 101, "noise": 202, "training": 303},
        "rigs": {
            "row_count": 10,
            "column_count": 8,
            "split_unit": "COMPLETE_RIG",
            "random_ray_split_used": False,
            "assignments": {
                "train-00": "train",
                "train-01": "train",
                "train-02": "train",
                "cal-00": "safety_calibration",
                "cal-01": "safety_calibration",
                "ood-00": "fresh_geometry_ood",
                "ood-01": "fresh_geometry_ood",
            },
        },
        "estimator": {"hidden_dim": 10, "steps": 24, "learning_rate": 0.02},
        "calibration": {"envelope_margin": 1.02, "fresh_exact_access_forbidden": True},
        "comparators": {
            "scalar_factor_grid": [0.75, 1.0, 1.25],
            "exact_factor_alpha_grid": [0.0, 0.5, 1.0],
            "selection_metric": "FINAL_FIELD_RELATIVE_L2_TRAIN_MEAN",
        },
        "solver": {"eta": 0.7, "theta": 1.0, "checkpoints": [0, 1, 2, 4, 8]},
        "runtime": {"device": "cpu", "dtype": "torch.float64", "timing_role": "MEASURED_SINGLE_RUN_NONCOMPARATIVE"},
        "claim_boundary": {"new_algorithm_claimed": False, "real_data_used": False, "generalization_claimed": False, "superiority_claimed": False},
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_smoke_freezes_git_state_before_writing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result"
    observations: list[bool] = []

    def fake_git_state() -> dict[str, object]:
        observations.append(output.exists())
        return {
            "source_commit": "0" * 40,
            "source_tree_clean": True,
            "source_snapshot_status": "COMMITTED_CLEAN_REPRODUCIBLE_FROM_COMMIT",
            "clean_rerun_required_after_commit": False,
            "source_file_sha256": {},
        }

    monkeypatch.setattr(runner_module, "_git_state", fake_git_state)
    report = runner_module.run_smoke(_config(), output_dir=output)

    assert observations == [False]
    assert report["provenance"]["source_tree_clean"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.__setitem__("evidence_scope", "REAL_DATA"), "evidence_scope"),
        (lambda value: value["runtime"].__setitem__("device", "mps"), "runtime"),  # type: ignore[index]
        (lambda value: value["calibration"].__setitem__("fresh_exact_access_forbidden", False), "fresh exact"),  # type: ignore[index]
        (lambda value: value["rigs"].__setitem__("random_ray_split_used", True), "complete-rig"),  # type: ignore[index]
        (lambda value: value.__setitem__("unexpected", 1), "keys differ"),
    ],
)
def test_load_config_fails_closed(tmp_path: Path, mutate: object, message: str) -> None:
    value = _config()
    mutate(value)  # type: ignore[operator]
    path = tmp_path / "config.json"
    _write(path, value)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_smoke_writes_traceable_comparator_evidence_bundle(tmp_path: Path) -> None:
    config = _config()
    output = tmp_path / "result"
    report = run_smoke(config, output_dir=output)

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["evidence_scope"] == EVIDENCE_SCOPE
    assert report["data_contract"]["primary_metric"] == "FINAL_FIELD_RELATIVE_L2"
    assert report["data_contract"]["inference_type"] == "InferenceRigFeatures_WITHOUT_A_EXACT_TRUTH_TARGET"
    assert report["split_contract"]["geometry_parameters_independently_sampled"] is True
    assert set(report["split_contract"]["train_rig_ids"]).isdisjoint(report["split_contract"]["fresh_geometry_ood_rig_ids"])
    assert report["calibration_envelope"]["fresh_exact_access"] is False
    assert report["calibration_envelope"]["exact_mass_materializations"] == 2
    assert report["calibration_envelope"]["factor_mass_vector_accesses"] == 4
    assert report["fresh_exact_access_instrumentation"]["fresh_candidate_exact_mass_access_count"] == 0
    assert report["fresh_exact_access_instrumentation"]["blocked_exact_mass_access_count"] == 0
    assert report["fresh_exact_access_instrumentation"]["fresh_guarded_candidate_scope_count"] == 8
    assert report["fresh_exact_access_instrumentation"]["guard_scope_count_by_phase"]["learned_prediction_setup"] == 2
    assert report["fresh_exact_access_instrumentation"]["guard_scope_count_by_phase"]["posthoc_schur_audit"] == 2 * len(METHODS)
    assert report["feature_cost_contract"]["end_to_end_cost_reduction_claimed"] is False
    for method in ("learned_oracle_free", "calibrated_envelope"):
        ledger = report["call_ledger"]["fresh_by_method"][method]
        assert ledger["factor_mass_vector_accesses"] == 4
        assert ledger["factor_feature_construction_calls"] == 2
        assert report["timing"]["fresh_method_seconds"][method][
            "factor_feature_construction_is_setup_subcomponent"
        ] is True
    assert report["method_contracts"]["exact_oracle"].startswith("NONDEPLOYABLE")
    assert report["method_contracts"]["learned_oracle_free"].startswith("DEPLOYABLE")
    assert report["decision"]["research_claim_authorized"] is False
    assert report["decision"]["next_gate"] == "SUPPORT_OOD_DETECTION_PLUS_FACTOR_FALLBACK_AND_STRUCTURED_SAFE_PARAMETERIZATION"
    rule_passed = (
        report["decision"]["calibrated_envelope_all_fresh_schur_safe"]
        and report["decision"]["calibrated_envelope_beats_factor_and_simple_on_each_fresh_rig"]
    )
    assert report["decision"]["metric_substitution_authorized"] is rule_passed
    if not all(report["decision"]["per_fresh_rig_stable_win_flags"].values()):
        assert report["decision"]["metric_substitution_authorized"] is False

    expected_files = {"report.json", "geometry_manifest.csv", "metric_rows.csv", "trajectory_rows.csv", "predictions.csv", "model_parameters.csv", "checksums.sha256"}
    assert {path.name for path in output.iterdir()} == expected_files
    metric_rows = list(csv.DictReader((output / "metric_rows.csv").open(encoding="utf-8")))
    trajectory_rows = list(csv.DictReader((output / "trajectory_rows.csv").open(encoding="utf-8")))
    predictions = list(csv.DictReader((output / "predictions.csv").open(encoding="utf-8")))
    assert {row["method"] for row in metric_rows} == set(METHODS)
    assert {row["method"] for row in trajectory_rows} == set(METHODS)
    assert {row["method"] for row in predictions} == set(METHODS)
    assert len(metric_rows) == 2 * len(METHODS)
    assert len(trajectory_rows) == 2 * len(METHODS) * 5
    assert all(row["field_relative_l2"] for row in trajectory_rows)
    assert all(int(row["spectral_violation_count"]) >= 0 for row in metric_rows)
    assert all(
        int(row["total_violation_count"]) == 0
        for row in metric_rows
        if row["method"] in {"factor", "exact_oracle"}
    )
    assert any(
        int(row["spectral_violation_count"]) == 1
        for row in metric_rows
        if row["method"] == "learned_oracle_free"
    )
    assert all(row["exact_masses_recomputed_from_signed_a"] == "1" for row in metric_rows)

    provenance = report["provenance"]
    assert len(provenance["source_commit"]) == 40
    if provenance["source_tree_clean"]:
        assert provenance["source_snapshot_status"] == "COMMITTED_CLEAN_REPRODUCIBLE_FROM_COMMIT"
        assert provenance["clean_rerun_required_after_commit"] is False
    else:
        assert provenance["source_snapshot_status"] == "UNCOMMITTED_SOURCE_SNAPSHOT_NOT_REPRODUCIBLE_FROM_COMMIT_ALONE"
        assert provenance["clean_rerun_required_after_commit"] is True
    assert len(provenance["config_sha256"]) == 64
    assert provenance["geometry_manifest_sha256"] == hashlib.sha256((output / "geometry_manifest.csv").read_bytes()).hexdigest()
    assert provenance["model_parameters_sha256"] == hashlib.sha256((output / "model_parameters.csv").read_bytes()).hexdigest()
    assert provenance["fresh_predictions_sha256"] == hashlib.sha256((output / "predictions.csv").read_bytes()).hexdigest()
    checksum_lines = (output / "checksums.sha256").read_text(encoding="ascii").splitlines()
    assert len(checksum_lines) == 6
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        assert digest == hashlib.sha256((output / relative).read_bytes()).hexdigest()

    repeated = tmp_path / "repeated"
    repeated_report = run_smoke(config, output_dir=repeated)
    for filename in ("geometry_manifest.csv", "metric_rows.csv", "trajectory_rows.csv", "predictions.csv", "model_parameters.csv"):
        assert (repeated / filename).read_bytes() == (output / filename).read_bytes()
    for stable_key in ("schema_version", "status", "evidence_scope", "claim_boundary", "provenance", "split_contract", "training", "simple_control_selection", "calibration_envelope", "fresh_exact_access_instrumentation", "feature_cost_contract", "method_contracts", "evidence_counting", "aggregate_fresh_ood", "call_ledger", "decision"):
        assert repeated_report[stable_key] == report[stable_key]


def test_exact_factor_alpha_one_is_counted_as_duplicate_not_independent(tmp_path: Path) -> None:
    report = run_smoke(_config(), output_dir=tmp_path / "result")
    if report["simple_control_selection"]["selected_exact_factor_alpha"] == 1.0:
        assert report["simple_control_selection"]["selected_exact_factor_duplicate_of_exact_oracle"] is True
        assert report["evidence_counting"]["duplicate_methods"]["exact_factor_interpolation_oracle"]["duplicate_of_exact_oracle"] is True
        assert report["evidence_counting"]["raw_method_count"] == 6
        assert report["evidence_counting"]["independent_method_count"] == 5
        assert "exact_factor_interpolation_oracle" not in report["evidence_counting"]["independent_method_names"]


def test_frozen_repository_config_retains_the_audited_no_go_regression(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(
        repository_root
        / "demo_t16_operator/configs/cancellation_aware_metric_surrogate_smoke_v1.json"
    )
    output = tmp_path / "formal-regression"
    report = run_smoke(config, output_dir=output)
    rows = list(csv.DictReader((output / "metric_rows.csv").open(encoding="utf-8")))
    calibrated = [row for row in rows if row["method"] == "calibrated_envelope"]
    assert len(calibrated) == 4
    assert all(int(row["total_violation_count"]) > 0 for row in calibrated)
    assert sum(int(row["total_violation_count"]) for row in calibrated) == 39
    assert report["decision"]["metric_substitution_authorized"] is False
    assert report["decision"]["research_claim_authorized"] is False
