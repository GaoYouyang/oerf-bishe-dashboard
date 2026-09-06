"""Verify reference scope, whole-sequence arithmetic and bilingual boundaries."""
import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = "poolfire_fixed512_reference_20260906"


def test_reference_is_not_learned_success():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["latest_reference_qualification"]["status"] == data["status"]
    assert current["latest_prediction"]["matched_cells"] == 367
    assert data["total"] == data["new_nonpilot_samples"] + data["reused_pilot_samples"] == 505
    assert all(data["checks"].values())
    for path in ("formal", "independent"):
        for arm in ("cgls", "jacobi_pcgls"):
            result = data["summaries"][path][arm]
            assert result["passing"] == 505 and result["complete_trajectories"] == 5
            for trajectory in result["trajectories"]:
                assert trajectory["passing"] == trajectory["total"] == 101
                assert all(v["worst"] <= .01 for v in trajectory["tails"].values())
    for key in ("learned_algorithm", "minimum_calls_proven", "resource_speedup", "external_generalization",
                "real_bost", "variable_camera_validated", "independent_test", "previous_cost_curves_requalified"):
        assert data[key] is False
    assert data["new_solver_calls"] == {"A": 2000 * 512, "AT": 2000 * 512}
    assert data["logical_calls_per_endpoint"] == {"A": 512, "AT": 512}


def test_bilingual_notes_and_no_private_identifiers():
    for file in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / file).read_text(), "html.parser")
        notes = soup.select("#fixed512-reference-result")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(word in notes[0][f"data-i18n-{lang}"] for word in ("512", "505/505", "5/5", "1%"))
        assert "not minimum calls" in notes[0]["data-i18n-en"]
        assert "不是最小调用数" in notes[0]["data-i18n-zh"]
    text = (ROOT / f"docs/{STEM}.md").read_text()
    assert "## 中文" in text and "## English" in text and "pilot-informed" in text
    assert "同一离散forward" in text and "same discrete forward" in text
    for suffix in ("md", "json"):
        content = (ROOT / f"docs/{STEM}.{suffix}").read_text()
        assert all(term not in content for term in ("/Users/", "/Volumes/", "private_results", "sha256", ".pt"))
    assert (ROOT / f"assets/figures/{STEM}.png").stat().st_size > 10000


def test_actual_residual_bound_has_narrow_scope():
    public = json.loads((ROOT / f"docs/{STEM}.json").read_text())
    data = public["direction_diagnostic"]
    assert data["cells"] == 505 and data["complete_trajectories"] == 5
    assert .2957 < data["minimum_field_error_lower_bound"] < .2958
    assert data["field_target"] == data["reference_field_error_guarantee"] == .01
    assert data["excludes_only_single_scalar_completion"]
    assert not data["excludes_further_cgls"] and not data["excludes_multidirection_warm_start"]
    assert not data["CFD_truth_parsed"] and not data["learned_speedup"]
    assert len(data["trajectories"]) == 5
    for row in data["trajectories"]:
        assert row["minimum_actual_residual_scalar_bound"] >= data["minimum_field_error_lower_bound"]
        assert 0 < row["median_rayleigh_ratio"] < 1
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#late-direction-diagnostic")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert "29.57%" in notes[0][f"data-i18n-{lang}"]
            assert "CGLS" in notes[0][f"data-i18n-{lang}"]
    text = (ROOT / f"docs/{STEM}.md").read_text()
    assert "lower bounds, not measured-error ranges" in text
    assert "preprocessed gauge-fixed" in text


def test_fsai_sentinel_is_not_complete_sequence_or_learning_success():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["fsai_sentinel"]
    assert data["sentinel_points"] == 5 and data["primary_passing_points"] == 3
    assert data["jacobi_passing_points"] == 0
    assert data["primary_no_worse_all_four_metrics_points"] == 5
    assert data["per_endpoint_exact_calls"] == {"A": 256, "AT": 256}
    assert not any(data[key] for key in ("complete_sequence_verified", "learned_algorithm",
        "resource_speedup", "external_generalization", "full_sequence_launch_authorized"))
    assert len(data["rows"]) == 5
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#fsai-sentinel-result")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(value in notes[0][f"data-i18n-{lang}"] for value in ("3/5", "0/5", "256"))


def test_field81_is_a_failed_necessary_gate_not_505_successes():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["field81_learning"]
    assert data["status"] == "FAIL_FIELD81_NECESSARY_MIDPOINTS"
    assert data["parameters_per_model"] == 81 and data["outer_fits"] == 10
    assert data["predictions_sealed"] == 505 and data["evaluated_midpoints"] == 5
    assert data["primary_passing"] == 0 and data["remaining_refinement_skipped"] == 500
    assert data["per_deployment_exact_calls"] == {"A": 258, "AT": 258}
    assert data["zero_bp_ridge_dominate_primary_all_four_points"] == 5
    assert not any(data[k] for k in ("complete_sequence_verified", "learned_advantage", "resource_speedup", "external_generalization", "real_bost"))
    assert all(row["formal"][2] > .01 and row["independent"][2] > .01 for row in data["rows"])
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#field81-learning-result")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(value in notes[0][f"data-i18n-{lang}"] for value in ("81", "505", "0/5", "258", "500"))


def test_direct_factor_is_classical_cache_ready_not_learned_speedup():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["direct_factor_resource"]
    assert data["accuracy_passed"] == data["independent_accuracy_passed"] == 505
    assert data["complete_trajectories"] == 5 and data["repetitions_per_arm"] == 3
    assert len(data["measurements"]) == 9 and data["retained_factor_bytes"] == 276595200
    assert data["factor_returned_residual_calls"] == {"A": 1, "AT": 1, "triangular_solves": 2}
    assert all(data["summary"][arm]["median_wall_seconds"] > 0 for arm in ("factor", "cgls", "pcgls"))
    assert not any(data[k] for k in ("learned_advantage", "algorithm_breakthrough", "whole_bos_pipeline_speedup", "filesystem_cache_cold", "external_generalization", "real_bost"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#direct-factor-resource-result")
        assert len(notes) == 1
        assert "不是学习算法加速" in notes[0]["data-i18n-zh"]
        assert "not learned acceleration" in notes[0]["data-i18n-en"]


def test_noisy_inverse_failure_does_not_erase_clean_result():
    doc = json.loads((ROOT / f"docs/{STEM}.json").read_text())
    data = doc["direct_inverse_noise_boundary"]
    assert data["status"] == "FAIL_FIXED_1PCT_DIRECT_INVERSE"
    assert data["primary_passing"] == 0 and data["noisy_samples"] == 1515
    assert data["complete_trajectories"] == 0 and data["unique_opened_frames"] == 505
    assert data["four_metric_pass_counts"] == [0, 0, 0, 1515]
    assert data["control_midpoints_per_arm"] == 15 and not data["control_full_sequences_evaluated"]
    assert data["clean_result_still_valid"] and doc["direct_factor_resource"]["accuracy_passed"] == 505
    assert not data["measured_experimental_noise"] and not data["learned_advantage"]
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#direct-noise-boundary")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(value in notes[0][f"data-i18n-{lang}"] for value in ("1%", "1515", "0/5", "10.10%", "15"))


def test_haar4_prediction_count_is_not_reconstruction_success():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["haar4_noise_initializer"]
    assert data["status"] == "FAIL_FIXED_HAAR4_NOISE_INITIALIZER"
    assert data["trainable_parameters"] == 4 and data["complete_trajectory_outer_folds"] == 5
    assert data["sealed_outer_predictions_per_method"] == 1515
    assert data["evaluated_midpoints_per_arm"] == 15 and data["primary_passing"] == 0
    assert data["skipped_primary_refinements"] == 1500 and not data["full_refinement_completed"]
    assert data["primary_better_than_direct_by_metric"] == [15, 15, 15, 0]
    assert data["primary_online"] == dict(A=3, AT=3, triangular_solves=4)
    assert not data["learned_advantage_established"] and not data["resource_speedup"] and not data["real_bost"]
    assert all(v["all_four_passing"] == 0 for v in data["arms"].values())
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#haar4-noise-result")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(value in notes[0][f"data-i18n-{lang}"] for value in ("1515", "0/15", "1500", "10.13%"))


def test_local_gaussian_failure_and_truth_aware_attribution():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["local_gaussian_noise_prior"]
    assert data["status"] == "FAIL_FIXED_PATCH_GAUSSIAN_PRIOR"
    assert data["primary_passing"] == 0 and data["evaluated_midpoints_per_arm"] == 15
    assert data["sealed_predictions_per_prior"] == 1515 and data["skipped_primary_refinements"] == 1500
    assert not data["complete_refinement_done"] and not data["neural_training"]
    assert data["attribution_truth_aware"] and data["attribution_is_not_deployment"]
    assert data["component_norms_are_not_additive_shares"]
    assert all(row["propagated_error_energy_exceeds_bias_cells"] == 15 for row in data["attribution"])
    assert not data["learned_advantage"] and not data["resource_speedup"] and not data["real_bost"]
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        note = soup.select("#patch-gaussian-result")
        assert len(note) == 1
        for lang in ("zh", "en"):
            assert all(value in note[0][f"data-i18n-{lang}"] for value in ("0/15", "9.08%", "1500"))


def test_oracle_noise_gate_issue_does_not_rescue_old_failures():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["oracle_noise_gate_audit"]
    assert data["status"] == "FAIL_NOISY_OBSERVATION_GATE_ORACLE_COMPATIBILITY"
    assert data["oracle"]["cells"] == 1515 and data["oracle"]["rejected"] == 308
    assert data["oracle"]["by_seed_rejected"] == [95, 112, 101]
    assert data["old_scientific_decisions_unchanged"] and data["future_metric_is_not_retroactive"]
    assert data["original_execution_exit_code"] == 1 and data["report_only_recovery_verified"]
    assert all(v["counterfactual_budget_joint_passing"] == v["field_and_gradients_passing"] == 0 for v in data["oracle"]["candidates"].values())
    assert not data["reconstruction_success"] and not data["algorithm_breakthrough"] and not data["real_bost"]
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        note = soup.select("#oracle-noise-gate-result")
        assert len(note) == 1
        for lang in ("zh", "en"):
            assert all(value in note[0][f"data-i18n-{lang}"] for value in ("308/1515", "0.000141", "0/15"))
