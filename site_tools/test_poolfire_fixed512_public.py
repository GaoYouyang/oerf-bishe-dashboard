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


def test_actual_source_pair_audit_is_not_a_learned_success():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["actual_CFD_pair_noise_audit"]
    assert data["status"] == "NO_1PCT_AMBIGUITY_FOUND_IN_OPENED_CFD_ROSTER"
    assert data["pair_count"] == 127260 and data["conflicts"] == data["observation_overlap_pairs"] == 0
    assert .08928 < data["minimum_relative_noise"] < .08930
    assert data["amplitude_only_conflicts"] == 11468 and data["nearest_same_trajectory"] == 505
    assert data["finite_opened_roster_only"] and data["old_decisions_unchanged"]
    assert not data["continuous_source_identifiability_proven"] and not data["random_noise_risk_claim"]
    assert not data["learned_advantage"] and not data["algorithm_breakthrough"] and not data["real_bost"]
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        note = soup.select("#physical-pair-result")
        assert len(note) == 1
        for lang in ("zh", "en"):
            assert all(value in note[0][f"data-i18n-{lang}"] for value in ("127,260", "8.93%", "505/505"))


def test_correlated_nlm_is_a_fixed_sentinel_failure():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["correlated_nonlocal_means"]
    assert data["status"] == "FAIL_FIXED_CORRELATED_NLM" and data["passing"] == 0
    assert data["cells_per_arm"] == 15 and data["unexecuted_remaining_cells"] == 1500
    assert all(row["passing"] == 0 for row in data["summaries"].values())
    assert data["primary_harm_vs_diagonal"] == [15, 13, 15, 4]
    assert data["new_observation_metric"] == "clean_projection_relative_error" and data["old_decisions_unchanged"]
    assert data["primary_online"] == {"A": 4, "AT": 3, "triangular": 4}
    assert data["fitted_parameters"] == 0 and data["no_other_frame_input"]
    assert not data["complete_sequence_evaluated"] and not data["algorithm_breakthrough"] and not data["real_bost"]
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        note = soup.select("#correlated-nlm-result")
        assert len(note) == 1
        for lang in ("zh", "en"):
            assert all(value in note[0][f"data-i18n-{lang}"] for value in ("0/15", "15.39%", "1500"))


def test_camera_dropout_reference_has_complete_but_clean_limited_scope():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["camera_dropout_reference"]
    assert data["status"] == "PASS_FIXED_CAMERA_DROPOUT_DIRECT_REFERENCE"
    assert data["new_primary_cells"] == data["passing"] == 1010
    assert data["complete_trajectory_camera_groups"] == 10 and data["camera_counts"] == [5, 7]
    for row in data["subsets"].values():
        assert row["primary_passing"] == 505 and row["control_passing"] == 0
        assert row["complete_trajectories"] == 5 and row["numerical_rank"] == 5880
        assert max(row["primary_worst"]) < .01
    assert data["separate_geometry_factor_required"] and data["equal_exact_calls_not_equal_work"]
    assert data["clean_only"] and data["inherited_nine_camera_not_rerun"] and data["old_failures_unchanged"]
    assert not any(data[key] for key in ("learned_advantage", "twelve_cameras_tested", "arbitrary_subsets_tested", "noise_robustness", "real_bost", "resource_speedup", "paper_success"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        note = soup.select("#camera-dropout-reference")
        assert len(note) == 1
        for lang in ("zh", "en"):
            assert all(value in note[0][f"data-i18n-{lang}"] for value in ("505/505", "1010/1010", "0/505", "12"))


def test_factor_transfer_reports_actual_learning_and_necessary_failure():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["factor_transfer_learner"]
    assert data["status"] == "FAIL_FIXED_FACTOR_TRANSFER_LEARNER"
    assert data["primary_parameters"] == 64 and data["linear_control_parameters"] == 44
    assert data["outer_folds"] == 5 and data["predicted_cases"] == 2525
    assert data["necessary_cases_evaluated"] == 25 and data["primary_passing"] == 5
    assert data["full_cases_evaluated"] == 0 and data["remaining_refinements_not_run"] == 2500
    assert data["primary_camera_removal_passing"] == 0 and data["primary_camera_removal_tested"] == 20
    assert data["arms"]["unlearned_mix"]["passing"] == data["arms"]["linear44"]["passing"] == 5
    assert data["arms"]["direct"]["passing"] == 25
    assert data["full_recipe_closed"] and data["geometry_factor_setup_not_free"]
    assert not any(data[k] for k in ("learned_advantage", "full_complete_trajectory_claim", "resource_speedup", "paper_success", "real_bost", "external_generalization", "larger_model_rescue_authorized"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        note = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#factor-transfer-learner")
        assert len(note) == 1
        for lang in ("zh", "en"):
            assert all(value in note[0][f"data-i18n-{lang}"] for value in ("64", "2525", "5/25", "25/25", "2500"))


def test_transfer_capacity_is_a_post_open_lower_bound_not_new_predictor():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["factor_transfer_capacity"]
    assert data["status"] == "FAIL_FIXED_TRANSFER_REACHABLE_CAPACITY"
    assert data["evaluated_cases"] == 25 and data["unavoidable_miss_cases"] == 20
    assert data["conditioning_qualified"] and data["optimistic_separate_metric_oracles"]
    assert data["no_training"] and data["diagnostic_not_deployment"] and data["post_open_only"]
    assert not data["actual_initializer_expanded"] and not data["new_predictor_authorized"]
    assert not any(data[k] for k in ("complete_trajectory_accuracy_claim", "algorithm_breakthrough", "paper_success", "external_generalization", "resource_speedup", "real_bost"))
    for row in data["geometry"]:
        assert row["evaluated"] == 5
        if row["camera_count"] != 9:
            assert row["unavoidable_miss_cases"] == 5
            assert row["smallest_primary_metric_condition_ratio"] > .07
            assert row["relaxed_post_K1_p90"][0] > .5
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        notes = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#factor-transfer-capacity")
        assert len(notes) == 1
        assert all("20/20" in notes[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_shared3_necessary_veto_does_not_rehabilitate_family():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["shared3_necessary_audit"]
    assert data["status"] == "FAIL_SEALED_PRIMARY_NECESSARY_ACCURACY"
    assert data["original_family_status"] == "INCONCLUSIVE_SHARED3_NUMERICAL"
    assert data["reference_numeric"] and data["primary_numeric"] and data["reference_pass"]
    assert data["parameters_per_model"] == 3 and data["folds"] == 5
    assert len(data["rows"]) == 8
    assert data["rows"][0]["passing"] == 0 and data["rows"][-1]["passing"] == 25
    assert data["rows"][6]["passing"] is None and data["rows"][6]["numerical_valid_pairs"] == 8
    assert data["independent_replay"]["states"] == 400
    assert data["training_accounting"]["teacher_image_A_previously_omitted"] == 6414
    assert not any(data[k] for k in ("primary_pass", "training9camera", "training_rerun", "full_trajectory_evaluation", "family_rehabilitated", "continuation_authorized", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    for rel in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        note = BeautifulSoup((ROOT / rel).read_text(), "html.parser").select("#shared3-primary-audit")
        assert len(note) == 1
        assert all("0/25" in note[0][f"data-i18n-{lang}"] and "44.69%" in note[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_dual_monitor_is_reference_relative_and_keeps_abstentions():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["dual_error_monitor"]
    assert data["status"] == "PASS_FIXED_ALGEBRAIC_ERROR_MONITOR"
    assert data["interval_coverage"] == data["intervals"] == 690
    assert data["false_accepts"] == 0 and data["fixed_queries_per_path"] == 115
    assert sum(row["raw_false_accepts"][0] for row in data["rows"]) == 11
    assert sum(row["adjoint_false_accepts"][0] for row in data["rows"]) == 15
    assert data["rows"][-1]["accepted"] == [1, 1] and data["rows"][-1]["abstained"] == [4, 4]
    assert not any(data[k] for k in ("CFD_accuracy_certified", "adaptive_queries_certified", "training_probe_reuse_authorized", "predictor_training_authorized", "new_field_predictions", "new_CFD_truth_reads", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    for rel in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        note = BeautifulSoup((ROOT / rel).read_text(), "html.parser").select("#dual-error-monitor")
        assert len(note) == 1
        assert all("690" in note[0][f"data-i18n-{lang}"] and "115" in note[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_projected_factor_audit_never_reopens_parent_gate():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["projected_factor_numerical_audit"]
    assert data["status"] == "PASS_READ_ONLY_PROJECTED_FACTOR_FAILURE_AUDIT"
    assert data["parent_status"] == "INCONCLUSIVE_PROJECTED_FACTOR_ACTION"
    assert data["original_ratio_difference"] > data["original_tolerance"] == 1e-6
    assert data["diagnostic_only"] and data["original_absolute_gate_unchanged"]
    assert data["native_replays"] == 200 and data["ratio_groups"] == 150
    assert data["rows"][2]["min"][0] > 1 and data["rows"][2]["max"][3] < .05
    assert not any(data[k] for k in ("parent_gate_reopened", "formal_contraction_verdict", "new_actions", "new_factor_solves", "new_CFD_truth_reads", "predictor_training_authorized", "neuralif_implemented", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        note = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#projected-factor-audit")
        assert len(note) == 1
        assert all("3.23e-4" in note[0][f"data-i18n-{lang}"] and "1.137" in note[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_causal_error_attribution_is_not_new_accuracy_or_causal_shares():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["causal_error_attribution"]
    assert data["status"] == "PASS_FIXED_CAUSAL_ERROR_ATTRIBUTION"
    assert data["transitions"] == 2500 and data["midpoint_transitions"] == 25
    assert data["all"]["inherited_larger"] == [2449, 2444, 2445, 2440]
    assert data["midpoints"]["inherited_larger"] == [25] * 4
    assert data["paired_norm_max"] < 1e-6 and data["paired_signed_cross_max"] < 1e-6
    assert not any(data[k] for k in ("recomputed_steps", "new_predictions", "new_truth_reads", "counterfactual_reference_refresh", "new_factor_solves", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost", "predictor_training_authorized", "positive_causal_shares", "new_reconstruction_accuracy_result", "old_recipe_reopened"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        note = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#causal-error-attribution")
        assert len(note) == 1
        assert all("2449/2500" in note[0][f"data-i18n-{lang}"] and "91.63%" in note[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_causal_carry_separates_improvement_equivalence_and_accuracy():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["causal_state_carry"]
    assert data["status"] == "FAIL_CAUSAL_ONE_STATE_K1_ACCURACY"
    assert data["predicted_cases"] == 2525 and data["scored_cases"] == 25
    assert data["skipped_scores"] == 2500 and data["primary_passes"] == 0
    assert data["cheaper_field_passes"] == 0 and data["direct_reference_passes"] == 25
    assert data["primary_no_worse_CGLS2_all_four"] == 25
    assert data["primary_field_better_CGLS2"] == 25
    assert data["cheaper_field_equivalence_max"] < 1e-12
    assert data["primary_warm_calls"] == {"A": 2, "AT": 2, "triangular": 0}
    assert data["cheaper_field_warm_calls"] == {"A": 2, "AT": 1, "triangular": 0}
    assert data["cold_factor_setup_not_free"] and data["geometry_fixed_within_sequence"]
    assert not any(data[k] for k in ("complete_sequence_passed", "future_frame_or_truth_in_prediction", "learned_model", "trainable_parameters", "algorithm_breakthrough", "paper_success", "external_generalization", "resource_speedup", "real_bost", "neural_training_authorized", "full_sequence_accuracy_evaluated", "abrupt_camera_change_evaluated", "retuning_authorized", "predictor_training_authorized", "gpu_rental_authorized", "inherited_ridge_is_matched_cost", "dual_advantage"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        note = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#causal-state-carry")
        assert len(note) == 1
        assert all("0/25" in note[0][f"data-i18n-{lang}"] and "57.46%" in note[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_ic0_construction_does_not_claim_reconstruction_or_speed():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["ic0_construction"]
    assert data["status"] == "FAIL_NATURAL_UNSHIFTED_IC0_CONSTRUCTION"
    assert data["certified_breakdowns"] == 5 and data["constructible_sets"] == 0
    assert all(r["valid"] and r["jacobi_positive"] for r in data["geometries"])
    assert all(c["valid"] for r in data["geometries"] for c in r["dense_cholesky_controls"])
    assert data["independent_post_exit"]["construction_only"]
    assert not any(data[k] for k in ("observations_read", "truth_arrays_read", "predictions", "physical_replays", "A", "AT", "trainable_parameters", "predictor_training_authorized", "physical_replay_authorized", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost", "retuning_authorized", "full_reconstruction_evaluated", "all_sparse_factors_rejected"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        note = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#ic0-construction-gate")
        assert len(note) == 1 and all("0/5" in note[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_inverse_attribution_keeps_diagnosis_distinct_from_algorithm_success():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["monarch_attribution"]
    assert data["status"] == "PASS_FIXED_RECIPE_ERROR_ATTRIBUTION"
    assert data["conclusion"] == "ERROR_PRESENT_BEFORE_LIFT_ALL25"
    assert data["primary_first_action_passing"] == 0 and data["cases"] == 25
    assert data["original_float64_audit_inconclusive"] and data["prior_recipe_still_closed"]
    assert data["unchanged_tolerance"] == 1e-6 and data["reference_defect_max"] < 1e-9
    assert data["independent_post_exit"]["decimal_raw_coordinates"] == 819200
    assert not any(data[k] for k in ("new_candidate", "truth_arrays_read", "retuning_authorized", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost", "new_CFD_accuracy_claim", "complete_sequence_claim"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        notes = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#inverse-error-attribution")
        assert len(notes) == 1
        assert all("8724" in notes[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_monarch_projection_failure_keeps_physical_and_resource_boundaries():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["monarch_inverse"]
    assert data["status"] == "FAIL_FIXED_MONARCH_INVERSE_PHYSICAL_GATE"
    assert data["primary_passing"] == data["block_passing"] == 0 and data["direct_passing"] == 25
    assert data["zero3_dominates_primary_cases"] == 25 and data["source_passing"] == 5
    assert data["predicted_cases"] == 2525 and data["unrun_refinements"] == 2500
    assert data["geometry_coefficients"] == 1572864 and data["coefficient_bytes"] == 12582912
    assert data["online_A"] == data["online_AT"] == 3 and data["online_block_multiply_stages"] == 4
    assert data["offline_triangular_solves"] == 117600 and data["offline_block_svd"] == 40960
    assert data["trainable_parameters"] == 0 and data["full_rank_capable"]
    assert data["coefficient_payload_not_whole_pipeline_memory"] and data["fixed_recipe_closed"]
    assert not any(data[k] for k in ("all_monarch_weights_rejected", "physical_optimality_proved", "global_rank_truncation", "dense_factor_in_prediction", "setup_free", "complete_sequence_claim", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        notes = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#monarch-inverse-gate")
        assert len(notes) == 1
        assert all("1572864" in notes[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_downdate_cache_bound_does_not_claim_reconstruction_or_all_updates():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["exact_downdate_cache"]
    assert data["status"] == "FAIL_GENERIC_DENSE_WOODBURY_INCREMENTAL_CACHE_SAVING"
    assert data["certified_sets"] == data["total_removal_sets"] == 4
    assert data["generic_dense_correction_only"] and data["packed_representation_comparison"]
    assert data["shared_source_factor_charged_separately"] and data["exact_full_rank_not_estimated"]
    assert all(row["rank_lower_bound"] == 2436 and row["incremental_bytes_lower_bound"] >= row["packed_direct_bytes"] for row in data["geometry"])
    assert all(data[k] == 0 for k in ("observations_read", "truth_arrays_read", "trainable_parameters", "predictions", "physical_replays", "A", "AT"))
    assert not any(data[k] for k in ("all_downdates_rejected", "measured_storage_speedup", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        notes = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#exact-downdate-cache")
        assert len(notes) == 1
        assert all("2436" in notes[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_sine_diagonal_preserves_failure_control_and_setup_cost():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["sine_normal_diagonal"]
    assert data["status"] == "FAIL_FIXED_FULL_MODE_SINE_NORMAL_DIAGONAL"
    assert data["retained_modes"] == 5880 and data["modes_truncated"] == data["trainable_parameters"] == 0
    assert data["primary_passing"] == data["nodal_passing"] == 0 and data["direct_passing"] == 25
    assert data["zero3_dominates_primary_cases"] == 25 and data["source_passing"] == 5
    assert data["predicted_cases"] == 2525 and data["unrun_refinements"] == 2500
    assert data["subset_diagonal_bytes"] == 47040 and data["setup_full9_A_per_implementation"] == 5880
    assert data["online_A"] == data["online_AT"] == 3 and data["online_DST"] == 2 and data["online_triangular"] == 0
    assert data["post_open_only"] and data["recipe_closed"] and data["subset_setup_not_free"]
    assert not any(data[k] for k in ("dense_factor_in_prediction", "complete_sequence_claim", "all_nonlocal_models_rejected", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        notes = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#sine-normal-diagonal")
        assert len(notes) == 1
        assert all("47040" in notes[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_fixed_point_capacity_is_post_open_lower_bound_not_predictor():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["point49_post_k1_capacity"]
    assert data["status"] == "FAIL_FROZEN_POINT49_POST_K1_REACHABLE_CAPACITY"
    assert data["evaluated_cases"] == 25 and data["primary_certified_misses"] == data["linear_certified_misses"] == 20
    assert data["post_open_only"] and data["fixed_hidden_features_only"] and data["after_alpha_and_K1"]
    assert data["original_feature_sets_unchanged"] and data["no_shared_head_or_joint_metric_feasibility_claim"]
    assert data["no_complete_sequence_claim"] and data["factor_setup_not_free"]
    assert not any(data[k] for k in ("new_predictions", "new_models", "parent_verdict_changed", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    for row in data["geometry"]:
        for model in row["models"]:
            assert model["certified_miss_cases"] == (0 if row["camera_count"] == 9 else 5)
    assert data["all_metric_projections_recomputed"] == 800
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        notes = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#point49-post-k1-capacity")
        assert len(notes) == 1
        assert all("20/20" in notes[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_head_diagnostic_does_not_rescue_old_prediction():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["point49_head_optimality"]
    assert data["status"] == "PASS_TRAINING_ONLY_FROZEN_HEAD_DIAGNOSTIC"
    assert data["folds"] == 5 and data["training_queries_per_fold"] == 1212
    assert data["no_heldout_truth"] and data["raw_before_alpha_and_K1"] and data["hidden_weights_fixed"]
    assert data["no_global_hidden_optimum_claim"] and data["no_new_models"]
    assert not any(data[k] for k in ("new_predictions", "new_checkpoints", "retraining", "parent_verdict_changed", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    assert data["offline_A"] == 246440 and data["offline_AT"] == 117160 and data["offline_triangular"] == 234320
    for row in data["records"]:
        for arm in ("primary", "linear"):
            assert row[arm]["material_gap"] and row[arm]["unregularized_minimum_certified"]
            assert row[arm]["unregularized_minimum"] > 1e-4
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        notes = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#point49-head-diagnostic")
        assert len(notes) == 1
        assert all("K1" in notes[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))


def test_point49_reports_fixed_negative_and_factor_cost():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["source_conditioned_point49"]
    assert data["status"] == "FAIL_FIXED_SOURCE_CONDITIONED_POINT49"
    assert data["evaluated_cases"] == 25 and data["predicted_cases"] == 2525
    assert data["unrun_refinements"] == 2500
    assert data["primary_passing"] == data["linear_passing"] == data["unlearned_passing"] == 5
    assert data["direct_passing"] == 25 and data["removal_cases_passing"] == 0
    assert data["model_parameters"] == 49 and data["linear_parameters"] == 15
    assert data["online_A"] == data["online_AT"] == 3 and data["online_triangular"] == 4
    assert data["shared_factor_bytes"] > 270000000 and data["factor_setup_not_free"]
    assert data["nine_camera_identity_not_learning"] and data["frozen_recipe_closed"]
    assert not any(data[k] for k in ("complete_sequence_accuracy_established", "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "real_bost"))
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        notes = BeautifulSoup((ROOT / relative).read_text(), "html.parser").select("#source-conditioned-point49")
        assert len(notes) == 1
        assert all("20/20" in notes[0][f"data-i18n-{lang}"] for lang in ("zh", "en"))
