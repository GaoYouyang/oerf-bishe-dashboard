import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
NAME = "poolfire_signed_prolongation_learning_2026-10-08_public_summary.json"
MARKER = "poolfire-signed-prolongation-learning-20261008"


def test_full_fresh_learned_loto_result_and_independence():
    d = json.loads((SITE / "docs" / NAME).read_text())
    assert d["decision"] == "FAIL_SIGNED_PROLONG29_FIXED_RECIPE_CLOSED"
    assert d["independent_checks_passed"] == d["independent_checks_total"] == 28
    assert (d["trajectories"], d["frames_per_trajectory"], d["cells"], d["complete_strata"]) == (11, 101, 3333, 33)
    assert d["camera_counts"] == [5, 7, 9] and d["train_sentinels_per_fold"] == 90
    assert d["shared_primary_parameters_per_fold"] == 29
    assert d["shared_linear_parameters_per_fold"] == 6
    assert d["optimizer_updates_per_fit"] == 80 and d["fresh_fits_each_mode"] == 22
    assert d["held_teacher_reads_for_fit_and_prediction"] == d["CFD_truth_reads_for_fit_and_prediction"] == 0
    assert d["all_fits_sealed_before_held_prediction"] and d["all_new_fields_sealed_before_scoring"]
    assert d["reference_adequate"] and d["camera_permutation_equivariance_checked"]
    assert max(d["independent_maxima"].values()) < 1e-7
    for m in ("CGLS31", "Jacobi2-Warm28", "ClassicalProlong-Warm28",
              "LinearProlong6-Warm28", "SignedProlong29-Warm28"):
        assert d["matched_cells"][m] == 0 and d["absolute_strata"][m] == 33
    assert all(v == 0 for v in d["primary_individual_metric_matches"].values())
    assert all(v == 0 for v in d["primary_matched_cells_by_count"].values())
    assert all(v == 1111 for v in d["primary_cells_by_count"].values())
    assert d["matched_cells"]["CGLS128"] == d["matched_cells"]["DirectRidge-Warm34"] == 3333
    assert d["same_or_cheaper_passing_controls"] == []


def test_seed_loss_is_not_final_accuracy_or_free_cost():
    d = json.loads((SITE / "docs" / NAME).read_text())
    desc = d["post_evaluation_descriptive_diagnostics"]
    loss = desc["mean_train_seed_loss"]
    assert loss["primary_final"] < loss["linear_final"] < loss["initial"]
    ratios = desc["p90_higher_error_ratio_to_CGLS128"]
    assert all(a > b for a, b in zip(ratios["SignedProlong29-Warm28"], ratios["CGLS31"]))
    assert desc["not_new_gates"] and desc["error_ratios_are_not_percentages"]
    assert d["logical_online_calls_each_new_arm"] == {"A": 31, "AT": 31}
    assert d["strong_direct_control_calls"] == {"A": 35, "AT": 35}
    c = d["costs_each_mode"]
    assert c["new_predictions"] == {"A": 413292, "AT": 413292}
    assert c["train_teacher_images"] == {"A": 990, "AT": 0}
    assert c["new_coarse_operator_builds"] == 99
    assert c["optimizer_objective_evaluations"] == 1782
    assert d["geometry_fit_coarse_setup_and_solves_nonfree"] and d["reused_comparator_computation_not_free"]
    n = d["numerical_scope"]
    assert n["retained_artificial_refinement_closure_failures"] == 4
    assert n["separate_native_probe_checks"] == 9 and n["full_native_independent_closure_passed"]
    assert not n["tolerances_changed"] and not n["universal_numerical_stability_claim"]
    assert d["learned_initializer_tested"] and d["whole_sequence_evaluated"] and d["primary_recipe_closed"]
    assert all(not d[k] for k in ("hyperparameter_or_larger_network_rescue_authorized",
        "matched_exact_call_reduction_established", "fresh_deployment_benchmark", "fresh_wall_RSS_tested",
        "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization",
        "real_bost", "curved_ray_validated", "native_12_camera_tested", "main_goal_complete"))


def test_bilingual_note_privacy_and_old_evidence_preserved():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-causal-innovation-history-20261008"')]
        assert note.count("data-i18n-zh=") >= 7 and note.count("data-i18n-en=") >= 7
        assert all(s in note for s in ("28/28", "0/3333", "33/33", "31A+31AT", "0.67006", "data-i18n-alt-en="))
        assert "not an established causal root cause" in note
        assert "The learned C-route goal remains incomplete" in note and NAME in note
    payload = (SITE / "docs" / NAME).read_text()
    assert not any(s in payload for s in ("/Users/", "private_results/", "private_data/", "sha256", "checkpoint"))
    current = json.loads((SITE / "operator-learning/current-evidence.json").read_text())
    assert current["latest_signed_prolongation_learning"]["matched_cells"] == 0
    assert not current["latest_signed_prolongation_learning"]["main_goal_complete"]
    assert current["latest_causal_innovation_history"]["post_bootstrap_matched_cells"] == 0
    assert (SITE / "assets/poolfire_signed_prolongation_learning_2026-10-08.png").is_file()
