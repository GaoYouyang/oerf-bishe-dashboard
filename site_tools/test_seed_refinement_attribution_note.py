import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
NAME = "poolfire_seed_refinement_attribution_2026-10-09_public_summary.json"
MARKER = "poolfire-seed-refinement-attribution-20261009"


def test_independent_postseal_field_component_diagnosis():
    d = json.loads((SITE / "docs" / NAME).read_text())
    assert d["cells"] == 3333 and d["trajectories"] == 11 and d["frames_per_trajectory"] == 101
    assert d["camera_counts"] == [5, 7, 9] and d["complete_strata"] == 33
    assert d["independent_checks_passed"] == d["independent_checks_total"] == 20
    assert max(d["independent_maxima"].values()) < 1e-10
    for arm, worsened, improved in (("LinearProlong6-Warm28", 3320, 13), ("SignedProlong29-Warm28", 3304, 29)):
        record = d["pooled"][arm]
        assert record["seed_field_improved"] == 3333
        assert record["final_teacher_field_improved"] == improved
        assert record["quadrants_seed_then_final"]["improved/worsened"] == worsened
        assert record["quadrants_seed_then_final"]["improved/improved"] == improved
        assert sum(record["quadrants_seed_then_final"].values()) == 3333
        assert d["strata_with_positive_mean_seed_benefit"][arm] == 33
        assert d["strata_with_positive_mean_final_benefit"][arm] == 0
        assert record["mean_seed_benefit"] > 0 and record["mean_final_benefit"] < 0
        assert record["original_joint_matched_cells"] == 0
    assert [d["count_strata"]["SignedProlong29-Warm28"][str(c)]["final_teacher_field_improved"] for c in (5, 7, 9)] == [24, 4, 1]
    assert list(d["pooled"]["SignedProlong29-Warm28"]["physical_error_improved"].values()) == [34, 76, 61, 130]


def test_no_new_queries_rescue_or_causal_claim():
    d = json.loads((SITE / "docs" / NAME).read_text())
    assert all(v == 0 for v in d["call_ledger"].values())
    assert d["inherited_reconstruction_and_training_nonfree"] and d["parent_recipe_closed"]
    assert d["serialization_only_report_repair"] and d["original_report_failure_preserved"]
    assert not d["reducer_outputs_repeated"] and d["partial_seed_image_error_not_reconstructed"]
    assert "Not CFD error or complete field+image seed loss" in d["metric"]
    assert all(not d[k] for k in ("predictor_recipe_rescue_authorized", "causal_root_cause_established",
        "algorithm_breakthrough", "resource_speedup", "paper_success", "fresh_wall_RSS_tested",
        "external_generalization", "real_bost", "curved_ray_validated", "native_12_camera_tested", "main_goal_complete"))
    assert not any(s in (SITE / "docs" / NAME).read_text() for s in ("/Users/", "private_results/", "private_data/", "sha256", "checkpoint"))


def test_bilingual_attribution_note_and_prior_closure_preserved():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-signed-prolongation-learning-20261008"')]
        assert note.count("data-i18n-zh=") >= 7 and note.count("data-i18n-en=") >= 7
        assert all(s in note for s in ("20/20", "3333/3333", "3304", "3320", "24/4/1", "34/76/61/130", "0A+0AT", "0/3333"))
        assert "not CFD error or the complete field-plus-image seed loss" in note
        assert "It does not establish a causal root cause" in note
        assert "The learned C-route goal remains incomplete" in note
        assert NAME in note and "data-i18n-alt-en=" in note
    current = json.loads((SITE / "operator-learning/current-evidence.json").read_text())
    assert current["latest_seed_refinement_attribution"]["seed_improved_final_worsened"] == 3304
    assert not current["latest_seed_refinement_attribution"]["causal_root_cause_established"]
    assert current["latest_signed_prolongation_learning"]["matched_cells"] == 0
    assert (SITE / "assets/poolfire_seed_refinement_attribution_2026-10-09.png").is_file()
