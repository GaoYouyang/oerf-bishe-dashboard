import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = "poolfire-full-temporal-direct-control-20261007"
SUMMARY = "poolfire_full_temporal_direct_control_2026-10-07_public_summary.json"


def test_full_temporal_classic_matches_are_not_learned_claims():
    text = (ROOT / "docs" / SUMMARY).read_text()
    data = json.loads(text)
    assert data["decision"] == "PASS_FIXED_DIRECT_RIDGE_FULL_TEMPORAL_STRONG_CONTROL_ONLY"
    assert data["trajectories"] == 11 and data["frames_per_trajectory"] == 101
    assert data["source_fields"] == 1111 and data["cells"] == 3333
    assert data["independent_checks_passed"] == data["independent_checks_total"] == 29
    assert data["matched_cells"] == {"CGLS128": 3333, "DirectRidge-Warm34": 3333}
    for group in ("absolute_strata", "all_frame_absolute_strata", "complete_matched_strata"):
        assert data[group] == {"CGLS128": 33, "DirectRidge-Warm34": 33}
    assert all(v == 3333 for v in data["individual_matched"][data["primary"]].values())
    assert data["evaluation_gate"]["matched_factor"] == 1.01 and data["evaluation_gate"]["matched_floor"] == 1e-10
    assert data["descriptive_harm"]["all_four_strictly_better"] == 3333 and data["descriptive_harm"]["any_metric_strict_harm"] == 0
    assert data["new_model_fits"] == 0 and data["reference_self_match_tautological"]
    assert all(data[n] is False for n in ("learned_initializer_tested", "algorithm_breakthrough", "resource_speedup",
        "external_generalization", "paper_success", "real_bost", "curved_ray_validated", "main_goal_complete"))
    assert not any(token in text for token in ("/Users/", "private_results/", "private_data/", "sha256", "checkpoint"))


def test_resource_and_reference_limits_are_preserved():
    data = json.loads((ROOT / "docs" / SUMMARY).read_text())
    assert data["standalone_candidate"] == {"A": 35, "AT": 35, "factor_solves": 1}
    assert data["standalone_reference"] == {"A": 128, "AT": 128}
    assert data["new_counted_each_mode"] == {"A": 137808, "AT": 117810} and data["factor_solves_each_mode"] == 3375
    assert data["full_factor_setup_storage_and_solves_nonfree"] and data["old99_query_resource_failure_preserved"]
    assert data["mixed_audit_wall_RSS_not_deployment"] and data["reference_row_permutation_not_requalified"]
    assert data["frame_zero_permutation_audits"] == 33 and data["original99_initial_final_anchors_bitwise_preserved"]
    assert data["maximum_numeric_differences"]["metric_absolute_max"] < 1e-11
    assert set(data["pooled_camera"]) == {f"{c}/{m}" for c in (5, 7, 9) for m in ("CGLS128", "DirectRidge-Warm34")}


def test_bilingual_scope_and_figures():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        nodes = BeautifulSoup((ROOT / name).read_text(), "html.parser").select("#" + MARKER)
        assert len(nodes) == 1
        translated = nodes[0].select("[data-i18n-zh][data-i18n-en]")
        assert len(translated) == 7
        for lang in ("zh", "en"):
            text = " ".join(n["data-i18n-" + lang] for n in translated)
            for token in ("101", "5/7/9", "1,111", "3,333", "29/29", "33/33", "0.07908", "35A+35AT", "128A+128AT"):
                assert token in text
            image = nodes[0].select_one("img")
            assert image["data-i18n-alt-" + lang]
            assert (ROOT / name).parent.joinpath(image["data-i18n-src-" + lang]).exists()
        assert "主目标仍未完成" in nodes[0].get_text()
        assert (ROOT / name).parent.joinpath(nodes[0].select_one("a")["href"]).resolve() == ROOT / "docs" / SUMMARY


def test_old_failures_and_full_reference_are_not_overwritten():
    data = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    new = data["latest_full_temporal_direct_control"]
    assert new["primary_matched_cells"] == 3333 and new["complete_matched_strata"] == 33
    assert new["new_model_fits"] == 0 and not new["learned_initializer_tested"] and new["old99_resource_failure_preserved"]
    assert data["latest_full_temporal_reference"]["complete_absolute_strata"] == 33
    assert data["latest_query_residual_dual"]["primary_matched_cells"] == 0
    text = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert "29/29独立核验全真" in text and "主目标仍未完成" in text
    assert "the original whole goal remains active and unmet" in text
