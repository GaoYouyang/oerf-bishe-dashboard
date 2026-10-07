import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = "poolfire-full-temporal-reference-20261007"
SUMMARY = "poolfire_full_temporal_reference_2026-10-07_public_summary.json"


def test_complete_reference_scope_is_not_learned_success():
    text = (ROOT / "docs" / SUMMARY).read_text()
    data = json.loads(text)
    assert data["decision"] == "PASS_FIXED_CGLS128_FULL_TEMPORAL_REFERENCE_ONLY"
    assert data["trajectories"] == 11 and data["frames_per_trajectory"] == 101
    assert data["source_fields"] == 1111 and data["operator_cells"] == 3333
    assert data["independent_checks_passed"] == data["independent_checks_total"] == 25
    assert data["absolute_p90_strata"] == data["all_frame_absolute_strata"] == {"Zero": 0, "BP-LS1": 0, "CGLS128": 33}
    assert data["matched_cells"] == {"Zero": 0, "BP-LS1": 0, "CGLS128": 3333}
    assert data["reference_self_match_is_tautological"] and data["new_model_fits"] == 0
    assert data["original_anchor_observations_bitwise_preserved"] == data["original_anchor_reference_states_bitwise_preserved"] == 99
    assert data["standalone_reference"] == {"A": 128, "AT": 128}
    assert data["actual_counted_each_mode"] == {"A": 456621, "AT": 429957}
    assert data["full_temporal_reference_evaluated"] and data["finite_comparator_not_stationary_inverse"]
    assert all(data[n] is False for n in ("learned_full_sequence_evaluated", "closed_learner_reopened", "algorithm_breakthrough",
        "resource_speedup", "paper_success", "external_generalization", "real_bost", "curved_ray_validated", "main_goal_complete"))
    assert not any(token in text for token in ("/Users/", "private_results/", "private_data/", "sha256", "checkpoint"))


def test_full_tails_do_not_hide_frame_failures_or_pool_only():
    data = json.loads((ROOT / "docs" / SUMMARY).read_text())
    assert data["absolute_limits"] == dict(zip(data["absolute_limits"], (.5, .75, .75, .2)))
    for metric, limit in data["absolute_limits"].items():
        assert data["reference_worst_stratum_p90"][metric] <= data["reference_worst_individual_frame"][metric] <= limit
    assert set(data["reference_pooled_camera"]) == {"5", "7", "9"}
    assert data["maximum_numeric_differences"]["field_relative_max"] == 0
    assert data["maximum_numeric_differences"]["metric_absolute_max"] < 3e-14


def test_bilingual_notes_and_figure_boundaries():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        nodes = BeautifulSoup((ROOT / name).read_text(), "html.parser").select("#" + MARKER)
        assert len(nodes) == 1
        translated = nodes[0].select("[data-i18n-zh][data-i18n-en]")
        assert len(translated) == 7
        for language in ("zh", "en"):
            text = " ".join(n["data-i18n-" + language] for n in translated)
            for token in ("101", "5/7/9", "1,111", "3,333", "25/25", "33/33", "0/33", "0.44601", "0.67118", "128A+128AT"):
                assert token in text
            image = nodes[0].select_one("img")
            assert image["data-i18n-alt-" + language]
            assert (ROOT / name).parent.joinpath(image["data-i18n-src-" + language]).exists()
        assert "主目标仍未完成" in nodes[0].get_text()
        assert (ROOT / name).parent.joinpath(nodes[0].select_one("a")["href"]).resolve() == ROOT / "docs" / SUMMARY


def test_old_neural_failures_are_not_promoted_by_baseline_pass():
    data = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    new = data["latest_full_temporal_reference"]
    assert new["cells"] == 3333 and new["complete_absolute_strata"] == new["all_frame_absolute_strata"] == 33
    assert new["old_closed_models_remain_closed"] and not new["learned_full_sequence_evaluated"]
    assert data["latest_query_residual_dual"]["primary_matched_cells"] == 0
    assert data["latest_learned_dual_pcg"]["primary_individual_matched"] == [71, 54, 56, 0]
    text = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert "25/25独立核验全真" in text and "下一模型仍需物理上不同且结果前冻结" in text
    assert "the original whole goal remains active and unmet" in text
