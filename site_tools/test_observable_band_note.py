import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NAME = "poolfire_observable_band_2026-10-07_public_summary.json"
MARKER = "poolfire-observable-band-20261007"


def evidence():
    return json.loads((ROOT / "docs" / NAME).read_text())


def test_scope_and_independent_result():
    d = evidence()
    assert d["decision"] == "FAIL_FIXED_OBSERVABLE_RESOLVENT128_WARM34_CLOSED"
    assert d["cells"] == 99 and d["sampled_three_frame_strata"] == 33
    assert d["frames"] == [0, 50, 100] and d["camera_counts"] == [5, 7, 9]
    assert d["independent_checks_passed"] == d["independent_checks_total"] == 60
    assert d["reference_adequate"] and d["matched_cells"][d["primary"]] == 0
    assert d["absolute_strata"][d["primary"]] == 33


def test_controls_and_cost_not_resource_claims():
    d = evidence()
    assert d["uniform_control_comparison"]["all_four_better"] == 99
    assert d["descriptive_controls"]["CGLS35"]["all_four_better"] == 65
    assert d["matched_cells"]["NormalFactor-PCGLS34"] == d["matched_cells"]["DirectRidge-Warm34"] == 99
    assert d["standalone"] == {"A": 35, "AT": 35, "coarse_actions": 1}
    assert d["geometry_factorization_and_rhs_solves_nonfree"]
    assert d["query_full_factor_solves"] == d["shared_trainable_parameters"] == 0
    assert d["old_invalid_control_not_retroactively_passed"]
    assert d["primary_geometry_and_thresholds_unchanged"] and d["fixed_recipe_closed_without_tuning"]
    for key in ("learned_initializer_tested", "full_sequence_tested", "algorithm_breakthrough", "paper_success", "fresh_wall_RSS_tested", "resource_speedup", "external_generalization", "real_bost", "curved_ray_validated", "main_goal_complete"):
        assert d[key] is False


def test_four_bilingual_notes_and_links():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        page = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        notes = page.select("#" + MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select("[data-i18n-zh][data-i18n-en]")) == 6
        assert all(node["data-i18n-zh"] and node["data-i18n-en"] for node in note.select("[data-i18n-zh]"))
        assert all(text in note.get_text() for text in ("60/60", "0/99", "33/33", "5/7/9", "99/99", "35A+35AT"))
        assert (ROOT / name).parent.joinpath(note.find("a")["href"]).resolve().is_file()


def test_aggregate_history_log_and_privacy():
    d = evidence()
    aggregate = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert aggregate["latest_observable_band_seed"]["independent_checks_passed"] == 60
    assert aggregate["latest_observable_band_seed"]["matched_cells"] == 0
    assert "latest_conditional_inverse_seed" in aggregate
    assert "## 2026-10-07: 可观测弱模态带来小幅改善" in (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert d["final_field_max_relative"] < 1e-7 and d["metric_max_absolute"] < 1e-7
    assert not any(word in (ROOT / "docs" / NAME).read_text() for word in ("/Users/", "private_results/", "private_data/", "sha256", "checkpoint"))
