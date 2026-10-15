import json
from pathlib import Path
import statistics

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]


def evidence():
    return json.loads((ROOT / "docs/potential_lift_resource_2026-10-15_public_summary.json").read_text())


def test_all_fixed_trials_and_resource_gates():
    data = evidence()
    assert data["decision"] == "PASS_SCOPED_THREE_WAY_LIFT_RESOURCE_TRADEOFF"
    assert len(data["all_trials"]) == 9 and data["resource_repetitions_per_method"] == 3
    for name in ("Potential0", "Range0", "GMM0"):
        rows = [v for v in data["all_trials"] if v["method"] == name]
        assert len(rows) == 3
        assert statistics.median(v["whole_process_wall_seconds"] for v in rows) == data["median_whole_process_wall_seconds"][name]
        assert max(v["lifetime_peak_RSS_bytes"] for v in rows) == data["maximum_lifetime_RSS_bytes"][name]
        assert data["matched_queries"][name] == 909 and data["absolute_trajectory_tails"][name] == 9
    assert all(data["resource_gates"].values())
    assert data["independent_checks_pass"] and data["terminal_readback_pass"]
    assert data["maximum_metric_relative_gap"] < 1e-7


def test_scoped_tradeoff_not_control_dominance():
    data = evidence()
    joint = data["joint_field_full_gradient_relative_MSE"]
    wall = data["median_whole_process_wall_seconds"]
    assert joint["GMM0"] < joint["Potential0"] < joint["Range0"]
    assert wall["Range0"] < wall["Potential0"] < wall["GMM0"]
    assert round(100 * (1 - data["wall_ratio_learned_to_GMM0"]), 2) == 52.35
    assert round(100 * (1 - data["peak_RSS_ratio_learned_to_GMM0"]), 2) == 4.59
    assert data["no_refinement_CGLS_steps"] and data["new_training_updates"] == 0
    assert data["previously_development_visible"] and data["exploratory_complete_trajectory_LOTO"]
    assert not any(data["claims"].values())
    assert data["actual_calls_per_nonzero_query"]["Potential0"]["QT"] == 7
    assert data["actual_calls_per_nonzero_query"]["Potential0"]["small_Poisson"] == 10
    assert data["actual_calls_per_nonzero_query"]["Range0"]["A"] == 0


def test_bilingual_milestone_and_no_private_material():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        soup = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        section = soup.select_one("#potential-resource-20261015")
        assert section
        assert "GMM0" in section.get_text() and "Range0" in section.get_text()
        for node in section.select("[data-i18n-zh]"):
            assert node.get_text() == node["data-i18n-zh"] and node.get("data-i18n-en")
    latest = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())["latest_potential_resource"]
    assert latest["matched_queries"] == 909 and not latest["algorithm_breakthrough"]
    text = (ROOT / "docs/potential_lift_resource_2026-10-15_public_summary.json").read_text()
    for token in ("/Users/", "/Volumes/", "private_results", "sha256", "field_sha", "weights_1024"):
        assert token not in text
