import json
from pathlib import Path

from bs4 import BeautifulSoup
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def evidence():
    return json.loads((ROOT / "docs/potential_primal_loto_2026-10-15_public_summary.json").read_text())


def test_exploratory_held_scope_and_seals():
    data = evidence()
    assert data["decision"] == "PASS_POTENTIAL_EXPLORATORY_COMPLETE_TRAJECTORY_LOTO"
    assert data["queries"] == 909 and data["complete_trajectories"] == 9
    assert data["train_fields_per_fold"] == 808 and data["frames_per_trajectory"] == 101
    assert data["exploratory_complete_trajectory_LOTO"] and not data["source_in_sample"]
    assert data["trajectories_previously_development_visible"]
    assert data["every_fitted_statistic_and_control_train_only"]
    assert data["all_double_predictions_sealed_before_own_held_truth_scoring"]
    assert data["all_source_gates_pass_before_held_prediction"]
    assert all(data["independent_checks"].values()) and all(data["gates"].values())
    assert not any(data["claims"].values())
    assert data["known_clean_cameras"] == 5 and not data["native_variable_geometry_or_noise_established"]


def test_gains_counts_and_stronger_control():
    data = evidence()
    joint = np.asarray(data["joint_relative_MSE"])
    assert round(100 * (1 - joint[0] / joint[1]), 2) == 34.78
    assert round(100 * (1 - joint[0] / joint[2]), 2) == 34.83
    assert data["matched_queries"][0] == 909 and data["absolute_trajectories"][0] == 9
    assert data["matched_queries"][4:8] == [909] * 4
    assert data["absolute_trajectories"][4:8] == [9] * 4
    assert joint[4] < joint[0] and data["primary_observation_MSE_ratio_vs_GMM0"] > 4
    assert data["primary_harms_vs_GMM0"] == [882, 891, 872, 909]
    assert data["standalone_cost_vectors"][0] == [4, 3, 1, 7, 10]
    assert data["standalone_cost_vectors"][4] == [1, 2, 2, 0, 0]
    assert data["heterogeneous_cost_vectors_are_not_speed_evidence"]
    assert data["componentwise_dominators"] == []


def test_bilingual_pages_and_current_labels():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        text = (ROOT / name).read_text()
        soup = BeautifulSoup(text, "html.parser")
        section = soup.select_one("#potential-loto-20261015")
        assert section
        assert "34.8" in section.get_text() or "34.78" in section.get_text()
        for node in section.select("[data-i18n-zh]"):
            assert node.get_text() == node["data-i18n-zh"] and node.get("data-i18n-en")
        assert "latest_potential_loto" in text if name in ("index.html", "operator-learning/index.html") else True
    data = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())["latest_potential_loto"]
    assert data["exploratory_complete_trajectory_loto"] and not data["source_in_sample"]
    assert "GMM0" in data["headline_zh"] and "GMM0" in data["headline_en"]
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert log.index("完整留轨迹的学习收益") < log.index("实际小神经模型获得完整源内空间收益")
    assert "10.18倍" in log
    old = json.loads((ROOT / "docs/multilevel_graph_source_2026-10-15_public_summary.json").read_text())
    assert old["decision"] == "FAIL_MULTILEVEL_GRAPH_FINITE_ACTION_SOURCE_VALUE"


def test_summary_has_no_private_identifiers():
    text = (ROOT / "docs/potential_primal_loto_2026-10-15_public_summary.json").read_text()
    for token in ("/Users/", "/Volumes/", "private_results", "checkpoint", "sha256", "weights_1024"):
        assert token not in text
