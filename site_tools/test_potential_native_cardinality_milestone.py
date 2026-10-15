import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]


def evidence():
    return json.loads((ROOT / "docs/potential_native_cardinality_2026-10-15_public_summary.json").read_text())


def test_complete_native_population_and_independence():
    data = evidence()
    assert data["decision"] == "PASS_FROZEN_NATIVE_CARDINALITY_TRANSFER"
    assert data["new_queries"] == 1818 and data["new_complete_trajectory_count_strata"] == 18
    assert data["new_neural_training_updates"] == data["CGLS_refinement_steps"] == 0
    assert data["learned_parameter_count"] == 225 and data["inherited_five_camera_evidence_not_rerun"]
    assert data["independent_checks_pass"] and data["terminal_readback_pass"]
    assert data["maximum_per_cell_metric_absolute_gap"] <= 1e-7
    for count in ("7", "9"):
        row = data["counts"][count]
        assert all(row["gates"].values()) and row["matched"][:5] == [909] * 5
        assert row["absolute"][:5] == [9] * 5
        assert row["primary_harms_vs_GMM0"] == [909] * 4
        assert row["joint"][4] < row["joint"][0] < min(row["joint"][1:4])
        for index, label in ((1, "Linear0"), (2, "NativeCachedMean0"), (3, "Range0")):
            assert abs(row["primary_joint_reduction_percent"][label] - 100 * (1 - row["joint"][0] / row["joint"][index])) < 1e-12
        assert row["standalone_cost_vectors"][0] == [1, 1, 1, 7, 2 * int(count)]
        assert row["standalone_cost_vectors"][3] == [0, 1, 1, 0, 0]
    assert not any(data["claims"].values())


def test_substantive_bilingual_milestone_without_private_material():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        text = (ROOT / name).read_text()
        section = BeautifulSoup(text, "html.parser").select_one("#potential-native-20261015")
        assert section is not None
        assert "Range0" in section.get_text() and "GMM0" in section.get_text()
        for node in section.select("[data-i18n-zh]"):
            assert node.get_text() == node["data-i18n-zh"] and node.get("data-i18n-en")
        if name in ("index.html", "operator-learning/index.html"):
            assert "evidence.latest_potential_native || evidence.latest_potential_resource" in text
    latest = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())["latest_potential_native"]
    assert latest["queries"] == latest["matched_queries"] == 1818
    assert latest["absolute_strata"] == latest["complete_strata"] == 18
    assert not latest["algorithm_breakthrough"]
    summary = (ROOT / "docs/potential_native_cardinality_2026-10-15_public_summary.json").read_text()
    for token in ("/Users/", "/Volumes/", "private_results", "sha256", "field_sha", "weights_1024"):
        assert token not in summary
