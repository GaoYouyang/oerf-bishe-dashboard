import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np


ROOT = Path(__file__).absolute().parents[1]
SLUG = "noise_band_native_cardinality_2026-10-16"
ANCHOR = "noise-band-native-20261016"


def evidence():
    return json.loads((ROOT / f"docs/{SLUG}_public_summary.json").read_text())


def test_exact_population_and_scope():
    data = evidence()
    assert data["decision"] == "PASS_NOISE_BAND_NATIVE_CARDINALITY"
    assert data["new_queries"] == 1818 and data["new_trajectory_count_strata"] == 18
    assert data["known_native_camera_counts"] == [7, 9]
    assert data["new_training_updates"] == data["CGLS_steps"] == 0
    assert data["unchanged_five_camera_normalization_and_weights"]
    assert data["all_four_population_seals_before_truth"] and all(data["independent_checks"].values())
    assert data["terminal_readback_pass"] and data["complete_trajectory_exclusion"]
    assert data["exploratory_development_visible_trajectories"]
    assert data["simulator_reported_variance_conditional_input"]
    assert not data["fresh_resource_measured"] and not any(data["claims"].values())
    assert data["original_noise_failure_preserved"] and data["original_engineering_invalidity_preserved"]


def test_four_metric_tails_and_control_boundaries():
    data = evidence()
    for count in ("7", "9"):
        s = data["by_camera_count"][count]
        assert s["matched"][0] == 909 and s["absolute"][0] == 9 and all(s["gates"].values())
        assert np.asarray(s["matched_by_group"])[:, 0].tolist() == [101] * 9
        assert np.asarray(s["absolute_by_group"])[:, 0].all()
        assert (np.asarray(s["p90"])[:, 0] <= [.5, .75, .75, .2]).all()
        assert (np.asarray(s["worst"])[:, 0] <= [1., 1., 1., .5]).all()
        assert s["dominators"] == []
        assert abs(s["gains_percent_vs_identity"] - 100 * (1 - s["joint"][0] / s["joint"][1])) < 1e-12
        assert abs(s["joint"][3] / s["joint"][0] - 1) < 3e-5
        assert s["matched"][6] == 909 and s["primary_harms_vs_raw_GMM"][3] == 909
        assert data["independent_gaps"][count]["maximum_metric_absolute"] < 1e-7
    assert data["by_camera_count"]["7"]["joint"][7] < data["by_camera_count"]["7"]["joint"][0]
    nine = data["by_camera_count"]["9"]
    assert nine["joint"][0] < nine["joint"][7] and nine["matched"][7] == 907
    assert data["primary_per_query"]["detector_Poisson_by_count"] == {"7": 14, "9": 18}
    assert data["primary_per_query"]["full_U_products"] == 3


def test_bilingual_publication_preserves_private_boundary():
    paragraphs = []
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        source = (ROOT / name).read_text()
        soup = BeautifulSoup(source, "html.parser")
        section = soup.select(f"#{ANCHOR}")
        assert len(section) == 1
        assert "28.94%/29.63%" in section[0].get_text()
        paragraphs.append([n["data-i18n-zh"] for n in section[0].select("p[data-i18n-zh]")])
        for node in section[0].select("[data-i18n-zh]"):
            assert node.get_text() == node["data-i18n-zh"] and len(node["data-i18n-en"]) > 8
        assert section[0].select_one("img").get("data-i18n-alt-en")
        if name in ("index.html", "operator-learning/index.html"):
            assert "evidence.latest_noise_native || evidence.latest_noise_band" in source
    assert all(row == paragraphs[0] for row in paragraphs)
    source = (ROOT / f"docs/{SLUG}_public_summary.json").read_text()
    assert not re.search(r"\b[0-9a-f]{64}\b", source)
    assert not any(token in source for token in ("/Users/", "/Volumes/", "private_results/", "private_notes/", "checkpoint"))
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())["latest_noise_native"]
    assert current["queries"] == current["matched_queries"] == 1818
    assert current["absolute_strata"] == current["complete_strata"] == 18
