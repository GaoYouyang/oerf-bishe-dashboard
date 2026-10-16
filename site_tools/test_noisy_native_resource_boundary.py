import json
from pathlib import Path
import re

from bs4 import BeautifulSoup


ROOT = Path(__file__).absolute().parents[1]
SLUG = "noise_band_native_resource_2026-10-16"


def data():
    return json.loads((ROOT/f"docs/{SLUG}_public_summary.json").read_text())


def test_fifty_fixed_trials_and_original_resource_gates():
    row = data()
    assert row["actual_fresh_processes"] == 50 and row["repetitions_each"] == 5
    assert row["known_native_camera_counts"] == [7, 9]
    assert row["independent_all50_pass"] and row["registered_terminal_readback_pass"]
    assert row["new_training_updates"] == row["CGLS_steps"] == 0
    assert row["original_five_camera_resource_results_not_pooled"]
    assert row["filesystem_cache_unmanaged"] and row["source_fitting_historical_nonfree"]
    for count in ("7", "9"):
        s = row["by_camera_count"][count]
        assert all(not v["scoped_faster_and_no_more_RSS"] for v in s["comparisons"].values())
        assert s["quality"]["BandNN"]["matched"] == s["quality"]["BandRange"]["matched"] == 909
        assert s["quality"]["BandNN"]["absolute"] == 9
        assert s["statistics_by_method"]["BandRange"]["wall_seconds"]["median"] < s["statistics_by_method"]["BandNN"]["wall_seconds"]["median"]
        assert s["quality"]["BandNN"]["joint"] < s["quality"]["BandRange"]["joint"]
        assert s["quality"]["RawGMM0"]["matched"] < 909
        for method in s["statistics_by_method"]:
            trials = [r for r in row["all_fresh_trials"] if str(r["count"]) == count and r["method"] == method]
            assert [r["repeat"] for r in trials] == list(range(5))
    assert row["by_camera_count"]["7"]["comparisons"]["BandGMM0"]["gates"]["every_repeat_wall_5percent"]
    assert not row["by_camera_count"]["7"]["comparisons"]["BandGMM0"]["gates"]["every_repeat_RSS_nonharm"]
    assert not row["by_camera_count"]["9"]["comparisons"]["BandGMM0"]["gates"]["median_wall_5percent"]


def test_no_global_speed_or_private_export():
    row = data()
    assert row["no_stable_joint_resource_pair_passed"] and row["no_global_speedup_claim"]
    assert row["simulator_variance_is_conditional_side_information"]
    assert all(not row[k] for k in ("algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization", "curved_ray_validated", "real_bost", "full_goal_complete"))
    source = (ROOT/f"docs/{SLUG}_public_summary.json").read_text()
    assert not re.search(r"\b[0-9a-f]{64}\b", source)
    assert not any(token in source for token in ("/Users/", "/Volumes/", "private_results", "private_notes", "checkpoint", "field_sha"))


def test_cost_boundary_stays_inside_existing_positive_milestone():
    paragraphs = []
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        soup = BeautifulSoup((ROOT/name).read_text(), "html.parser")
        boundary = soup.select_one("#native-resource-boundary-20261016")
        assert boundary and boundary.find_parent("section")["id"] == "noise-band-native-20261016"
        assert boundary.get_text() == boundary["data-i18n-zh"]
        assert all(value in boundary.get_text() for value in ("128.68/264.10", "32.60%/17.65%", "不是全面加速"))
        assert "not global acceleration" in boundary["data-i18n-en"]
        paragraphs.append(boundary["data-i18n-zh"])
    assert len(set(paragraphs)) == 1
    current = json.loads((ROOT/"operator-learning/current-evidence.json").read_text())["latest_noise_native"]
    assert current["fresh_native_resource_measured"] and not current["stable_joint_resource_gate_passed"]
