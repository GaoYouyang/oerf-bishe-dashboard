"""Check the public pilot scope, arithmetic and bilingual boundaries."""
import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = "poolfire_endpoint_pilot_20260906"


def test_pilot_does_not_replace_whole_trajectory_prediction():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert data["passing_points"] == len(data["points"]) == 5
    assert not data["whole_trajectory_qualified"] and data["same_discretization_inverse_consistency"]
    assert not any(data[k] for k in ("learned_algorithm", "resource_speedup", "call_reduction", "real_bost"))
    assert current["latest_diagnostic"]["status"] == data["status"]
    assert current["latest_prediction"]["matched_cells"] == 367
    assert current["latest_prediction"]["complete_trajectories"] == 1
    assert current["previous_loss_diagnostic"]["floor_dominant_query_cells"] == 505
    for row in data["points"]:
        for arm in ("lsmr", "lsqr"):
            assert all(0 <= value <= .01 for value in row[arm].values())
            calls = row[arm + "_calls_including_numerical_checks"]
            assert calls["A"] == row[arm + "_iterations"] + 1
            assert calls["AT"] == row[arm + "_iterations"] + 3
        assert .4 < row["cgls4"]["field"] < .5
    assert min(r["lsmr_iterations"] for r in data["points"]) == 1945
    assert max(r["lsmr_iterations"] for r in data["points"]) == 2122


def test_pilot_notes_are_substantively_bilingual():
    for filename in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / filename).read_text(), "html.parser")
        notes = soup.select("#endpoint-pilot-result")
        assert len(notes) == 1
        note = notes[0]
        for key in ("data-i18n-zh", "data-i18n-en"):
            assert all(term in note[key] for term in ("1%", "1945", "2122", "LSMR", "LSQR"))
        assert "not learning" in note["data-i18n-en"]
        assert "不是学习" in note["data-i18n-zh"]
    text = (ROOT / f"docs/{STEM}.md").read_text()
    assert "五点" in text and "five points, not five completed trajectories" in text
    assert "same discrete forward" in text and "同一离散forward" in text
    assert "not minimum calls" in text and "最小调用数" in text
    for forbidden in ("/Users/", "/Volumes/", "private_results", "sha256", "cameraData", "checkpoint", ".pt"):
        assert forbidden not in text
        assert forbidden not in (ROOT / f"docs/{STEM}.json").read_text()
    assert (ROOT / f"assets/figures/{STEM}.png").stat().st_size > 10000
