"""Verify reference scope, whole-sequence arithmetic and bilingual boundaries."""
import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = "poolfire_fixed512_reference_20260906"


def test_reference_is_not_learned_success():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["latest_reference_qualification"]["status"] == data["status"]
    assert current["latest_prediction"]["matched_cells"] == 367
    assert data["total"] == data["new_nonpilot_samples"] + data["reused_pilot_samples"] == 505
    assert all(data["checks"].values())
    for path in ("formal", "independent"):
        for arm in ("cgls", "jacobi_pcgls"):
            result = data["summaries"][path][arm]
            assert result["passing"] == 505 and result["complete_trajectories"] == 5
            for trajectory in result["trajectories"]:
                assert trajectory["passing"] == trajectory["total"] == 101
                assert all(v["worst"] <= .01 for v in trajectory["tails"].values())
    for key in ("learned_algorithm", "minimum_calls_proven", "resource_speedup", "external_generalization",
                "real_bost", "variable_camera_validated", "independent_test", "previous_cost_curves_requalified"):
        assert data[key] is False
    assert data["new_solver_calls"] == {"A": 2000 * 512, "AT": 2000 * 512}
    assert data["logical_calls_per_endpoint"] == {"A": 512, "AT": 512}


def test_bilingual_notes_and_no_private_identifiers():
    for file in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / file).read_text(), "html.parser")
        notes = soup.select("#fixed512-reference-result")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(word in notes[0][f"data-i18n-{lang}"] for word in ("512", "505/505", "5/5", "1%"))
        assert "not minimum calls" in notes[0]["data-i18n-en"]
        assert "不是最小调用数" in notes[0]["data-i18n-zh"]
    text = (ROOT / f"docs/{STEM}.md").read_text()
    assert "## 中文" in text and "## English" in text and "pilot-informed" in text
    assert "同一离散forward" in text and "same discrete forward" in text
    for suffix in ("md", "json"):
        content = (ROOT / f"docs/{STEM}.{suffix}").read_text()
        assert all(term not in content for term in ("/Users/", "/Volumes/", "private_results", "sha256", ".pt"))
    assert (ROOT / f"assets/figures/{STEM}.png").stat().st_size > 10000
