import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).absolute().parents[1]
SLUG = "pinhole_forward_reference_2026-10-16"


def test_physical_and_reference_scope():
    p = json.loads((ROOT/f"docs/{SLUG}_public_summary.json").read_text())
    assert p["physical_model_comparison"]["within_count"] == 0
    assert p["physical_model_comparison"]["total"] == 2727
    assert p["qualified_complete_reference_strata"] == 27
    assert p["reference_summary"]["reference_all"]
    assert p["reference_summary"]["matched_cells"] == [0, 0, 0, 0, 0, 2727]
    assert p["reference_summary"]["absolute_counts"] == [0, 0, 0, 3, 0, 27]
    assert p["new_training_updates"] == 0
    for k in ("new_learned_initializer_tested", "algorithm_breakthrough", "resource_speedup", "fresh_resource_measured",
              "paper_success", "external_generalization", "curved_ray_validated", "finite_aperture_validated", "real_bost", "full_goal_complete"):
        assert p[k] is False


def test_bilingual_pages_have_same_physical_milestone():
    paragraphs = []
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        soup = BeautifulSoup((ROOT/name).read_text(), "html.parser")
        sections = soup.select("#pinhole-forward-reference-20261016")
        assert len(sections) == 1
        section = sections[0]
        paragraphs.append([(n["data-i18n-zh"], n["data-i18n-en"]) for n in section.select("p[data-i18n-zh]")])
        assert len(paragraphs[-1]) == 4
        assert "27/27" in section.get_text() and "2727" in section.get_text()
        assert "CGLS128" in section.get_text()
        assert all(n.get("data-i18n-en") for n in section.select("[data-i18n-zh]"))
        assert section.select_one("img")["data-i18n-alt-en"]
        assert len(section.select("a")) == 2
    assert all(value == paragraphs[0] for value in paragraphs)


def test_public_payload_privacy_and_retained_old_frontier():
    raw = (ROOT/f"docs/{SLUG}_public_summary.json").read_text()
    for forbidden in ("/Users/", "/Volumes/", "private_results/", "private_notes/", "checkpoint", "weights", "source_array_sha"):
        assert forbidden not in raw
    old = json.loads((ROOT/"docs/noise_band_native_resource_2026-10-16_public_summary.json").read_text())
    assert old["no_stable_joint_resource_pair_passed"] and old["resource_speedup"] is False
    current = json.loads((ROOT/"operator-learning/current-evidence.json").read_text())
    assert current["latest_noise_native"]["matched_queries"] == 1818
    assert current["latest_observation_interface"]["learned_initializer_tested"] is False
    assert (ROOT/f"assets/{SLUG}.png").stat().st_size > 10000
