import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
SLUG = "pinhole_flat_loto_2026-10-17"


def payload():
    return json.loads((ROOT / f"docs/{SLUG}_public_summary.json").read_text())


def test_nine_complete_development_trajectories_and_scope():
    p = payload()
    assert (p["queries"], p["complete_trajectories"], p["known_camera_count"]) == (
        909,
        9,
        5,
    )
    assert (
        p["all_method_matched_counts"][0] == 909
        and p["all_method_absolute_counts"][0] == 9
    )
    assert (
        p["all_source_pilot_informed_family_selection"]
        and p["first_static_control_post_open"]
    )
    assert p["remaining_eight_static_control_fixed_before_results"]
    for key in (
        "unbiased_nested_selection",
        "new_intrinsic_method",
        "actual_error_equivalence",
        "fresh_resource_measured",
        "resource_speedup",
        "external_generalization",
        "variable_pinhole_count_validated",
        "perturbed_pinhole_validated",
        "algorithm_breakthrough",
        "paper_success",
        "real_bost",
        "curved_ray_validated",
        "finite_aperture_validated",
        "full_goal_complete",
    ):
        assert p[key] is False


def test_all_controls_and_conditional_margin_retained():
    p = payload()
    m = np.array(p["group_four_mse"])
    assert m.shape == (9, 21, 4)
    joint = m[:, :, :2].mean(2)
    assert np.allclose(joint, p["group_joint_mse"], rtol=0, atol=1e-14)
    assert np.allclose(joint.mean(0), p["global_joint_mse"], rtol=0, atol=1e-14)
    assert np.all(joint[:, 0] <= 0.95 * joint[:, 20])
    assert np.all(m[:, 0] <= 1.01 * m[:, 20])
    assert joint.mean(0)[6] < joint.mean(0)[0] and joint.mean(0)[9] < joint.mean(0)[0]
    assert p["all_method_matched_counts"][4] == 909
    assert 30 < p["gain_vs_static_percent"] < 31
    assert (
        p["cost"]["inverse_not_a_free_primitive"]
        and p["cost"]["dense_geometry_factors_nonfree"]
    )
    assert p["cost"]["neural_logical_online"]["full_inverse"] == 1


def test_four_bilingual_pages_agree_and_latest_navigation():
    paragraphs = []
    for name in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
        "learning_log.html",
    ):
        soup = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        sections = soup.select("#pinhole-flat-loto-20261017")
        assert len(sections) == 1
        section = sections[0]
        paragraphs.append(
            [
                (n["data-i18n-zh"], n["data-i18n-en"])
                for n in section.select("p[data-i18n-zh]")
            ]
        )
        assert len(paragraphs[-1]) == 4
        assert all(n.get("data-i18n-en") for n in section.select("[data-i18n-zh]"))
        assert section.select_one("img")["data-i18n-alt-en"]
        assert "909" in section.get_text() and "GMM0" in section.get_text()
        assert (
            section.find_next("section")["id"] == "pinhole-forward-reference-20261016"
        )
    assert all(v == paragraphs[0] for v in paragraphs)
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["latest_pinhole_flat_loto"]["queries"] == 909
    assert current["latest_noise_native"]["matched_queries"] == 1818
    assert (
        current["latest_observation_interface"]["learned_initializer_tested"] is False
    )


def test_public_payload_has_no_private_material():
    raw = (ROOT / f"docs/{SLUG}_public_summary.json").read_text()
    assert not re.search(r"\b[a-f0-9]{64}\b", raw)
    for token in (
        "/Users/",
        "/Volumes/",
        "private_results/",
        "private_notes/",
        "checkpoint",
        "weights",
        "source_array_sha",
    ):
        assert token not in raw
    assert (ROOT / f"assets/{SLUG}.png").stat().st_size > 10000
