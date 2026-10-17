"""Source acquisition gain cannot be promoted to held neural or speed evidence."""

import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
SLUG = "paired_source_information_2026-10-17"
ID = "paired-source-information-20261017"


def data():
    return json.loads((ROOT / f"docs/{SLUG}_public_summary.json").read_text())


def test_source_scope_and_strong_controls():
    p = data()
    assert p["source_frames"] == 808 and p["source_trajectories"] == 8
    assert p["all_frames_used_for_fitting"] and not p["held_evidence"]
    mse = np.asarray(p["group_four_mse"])
    assert mse.shape == (8, 11, 4)
    joint = mse[:, :, :2].mean((0, 2))
    assert np.allclose(joint, p["global_joint_mse"], rtol=0, atol=1e-14)
    near = np.array(p["near_gamma_beta_spatial_joint_mse"])
    assert np.allclose(100 * (1 - joint[0] / near), p["paired_gain_vs_near_percent"], rtol=0, atol=1e-12)
    assert min(p["paired_gain_vs_near_percent"]) > 85
    assert p["all_arm_matched_counts"][:5] == [808] * 5
    assert p["all_arm_absolute_counts"][:5] == [8] * 5


def test_cost_and_unearned_claims_remain_visible():
    p = data()
    assert p["raw_capture_and_row_multiplier"] == 2
    assert p["GMM0_already_matches_reference"]
    assert p["cost"]["GMM0"] == dict(A=0, AT=1, full_normal_inverses=2)
    assert p["near_observation_not_compared_as_paired"]
    for k in ("learned_initializer_tested", "external_generalization", "resource_speedup", "real_bost", "curved_ray_validated", "algorithm_breakthrough", "paper_success", "full_goal_complete"):
        assert p[k] is False
    assert p["new_neural_updates"] == 0
    assert max(p["independent_physical_maximum"].values()) < 1e-7


def test_four_bilingual_views_and_original_neural_milestone():
    for file in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        soup = BeautifulSoup((ROOT / file).read_text(), "html.parser")
        assert len(soup.select("#" + ID)) == 1
        section = soup.select_one("#" + ID)
        assert len(section.select("p[data-i18n-zh][data-i18n-en]")) == 2
        assert "808" in section.get_text() and "GMM0" in section.get_text()
        assert section.select_one("img")["data-i18n-alt-en"]
        assert soup.select_one("#pinhole-camera-deletion-20261017")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert "paired source information" in log and "不是跨轨迹泛化" in log


def test_aggregate_only_and_figure_exists():
    text = (ROOT / f"docs/{SLUG}_public_summary.json").read_text()
    assert not re.search(r"\b[a-f0-9]{64}\b", text)
    for token in ("/Users/", "/Volumes/", "private_results/", "private_notes/", "checkpoint"):
        assert token not in text
    assert (ROOT / f"assets/{SLUG}.png").stat().st_size > 10000
