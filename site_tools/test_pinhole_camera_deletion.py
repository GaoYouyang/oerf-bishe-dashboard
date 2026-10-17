"""Aggregate scientific claims, counterevidence, scope and bilingual publication."""

import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
SLUG = "pinhole_camera_deletion_2026-10-17"
ID = "pinhole-camera-deletion-20261017"


def payload():
    return json.loads((ROOT / f"docs/{SLUG}_public_summary.json").read_text())


def test_actual_camera_removal_not_new_physical_population():
    p = payload()
    assert p["physical_camera_ids"] == [0, 1, 3, 4, 5, 6, 7, 8]
    assert p["physical_camera_count"] == 8 and p["dropped_camera_id"] == 2
    assert p["actual_removal_not_padding"] and p["known_hardware_layout"]
    assert (
        p["unique_development_frames"],
        p["complete_held_blocks"],
        p["frames_per_block"],
    ) == (909, 9, 101)
    assert p["original_neural_train_camera_count"] == 5 and not p["neural_retraining"]
    assert p["original_five_camera_neural_statistics_unchanged"]
    assert p["source_frames_per_fold"] == 808
    assert (
        p["all_source_pilot_family_selection_exposure"]
        and not p["unbiased_nested_selection"]
    )


def test_all_controls_and_unchanged_per_block_value():
    p = payload()
    mse = np.array(p["group_four_mse"])
    joint = mse[:, :, :2].mean(2)
    assert mse.shape == (9, 8, 4)
    assert np.allclose(joint, p["group_joint_mse"], rtol=0, atol=1e-14)
    assert np.allclose(joint.mean(0), p["global_joint_mse"], rtol=0, atol=1e-14)
    assert p["all_method_matched_counts"] == [909] * 8
    assert p["all_method_absolute_counts"] == [9] * 8
    for control in (1, 2, 3, 4, 7):
        assert np.all(joint[:, 0] <= 0.95 * joint[:, control])
    for control in (2, 3, 4, 7):
        assert np.all(mse[:, 0] <= 1.01 * mse[:, control])
    gains = 100 * (1 - joint[:, 0] / joint[:, 7])
    assert np.allclose(gains, p["group_gain_vs_static_percent"], atol=1e-12, rtol=0)
    assert gains.min() > 9 and gains.max() < 39


def test_stronger_and_cheaper_counterevidence():
    p = payload()
    risk = np.array(p["global_joint_mse"])
    assert risk[5] < risk[0] and risk[6] < risk[0]
    assert p["gamma_GMM_more_accurate"] and p["beta_GMM_more_accurate"]
    assert p["range_also_matches"] and p["range_is_cheaper"]
    assert p["neural_over_gamma_GMM_joint_risk_ratio"] > 2.3


def test_reference_context_uses_actual_frozen_column_order():
    p = payload()["classical_control_context"]
    assert p["arms"] == [
        "Zero",
        "BP-LS1",
        "CGLS5",
        "Jacobi5",
        "DualRidge5",
        "Reference128",
    ]
    p90, worst = np.array(p["group_p90"]), np.array(p["group_worst"])
    assert p90.shape == worst.shape == (9, 6, 4)
    assert np.all(p90[:, -1] <= [0.5, 0.75, 0.75, 0.2])
    assert np.all(worst[:, -1] <= [1, 1, 1, 0.5])


def test_cost_and_evidence_limits():
    p = payload()
    assert p["cost"]["neural_query"] == dict(
        A=0,
        AT=1,
        full_normal_inverse=1,
        transpose_splats=7,
        small_camera_RHS=16,
        CGLS_steps=0,
    )
    assert p["cost"]["GMM_full_normal_inverses"] == 2
    assert p["cost"]["geometry_factor_source_fit_nonfree"]
    assert not p["cost"]["fresh_resource_measured_for_deletion"]
    assert max(p["numerical_maximum"].values()) < 1e-7
    assert p["other7_controls_double_scored_then_RAM_released_not_third_replayed"]
    for key in (
        "arbitrary_camera_deletion_validated",
        "noisy_deletion_validated",
        "calibration_error_deletion_validated",
        "actual_error_equivalence",
        "resource_speedup",
        "external_generalization",
        "curved_ray_validated",
        "real_bost",
        "algorithm_breakthrough",
        "paper_success",
        "full_goal_complete",
    ):
        assert p[key] is False


def test_public_aggregate_privacy_and_figure():
    raw = (ROOT / f"docs/{SLUG}_public_summary.json").read_text()
    assert not re.search(r"\b[a-f0-9]{64}\b", raw)
    for token in (
        "/Users/",
        "/Volumes/",
        "private_results/",
        "private_notes/",
        "checkpoint",
    ):
        assert token not in raw
    assert (ROOT / f"assets/{SLUG}.png").stat().st_size > 10000


def test_four_substantive_bilingual_views_and_history_retained():
    for name in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
        "learning_log.html",
    ):
        soup = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        sections = soup.select(f"#{ID}")
        assert len(sections) == 1
        section = sections[0]
        assert len(section.select("p[data-i18n-zh][data-i18n-en]")) >= 2
        assert "909" in section.get_text() and "GMM0" in section.get_text()
        assert section.select_one("img")["data-i18n-alt-en"]
        assert soup.select_one("#pinhole-native-loto-20261017")
        assert soup.select_one("#pinhole-flat-loto-20261017")
    assert "实际删相机" in (ROOT / "docs/operator_3d_learning_log.md").read_text()


def test_latest_state_not_stale_five_camera_only():
    evidence = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text()
    )
    current = evidence["latest_pinhole_camera_deletion"]
    assert current["queries"] == current["matched_queries"] == 909
    assert current["absolute_strata"] == current["complete_strata"] == 9
    for name in ("index.html", "operator-learning/index.html"):
        assert (
            "const graph = evidence.latest_pinhole_camera_deletion ||"
            in (ROOT / name).read_text()
        )
    root = BeautifulSoup((ROOT / "index.html").read_text(), "html.parser")
    assert root.select_one("#latestEvidenceLink")["href"] == "#" + ID
    assert root.select_one("#witnessMetric").text == "909"
    assert root.select_one("#callMetric").text == "909 / 909"
