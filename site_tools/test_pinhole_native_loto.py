import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np

ROOT = Path(__file__).absolute().parents[1]
SLUG = "pinhole_native_loto_2026-10-17"


def payload():
    return json.loads((ROOT / f"docs/{SLUG}_public_summary.json").read_text())


def test_same_frames_frozen_fold_heads_and_known_count_scope():
    p = payload()
    assert p["known_camera_counts"] == [5, 7, 9]
    assert (
        p["unique_development_frames"],
        p["complete_trajectories"],
        p["newly_verified_count_frame_queries"],
    ) == (909, 9, 1818)
    assert (
        p["total_count_frame_queries"] == 2727
        and p["total_complete_count_trajectory_strata"] == 27
    )
    assert (
        p["neural_train_camera_count"] == 5
        and p["dynamic_head_parameters_per_fold"] == 225
    )
    assert not p["neural_retraining_for_native_counts"]
    assert p["model_dispatch_is_cross_validation_not_a_deployment_ID_feature"]
    assert (
        p["source_mean_field_and_RMS_nonfree"]
        and p["source_family_selection_used_all_source_pilot"]
    )
    for key in (
        "unbiased_nested_selection",
        "arbitrary_camera_layout_validated",
        "native_camera_permutation_validated",
        "perturbed_pinhole_validated",
        "native7_9_fresh_resource_measured",
        "actual_error_equivalence",
        "resource_speedup",
        "external_generalization",
        "curved_ray_validated",
        "finite_aperture_validated",
        "real_bost",
        "algorithm_breakthrough",
        "paper_success",
        "full_goal_complete",
    ):
        assert p[key] is False


def test_all_eight_methods_all_nine_tails_and_strong_counters():
    p = payload()
    assert len(p["arms"]) == 8
    for count in ("5", "7", "9"):
        item = p["by_camera_count"][count]
        mse = np.array(item["group_four_mse"])
        assert mse.shape == (9, 8, 4)
        joint = mse[:, :, :2].mean(2)
        assert np.allclose(joint, item["group_joint_mse"], rtol=0, atol=1e-14)
        assert np.allclose(joint.mean(0), item["global_joint_mse"], rtol=0, atol=1e-14)
        assert (
            item["all_method_matched_counts"][0] == 909
            and item["all_method_absolute_counts"][0] == 9
        )
        assert np.all(joint[:, 0] <= 0.95 * joint[:, 7]) and np.all(
            mse[:, 0] <= 1.01 * mse[:, 7]
        )
        assert (
            joint.mean(0)[5] < joint.mean(0)[0] and joint.mean(0)[6] < joint.mean(0)[0]
        )
        assert item["range_also_matches"]
        classical = p["classical_control_context_by_camera_count"][count]
        assert classical["arms"] == [
            "Zero",
            "BP-LS1",
            "CGLS5",
            "Jacobi5",
            "DualRidge5",
            "Reference128",
        ]
        assert np.array(classical["group_four_mse"]).shape == (9, 6, 4)
        assert classical["all_method_absolute_counts"][-1] == 9


def test_unit_audit_and_complete_resource_counterevidence():
    p = payload()
    unit = p["raw_CFD_unit_replay"]
    assert unit["retained_neural_fields"] == 3636
    assert (
        unit["unit_conversion_not_new_raw_input_prediction"]
        and unit["constant_density_gauge_not_recovered"]
    )
    assert max(unit["maximum"].values()) <= 1e-10
    assert p["manufactured_positive_row_scale"]["checks"] == 162
    resource = p["five_camera_fresh_resource"]
    assert resource["repetitions"] == 3 and len(resource["statistics_by_method"]) == 8
    assert len(resource["all_pairwise_comparisons"]) == 7
    assert not any(
        v["scoped_pairwise_resource_win"]
        for v in resource["all_pairwise_comparisons"].values()
    )
    assert all(
        len(v["wall_seconds"]["all3"]) == 3
        for v in resource["statistics_by_method"].values()
    )
    assert (
        p["cost"]["full_inverse_not_free"]
        and p["cost"]["geometry_dense_factor_source_calibration_and_training_nonfree"]
    )


def test_four_bilingual_pages_and_current_headline():
    records = []
    for name in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
        "learning_log.html",
    ):
        soup = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        sections = soup.select("#pinhole-native-loto-20261017")
        assert len(sections) == 1
        section = sections[0]
        records.append(
            [
                (n["data-i18n-zh"], n["data-i18n-en"])
                for n in section.select("p[data-i18n-zh]")
            ]
        )
        assert (
            len(records[-1]) == 4
            and section.find_next("section")["id"] == "pinhole-flat-loto-20261017"
        )
        assert section.select_one("img")["data-i18n-alt-en"]
        assert len(section.select("tbody tr")) == 8
        assert "GMM0" in section.get_text() and "909/909" in section.get_text()
    assert all(value == records[0] for value in records)
    evidence = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text()
    )
    assert evidence["latest_pinhole_native_loto"]["queries"] == 1818
    assert evidence["latest_pinhole_flat_loto"]["queries"] == 909
    for name in ("index.html", "operator-learning/index.html"):
        assert (
            "evidence.latest_pinhole_native_loto || evidence.latest_pinhole_flat_loto ||"
            in (ROOT / name).read_text()
        )


def test_public_whitelist_and_figure():
    raw = (ROOT / f"docs/{SLUG}_public_summary.json").read_text()
    assert not re.search(r"\b[a-f0-9]{64}\b", raw)
    for token in (
        "/Users/",
        "/Volumes/",
        "private_results/",
        "private_notes/",
        "checkpoint",
        "source_array_sha",
    ):
        assert token not in raw
    assert (ROOT / f"assets/{SLUG}.png").stat().st_size > 10000
