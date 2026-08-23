from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_five_camera_subset_oracle_v170_public_summary.json"
RESULT = ROOT / "docs/real_bost_five_camera_subset_oracle_v170_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/real_bost_five_camera_subset_oracle_v170.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_preserves_truth_aware_capacity_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert (
        payload["scientific_decision"]
        == "PASS_GEOMETRY_ONLY_SHARED_FIVE_CAMERA_SUBSET_CAPACITY_V170"
    )
    assert payload["capacity"]["calibration_shared"]["strata_passed"] == 4
    assert payload["capacity"]["calibration_shared"]["strata_total"] == 4
    assert payload["capacity"]["calibration_shared"]["passed"] is True
    assert (
        payload["capacity"]["calibration_shared"]["strata"][2]["gradient"]["p90_higher"]
        == 0.7489533005177225
    )
    assert payload["post_open_robustness_audit"]["minimum"] == 12
    assert (
        payload["post_open_robustness_audit"]["all_calibrations_have_at_least_one"]
        is True
    )
    assert payload["execution"]["candidate_cell_count"] == 58968
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["claim_limits"]["deployable_selector_established"] is False
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "五相机并非没有容量" in text
    assert "five cameras have capacity" in text
    assert "58,968" in text
    assert "PASS_GEOMETRY_ONLY_SHARED_FIVE_CAMERA_SUBSET_CAPACITY_V170" in text
    assert "algorithm_breakthrough=false" in text


def test_public_surfaces_expose_the_same_capacity_verdict() -> None:
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        assert "v170" in text
        assert "real_bost_five_camera_subset_oracle_v170_result_2026-08-21.md" in text
        assert "algorithm_breakthrough" in text
    for surface in SURFACES[:2]:
        text = surface.read_text(encoding="utf-8")
        assert "PASS_GEOMETRY_ONLY_SHARED_FIVE_CAMERA_SUBSET_CAPACITY_V170" in text
        assert "0.748953" in text


def test_current_evidence_preserves_v170_but_points_to_v181() -> None:
    payload = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert payload["v170_five_camera_subset_oracle_scientific_decision"] == (
        "PASS_GEOMETRY_ONLY_SHARED_FIVE_CAMERA_SUBSET_CAPACITY_V170"
    )
    assert (
        payload["current_decision"]["v170_calibration_shared_capacity_passed"] is True
    )
    assert payload["current_decision"]["v170_deployable_selector_established"] is False
    assert (
        payload["engineering_status"]
        == "PASS_INDEPENDENT_RECOMPUTATION_ALL_NINE_CONTROL_ATTRIBUTION_V204"
    )
    assert (
        payload["formal_status"]
            == "PASS_FORMAL_POOLFIRE_ALL_NINE_CONTROL_ATTRIBUTION_P14_V204"
    )
    assert payload["scientific_status"] == (
        "PASS_ALL_NINE_DENSE_REPRESENTATION_CALL_HEADROOM_V204"
    )


def test_public_figure_is_large_and_nonblank() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.convert("RGB").getbbox() is not None


def test_public_payload_does_not_expose_private_execution_material() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = [
        "/Users/",
        "private_results",
        "private_data",
        "model_tree_seal",
        "calibration_tree_seal",
        "checkpoint",
        "cameraData",
        "hexplane",
    ]
    for value in forbidden:
        assert value not in text
