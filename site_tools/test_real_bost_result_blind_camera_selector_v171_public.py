from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_result_blind_camera_selector_v171_public_summary.json"
RESULT = ROOT / "docs/real_bost_result_blind_camera_selector_v171_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/real_bost_result_blind_camera_selector_v171.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_preserves_result_blind_headroom_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert (
        payload["scientific_decision"]
        == "PASS_RESULT_BLIND_GEOMETRY_SELECTOR_HEADROOM_V171"
    )
    assert payload["selector"]["strict_local_safe_count"] == 13
    assert payload["selector"]["strict_local_safe_total"] == 13
    assert payload["selector"]["trainable_parameters_max"] == 357
    assert payload["selector"]["heldout_outcomes_available_to_prediction"] is False
    assert payload["controls"]["fit_static"]["strict_local_safe_count"] == 2
    assert payload["controls"]["v169_fixed_geometry"]["strict_local_safe_count"] == 0
    assert payload["controls"]["passing_cheap_controls"] == []
    assert all(row["passed"] for row in payload["primary_strata"])
    assert payload["primary_strata"][2]["gradient"]["p90_higher"] == 0.6303836430312433
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["independent_recomputation"]["check_count"] == 21
    assert payload["claim_limits"]["external_generalization"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "结果不可见的几何选择器找回了五相机容量" in text
    assert "a result-blind geometry selector recovers five-camera capacity" in text
    assert "13/13" in text
    assert "357" in text
    assert "PASS_RESULT_BLIND_GEOMETRY_SELECTOR_HEADROOM_V171" in text
    assert "algorithm_breakthrough=false" in text


def test_public_surfaces_expose_the_same_v171_verdict() -> None:
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        assert "PASS_RESULT_BLIND_GEOMETRY_SELECTOR_HEADROOM_V171" in text
        assert "13/13" in text
        assert "algorithm_breakthrough" in text
    for surface in SURFACES[1:]:
        assert "0.630384" in surface.read_text(encoding="utf-8")


def test_current_evidence_preserves_v171_but_points_to_v181() -> None:
    payload = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert (
        payload["v221_low64_exact_rowspace_lift_independent_status"]
        == "PASS_INDEPENDENT_RECOMPUTATION_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert (
        payload["v221_low64_exact_rowspace_lift_formal_status"]
            == "FORMAL_PENDING_INDEPENDENT_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert payload["v221_low64_exact_rowspace_lift_scientific_decision"] == (
        "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert payload["current_decision"]["v171_result_blind_selector_passed"] is True
    assert payload["current_decision"]["v171_external_generalization"] is False
    assert payload["current_decision"]["algorithm_breakthrough"] is False


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
        "7c887937",
        "0463b996",
    ]
    for value in forbidden:
        assert value not in text
