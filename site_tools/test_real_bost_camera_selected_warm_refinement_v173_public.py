from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/real_bost_camera_selected_warm_refinement_v173_public_summary.json"
)
RESULT = (
    ROOT / "docs/real_bost_camera_selected_warm_refinement_v173_result_2026-08-21.md"
)
FIGURE = ROOT / "assets/figures/real_bost_camera_selected_warm_refinement_v173.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_the_negative_scientific_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == (
        "FAIL_CLASSICAL_CONTROL_EXPLAINS_CAMERA_SELECTED_WARM_V173"
    )
    assert payload["primary_h1_k1"]["strict_safe_cells"] == 468
    assert payload["blocking_h1_k0"]["strict_safe_cells"] == 468
    assert payload["primary_h1_k1"]["complete_pass"] is True
    assert payload["blocking_h1_k0"]["complete_pass"] is True
    assert payload["primary_h1_k1"]["exact_forward_calls"] == 2
    assert payload["blocking_h1_k0"]["exact_forward_calls"] == 1
    assert payload["controls"]["blocking_controls_that_passed"] == ["primary_h1_k0"]
    assert payload["independent_recomputation"]["check_count"] == 21
    assert payload["claim_limits"]["cgls_k1_advantage_established"] is False
    assert (
        payload["claim_limits"]["selector_advantage_under_equal_h1_k0_cost_established"]
        is False
    )
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["resource_speedup"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v173：完整 warm refinement 过门" in text
    assert "# v173: the full warm refinement passes" in text
    assert "FAIL_CLASSICAL_CONTROL_EXPLAINS_CAMERA_SELECTED_WARM_V173" in text
    assert "H1-K0" in text
    assert "1A+1A^T" in text
    assert "21/21" in text
    assert "algorithm_breakthrough=false" in text
    assert "相机选择 headroom" in text
    assert "camera-selection headroom" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_preserves_v173_but_points_to_v181() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == (
        "FAIL_SIGNED_LINE_CANCELLATION_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V212"
    )
    assert payload["metrics"]["v173_primary_strict_safe_count"] == 468
    assert payload["metrics"]["v173_h1_k0_strict_safe_count"] == 468
    assert payload["metrics"]["v173_h1_k0_exact_A"] == 1
    assert payload["metrics"]["v173_independent_check_count"] == 21
    assert "fixed 64-mode spectral floor" in payload["next_scientific_gate_en"]
    assert "paired real-BOST physical data" in payload["next_scientific_gate_en"]
    assert "配对真实 BOST 物理数据" in payload["next_scientific_gate_zh"]
    assert "真实 BOST" in payload["next_scientific_gate_zh"]


def test_primary_pages_reference_v173_in_both_languages() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily, home):
        assert "v173" in text
    assert "更便宜的 H1-K0" in operator
    assert "cheaper H1-K0" in operator
    assert (
        "real_bost_camera_selected_warm_refinement_v173_result_2026-08-21.md"
        in operator
    )
    assert daily.count("FAIL_CLASSICAL_CONTROL_EXPLAINS_CAMERA_SELECTED_WARM_V173") == 1


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
