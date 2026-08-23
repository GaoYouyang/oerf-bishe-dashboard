from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_affine_coordinate_observability_v179_public_summary.json"
RESULT = ROOT / "docs/poolfire_affine_coordinate_observability_v179_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/poolfire_affine_coordinate_observability_v179.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_observability_positive_and_cost_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["formal_status"] == "PASS_FORMAL_POOLFIRE_AFFINE_COORDINATE_OBSERVABILITY_EXECUTION_V179"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_AFFINE_COORDINATE_OBSERVABILITY_V179"
    assert payload["scientific_decision"] == "PASS_AFFINE_MEASUREMENT_INVERSE_HEADROOM_V179"
    assert payload["evaluation"]["stable_affine_rank"] == 1009
    assert payload["evaluation"]["measurement_rank_minimum"] == 1009
    assert payload["five_camera_measurement_pseudoinverse_k0"]["strict_cells_safe"] == 52
    assert payload["five_camera_measurement_pseudoinverse_k1"]["strict_cells_safe"] == 52
    assert payload["five_camera_cheap_controls"]["one_step_coordinate_k0"]["strict_cells_safe"] == 0
    assert payload["five_camera_cheap_controls"]["one_step_coordinate_k1"]["strict_cells_safe"] == 0
    assert payload["independent_recomputation"]["check_count"] == 36
    assert payload["cost_disclosure"]["geometry_cache_forward_equivalents"] == 26260
    assert payload["claim_limits"]["compact_predictor_established"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v179：五相机观测" in text
    assert "# v179: five-camera observations" in text
    assert "PASS_AFFINE_MEASUREMENT_INVERSE_HEADROOM_V179" in text
    assert "1009/1009" in text.replace(",", "")
    assert "52/52" in text
    assert "0/52" in text
    assert "36/36" in text
    assert "26,260" in text
    assert "不是低成本 warm initializer" in text
    assert "not a low-cost warm initializer" in text
    assert "algorithm_breakthrough=false" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_preserves_v179_as_parent_after_v181() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == "FAIL_LOCAL_RAY_COVERAGE_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V211"
    assert payload["metrics"]["v179_measurement_rank_minimum"] == 1009
    assert payload["metrics"]["v179_five_pseudoinverse_k0_strict_safe_count"] == 52
    assert payload["metrics"]["v179_five_coordinate_cgls1_k0_strict_safe_count"] == 0
    assert payload["metrics"]["v179_geometry_cache_forward_equivalents"] == 26260
    assert payload["metrics"]["v179_independent_check_count"] == 36
    assert payload["current_decision"]["v179_affine_measurement_observability_passed"] is True
    assert payload["current_decision"]["v179_compact_predictor_established"] is False
    assert payload["current_decision"]["v179_algorithm_breakthrough"] is False


def test_primary_pages_reference_v179_in_both_languages() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily, home):
        assert "v179" in text
    for text in (operator, daily):
        assert "PASS_AFFINE_MEASUREMENT_INVERSE_HEADROOM_V179" in text
    assert "可辨识" in operator
    assert "observable" in operator
    assert FIGURE.exists()
    assert daily.count('data-date="2026-08-21"') == 1


def test_route_metadata_preserves_v179_after_v181_advance() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    curriculum = (ROOT / "operator-learning/curriculum.js").read_text()
    assert "curriculum.js?v=20260822-v196" in operator
    assert 'version: "2026.08.22-c-v185-potential-affine-capacity"' in curriculum
    assert 'previousVersion: "2026.08.22-c-v184-projection-potential-negative"' in curriculum
    assert 'updated: "2026-08-22"' in curriculum


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "ae829bb5",
        "47a5b648",
        "cb0b304f",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
