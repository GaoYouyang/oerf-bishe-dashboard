from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/poolfire_train_field_affine_span_capacity_v178_public_summary.json"
)
RESULT = (
    ROOT / "docs/poolfire_train_field_affine_span_capacity_v178_result_2026-08-21.md"
)
FIGURE = ROOT / "assets/figures/poolfire_train_field_affine_span_capacity_v178.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_capacity_positive_and_rank_limitation() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert (
        payload["formal_status"]
        == "PASS_FORMAL_POOLFIRE_TRAIN_FIELD_AFFINE_SPAN_EXECUTION_V178"
    )
    assert (
        payload["independent_status"]
        == "PASS_INDEPENDENT_RECOMPUTATION_TRAIN_FIELD_AFFINE_SPAN_V178"
    )
    assert (
        payload["scientific_decision"] == "PASS_TRAIN_FIELD_AFFINE_SPAN_HEADROOM_V178"
    )
    assert payload["evaluation"]["fit_fields"] == 1010
    assert payload["evaluation"]["stable_affine_rank"] == 1009
    assert payload["five_camera_affine_projection_k0"]["strict_cells_safe"] == 52
    assert payload["five_camera_affine_projection_k1"]["strict_cells_safe"] == 52
    assert payload["five_camera_static_mean_controls"]["k0_strict_cells_safe"] == 0
    assert payload["five_camera_static_mean_controls"]["k1_strict_cells_safe"] == 0
    assert payload["independent_recomputation"]["check_count"] == 26
    assert payload["claim_limits"]["compact_representation_established"] is False
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v178：训练场仿射张成空间" in text
    assert "# v178: the training-field affine span" in text
    assert "PASS_TRAIN_FIELD_AFFINE_SPAN_HEADROOM_V178" in text
    assert "1009/1010" in text
    assert "52/52" in text
    assert "0/52" in text
    assert "26/26" in text
    assert "不是紧凑表示" in text
    assert "not a compact representation" in text
    assert "algorithm_breakthrough=false" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_preserves_v178_as_parent_and_advances_to_v181() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == "PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    assert payload["v178_affine_span_scientific_decision"] == "PASS_TRAIN_FIELD_AFFINE_SPAN_HEADROOM_V178"
    assert payload["metrics"]["v178_fit_field_count"] == 1010
    assert payload["metrics"]["v178_stable_affine_rank"] == 1009
    assert payload["metrics"]["v178_five_projection_k1_strict_safe_count"] == 52
    assert payload["metrics"]["v178_five_mean_k1_strict_safe_count"] == 0
    assert payload["metrics"]["v178_independent_check_count"] == 26
    assert payload["current_decision"]["v178_linear_field_span_capacity_passed"] is True
    assert (
        payload["current_decision"]["v178_compact_representation_established"] is False
    )
    assert payload["current_decision"]["v178_predictor_training_authorized"] is False
    assert payload["current_decision"]["v178_algorithm_breakthrough"] is False


def test_primary_pages_reference_v178_in_both_languages() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily):
        assert "v178" in text
    for text in (operator, daily, home):
        assert "v181" in text
    assert "PASS_TRAIN_FIELD_AFFINE_SPAN_HEADROOM_V178" in operator
    assert "PASS_TRAIN_FIELD_AFFINE_SPAN_HEADROOM_V178" in daily
    assert "近满秩" in operator
    assert "near-full-rank" in operator
    assert FIGURE.exists()
    assert daily.count('data-date="2026-08-21"') == 1


def test_route_metadata_preserves_v178_as_previous_release() -> None:
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
        "3494770f",
        "20972d29",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
