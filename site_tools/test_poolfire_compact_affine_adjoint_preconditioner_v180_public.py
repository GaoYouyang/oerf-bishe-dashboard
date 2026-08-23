from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_compact_affine_adjoint_preconditioner_v180_public_summary.json"
RESULT = ROOT / "docs/poolfire_compact_affine_adjoint_preconditioner_v180_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/poolfire_compact_affine_adjoint_preconditioner_v180.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_failure_and_parent_observability_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["formal_status"] == "PASS_FORMAL_POOLFIRE_COMPACT_AFFINE_ADJOINT_EXECUTION_V180"
    assert (
        payload["original_independent_status"]
        == "INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_COMPACT_AFFINE_ADJOINT_V180"
    )
    assert payload["zero_mean_audit_status"] == "PASS_ZERO_MEAN_COMPARISON_AUDIT_V180_1"
    assert payload["scientific_decision"] == "FAIL_SHARED_COMPACT_ADJOINT_PRECONDITIONER_V180"
    assert payload["evaluation"]["stable_affine_rank"] == 1009
    assert payload["evaluation"]["primary_residual_rank"] == 16
    assert payload["five_camera_primary_k1"]["strict_cells_safe"] == 4
    assert payload["all_nine_primary_k1"]["strict_cells_safe"] == 7
    assert payload["five_camera_primary_k1"]["frame_strata_passed"] == 0
    assert payload["all_nine_primary_k1"]["frame_strata_passed"] == 0
    assert payload["five_camera_primary_k1"]["global_p90"]["field_relative_l2"] < 0.5
    assert payload["five_camera_primary_k1"]["global_p90"]["gradient_relative_l2"] < 0.75
    assert payload["five_camera_primary_k1"]["global_p90"]["observation_relative_l2"] > 0.2
    assert payload["all_nine_primary_k1"]["global_p90"]["observation_relative_l2"] > 0.2
    assert payload["independent_recomputation"]["corrected_zero_mean_audit_checks_passed"] == 24
    assert payload["independent_recomputation"]["original_inconclusive_preserved"] is True
    assert payload["independent_recomputation"]["formal_or_validator_rerun"] is False
    assert payload["claim_limits"]["shared_linear_family_closed"] is True
    assert payload["claim_limits"]["all_compact_mechanisms_ruled_out"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v180：精确逆可观测" in text
    assert "# v180: the exact inverse is observable" in text
    assert "FAIL_SHARED_COMPACT_ADJOINT_PRECONDITIONER_V180" in text
    assert "4/52" in text
    assert "7/52" in text
    assert "0/4" in text
    assert "24/24" in text
    assert "observation" in text
    assert "不可能" not in text
    assert "not an impossibility result" in text
    assert "algorithm_breakthrough=false" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_preserves_v180_after_v181_advance() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == "PARTIAL_OVERLAPPING_GEOMETRY_ONLY_OBSERVABILITY_EVIDENCE_V210"
    assert payload["metrics"]["v180_five_primary_k1_strict_safe_count"] == 4
    assert payload["metrics"]["v180_all_nine_primary_k1_strict_safe_count"] == 7
    assert payload["metrics"]["v180_five_primary_k1_observation_p90"] > 0.2
    assert payload["metrics"]["v180_all_nine_primary_k1_observation_p90"] > 0.2
    assert payload["metrics"]["v180_zero_mean_audit_check_count"] == 24
    assert payload["current_decision"]["v180_shared_linear_family_closed"] is True
    assert payload["current_decision"]["v180_all_compact_mechanisms_ruled_out"] is False
    assert payload["current_decision"]["v180_gpu_rental_authorized"] is False
    assert payload["current_decision"]["v180_algorithm_breakthrough"] is False


def test_primary_pages_reference_v180_in_both_languages() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily):
        assert "v180" in text
        assert "FAIL_SHARED_COMPACT_ADJOINT_PRECONDITIONER_V180" in text
    for text in (operator, daily, home):
        assert "v181" in text
    assert "共享紧凑线性" in operator
    assert "shared compact linear" in operator
    assert FIGURE.exists()
    assert daily.count('data-date="2026-08-21"') == 1


def test_route_metadata_and_cachebuster_advance_to_v181() -> None:
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
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
