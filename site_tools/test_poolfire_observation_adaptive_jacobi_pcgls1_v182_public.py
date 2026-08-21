from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_observation_adaptive_jacobi_pcgls1_v182_public_summary.json"
RESULT = ROOT / "docs/poolfire_observation_adaptive_jacobi_pcgls1_v182_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/poolfire_observation_adaptive_jacobi_pcgls1_v182.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_jacobi_pcgls1_failure_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["formal_status"] == "PASS_FORMAL_POOLFIRE_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_EXECUTION_V182"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182"
    assert payload["scientific_decision"] == "FAIL_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182"
    assert payload["mechanism"]["affine_coordinate_count"] == 1009
    assert payload["mechanism"]["target_truth_used_for_update"] is False
    assert payload["five_camera_primary_k1"]["strict_cells_safe"] == 0
    assert payload["all_nine_primary_k1"]["strict_cells_safe"] == 0
    assert payload["five_camera_primary_k1"]["global_p90"]["field_relative_l2"] < 0.5
    assert payload["five_camera_primary_k1"]["global_p90"]["gradient_relative_l2"] < 0.75
    assert payload["five_camera_primary_k1"]["global_p90"]["observation_relative_l2"] > 0.2
    assert payload["all_nine_primary_k1"]["global_p90"]["field_relative_l2"] < 0.5
    assert payload["all_nine_primary_k1"]["global_p90"]["gradient_relative_l2"] < 0.75
    assert payload["all_nine_primary_k1"]["global_p90"]["observation_relative_l2"] > 0.2
    assert payload["five_camera_primary_k1"]["global_p90"]["observation_relative_l2"] < payload["five_camera_k0"]["global_p90"]["observation_relative_l2"]
    assert payload["all_nine_primary_k1"]["global_p90"]["observation_relative_l2"] < payload["all_nine_k0"]["global_p90"]["observation_relative_l2"]
    assert payload["independent_recomputation"]["checks_passed"] == 47
    assert payload["claim_limits"]["one_step_jacobi_pcgls1_family_closed"] is True
    assert payload["claim_limits"]["full_c_route_closed"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v182：一步可观测 Jacobi-PCGLS" in text
    assert "# v182: one observable Jacobi-PCGLS" in text
    assert "FAIL_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182" in text
    assert "0/52" in text
    assert "47/47" in text
    assert "INCONCLUSIVE_ENGINEERING_FAILURE_BEFORE_REPORT_SEAL" in text
    assert "不可能性" in text
    assert "not an impossibility result" in text
    assert "algorithm_breakthrough=false" in text
    assert "exact_call_reduction=false" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_preserves_v182_history_and_limitations() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["v182_observation_adaptive_jacobi_pcgls1_scientific_decision"] == "FAIL_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182"
    assert payload["metrics"]["v182_five_primary_k1_strict_safe_count"] == 0
    assert payload["metrics"]["v182_all_nine_primary_k1_strict_safe_count"] == 0
    assert payload["metrics"]["v182_five_primary_k1_observation_p90"] > 0.2
    assert payload["metrics"]["v182_all_nine_primary_k1_observation_p90"] > 0.2
    assert payload["current_decision"]["v182_one_step_jacobi_pcgls1_family_closed"] is True
    assert payload["current_decision"]["v182_full_c_route_closed"] is False
    assert payload["current_decision"]["v182_exact_call_reduction_established"] is False
    assert payload["current_decision"]["v182_gpu_rental_authorized"] is False
    assert payload["current_decision"]["v182_algorithm_breakthrough"] is False


def test_primary_pages_preserve_v182_as_bilingual_parent_evidence() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    learning = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    for text in (operator, daily, learning):
        assert "v182" in text
    assert "FAIL_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182" in daily
    assert "FAIL_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182" in learning
    assert "一步可观测 Jacobi-PCGLS" in operator
    assert "observable Jacobi-PCGLS" in operator
    assert "poolfire_observation_adaptive_jacobi_pcgls1_v182_result_2026-08-21.md" in operator
    assert daily.count('data-date="2026-08-21"') == 1


def test_route_metadata_keeps_v182_in_history_after_v183_advance() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    curriculum = (ROOT / "operator-learning/curriculum.js").read_text()
    assert "curriculum.js?v=20260822-v185" in operator
    assert 'version: "2026.08.22-c-v185-potential-affine-capacity"' in curriculum
    assert 'previousVersion: "2026.08.22-c-v184-projection-potential-negative"' in curriculum
    assert 'version: "2026.08.21-c-v182-observation-adaptive-jacobi-pcgls1-negative"' in curriculum
    assert 'updated: "2026-08-21"' in curriculum


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "c23b528f",
        "f6e54fba",
        "SCIENTIFIC_DECISION_V182.json",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
