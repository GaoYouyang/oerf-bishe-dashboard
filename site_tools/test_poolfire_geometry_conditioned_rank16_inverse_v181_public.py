from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_geometry_conditioned_rank16_inverse_v181_public_summary.json"
RESULT = ROOT / "docs/poolfire_geometry_conditioned_rank16_inverse_v181_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/poolfire_geometry_conditioned_rank16_inverse_v181.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_geometry_rank16_failure_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["formal_status"] == "PASS_FORMAL_POOLFIRE_GEOMETRY_CONDITIONED_RANK16_EXECUTION_V181"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_GEOMETRY_CONDITIONED_RANK16_V181"
    assert payload["scientific_decision"] == "FAIL_GEOMETRY_CONDITIONED_RANK16_INVERSE_V181"
    assert payload["evaluation"]["stable_affine_rank"] == 1009
    assert payload["evaluation"]["spectral_correction_rank"] == 16
    assert payload["five_camera_primary_k1"]["strict_cells_safe"] == 0
    assert payload["all_nine_primary_k1"]["strict_cells_safe"] == 0
    assert payload["five_camera_primary_k1"]["global_p90"]["field_relative_l2"] > 0.5
    assert payload["five_camera_primary_k1"]["global_p90"]["gradient_relative_l2"] > 0.75
    assert payload["five_camera_primary_k1"]["global_p90"]["observation_relative_l2"] > 0.2
    assert payload["all_nine_primary_k1"]["global_p90"]["field_relative_l2"] < 0.5
    assert payload["all_nine_primary_k1"]["global_p90"]["gradient_relative_l2"] < 0.75
    assert payload["all_nine_primary_k1"]["global_p90"]["observation_relative_l2"] > 0.2
    assert payload["mechanism_diagnostics"]["relative_p90_reduction"] < 0.01
    assert payload["independent_recomputation"]["checks_passed"] == 48
    assert payload["claim_limits"]["fixed_geometry_rank16_family_closed"] is True
    assert payload["claim_limits"]["all_compact_mechanisms_ruled_out"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v181：显式几何条件化" in text
    assert "# v181: explicit geometry conditioning" in text
    assert "FAIL_GEOMETRY_CONDITIONED_RANK16_INVERSE_V181" in text
    assert "0/52" in text
    assert "48/48" in text
    assert "0.71%" in text
    assert "不可能性" in text
    assert "not an impossibility result" in text
    assert "algorithm_breakthrough=false" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_preserves_v181_and_points_to_v182() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == "FAIL_OBSERVATION_BLOCK_GALERKIN_V183"
    assert payload["metrics"]["v181_five_primary_k1_strict_safe_count"] == 0
    assert payload["metrics"]["v181_all_nine_primary_k1_strict_safe_count"] == 0
    assert payload["metrics"]["v181_rank16_inverse_residual_p90"] > 1.0
    assert payload["current_decision"]["v181_fixed_geometry_rank16_family_closed"] is True
    assert payload["current_decision"]["v181_all_compact_mechanisms_ruled_out"] is False
    assert payload["current_decision"]["v181_gpu_rental_authorized"] is False
    assert payload["current_decision"]["v181_algorithm_breakthrough"] is False


def test_primary_pages_reference_v181_in_both_languages() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    learning = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    for text in (operator, daily, home, learning):
        assert "v181" in text
        assert "FAIL_GEOMETRY_CONDITIONED_RANK16_INVERSE_V181" in text
    assert "显式几何条件" in operator
    assert "explicit geometry conditioning" in operator
    assert '<span class="evidence-state fail" data-i18n-zh="几何条件 rank-16 负结果' in operator
    assert RESULT.is_file()
    assert daily.count('data-date="2026-08-21"') == 1


def test_route_metadata_and_cachebuster_advance_to_v181() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    curriculum = (ROOT / "operator-learning/curriculum.js").read_text()
    assert "curriculum.js?v=20260821-v183" in operator
    assert 'version: "2026.08.21-c-v183-observation-block-galerkin-negative"' in curriculum
    assert 'updated: "2026-08-21"' in curriculum


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = ["/Users/", "private_results", "private_worktrees"]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
