"""Public evidence checks for the sealed v222.1 post-open attribution."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_algebraic_nullspace_attribution_v222_1_public_summary.json"
RESULT = ROOT / "docs/blastnet_case2_case5_low64_algebraic_nullspace_attribution_v222_1_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case2_case5_low64_algebraic_nullspace_attribution_v222_1.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v222_boundary_is_preserved_and_v222_1_role_is_post_open() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["v222_boundary"]["scientific_status"] == "INCONCLUSIVE_INVALID_ORTHOGONAL_ROWSPACE_ATTRIBUTION_V222"
    assert payload["v222_boundary"]["maximum_direct_vs_projected_k11_residual_relative_difference"] > 1e-9
    assert payload["v222_boundary"]["relabelled_as_preregistered_pass"] is False
    assert payload["algebraic_attribution"]["deployment_method"] is False
    assert payload["algebraic_attribution"]["fresh_validation"] is False


def test_v222_1_attribution_preserves_case5_and_case2_harm() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    case5 = payload["outcomes"]["case5"]
    case2 = payload["outcomes"]["case2"]
    assert (case5["matched_strict_safe"], case5["complete_rigs_passed"]) == (546, 13)
    assert (case2["matched_strict_safe"], case2["complete_rigs_passed"]) == (518, 0)
    assert case5["matched_strict_safe"] == case5["direct_low64_k11_matched_strict_safe"]
    assert case2["matched_strict_safe"] == case2["direct_low64_k11_matched_strict_safe"]
    adjudication = payload["adjudication"]
    assert adjudication["nullspace_component_required_for_case5_benefit"] is False
    assert adjudication["nullspace_component_explains_case2_harm"] is False
    assert adjudication["spectral_reweighting_attribution_supported"] is True


def test_v222_1_independent_validation_and_claim_limits() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = payload["independent_validation"]
    assert validation["status"] == "PASS_INDEPENDENT_ALGEBRAIC_NULLSPACE_ATTRIBUTION_V222_1"
    assert validation["scientific_decision"] == "POST_OPEN_ROWSPACE_PRESERVES_CASE5_BUT_CASE2_HARM_REMAINS_V222_1"
    assert (validation["checks_passed"], validation["checks_failed"]) == (16, 0)
    assert validation["end_to_end_physics_independence_proven"] is False
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v222_1_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v222.1：" in text
    assert "# v222.1:" in text
    assert "INCONCLUSIVE_INVALID_ORTHOGONAL_ROWSPACE_ATTRIBUTION_V222" in text
    assert "POST_OPEN_ROWSPACE_PRESERVES_CASE5_BUT_CASE2_HARM_REMAINS_V222_1" in text
    assert "546/546" in text and "518/715" in text
    assert "algorithm_breakthrough=false" in text


def test_v222_1_figure_is_rendered() -> None:
    assert FIGURE.is_file()
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 650
        assert image.mode == "RGB"


def test_v222_1_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert (
        current["v222_1_algebraic_nullspace_scientific_decision"]
        == "POST_OPEN_ROWSPACE_PRESERVES_CASE5_BUT_CASE2_HARM_REMAINS_V222_1"
    )
    assert current["current_decision"]["v222_inconclusive_preserved"] is True
    assert current["current_decision"]["v222_1_nullspace_explanation_closed"] is True
    assert current["current_decision"]["v222_1_algorithm_breakthrough"] is False
    for page in [ROOT / "index.html", ROOT / "operator-learning/index.html"]:
        content = page.read_text(encoding="utf-8")
        assert "v222.1" in content
        assert "546/546" in content
        assert "518/715" in content
        assert "algorithm_breakthrough=false" in content
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert "blastnet_case2_case5_low64_algebraic_nullspace_attribution_v222_1_result_2026-08-24.md" in daily
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v222.1 正交去除 null(A)" in log
    assert "v222.1 orthogonal null(A) removal" in log
