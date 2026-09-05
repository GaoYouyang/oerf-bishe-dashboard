from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_galerkin_pyramid_v256_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_galerkin_pyramid_v256_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_galerkin_pyramid_v256.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_galerkin_pyramid_v256_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v256_summary_records_the_single_failed_independent_gate() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    diagnostic = data["diagnostic_only"]
    assert data["scope"]["cells"] == 13
    assert data["scope"]["full_sequence_run"] is False
    assert data["scope"]["truth_available_before_prediction_barrier"] is False
    assert validation["completed"] is True
    assert validation["passed"] is False
    assert validation["checks_passed"] == 19
    assert validation["checks_total"] == 20
    assert validation["failed_checks"] == ["residuals_agree"]
    assert validation["maximum_residual_relative_difference"] > validation["residual_relative_tolerance"]
    assert validation["residual_limit_ratio"] > 2.95
    assert validation["maximum_final_field_relative_difference"] <= validation["field_relative_tolerance"]
    assert diagnostic["admissible_as_scientific_performance"] is False
    assert diagnostic["primary"]["absolute_safe_cells"] == 13
    assert diagnostic["primary"]["matched_safe_cells"] == 13
    assert data["adjudication"]["scientific_pass_claimed"] is False
    assert data["adjudication"]["scientific_fail_claimed"] is False
    assert data["adjudication"]["numeric_tolerance_relaxed"] is False
    assert data["adjudication"]["full_sequence_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v256_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v256：" in text and "# v256:" in text
    for token in ("19/20", "5.91005e-7", "2.00000e-7", "2.95502", "9.375%"):
        assert token in text
    assert "13/13 只能作诊断" in text
    assert "13/13 count is diagnostic only" in text
    assert "不是有效 exact-call 减少" in text
    assert "not effective exact-call reduction" in text
    assert "algorithm_breakthrough=false" in text


def test_v256_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v256_remains_historical_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert current["scientific_status"] == "PASS_INDEPENDENT_LINEAR_SOURCE_BUDGET_V280"
    assert metrics["v256_independent_checks_passed"] == 19
    assert metrics["v256_independent_checks_total"] == 20
    assert metrics["v256_failed_numeric_checks"] == 1
    assert decision["v256_independent_validation_passed"] is False
    assert decision["v256_scientific_result_inconclusive"] is True
    assert decision["v256_discrete_diagnostic_admissible"] is False
    assert decision["v256_full_sequence_authorized"] is False
    assert decision["v256_effective_exact_call_reduction_established"] is False
    assert decision["v256_algorithm_breakthrough"] is False
    for page in (FOCUS, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_galerkin_pyramid_v256" in text
        assert "19/20" in text and "INCONCLUSIVE" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert "v256" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v256_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)
