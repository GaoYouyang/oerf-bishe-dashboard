from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case19_cold_start_linear_hybrid_attribution_v251_public_summary.json"
)
RESULT = (
    ROOT
    / "docs/blastnet_case19_cold_start_linear_hybrid_attribution_v251_result_2026-08-26.md"
)
FIGURE = (
    ROOT
    / "assets/figures/blastnet_case19_cold_start_linear_hybrid_attribution_v251.png"
)
BUILDER = (
    ROOT
    / "site_tools/build_blastnet_case19_cold_start_linear_hybrid_attribution_v251_figure.py"
)
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v251_summary_preserves_matched_accuracy_failure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["independent_results"]
    adjudication = data["adjudication"]
    assert data["scope"]["cells"] == 429
    assert data["scope"]["new_operator_or_solver_calls_A_At"] == [0, 0]
    assert data["integrity_boundary"]["outcome_observed_before_formalization"] is True
    assert data["integrity_boundary"]["result_blind_preregistration_claimed"] is False
    assert data["composition"]["hybrid_sequence_calls_A_At_per_rig"] == [495, 462]
    assert data["composition"]["k16_sequence_calls_A_At_per_rig"] == [528, 528]
    assert data["execution"]["independent_checks_passed"] == 17
    assert data["execution"]["independent_checks_total"] == 17
    assert data["execution"]["maximum_formal_independent_absolute_difference"] == 0
    assert results["hybrid_robust_absolute_cells"] == 429
    assert results["hybrid_robust_matched_cells"] == 416
    assert results["hybrid_robust_absolute_complete_rigs"] == 13
    assert results["hybrid_robust_matched_complete_rigs"] == 0
    assert results["matched_failure_counts_by_metric"] == [0, 0, 0, 13]
    assert results["matched_failure_counts_by_frame"][0] == 13
    assert sum(results["matched_failure_counts_by_frame"][1:]) == 0
    assert adjudication["scientific_decision"] == (
        "FAIL_CASE19_COLD_START_LINEAR_HYBRID_MATCHED_ACCURACY_V251"
    )
    assert adjudication["independent_validation_passed"] is True
    assert adjudication["absolute_accuracy_passed"] is True
    assert adjudication["matched_accuracy_passed"] is False
    assert adjudication["nominal_call_arithmetic_is_effective_reduction"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v251_result_is_bilingual_and_discloses_retrospective_scope() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v251：" in text and "# v251:" in text
    for token in ("17/17", "429/429", "416/429", "0/13", "9.375%"):
        assert token in text
    assert "不是结果盲的前瞻试验" in text
    assert "not a result-blind prospective experiment" in text
    assert "名义调用差" in text
    assert "nominal call difference" in text
    assert "algorithm_breakthrough=false" in text


def test_v251_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v251_is_latest_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == (
        "FAIL_CASE19_COLD_START_LINEAR_HYBRID_MATCHED_ACCURACY_V251"
    )
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert metrics["v251_independent_checks_passed"] == 17
    assert metrics["v251_hybrid_robust_absolute_cells"] == 429
    assert metrics["v251_hybrid_robust_matched_cells"] == 416
    assert metrics["v251_hybrid_robust_matched_complete_rigs"] == 0
    assert decision["v251_independent_validation_passed"] is True
    assert decision["v251_matched_accuracy_passed"] is False
    assert decision["v251_effective_exact_call_reduction_established"] is False
    assert decision["v251_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_cold_start_linear_hybrid_attribution_v251.png"
    )
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_cold_start_linear_hybrid_attribution_v251" in text
        assert "429/429" in text and "416/429" in text and "0/13" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v251" in log
    assert "post-open retrospective" in log


def test_v251_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)
