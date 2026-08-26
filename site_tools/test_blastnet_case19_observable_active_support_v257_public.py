from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_observable_active_support_v257_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_observable_active_support_v257_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_observable_active_support_v257.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_observable_active_support_v257_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v257_summary_records_the_single_failed_independent_gate() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    interpretation = data["performance_interpretation"]
    assert data["scope"]["cells"] == 13
    assert data["scope"]["full_sequence_run"] is False
    assert data["scope"]["truth_available_before_prediction_barrier"] is False
    assert validation["completed"] is True
    assert validation["passed"] is False
    assert validation["checks_passed"] == 23
    assert validation["checks_total"] == 24
    assert validation["failed_checks"] == ["observation_normalized_residuals_agree"]
    assert validation["maximum_observation_normalized_residual_difference"] > validation[
        "observation_normalized_residual_tolerance"
    ]
    assert 1.54 < validation["residual_limit_ratio"] < 1.55
    assert validation["support_masks_agree_exactly"] is True
    assert validation["selected_counts_agree_exactly"] is True
    assert validation["maximum_final_field_relative_difference"] <= validation["field_relative_tolerance"]
    assert interpretation["formal_scientific_arrays_admissible"] is False
    assert interpretation["independent_scientific_arrays_admissible"] is False
    assert interpretation["discrete_counts_published_as_performance"] is False
    assert data["adjudication"]["scientific_pass_claimed"] is False
    assert data["adjudication"]["scientific_fail_claimed"] is False
    assert data["adjudication"]["numeric_tolerance_relaxed"] is False
    assert data["adjudication"]["full_sequence_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v257_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v257：" in text and "# v257:" in text
    for token in ("23/24", "3.08360e-8", "2.00000e-8", "1.54180", "15.625%"):
        assert token in text
    assert "不可用于性能解释" in text
    assert "not effective exact-call reduction" in text
    assert "algorithm_breakthrough=false" in text


def test_v257_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v257_is_latest_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert current["scientific_status"] == "INCONCLUSIVE_INVALID_CASE19_OBSERVABLE_ACTIVE_SUPPORT_FRAME_ZERO_V257"
    assert metrics["v257_independent_checks_passed"] == 23
    assert metrics["v257_independent_checks_total"] == 24
    assert metrics["v257_failed_numeric_checks"] == 1
    assert decision["v257_independent_validation_passed"] is False
    assert decision["v257_scientific_result_inconclusive"] is True
    assert decision["v257_scientific_arrays_admissible"] is False
    assert decision["v257_full_sequence_authorized"] is False
    assert decision["v257_effective_exact_call_reduction_established"] is False
    assert decision["v257_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith("blastnet_case19_observable_active_support_v257.png")
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_observable_active_support_v257" in text
        assert "23/24" in text and "INCONCLUSIVE" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert "v257" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v257_public_artifacts_exclude_private_execution_details() -> None:
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
