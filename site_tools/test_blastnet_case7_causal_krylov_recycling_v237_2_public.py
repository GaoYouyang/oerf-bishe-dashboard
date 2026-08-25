from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_causal_krylov_recycling_v237_2_public_summary.json"
RESULT = ROOT / "docs/blastnet_case7_causal_krylov_recycling_v237_2_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case7_causal_krylov_recycling_v237_2.png"
BUILDER = ROOT / "site_tools/build_blastnet_case7_causal_krylov_recycling_v237_2_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v237_2_summary_records_the_validated_negative() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["new_condition_opened"] is False
    assert data["question"]["truth_used_for_cache_or_coefficients"] is False
    primary = data["results"]["causal_fifo16_primary"]
    assert primary["absolute_strict_safe_cells"] == 148
    assert primary["matched_cells"] == 13
    assert primary["matched_frame_zero_cells"] == 13
    assert primary["matched_non_anchor_cells"] == 0
    assert primary["matched_complete_rigs_passed"] == 0
    assert primary["accepted_cache_updates"] == 533
    assert data["independent_validation"]["erratum_checks_passed"] == 26
    assert data["independent_validation"]["erratum_checks_total"] == 26
    assert data["adjudication"]["scientific_decision"] == (
        "FAIL_CASE7_CAUSAL_KRYLOV_RECYCLING_V237"
    )
    assert data["cost_accounting"]["effective_exact_call_reduction_established"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v237_2_result_is_bilingual_and_keeps_the_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v237/v237.2：" in text and "# v237/v237.2:" in text
    for token in ("148/546", "13/546", "0/533", "0/13", "26/26", "88.47%"):
        assert token in text
    assert "post-result" in text and "结果后" in text
    assert "effective exact-call saving" in text and "**not**" in text
    assert "algorithm_breakthrough=false" in text


def test_v237_2_figure_and_source_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 1100


def test_v237_2_is_synchronized_as_the_latest_public_decision() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_CASE7_CAUSAL_KRYLOV_RECYCLING_V237"
    assert current["current_decision"]["v237_final_independent_validation_passed"] is True
    assert current["current_decision"]["v237_algorithm_breakthrough"] is False
    assert current["metrics"]["v237_primary_absolute_safe_cells"] == 148
    assert current["metrics"]["v237_primary_matched_cells"] == 13
    assert current["metrics"]["v237_primary_matched_non_anchor_cells"] == 0
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case7_causal_krylov_recycling_v237_2.png"
    )

    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case7_causal_krylov_recycling_v237_2" in text
        assert "148/546" in text and "0/533" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text

    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert log.index("v237.2 确认") < log.index("v239 排除")
    assert "causal Krylov recycling" in log


def test_v237_2_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
    )
    assert all(token not in text for token in forbidden)
