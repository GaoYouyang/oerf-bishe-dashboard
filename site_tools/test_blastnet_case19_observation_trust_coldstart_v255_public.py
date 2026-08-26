from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_observation_trust_coldstart_v255_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_observation_trust_coldstart_v255_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_observation_trust_coldstart_v255.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_observation_trust_coldstart_v255_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v255_summary_records_inconclusive_independent_contract() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    diagnostic = data["diagnostic_only"]
    assert data["scope"]["cells"] == 429
    assert data["scope"]["physical_replay_completed"] is True
    assert data["scope"]["truth_available_before_prediction_barrier"] is False
    assert validation["completed"] is True
    assert validation["passed"] is False
    assert validation["checks_passed"] == 26
    assert validation["checks_total"] == 29
    assert len(validation["failed_checks"]) == 3
    assert validation["maximum_alpha_absolute_difference"] > validation["alpha_absolute_tolerance"]
    assert validation["maximum_residual_relative_difference"] > validation["residual_relative_tolerance"]
    assert validation["maximum_metric_absolute_difference"] > validation["metric_absolute_tolerance"]
    assert diagnostic["admissible_as_scientific_performance"] is False
    assert diagnostic["primary"]["absolute_safe_cells"] == 429
    assert diagnostic["primary"]["matched_safe_cells"] == 429
    assert data["adjudication"]["scientific_pass_claimed"] is False
    assert data["adjudication"]["scientific_fail_claimed"] is False
    assert data["adjudication"]["numeric_tolerance_relaxed"] is False
    assert data["adjudication"]["resource_gate_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v255_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v255：" in text and "# v255:" in text
    for token in ("26/29", "6.73e-9", "7.51e-9", "1.02e-8", "8.8068%"):
        assert token in text
    assert "只能作诊断" in text
    assert "diagnostic only" in text
    assert "不是有效 exact-call 减少" in text
    assert "not effective exact-call reduction" in text
    assert "algorithm_breakthrough=false" in text


def test_v255_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v255_is_preserved_as_bilingual_historical_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert metrics["v255_independent_checks_passed"] == 26
    assert metrics["v255_independent_checks_total"] == 29
    assert metrics["v255_failed_numeric_checks"] == 3
    assert decision["v255_independent_validation_passed"] is False
    assert decision["v255_scientific_result_inconclusive"] is True
    assert decision["v255_discrete_diagnostic_admissible"] is False
    assert decision["v255_effective_exact_call_reduction_established"] is False
    assert decision["v255_algorithm_breakthrough"] is False
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_observation_trust_coldstart_v255" in text
        assert "v255" in text and "INCONCLUSIVE" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert "v255" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v255_public_artifacts_exclude_private_execution_details() -> None:
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
