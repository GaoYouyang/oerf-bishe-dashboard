from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_k1_anchor_identifiability_v253_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_k1_anchor_identifiability_v253_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_k1_anchor_identifiability_v253.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_k1_anchor_identifiability_v253_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v253_summary_records_valid_negative_identifiability_result() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["results"]
    validation = data["independent_validation"]
    adjudication = data["adjudication"]
    assert data["scope"]["cells"] == 429
    assert data["scope"]["truth_available_before_anchor_barrier"] is False
    assert data["scope"]["full_traversal_executed"] is False
    assert results["primary"]["safe_rigs"] == 9
    assert results["minimum_norm_control"]["safe_rigs"] == 9
    assert results["minimum_norm_control"]["identical_to_primary"] is True
    assert results["cosine_medoid_control"]["safe_rigs"] == 11
    assert results["fixed_midpoint_diagnostic"]["safe_rigs"] == 13
    assert results["fixed_midpoint_diagnostic"]["deployment_admissible"] is False
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 16
    assert validation["scientific_decision"] == "FAIL_CASE19_K1_RESIDUAL_CONTRACTION_ANCHOR_V253"
    assert adjudication["route_action"] == "CLOSE_K1_RESIDUAL_CONTRACTION_ANCHOR_HYPOTHESIS"
    assert adjudication["full_traversal_mechanism_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v253_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v253：" in text and "# v253:" in text
    for token in ("9/13", "11/13", "13/13", "33A+33A^T", "3.0303%", "16/16"):
        assert token in text
    assert "不是有效减调用或速度结果" in text
    assert "not effective call reduction or a speed result" in text
    assert "algorithm_breakthrough=false" in text


def test_v253_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v253_remains_preserved_as_historical_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_LOMO_DISCREPANCY_SENTINEL_V281"
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert metrics["v253_k1_anchor_safe_rigs"] == 9
    assert metrics["v253_minimum_norm_safe_rigs"] == 9
    assert metrics["v253_cosine_medoid_safe_rigs"] == 11
    assert metrics["v253_fixed_midpoint_diagnostic_safe_rigs"] == 13
    assert decision["v253_independent_validation_passed"] is True
    assert decision["v253_k1_residual_contraction_anchor_closed"] is True
    assert decision["v253_full_traversal_mechanism_authorized"] is False
    assert decision["v253_algorithm_breakthrough"] is False
    daily_text = DAILY.read_text(encoding="utf-8")
    assert "blastnet_case19_k1_anchor_identifiability_v253" in daily_text
    assert "9/13" in daily_text and "11/13" in daily_text and "13/13" in daily_text
    assert "data-i18n-zh" in daily_text and "data-i18n-en" in daily_text
    learning_log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v253" in learning_log and "FAIL_CASE19_K1_RESIDUAL_CONTRACTION_ANCHOR_V253" in learning_log


def test_v253_public_artifacts_exclude_private_execution_details() -> None:
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
