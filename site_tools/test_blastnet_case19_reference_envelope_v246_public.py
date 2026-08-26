from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_reference_envelope_v246_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_reference_envelope_v246_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_reference_envelope_v246.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_reference_envelope_v246_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v246_summary_preserves_the_negative_envelope_decision() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    v245 = data["v245_fail_closed_attempt"]
    v246 = data["v246_two_implementation_envelope"]
    blocker = data["blocking_component"]
    assert data["scope"]["cells"] == 429
    assert data["scope"]["new_operator_or_solver_calls"] == 0
    assert v245["independent_output_created"] is False
    assert v245["tolerance_relaxed"] is False
    assert v245["second_attempt_used"] is False
    assert v246["independent_checks_passed"] == 16
    assert v246["independent_checks_total"] == 16
    assert v246["formal_independent_maximum_absolute_difference"] == 0.0
    assert v246["k14_definitely_safe_cells"] == 313
    assert v246["k16_definitely_safe_cells"] == 417
    assert v246["gained_definitely_safe_cells"] == 104
    assert v246["lost_definitely_safe_cells"] == 0
    assert v246["positive_worst_case_gain_components"] == 11
    assert v246["nonpositive_worst_case_gain_components"] == 1
    assert blocker["metric"] == "interior_gradient_relative_l2"
    assert blocker["worst_case_gain"] <= 0.0
    assert data["adjudication"]["fixed_k20_reference_authorized"] is False
    assert data["adjudication"]["fixed_depth_reference_deepening_closed"] is True
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v246_result_is_bilingual_and_keeps_the_claim_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v246：" in text and "# v246:" in text
    for token in ("16/16", "313/429", "417/429", "104", "0.758223", "K20"):
        assert token in text
    assert "post-open" in text
    assert "algorithm_breakthrough=false" in text
    assert "resource_speedup=false" in text


def test_v246_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v246_is_preserved_as_historical_parent_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_CASE19_STRATIFIED_RAY_CORRECTION_FULL_SEQUENCE_V265_1"
    assert current["current_decision"]["v246_independent_validation_passed"] is True
    assert current["current_decision"]["v246_fixed_k20_reference_authorized"] is False
    assert current["current_decision"]["v246_fixed_depth_reference_deepening_closed"] is True
    assert current["current_decision"]["v246_algorithm_breakthrough"] is False
    assert current["metrics"]["v246_gained_definitely_safe_cells"] == 104
    assert current["metrics"]["v246_nonpositive_worst_case_gain_components"] == 1
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_stratified_ray_correction_full_sequence_v265_1.png"
    )
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_reference_envelope_v246" in text
        assert "16/16" in text and "104" in text and "K20" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    focus = FOCUS.read_text(encoding="utf-8")
    assert "v264 Case 19" in focus
    assert "full-sequence gate only" in focus
    assert "历史学习资料版本" in focus
    assert "Archived learning-guide version" in focus
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v246 双实现最坏包络" in log
    assert "fixed-depth reference deepening therefore close" in log


def test_v246_public_artifacts_exclude_private_execution_details() -> None:
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
