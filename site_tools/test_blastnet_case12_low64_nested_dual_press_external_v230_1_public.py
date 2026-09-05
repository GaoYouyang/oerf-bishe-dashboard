from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case12_low64_nested_dual_press_external_v230_1_public_summary.json"
RESULT = ROOT / "docs/blastnet_case12_low64_nested_dual_press_external_v230_1_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case12_low64_nested_dual_press_external_v230_1.png"


def test_v230_1_public_summary_preserves_reference_first_adjudication() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    numerical = data["numerical_adjudication"]
    reference = data["reference_adequacy"]
    adjudication = data["adjudication"]
    assert numerical["required_numerical_checks_passed"] == 18
    assert numerical["required_science_release_checks_passed"] == 7
    assert numerical["decision_mismatches"] == 0
    assert numerical["residual_vector_difference_is_decision_gate"] is False
    assert reference["status"] == "INCONCLUSIVE_INADEQUATE_CASE12_K16_REFERENCE_V230"
    assert reference["absolute_strict_safe_cells"] == 594
    assert reference["absolute_strict_total_cells"] == 598
    assert reference["complete_rigs_passed"] == 11
    assert len(reference["failed_cells"]) == 4
    assert {cell["rig"] for cell in reference["failed_cells"]} == {0, 12}
    assert {cell["frame"] for cell in reference["failed_cells"]} == {11, 42}
    assert adjudication["dual_press_policy_adjudicated"] is False
    assert adjudication["resource_gate_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v230_1_result_is_bilingual_and_reference_failure_is_precise() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v230.1：" in text and "# v230.1:" in text
    for token in ("594/598", "11/13", "18/18", "7/7", "0.754621"):
        assert token in text
    assert "策略比较的前提" in text
    assert "Because reference adequacy fails before policy adjudication" in text
    assert "algorithm_breakthrough=false" in text


def test_v230_1_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v230_1_historical_surfaces_and_log_are_preserved() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["updated"] == "2026-09-05"
    assert current["v230_scientific_decision"] == "INCONCLUSIVE_INADEQUATE_CASE12_K16_REFERENCE_V230"
    assert current["metrics"]["v230_case12_reference_absolute_strict_safe"] == 594
    assert current["metrics"]["v230_case12_reference_strict_total"] == 598
    assert current["metrics"]["v230_case12_reference_complete_rigs_passed"] == 11
    assert current["current_decision"]["v230_dual_press_policy_adjudicated"] is False
    assert current["current_decision"]["v230_resource_gate_authorized"] is False
    assert current["current_decision"]["v230_algorithm_breakthrough"] is False
    for relative in ("index.html", "operator-learning/index.html"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v230.1" in content
        assert "blastnet_case12_low64_nested_dual_press_external_v230_1_result_2026-08-25.md" in content
        assert "594/598" in content
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert "blastnet_case12_low64_nested_dual_press_external_v230_1_result_2026-08-25.md" in daily
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v230.1 先修正了一个病态数值比较" in log
    assert "v230.1 replaces the ill-conditioned" in log


def test_v230_1_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = ("/Users/", "private_results", "private_worktrees", "source_commit", "sha256", "checkpoint")
    assert all(token not in text for token in forbidden)
