"""Public evidence checks for the independently sealed v216 result."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case5_pcgls_reference_warm_readjudication_v216_public_summary.json"
RESULT = ROOT / "docs/blastnet_case5_pcgls_reference_warm_readjudication_v216_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case5_pcgls_reference_warm_readjudication_v216.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v216_summary_preserves_reference_and_negative_decision() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["scientific_decision"] == (
        "FAIL_FIXED_LOW64_PROXY_WARM_START_AGAINST_ADEQUATE_PCGLS_REFERENCE_V216"
    )
    reference = summary["adequate_reference"]
    assert reference["strict_cells_passed"] == 546
    assert reference["complete_geometries_passed"] == 13
    assert reference["adequate"] is True
    k8 = summary["proxy_checkpoints"][-1]
    assert k8["absolute_cells_passed"] == 546
    assert k8["absolute_geometries_passed"] == 13
    assert k8["matched_cells_passed"] == 0
    assert k8["matched_geometries_passed"] == 0


def test_v216_summary_preserves_failure_location_and_independence() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    diagnostic = summary["proxy_k8_matched_diagnostic"]
    assert diagnostic["field_violation_cells"] == 545
    assert diagnostic["full_gradient_violation_cells"] == 546
    assert diagnostic["interior_gradient_violation_cells"] == 23
    assert diagnostic["observation_violation_cells"] == 546
    independent = summary["independent_readjudication"]
    assert independent["checks_passed"] == independent["checks_total"] == 18
    assert independent["decision_exact_match"] is True
    assert independent["maximum_summary_absolute_difference"] < 2e-10
    assert all(value is False for value in summary["claims_fixed_false"].values())


def test_v216_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v216：" in text
    assert "# v216:" in text
    assert "546/546" in text
    assert "0/546" in text
    assert "algorithm_breakthrough=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 600
        extrema = image.convert("RGB").getextrema()
        assert any(low != high for low, high in extrema)


def test_v216_is_preserved_beneath_the_v217_1_current_headline() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-24"
    assert current["v221_low64_exact_rowspace_lift_scientific_decision"] == (
        "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert current["metrics"]["v216_reference_strict_cells_passed"] == 546
    assert current["metrics"]["v216_proxy_k8_absolute_cells_passed"] == 546
    assert current["metrics"]["v216_proxy_k8_matched_cells_passed"] == 0
    assert current["current_decision"]["v216_fixed_low64_proxy_closed"] is True
    assert current["current_decision"]["v216_resource_gate_authorized"] is False


def test_primary_pages_reference_v216_in_both_languages() -> None:
    for relative in (
        "index.html",
        "operator-learning/index.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "blastnet_case5_pcgls_reference_warm_readjudication_v216" in content
        assert "v216" in content
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v216 现在可以下负结论" in log
    assert "v216 can now make a valid negative decision" in log


def test_v216_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "d71c038f",
        "launch_formal",
    )
    for path in (SUMMARY, RESULT):
        content = path.read_text(encoding="utf-8")
        assert all(token not in content for token in forbidden)
