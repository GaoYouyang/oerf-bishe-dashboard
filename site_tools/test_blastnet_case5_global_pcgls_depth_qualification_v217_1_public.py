"""Public evidence checks for the independently sealed v217.1 result."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case5_global_pcgls_depth_qualification_v217_1_public_summary.json"
RESULT = ROOT / "docs/blastnet_case5_global_pcgls_depth_qualification_v217_1_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case5_global_pcgls_depth_qualification_v217_1.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v217_1_summary_preserves_depth_qualification() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["scientific_decision"] == (
        "PASS_K16_REMAINS_MINIMAL_ADEQUATE_GLOBAL_PCGLS_DEPTH_V217_1"
    )
    rows = summary["depth_qualification"]
    assert [row["depth"] for row in rows] == list(range(8, 17))
    assert rows[-2]["absolute_cells_passed"] == 544
    assert rows[-2]["absolute_geometries_passed"] == 11
    assert rows[-2]["matched_cells_passed"] == 0
    assert rows[-1]["absolute_cells_passed"] == 546
    assert rows[-1]["matched_cells_passed"] == 546
    assert summary["selected_reference"]["depth"] == 16


def test_v217_1_summary_preserves_invalid_history_and_independence() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    invalid = summary["invalid_execution_history"]
    assert invalid["scientific_status"] == (
        "INCONCLUSIVE_INVALID_GLOBAL_PCGLS_DEPTH_QUALIFICATION_V217"
    )
    repair = summary["execution_repair"]
    assert repair["thresholds_changed"] is False
    assert repair["solver_changed"] is False
    assert repair["scientific_arrays_reproduced"] is True
    independent = summary["independent_recomputation"]
    assert independent["checks_passed"] == independent["checks_total"] == 14
    assert independent["decision_exact_match"] is True
    assert independent["maximum_camera_permutation_k16_field_relative_difference"] == 0
    assert all(value is False for value in summary["claims_fixed_false"].values())


def test_v217_1_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v217.1：" in text
    assert "# v217.1:" in text
    assert "544/546" in text
    assert "546/546" in text
    assert "algorithm_breakthrough=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 600
        extrema = image.convert("RGB").getextrema()
        assert any(low != high for low, high in extrema)


def test_v217_1_history_is_preserved_beneath_v220_2() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-24"
    assert current["scientific_status"] == "INCONCLUSIVE_INVALID_OBSERVABLE_FALLBACK_V220_2"
    assert current["v218_1_scientific_decision"] == "FAIL_POTENTIAL_NORMAL_PCGLS_WARM_INSUFFICIENT_V218_1"
    assert current["metrics"]["v217_1_k15_absolute_cells_passed"] == 544
    assert current["metrics"]["v217_1_k15_matched_cells_passed"] == 0
    assert current["metrics"]["v217_1_k16_matched_cells_passed"] == 546
    assert current["current_decision"]["v217_1_k16_is_minimal_global_reference"] is True
    assert current["current_decision"]["v217_1_exact_call_reduction_established"] is False


def test_primary_pages_reference_v217_1_in_both_languages() -> None:
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "blastnet_case5_global_pcgls_depth_qualification_v217_1" in content
        assert "v217.1" in content
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v217.1 定下最低可靠的全局 PCGLS 深度" in log
    assert "v217.1 qualifies the lowest reliable global PCGLS depth" in log


def test_v217_1_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "cc0b0a59",
        "3467b7b9",
        "launch_formal",
    )
    for path in (SUMMARY, RESULT):
        content = path.read_text(encoding="utf-8")
        assert all(token not in content for token in forbidden)
