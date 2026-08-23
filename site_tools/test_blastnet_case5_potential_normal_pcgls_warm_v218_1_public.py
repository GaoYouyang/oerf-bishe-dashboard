"""Public evidence checks for the independently sealed v218.1 result."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case5_potential_normal_pcgls_warm_v218_1_public_summary.json"
RESULT = ROOT / "docs/blastnet_case5_potential_normal_pcgls_warm_v218_1_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case5_potential_normal_pcgls_warm_v218_1.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v218_1_summary_preserves_primary_failure_and_control_headroom() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_POTENTIAL_NORMAL_PCGLS_WARM_INSUFFICIENT_V218_1"
    assert payload["primary_result"]["matched_cells_passed"] == 0
    assert payload["primary_result"]["matched_geometries_passed"] == 0
    low64 = payload["deterministic_controls"]["low64_pcgls_k11"]
    assert low64["logical_exact_A"] == 12
    assert low64["logical_exact_AT"] == 11
    assert low64["matched_cells_passed"] == 546
    assert low64["matched_geometries_passed"] == 13
    assert payload["low64_k11_call_headroom"]["total_exact_call_reduction_fraction"] == 0.28125


def test_v218_1_independent_boundary_remains_explicit() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    independent = payload["independent_recomputation"]
    assert independent["all_checks_passed"] is True
    assert independent["decision_exact_match"] is True
    assert payload["validator_history"]["first_validator_scientific_arrays_changed"] is False
    assert payload["claims_fixed_false"]["algorithm_breakthrough"] is False
    assert payload["claims_fixed_false"]["resource_speedup"] is False
    assert payload["claims_fixed_false"]["external_generalization"] is False
    assert payload["claims_fixed_false"]["real_bost"] is False


def test_v218_1_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v218.1：" in text
    assert "# v218.1:" in text
    assert "0/546" in text
    assert "12A+11A^T" in text
    assert "28.125%" in text
    assert "algorithm_breakthrough=false" in text


def test_v218_1_figure_is_rendered() -> None:
    assert FIGURE.is_file()
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 650
        assert image.mode == "RGB"


def test_v218_1_current_evidence_and_primary_pages_are_synchronized() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_POTENTIAL_NORMAL_PCGLS_WARM_INSUFFICIENT_V218_1"
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_NORMAL_PCGLS_WARM_V218_1_1"
    )
    assert current["metrics"]["v218_1_low64_k11_matched_cells_passed"] == 546
    assert current["metrics"]["v218_1_low64_k11_total_exact_call_reduction_fraction"] == 0.28125
    assert current["current_decision"]["v218_1_potential_normal_representation_closed"] is True
    assert current["current_decision"]["v218_1_low64_k11_control_headroom"] is True
    assert current["current_decision"]["v218_1_algorithm_breakthrough"] is False
    for page in [ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html"]:
        content = page.read_text(encoding="utf-8")
        assert "v218.1" in content
        assert "12A+11A" in content
        assert "algorithm_breakthrough=false" in content


def test_v218_1_learning_log_records_both_route_actions() -> None:
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v218.1 关闭 potential-normal" in log
    assert "v218.1 closes potential-normal" in log
    assert "Low-64 K11" in log
