from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_low64_k11_external_v235_public_summary.json"
RESULT = ROOT / "docs/blastnet_case7_low64_k11_external_v235_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case7_low64_k11_external_v235.png"


def test_v235_summary_records_the_prospective_failure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["condition_was_unopened_before_release"] is True
    assert data["scope"]["result_dependent_replacement_used"] is False
    assert data["execution"]["formal_cells_completed"] == 546
    assert data["execution"]["independent_cells_recomputed"] == 546
    assert data["execution"]["first_independent_checks_true"] == 26
    assert data["execution"]["first_independent_checks_false"] == 2
    assert data["execution"]["validation_erratum_expected_polarity_checks_satisfied"] == 17
    assert data["absolute_accuracy"]["fixed_low64_geometry_jacobi_pcgls_k11_primary"] == {
        "strict_safe_cells": 546,
        "complete_rigs_passed": 13,
    }
    assert data["matched_to_k16"]["primary_strict_safe_cells"] == 330
    assert data["matched_to_k16"]["primary_complete_rigs_passed"] == 0
    assert data["matched_to_k16"]["equal_or_cheaper_control_passed"] is False
    assert data["adjudication"]["scientific_decision"] == "FAIL_CASE7_LOW64_K11_PROSPECTIVE_CONFIRMATION_V235"
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v235_result_is_bilingual_and_keeps_the_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v235/v235.1：" in text and "# v235/v235.1:" in text
    for token in ("546/546", "330/546", "0/13", "216", "28.125%", "2.18036e-16"):
        assert token in text
    assert "第一次验证记录" in text
    assert "first validation record" in text
    assert "algorithm_breakthrough=false" in text


def test_v235_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v235_historical_surfaces_and_log_remain_synchronized() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["scientific_status"] == "INCONCLUSIVE_CASE19_LORO_FUSION_REFERENCE_V274_2"
    assert current["v235_scientific_decision"] == "FAIL_CASE7_LOW64_K11_PROSPECTIVE_CONFIRMATION_V235"
    assert current["metrics"]["v235_primary_absolute_safe_cells"] == 546
    assert current["metrics"]["v235_primary_matched_safe_cells"] == 330
    assert current["metrics"]["v235_primary_matched_complete_rigs"] == 0
    assert current["current_decision"]["v235_fixed_direct_low64_k11_route_closed"] is True
    assert current["current_decision"]["v235_resource_gate_authorized"] is False
    assert current["current_decision"]["v244_2_fixed_confirmation_route_closed"] is True
    assert "case19_loro_fusion_reference_v274_2" in (
        current["public_evidence"]["result"]
    )
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v235" in content
        assert "blastnet_case7_low64_k11_external_v235.png" in content
        assert "330/546" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v235/v235.1" in log
    assert "none of the 13 rigs" in log


def test_v235_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "c4f0e39e",
        "b5a96c8b",
    )
    assert all(token not in text for token in forbidden)
