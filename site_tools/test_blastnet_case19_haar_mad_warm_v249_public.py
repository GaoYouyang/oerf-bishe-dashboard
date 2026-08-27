from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_haar_mad_warm_v249_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_haar_mad_warm_v249_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_haar_mad_warm_v249.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_haar_mad_warm_v249_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v249_summary_preserves_inconclusive_frame_zero_decision() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    execution = data["execution"]
    agreement = data["formal_independent_agreement"]
    diagnostic = data["diagnostic_only_not_an_admissible_performance_verdict"]
    adjudication = data["adjudication"]
    assert data["scope"]["cells"] == 13
    assert data["scope"]["trainable_parameters"] == 0
    assert data["mechanism"]["primary_logical_calls_A_At"] == [15, 14]
    assert execution["formal_validity_checks_passed"] == 22
    assert execution["independent_checks_passed"] == 33
    assert execution["independent_checks_total"] == 35
    assert execution["numeric_tolerance_relaxed"] is False
    assert agreement["haar_coefficient_relative_maximum"] > agreement[
        "haar_coefficient_relative_limit"
    ]
    assert agreement["haar_coefficient_agreement_passed"] is False
    assert diagnostic["primary_strict_safe_cells"] == 13
    assert diagnostic["approximation_only_control_strict_safe_cells"] == 0
    assert diagnostic["raw_k14_control_strict_safe_cells"] == 7
    assert adjudication["scientific_decision"] == (
        "INCONCLUSIVE_INVALID_CASE19_HAAR_MAD_WARM_FRAME_ZERO_V249"
    )
    assert adjudication["full_sequence_authorized"] is False
    assert adjudication["haar_mad_headroom_established"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v249_result_is_bilingual_and_keeps_claim_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v249：" in text and "# v249:" in text
    for token in ("22/22", "33/35", "1.8209834396e-11", "13/13", "0/13", "15A+14A^T"):
        assert token in text
    assert "完整 429 单元序列" in text
    assert "full 429-cell sequence" in text
    assert "algorithm_breakthrough=false" in text


def test_v249_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v249_is_preserved_as_historical_evidence_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "INCONCLUSIVE_CASE19_LORO_FUSION_REFERENCE_V274_2"
    assert current["current_decision"]["v249_independent_validation_passed"] is False
    assert current["current_decision"]["v249_full_sequence_authorized"] is False
    assert current["current_decision"]["v249_haar_mad_headroom_established"] is False
    assert current["current_decision"]["v249_algorithm_breakthrough"] is False
    assert current["metrics"]["v249_independent_checks_passed"] == 33
    assert current["metrics"]["v249_independent_checks_total"] == 35
    assert current["metrics"]["v249_primary_strict_safe_cells_diagnostic"] == 13
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_loro_fusion_reference_v274_2.png"
    )
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_haar_mad_warm_v249" in text
        assert "33/35" in text and "1.82e-11" in text and "INCONCLUSIVE" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v249 Haar-MAD" in log
    assert "v249 preregisters" in log


def test_v249_public_artifacts_exclude_private_execution_details() -> None:
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
