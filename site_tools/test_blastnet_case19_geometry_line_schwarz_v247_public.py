from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_geometry_line_schwarz_v247_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_geometry_line_schwarz_v247_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_geometry_line_schwarz_v247.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_geometry_line_schwarz_v247_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v247_summary_preserves_the_inconclusive_decision() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    execution = data["execution"]
    agreement = data["formal_independent_agreement"]
    adjudication = data["adjudication"]
    assert data["scope"]["cells"] == 429
    assert data["scope"]["trainable_parameters"] == 0
    assert execution["formal_validity_checks_passed"] == 21
    assert execution["independent_checks_passed"] == 25
    assert execution["independent_checks_total"] == 27
    assert execution["rerun_used"] is False
    assert execution["numeric_tolerance_relaxed"] is False
    assert agreement["discrete_summary_flags_equal"] is True
    assert agreement["residual_relative_maximum"] > agreement["residual_relative_limit"]
    assert adjudication["scientific_decision"] == "INCONCLUSIVE_INVALID_CASE19_GEOMETRY_LINE_SCHWARZ_V247"
    assert adjudication["line_schwarz_authorized"] is False
    assert adjudication["mechanism_operationally_retired"] is True
    assert adjudication["mathematical_impossibility_proven"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v247_result_is_bilingual_and_keeps_the_claim_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v247：" in text and "# v247:" in text
    for token in ("21/21", "25/27", "3.24e-8", "1e-8", "2.38e-9", "0/13"):
        assert token in text
    assert "不重跑、不改目录、不放宽容差" in text
    assert "not rerun, redirected, tolerance-relaxed" in text
    assert "algorithm_breakthrough=false" in text


def test_v247_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v247_is_preserved_as_historical_evidence_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_CASE19_STRATIFIED_RAY_CORRECTION_FULL_SEQUENCE_V265_1"
    assert current["current_decision"]["v247_independent_validation_passed"] is False
    assert current["current_decision"]["v247_line_schwarz_authorized"] is False
    assert current["current_decision"]["v247_mechanism_operationally_retired"] is True
    assert current["current_decision"]["v247_algorithm_breakthrough"] is False
    assert current["metrics"]["v247_independent_checks_passed"] == 25
    assert current["metrics"]["v247_independent_checks_total"] == 27
    assert current["metrics"]["v247_residual_relative_maximum"] > current["metrics"]["v247_residual_relative_limit"]
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_stratified_ray_correction_full_sequence_v265_1.png"
    )
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_geometry_line_schwarz_v247" in text
        assert "25/27" in text and "3.2449e-8" in text and "INCONCLUSIVE" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v247 精确坐标线 Schwarz" in log
    assert "exact coordinate-line Schwarz" in log


def test_v247_public_artifacts_exclude_private_execution_details() -> None:
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
