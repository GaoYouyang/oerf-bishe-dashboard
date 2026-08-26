from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_charbonnier_graph_diffusion_v250_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_charbonnier_graph_diffusion_v250_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_charbonnier_graph_diffusion_v250.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_charbonnier_graph_diffusion_v250_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v250_summary_preserves_control_explained_decision() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    execution = data["execution"]
    results = data["independent_results"]
    adjudication = data["adjudication"]
    assert data["scope"]["cells"] == 13
    assert data["scope"]["trainable_parameters"] == 0
    assert data["mechanism"]["primary_logical_calls_A_At"] == [15, 14]
    assert data["mechanism"]["equal_call_control_logical_calls_A_At"] == [15, 14]
    assert execution["formal_validity_checks_passed"] == 24
    assert execution["independent_checks_passed"] == 38
    assert execution["independent_checks_total"] == 38
    assert results["charbonnier_primary"]["strict_safe_cells"] == 13
    assert results["equal_call_linear_heat_control"]["strict_safe_cells"] == 13
    assert results["raw_k14_control"]["strict_safe_cells"] == 7
    assert results["k16_reference"]["strict_safe_cells"] == 12
    assert adjudication["scientific_decision"] == (
        "PASS_CASE19_FRAME_ZERO_BUT_CHARBONNIER_ADVANTAGE_NOT_ISOLATED_V250"
    )
    assert adjudication["independent_validation_passed"] is True
    assert adjudication["charbonnier_specific_advantage_isolated"] is False
    assert adjudication["equal_call_control_explains_result"] is True
    assert adjudication["full_sequence_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v250_result_is_bilingual_and_keeps_claim_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v250：" in text and "# v250:" in text
    for token in ("24/24", "38/38", "13/13", "7/13", "12/13", "15A+14A^T"):
        assert token in text
    assert "同价线性热扩散" in text
    assert "equal-call linear heat" in text
    assert "完整 429 单元序列" in text
    assert "full 429-cell sequence" in text
    assert "algorithm_breakthrough=false" in text


def test_v250_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v250_remains_preserved_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_CASE19_KRYLOV_COMPLEMENT_HEAT_FRAME_ZERO_V258"
    decision = current["current_decision"]
    assert decision["v250_independent_validation_passed"] is True
    assert decision["v250_equal_call_control_explains_result"] is True
    assert decision["v250_charbonnier_specific_advantage_isolated"] is False
    assert decision["v250_full_sequence_authorized"] is False
    assert decision["v250_algorithm_breakthrough"] is False
    metrics = current["metrics"]
    assert metrics["v250_independent_checks_passed"] == 38
    assert metrics["v250_independent_checks_total"] == 38
    assert metrics["v250_primary_strict_safe_cells"] == 13
    assert metrics["v250_equal_call_control_strict_safe_cells"] == 13
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_charbonnier_graph_diffusion_v250" in text
        assert "38/38" in text and "13/13" in text and "v251" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v250 Charbonnier" in log
    assert "v250 preregisters" in log


def test_v250_public_artifacts_exclude_private_execution_details() -> None:
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
