from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_causal_reachable_span_capacity_v240_public_summary.json"
RESULT = ROOT / "docs/blastnet_case7_causal_reachable_span_capacity_v240_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case7_causal_reachable_span_capacity_v240.png"
BUILDER = ROOT / "site_tools/build_blastnet_case7_causal_reachable_span_capacity_v240_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v240_summary_records_the_validated_span_negative() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["new_condition_opened"] is False
    assert data["question"]["metrics_optimized_separately"] is True
    assert data["question"]["joint_feasibility_tested"] is False
    assert data["results"]["necessary_safe_later_cells"] == 0
    assert data["results"]["later_cells_total"] == 533
    assert data["results"]["necessary_complete_rigs_passed"] == 0
    assert data["results"]["metric_cell_failures"]["observation"] == 533
    assert data["results"]["design_rank"] == 17
    assert data["independent_validation"]["checks_passed"] == 20
    assert data["independent_validation"]["checks_total"] == 20
    assert data["adjudication"]["scientific_decision"] == (
        "FAIL_CASE7_CAUSAL_REACHABLE_SPAN_NECESSARY_CAPACITY_V240"
    )
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v240_result_is_bilingual_and_keeps_the_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v240：" in text and "# v240:" in text
    for token in ("0/533", "0/13", "377/533", "251/533", "309/533", "533/533", "20/20"):
        assert token in text
    assert "different coefficient vectors" in text and "四组不同系数" in text
    assert "does not close the full C route" in text and "不关闭完整 C 路线" in text
    assert "algorithm_breakthrough=false" in text


def test_v240_figure_and_source_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 1100


def test_v240_remains_synchronized_as_parent_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["v240_scientific_decision"] == "FAIL_CASE7_CAUSAL_REACHABLE_SPAN_NECESSARY_CAPACITY_V240"
    assert current["current_decision"]["v240_independent_validation_passed"] is True
    assert current["current_decision"]["v240_reachable_span_capacity_passed"] is False
    assert current["current_decision"]["v240_algorithm_breakthrough"] is False
    assert current["metrics"]["v240_later_cells_necessary_safe"] == 0
    assert current["metrics"]["v240_metric_observation_failures"] == 533

    for page in (FOCUS, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case7_causal_reachable_span_capacity_v240" in text
        assert "0/533" in text and "533/533" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text

    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v240 冻结因果可达空间容量" in log
    assert "causal reachable span" in log


def test_v240_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "36dee8e4",
    )
    assert all(token not in text for token in forbidden)
