from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_maximal_strict_savings_span_capacity_v241_public_summary.json"
RESULT = ROOT / "docs/blastnet_case7_maximal_strict_savings_span_capacity_v241_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case7_maximal_strict_savings_span_capacity_v241.png"
BUILDER = ROOT / "site_tools/build_blastnet_case7_maximal_strict_savings_span_capacity_v241_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v241_summary_records_the_validated_necessary_headroom() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["new_condition_opened"] is False
    assert data["question"]["current_solver_depth"] == 14
    assert data["question"]["span_dimension"] == 30
    assert data["question"]["metrics_optimized_separately"] is True
    assert data["question"]["joint_feasibility_tested"] is False
    assert data["results"]["necessary_safe_later_cells"] == 533
    assert data["results"]["necessary_complete_rigs_passed"] == 13
    assert set(data["results"]["metric_cell_failures"].values()) == {0}
    assert data["results"]["design_rank"] == 30
    assert data["independent_validation"]["sealed_array_erratum_checks_passed"] == 35
    assert data["independent_validation"]["new_science_arrays_written_by_erratum"] == 0
    assert data["adjudication"]["scientific_decision"] == (
        "POST_OPEN_CASE7_MAXIMAL_STRICT_SAVINGS_SPAN_NECESSARY_HEADROOM_V241"
    )
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v241_result_is_bilingual_and_keeps_the_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v241：" in text and "# v241:" in text
    for token in ("533/533", "13/13", "0/533", "2,132", "35/35", "9.1518%"):
        assert token in text
    assert "四组不同系数" in text and "four different coefficient vectors" in text
    assert "第一次 inconclusive" in text and "original inconclusive" in text
    assert "algorithm_breakthrough=false" in text


def test_v241_figure_and_source_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 1100


def test_v241_is_preserved_as_historical_parent_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_CASE19_CAMERA_WEIGHTED_COMPLEMENT_FRAME_ZERO_V260"
    assert current["current_decision"]["v241_sealed_array_readjudication_passed"] is True
    assert current["current_decision"]["v241_necessary_span_capacity_passed"] is True
    assert current["current_decision"]["v241_joint_feasibility_tested"] is False
    assert current["current_decision"]["v241_algorithm_breakthrough"] is False
    assert current["metrics"]["v241_later_cells_necessary_safe"] == 533
    assert current["metrics"]["v241_complete_rigs_necessary_passed"] == 13
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_camera_weighted_complement_v260.png"
    )

    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case7_maximal_strict_savings_span_capacity_v241" in text
        assert "533/533" in text and "13/13" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text

    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v241 K14 当前 Krylov 空间" in log
    assert "maximal strict-savings K14" in log


def test_v241_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "6656873e",
        "dc54cd4a",
    )
    assert all(token not in text for token in forbidden)
