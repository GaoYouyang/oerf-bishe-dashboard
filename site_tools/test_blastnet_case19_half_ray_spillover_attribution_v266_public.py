from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_half_ray_spillover_attribution_v266_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_half_ray_spillover_attribution_v266_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_half_ray_spillover_attribution_v266.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PRIMARY_PAGES = (ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html")


def test_v266_summary_records_mixed_failure_attribution() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    attribution = data["attribution"]
    assert data["scope"]["cells"] == 429
    assert data["scope"]["matched_cells"] == 200
    assert data["scope"]["failed_cells"] == 229
    assert attribution["selected_residual_nonincrease_cells_vs_parent"] == 429
    assert attribution["cell_failure_classes"] == {
        "complement_only": 119,
        "both_selected_and_complement": 110,
        "selected_only": 0,
        "numerical_cancellation": 0,
    }
    assert attribution["p90_violation_classes"]["both_selected_and_complement"] == 12
    assert attribution["worst_violation_classes"]["both_selected_and_complement"] == 13
    assert attribution["cells_where_selected_fell_but_complement_rose_vs_parent"] == 0


def test_v266_summary_records_independence_cost_and_boundaries() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 18
    assert data["cost_ledger"]["new_exact_calls_A"] == 0
    assert data["cost_ledger"]["new_exact_calls_At"] == 0
    assert data["adjudication"]["pure_unselected_ray_spillover_explanation_rejected"] is True
    assert data["adjudication"]["fixed_even_quincunx_half_ray_route_closed"] is True
    assert data["adjudication"]["one_sided_objective_class_globally_closed"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v266_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v266：" in text and "# v266:" in text
    for token in ("18/18", "119", "110", "12", "13/13", "0A+0A^T", "MIXED_SELECTED_AND_UNSELECTED_RAY_DEFICIT_V266"):
        assert token in text
    assert "不是新候选" in text
    assert "not a new candidate" in text
    assert "algorithm_breakthrough=false" in text


def test_v266_figure_is_public_and_readable() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v266_remains_historical_after_v267() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-27"
    assert current["scientific_status"] == "FAIL_CASE19_COARSE_RESIDUAL_GALERKIN_FRAME_ZERO_V268_4"
    assert current["metrics"]["v266_cell_failure_complement_only"] == 119
    assert current["metrics"]["v266_cell_failure_both_positive"] == 110
    assert current["current_decision"]["v266_fixed_half_ray_route_closed"] is True
    assert current["current_decision"]["v266_new_candidate_authorized"] is False
    assert current["public_evidence"]["figure"].endswith("coarse_residual_galerkin_v268.png")
    for page in PRIMARY_PAGES:
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_half_ray_spillover_attribution_v266" in text
        assert "v265.1" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text


def test_v266_daily_progress_keeps_one_day_card() -> None:
    text = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert text.count('data-date="2026-08-27"') == 1
    assert "same-day-history" in text
    assert "v264" in text and "v265.1" in text and "v266" in text


def test_v266_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "protocol_sha256",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)
