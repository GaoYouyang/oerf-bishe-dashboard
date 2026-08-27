from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_two_color_additive_schwarz_v267_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_two_color_additive_schwarz_v267_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_two_color_additive_schwarz_v267.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
HISTORICAL_PAGES = (
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
)


def test_v267_summary_records_primary_control_and_failure_origin() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = data["results"]
    assert data["scope"]["cells"] == 429
    assert results["primary_absolute_pass_cells"] == 429
    assert results["primary_matched_pass_cells"] == 2
    assert results["primary_matched_pass_rigs"] == 0
    assert results["matched_failure_counts"] == {
        "field": 0,
        "full_gradient": 0,
        "interior_gradient": 0,
        "observation": 427,
    }
    assert results["same_work_full_row_control_matched_pass_cells"] == 219
    assert results["sealed_one_half_control_matched_pass_cells"] == 200


def test_v267_summary_records_cross_color_interference() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    interference = data["cross_color_interference"]
    assert interference["each_color_own_local_step_improved_cells"] == 429
    assert interference["combined_state_worse_than_each_color_own_diagonal_cells"] == {
        "color_0": 429,
        "color_1": 429,
    }
    assert interference["global_residual_vs_parent"]["worsened_cells"] == 419
    assert interference["global_residual_vs_parent"]["improved_cells"] == 10


def test_v267_summary_records_independence_cost_and_boundaries() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 24
    assert validation["maximum_camera_permutation_difference"] == 0
    assert data["cost_ledger"]["primary_calls_A"] == 16
    assert data["cost_ledger"]["primary_calls_At"] == 15
    assert data["adjudication"]["exact_synchronous_two_color_route_closed"] is True
    assert data["adjudication"]["effective_exact_call_reduction_established"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v267_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v267：" in text and "# v267:" in text
    for token in (
        "24/24",
        "2/429",
        "427",
        "419/429",
        "16A+15A^T",
        "FAIL_CASE19_TWO_COLOR_ADDITIVE_SCHWARZ_V267",
    ):
        assert token in text
    assert "post-open" in text
    assert "algorithm_breakthrough=false" in text


def test_v267_figure_is_public_and_readable() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v267_remains_historical_after_v268_becomes_latest() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-27"
    assert current["scientific_status"] == "FAIL_CASE19_DIRECT_VOLUME_HODGE_IS_POISSON_REPARAMETERIZATION_V273"
    assert current["metrics"]["v267_primary_matched_pass_cells"] == 2
    assert current["metrics"]["v267_global_residual_worsened_cells"] == 419
    assert current["current_decision"]["v267_exact_synchronous_two_color_route_closed"] is True
    assert current["current_decision"]["v267_predictor_training_authorized"] is False
    assert current["public_evidence"]["figure"].endswith("volume_hodge_equivalence_v273.png")
    for page in HISTORICAL_PAGES:
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_two_color_additive_schwarz_v267" in text
        assert "v266" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v267" in log and "v268" in log


def test_v267_daily_progress_keeps_one_day_card() -> None:
    text = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert text.count('data-date="2026-08-27"') == 1
    assert "same-day-history" in text
    assert all(version in text for version in ("v264", "v265.1", "v266", "v267"))


def test_v267_public_artifacts_exclude_private_execution_details() -> None:
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
