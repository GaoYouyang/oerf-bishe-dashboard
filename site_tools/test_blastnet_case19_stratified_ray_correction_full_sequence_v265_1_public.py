from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_stratified_ray_correction_full_sequence_v265_1_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_stratified_ray_correction_full_sequence_v265_1_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_stratified_ray_correction_full_sequence_v265_1.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PRIMARY_PAGES = (ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html")


def test_v265_1_summary_records_full_sequence_failure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    envelope = data["two_implementation_envelope"]
    assert data["scope"]["full_sequence_tested"] is True
    assert data["scope"]["rigs"] == 13
    assert data["scope"]["frames_per_rig"] == 33
    assert data["scope"]["cells"] == 429
    assert envelope["candidate_absolute_pass_cells"] == 429
    assert envelope["candidate_absolute_pass_rigs"] == 13
    assert envelope["candidate_matched_pass_cells"] == 200
    assert envelope["candidate_matched_pass_rigs"] == 0
    assert envelope["candidate_matched_fail_cells"] == 229
    assert envelope["candidate_matched_failure_cells_by_metric"] == {
        "field": 0,
        "full_gradient": 0,
        "interior_gradient": 0,
        "observation": 229,
    }
    assert envelope["candidate_matched_ratio_p90_higher"][-1] == 1.23502543
    assert envelope["candidate_matched_ratio_worst"][-1] == 1.85517339


def test_v265_1_summary_records_cost_controls_and_independent_closure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cost = data["cost_ledger"]
    assert cost["candidate_total_ray_equivalent_calls_A"] == 15.5
    assert cost["candidate_total_ray_equivalent_calls_At"] == 14.5
    assert cost["k16_reference_calls_A"] == cost["k16_reference_calls_At"] == 16.0
    assert data["controls"]["sealed_parent"]["matched_pass_cells"] == 4
    assert data["controls"]["zero_k15"]["absolute_pass_cells"] == 383
    validation = data["independent_validation"]
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 35
    assert validation["standalone_array_recomputation_agreed"] is True
    assert data["adjudication"]["fixed_even_quincunx_half_ray_route_closed"] is True
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v265_1_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v265.1：" in text and "# v265.1:" in text
    for token in ("35/35", "429/429", "200/429", "0/13", "1.23503", "1.85517"):
        assert token in text
    assert "路线现在关闭" in text
    assert "route now closes" in text
    assert "algorithm_breakthrough=false" in text


def test_v265_1_figure_is_public_and_readable() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v265_1_remains_historical_after_v266() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-27"
    assert current["scientific_status"].endswith("V273")
    assert current["metrics"]["v265_1_candidate_matched_pass_cells"] == 200
    assert current["current_decision"]["v265_1_fixed_half_ray_route_closed"] is True
    assert current["current_decision"]["v265_1_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith("volume_hodge_equivalence_v273.png")
    for page in PRIMARY_PAGES:
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_stratified_ray_correction_full_sequence_v265_1" in text
        assert "v264" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text


def test_v265_1_daily_progress_keeps_one_day_card() -> None:
    text = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert text.count('data-date="2026-08-27"') == 1
    assert "same-day-history" in text
    assert "v264" in text and "v265.1" in text


def test_v265_1_public_artifacts_exclude_private_execution_details() -> None:
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
