from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_stratified_ray_correction_v264_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_stratified_ray_correction_v264_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_stratified_ray_correction_v264.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PRIMARY_PAGES = (
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
)


def test_v264_summary_records_the_conservative_joint_pass() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    envelope = data["two_implementation_envelope"]
    assert data["scope"]["rigs"] == 13
    assert data["scope"]["selected_ray_fraction"] == 0.5
    assert envelope["candidate_absolute_pass_rigs"] == 13
    assert envelope["candidate_matched_pass_rigs"] == 13
    assert envelope["candidate_joint_pass_rigs"] == 13
    assert max(envelope["candidate_matched_ratio_p90_higher"]) <= 1.0
    assert max(envelope["candidate_matched_ratio_worst"]) <= 1.05
    assert data["controls"]["sealed_v258"]["joint_pass_rigs"] == 0
    assert data["controls"]["zero_pcgls_k15"]["joint_pass_rigs"] == 0


def test_v264_summary_records_cost_and_independent_closure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cost = data["cost_ledger"]
    assert cost["candidate_total_ray_equivalent_calls_A"] == 15.5
    assert cost["candidate_total_ray_equivalent_calls_At"] == 14.5
    assert cost["k16_reference_calls_A"] == cost["k16_reference_calls_At"] == 16.0
    validation = data["independent_validation"]
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 32
    assert validation["camera_permutation_field_difference"] == 0.0
    assert validation["camera_permutation_state_difference"] == 0.0
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v264_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v264：" in text and "# v264:" in text
    for token in ("32/32", "13/13", "15.5A+14.5A^T", "0.96580", "1.04717"):
        assert token in text
    assert "只有首帧" in text
    assert "only frame zero" in text
    assert "algorithm_breakthrough=false" in text


def test_v264_figure_is_public_and_readable() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v264_remains_historical_after_v265_1() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-27"
    assert current["scientific_status"].endswith("V268_4")
    assert current["metrics"]["v264_candidate_joint_pass_rigs"] == 13
    assert current["current_decision"]["v264_unchanged_full_sequence_gate_authorized"] is True
    assert current["current_decision"]["v264_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_coarse_residual_galerkin_v268.png"
    )
    for page in PRIMARY_PAGES:
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_stratified_ray_correction_v264" in text
        assert "blastnet_case19_stratified_ray_correction_full_sequence_v265_1" in text
        assert "32/32" in text and "35/35" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text


def test_v264_public_artifacts_exclude_private_execution_details() -> None:
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
