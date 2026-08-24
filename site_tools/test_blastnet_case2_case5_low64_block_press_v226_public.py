"""Public evidence checks for the sealed v226 block-PRESS diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_block_press_v226_public_summary.json"
RESULT = ROOT / "docs/blastnet_case2_case5_low64_block_press_v226_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case2_case5_low64_block_press_v226.png"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v226_primary_is_safe_but_misses_utility_by_one_frame() -> None:
    payload = _summary()
    primary = payload["primary_block_press_certificate"]
    assert primary["case2"]["accepted_cells"] == 297
    assert primary["case2"]["accepted_safe_cells"] == 297
    assert primary["case2"]["accepted_unsafe_cells"] == 0
    assert primary["case2"]["complete_rigs_accuracy_passed"] == 13
    assert primary["case5"]["minimum_rig_accepted_cells"] == 4
    assert primary["case5"]["required_minimum_rig_accepted_cells"] == 5
    assert primary["case5"]["minimum_rig_accept_fraction"] == 4 / 42
    assert primary["case5"]["support_gate_passed"] is False
    assert primary["selected_mixed_policy_passed"] is True
    assert primary["certificate_passed"] is False


def test_v226_cheap_control_does_not_explain_the_safe_transfer() -> None:
    control = _summary()["cheap_full_fit_residual_control"]
    assert control["case2"]["accepted_cells"] == 553
    assert control["case2"]["accepted_safe_cells"] == 492
    assert control["case2"]["accepted_unsafe_cells"] == 61
    assert control["case2"]["complete_rigs_accuracy_passed"] == 0
    assert control["policy_passed"] is False


def test_v226_independent_validation_and_boundaries() -> None:
    payload = _summary()
    validation = payload["independent_validation"]
    assert validation["status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_LOW64_BLOCK_PRESS_CERTIFICATE_V226"
    )
    assert (
        validation["scientific_decision"] == "FAIL_LOW64_BLOCK_PRESS_CERTIFICATE_V226"
    )
    assert validation["required_checks_passed"] == 16
    assert validation["required_checks_total"] == 16
    assert validation["discrete_decisions_match"] is True
    assert payload["adjudication"]["block_press_certificate_route_closed"] is True
    assert payload["adjudication"]["all_multiview_mechanisms_closed"] is False
    assert payload["claims_fixed_false"]["algorithm_breakthrough"] is False
    assert payload["claims_fixed_false"]["gpu_rental_authorized"] is False


def test_v226_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v226：" in text and "# v226:" in text
    assert "4/42=9.52%" in text
    assert "297" in text and "197" in text and "61" in text
    assert "16/16" in text
    assert "algorithm_breakthrough=false" in text


def test_v226_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v226_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["scientific_status"] == "FAIL_LOW64_BLOCK_PRESS_CERTIFICATE_V226"
    assert current["metrics"]["v226_primary_case2_accepted_unsafe"] == 0
    assert current["metrics"]["v226_primary_case5_minimum_rig_accepted"] == 4
    assert (
        current["current_decision"]["v226_block_press_certificate_route_closed"] is True
    )
    assert current["current_decision"]["v226_algorithm_breakthrough"] is False
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v226" in content
        assert "blastnet_case2_case5_low64_block_press_v226.png" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v226 相机分块 PRESS 零危险误接但效用门差一个帧" in log
    assert (
        "v226 camera-block PRESS has zero unsafe accepts but misses utility by one frame"
        in log
    )


def test_v226_public_artifacts_do_not_expose_private_execution_details() -> None:
    values = [
        SUMMARY.read_text(encoding="utf-8"),
        RESULT.read_text(encoding="utf-8"),
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"),
    ]
    forbidden = (
        "private_results",
        "private_worktrees",
        "/Users/",
        "source_commit",
        "checkpoint",
    )
    for value in values:
        assert not any(token in value for token in forbidden)
