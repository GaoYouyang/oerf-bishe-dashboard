"""Public evidence checks for the sealed v225 angular-spectrum diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case2_case5_low64_angular_spectrum_support_v225_public_summary.json"
)
RESULT = (
    ROOT
    / "docs/blastnet_case2_case5_low64_angular_spectrum_support_v225_result_2026-08-24.md"
)
FIGURE = (
    ROOT / "assets/figures/blastnet_case2_case5_low64_angular_spectrum_support_v225.png"
)


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v225_primary_and_control_fail_cross_condition_safety() -> None:
    payload = _summary()
    primary = payload["primary_angular_spectrum_policy"]
    control = payload["cheap_two_scalar_control"]
    assert primary["case2"]["accepted_cells"] == 523
    assert primary["case2"]["accepted_unsafe_cells"] == 145
    assert primary["case5"]["minimum_rig_accept_fraction"] == 0.0
    assert primary["policy_passed"] is False
    assert control["case2"]["accepted_cells"] == 186
    assert control["case2"]["accepted_unsafe_cells"] == 132
    assert control["policy_passed"] is False


def test_v225_independent_validation_and_boundaries() -> None:
    payload = _summary()
    validation = payload["independent_validation"]
    assert validation["status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_LOW64_ANGULAR_SPECTRUM_SUPPORT_V225"
    )
    assert validation["scientific_decision"] == (
        "FAIL_LOW64_ANGULAR_SPECTRUM_MUTUAL_SUPPORT_V225"
    )
    assert validation["required_checks_passed"] == 17
    assert validation["required_checks_total"] == 17
    assert validation["discrete_decisions_match"] is True
    assert (
        payload["adjudication"]["angular_spectrum_mutual_support_route_closed"] is True
    )
    assert payload["adjudication"]["all_multiview_mechanisms_closed"] is False
    assert payload["claims_fixed_false"]["algorithm_breakthrough"] is False
    assert payload["claims_fixed_false"]["gpu_rental_authorized"] is False


def test_v225_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v225：" in text and "# v225:" in text
    assert "145" in text and "132" in text
    assert "17 / 17" in text
    assert "algorithm_breakthrough=false" in text


def test_v225_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v225_historical_surfaces_and_log_remain_synchronized() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["v225_low64_angular_spectrum_scientific_decision"] == (
        "FAIL_LOW64_ANGULAR_SPECTRUM_MUTUAL_SUPPORT_V225"
    )
    assert current["metrics"]["v225_primary_case2_accepted_unsafe"] == 145
    assert current["metrics"]["v225_control_case2_accepted_unsafe"] == 132
    assert (
        current["current_decision"]["v225_angular_spectrum_support_route_closed"]
        is True
    )
    assert current["current_decision"]["v225_algorithm_breakthrough"] is False
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v225" in content
        assert "blastnet_case2_case5_low64_angular_spectrum_support_v225.png" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v225 完整九相机角谱互支持仍不安全" in log
    assert "v225 full nine-camera angular-spectrum support remains unsafe" in log


def test_v225_public_artifacts_do_not_expose_private_execution_details() -> None:
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
