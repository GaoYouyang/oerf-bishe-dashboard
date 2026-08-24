"""Public evidence checks for the sealed v227 studentized block-PRESS diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case2_case5_low64_studentized_block_press_v227_public_summary.json"
)
RESULT = (
    ROOT
    / "docs/blastnet_case2_case5_low64_studentized_block_press_v227_result_2026-08-24.md"
)
FIGURE = (
    ROOT / "assets/figures/blastnet_case2_case5_low64_studentized_block_press_v227.png"
)


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v227_whitening_adds_safe_accepts_but_still_misses_utility() -> None:
    payload = _summary()
    primary = payload["primary_studentized_block_press_certificate"]
    parent = payload["parent_v226_block_press_control"]
    assert primary["case2"]["accepted_safe_cells"] == 323
    assert primary["case2"]["accepted_unsafe_cells"] == 0
    assert parent["case2_accepted_safe_cells"] == 297
    assert parent["case2_accepted_unsafe_cells"] == 0
    assert primary["case5"]["accepted_cells"] == 123
    assert primary["case5"]["minimum_rig_accepted_cells"] == 4
    assert primary["case5"]["required_minimum_rig_accepted_cells"] == 5
    assert primary["case5"]["support_gate_passed"] is False
    assert primary["selected_mixed_policy_passed"] is True
    assert primary["certificate_passed"] is False


def test_v227_reproduces_parent_and_validates_independently() -> None:
    payload = _summary()
    parent = payload["parent_v226_block_press_control"]
    validation = payload["independent_validation"]
    assert parent["score_maximum_absolute_reproduction_difference"] == 0.0
    assert parent["threshold_maximum_absolute_reproduction_difference"] == 0.0
    assert parent["decisions_reproduced_exactly"] is True
    assert validation["status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_LOW64_STUDENTIZED_BLOCK_PRESS_V227"
    )
    assert validation["scientific_decision"] == (
        "FAIL_LOW64_STUDENTIZED_BLOCK_PRESS_CERTIFICATE_V227"
    )
    assert validation["required_checks_passed"] == 19
    assert validation["required_checks_total"] == 19
    assert validation["discrete_decisions_match"] is True


def test_v227_boundaries_remain_fail_closed() -> None:
    payload = _summary()
    adjudication = payload["adjudication"]
    claims = payload["claims_fixed_false"]
    assert adjudication["geometry_whitening_changes_acceptance"] is True
    assert (
        adjudication["geometry_whitening_solves_cross_rig_utility_stability"] is False
    )
    assert adjudication["studentized_block_press_certificate_route_closed"] is True
    assert adjudication["all_multiview_mechanisms_closed"] is False
    assert all(value is False for value in claims.values())


def test_v227_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v227：" in text and "# v227:" in text
    assert "323" in text and "297" in text and "4/42=9.52%" in text
    assert "19/19" in text
    assert "algorithm_breakthrough=false" in text


def test_v227_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v227_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["scientific_status"] == (
        "FAIL_LOW64_STUDENTIZED_BLOCK_PRESS_CERTIFICATE_V227"
    )
    assert current["metrics"]["v227_primary_case2_accepted_safe"] == 323
    assert current["metrics"]["v227_primary_case2_accepted_unsafe"] == 0
    assert current["metrics"]["v227_primary_case5_minimum_rig_accepted"] == 4
    assert (
        current["current_decision"][
            "v227_studentized_block_press_certificate_route_closed"
        ]
        is True
    )
    assert current["current_decision"]["v227_algorithm_breakthrough"] is False
    assert "v227" in current["next_scientific_gate"]
    assert "v226 closes" not in current["next_scientific_gate"]
    assert "studentized_block_press_v227" in current["public_evidence"]["result"]
    assert "studentized_block_press_v227" in current["public_evidence"]["summary"]
    assert "studentized_block_press_v227" in current["public_evidence"]["figure"]
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v227" in content
        assert "blastnet_case2_case5_low64_studentized_block_press_v227.png" in content
        assert "blastnet_case2_case5_low64_block_press_v226" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v227 几何白化提高安全接受但逐 rig 效用仍失败" in log
    assert (
        "v227 geometry whitening raises safe acceptance but per-rig utility still fails"
        in log
    )


def test_v227_public_artifacts_do_not_expose_private_execution_details() -> None:
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
