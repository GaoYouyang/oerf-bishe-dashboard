from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/blastnet_case2_case5_low64_dual_press_union_v228_public_summary.json"
)
RESULT = (
    ROOT / "docs/blastnet_case2_case5_low64_dual_press_union_v228_result_2026-08-24.md"
)
FIGURE = ROOT / "assets/figures/blastnet_case2_case5_low64_dual_press_union_v228.png"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v228_public_summary_preserves_retrospective_boundary() -> None:
    payload = _summary()
    scope = payload["scope"]
    policy = payload["fixed_union_policy"]
    assert (
        scope["diagnostic_type"]
        == "post-open retrospective fixed-OR mechanism diagnostic"
    )
    assert scope["new_score_fitted"] is False
    assert scope["new_threshold_fitted"] is False
    assert scope["new_condition_opened"] is False
    assert scope["deployment_algorithm_established"] is False
    assert scope["preregistered_success"] is False
    assert policy["case5"]["accepted_cells"] == 140
    assert policy["case5"]["minimum_rig_accepted_cells"] == 5
    assert policy["case5"]["accepted_unsafe_cells"] == 0
    assert policy["case2"]["accepted_safe_cells"] == 324
    assert policy["case2"]["accepted_unsafe_cells"] == 0
    assert policy["retrospective_gate_passed"] is True


def test_v228_complementarity_and_independent_recomputation_are_exact() -> None:
    payload = _summary()
    case5 = payload["parent_comparison"]["case5"]
    case2 = payload["parent_comparison"]["case2"]
    independent = payload["independent_validation"]
    assert case5["raw_only_accept"] == 17
    assert case5["studentized_only_accept"] == 14
    assert case5["rig4_raw_studentized_union"] == [5, 4, 5]
    assert case5["rig11_raw_studentized_union"] == [4, 6, 6]
    assert case2["raw_only_accept"] == 1
    assert case2["studentized_only_accept"] == 27
    assert (
        independent["required_checks_passed"]
        == independent["required_checks_total"]
        == 17
    )
    assert independent["discrete_decisions_match"] is True
    assert independent["maximum_formal_summary_absolute_difference"] <= 1e-10
    assert independent["end_to_end_physics_independence_proven"] is False


def test_v228_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v228：" in text and "# v228:" in text
    for token in ("140/546", "5/42=11.90%", "324/715", "17/17"):
        assert token in text
    assert "post-open retrospective diagnostic" in text
    assert "algorithm_breakthrough=false" in text


def test_v228_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v228_historical_surfaces_and_log_are_preserved() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["metrics"]["v228_union_case5_accepted"] == 140
    assert current["metrics"]["v228_union_case5_minimum_rig_accepted"] == 5
    assert current["metrics"]["v228_union_case2_accepted_safe"] == 324
    assert current["metrics"]["v228_union_case2_accepted_unsafe"] == 0
    assert current["current_decision"]["v228_retrospective_gate_passed"] is True
    assert current["current_decision"]["v228_deployment_policy_established"] is False
    assert current["current_decision"]["v228_algorithm_breakthrough"] is False
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v228" in content
        assert "blastnet_case2_case5_low64_dual_press_union_v228.png" in content
        assert "blastnet_case2_case5_low64_studentized_block_press_v227" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v228" in log
    assert "post-open" in log


def test_v228_public_artifacts_do_not_expose_private_execution_details() -> None:
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
