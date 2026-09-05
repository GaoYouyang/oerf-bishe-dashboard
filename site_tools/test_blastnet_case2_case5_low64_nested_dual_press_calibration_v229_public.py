from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_nested_dual_press_calibration_v229_public_summary.json"
RESULT = ROOT / "docs/blastnet_case2_case5_low64_nested_dual_press_calibration_v229_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case2_case5_low64_nested_dual_press_calibration_v229.png"


def test_public_summary_preserves_v229_evidence_boundary() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = data["primary_policy"]
    assert primary["case5"]["accepted_cells"] == 136
    assert primary["case5"]["minimum_rig_accepted_cells"] == 5
    assert primary["case2"]["accepted_cells"] == 318
    assert primary["case2"]["accepted_unsafe_cells"] == 0
    assert data["independent_validation"]["required_checks_passed"] == 17
    assert data["adjudication"]["unopened_condition_gate_authorized"] is True
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_public_artifacts_exist_and_do_not_disclose_private_paths() -> None:
    assert RESULT.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = ("/Users/", "private_results", "private_worktrees", "checkpoint", "sha256")
    assert all(token not in text for token in forbidden)


def test_v229_figure_is_rendered() -> None:
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v229_historical_surfaces_and_log_are_preserved() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["updated"] == "2026-09-05"
    assert current["v229_nested_dual_press_scientific_decision"] == (
        "POST_OPEN_FOLD_LOCAL_DUAL_PRESS_CALIBRATION_HEADROOM_V229"
    )
    assert current["metrics"]["v229_primary_case5_accepted"] == 136
    assert current["metrics"]["v229_primary_case5_minimum_rig_accepted"] == 5
    assert current["metrics"]["v229_primary_case2_accepted_safe"] == 318
    assert current["metrics"]["v229_primary_case2_accepted_unsafe"] == 0
    assert current["current_decision"]["v229_unopened_condition_gate_authorized"] is True
    assert current["current_decision"]["v229_algorithm_breakthrough"] is False
    assert RESULT.is_file() and FIGURE.is_file()
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v229" in content
        assert "blastnet_case2_case5_low64_nested_dual_press_calibration_v229" in content
        assert "blastnet_case2_case5_low64_dual_press_union_v228" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v229 把事后 OR 线索" in log
    assert "v229 converts the retrospective v228 OR lead" in log
