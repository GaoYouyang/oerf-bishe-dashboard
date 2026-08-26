from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_canonical_camera_actual_k14_solver_controls_v243_public_summary.json"
RESULT = ROOT / "docs/blastnet_case7_canonical_camera_actual_k14_solver_controls_v243_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case7_canonical_camera_actual_k14_solver_controls_v243.png"
BUILDER = ROOT / "site_tools/build_blastnet_case7_canonical_camera_actual_k14_solver_controls_v243_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v243_summary_records_validated_actual_solver_headroom() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = data["results"]["primary"]
    controls = data["results"]["equal_or_cheaper_controls"]
    assert data["scope"]["new_condition_opened"] is False
    assert data["question"]["deployment_visible_inputs_only_before_prediction_barrier"] is True
    assert data["question"]["truth_used_for_prediction_or_cache_update"] is False
    assert data["question"]["tolerance_relaxation"] is False
    assert primary["passed"] is True
    assert primary["absolute_cells_passed"] == 546
    assert primary["matched_cells_passed"] == 546
    assert primary["matched_complete_rigs_passed"] == 13
    assert all(row["passed"] is False for row in controls)
    assert all(row["matched_complete_rigs_passed"] == 0 for row in controls)
    assert data["independent_validation"]["independent_checks_passed"] == 39
    assert data["independent_validation"]["formal_independent_field_maximum"] == 0.0
    assert data["adjudication"]["scientific_decision"] == (
        "POST_OPEN_CASE7_CANONICAL_CAMERA_ACTUAL_K14_WARM_SPECIFIC_HEADROOM_V243"
    )
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v243_result_is_bilingual_and_keeps_the_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v243：" in text and "# v243:" in text
    for token in ("546/546", "13/13", "39/39", "9.1518%", "v242.1", "1.19e-6"):
        assert token in text
    assert "四个同价或更便宜 controls 均未通过" in text
    assert "All four equal-or-cheaper controls fail" in text
    assert "没有通过放宽容差" in text and "not relaxed" in text
    assert "algorithm_breakthrough=false" in text


def test_v243_figure_and_source_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 1100


def test_v243_is_preserved_as_historical_parent_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "MIXED_SELECTED_AND_UNSELECTED_RAY_DEFICIT_V266"
    assert current["current_decision"]["v243_actual_unchanged_k14_solver_passed"] is True
    assert current["current_decision"]["v243_equal_or_cheaper_controls_explain_result"] is False
    assert current["current_decision"]["v243_external_generalization"] is False
    assert current["current_decision"]["v243_algorithm_breakthrough"] is False
    assert current["metrics"]["v243_primary_matched_cells_passed"] == 546
    assert current["metrics"]["v243_primary_matched_complete_rigs_passed"] == 13
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_half_ray_spillover_attribution_v266.png"
    )

    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case7_canonical_camera_actual_k14_solver_controls_v243" in text
        assert "546/546" in text and "13/13" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text

    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v243 相机 ID 规范化" in log
    assert "v243 canonical camera IDs" in log


def test_v243_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
    )
    assert all(token not in text for token in forbidden)
