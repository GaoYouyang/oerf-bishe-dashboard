"""Public evidence checks for the independently sealed v215 result."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case5_observation_proxy_warm_replay_v215_public_summary.json"
RESULT = ROOT / "docs/blastnet_case5_observation_proxy_warm_replay_v215_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case5_observation_proxy_warm_replay_v215.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v215_summary_preserves_the_inconclusive_reference_decision() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["scientific_decision"] == (
        "INCONCLUSIVE_INVALID_OBSERVATION_PROXY_WARM_REPLAY_V215"
    )
    reference = summary["reference_adequacy"]
    assert reference["complete_geometries_passed"] == 1
    assert reference["complete_geometries_total"] == 13
    assert reference["strict_cells_passed"] == 466
    assert reference["strict_cells_total"] == 546
    assert reference["interior_gradient_violations"] == 80
    assert reference["field_violations"] == 0
    assert reference["full_gradient_violations"] == 0
    assert reference["observation_violations"] == 0
    assert reference["adequate"] is False


def test_v215_summary_preserves_independent_agreement_and_false_claims() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    independent = summary["independent_recomputation"]
    assert independent["scientific_decision_exact_match"] is True
    assert independent["all_arm_physical_fields_match"] is True
    assert independent["all_arm_metrics_match"] is True
    assert independent["all_call_ledgers_match"] is True
    assert independent["maximum_metric_absolute_difference"] < 2e-10
    adjudication = summary["adjudication"]
    assert adjudication["selected_proxy_arm"] is None
    assert adjudication["warm_start_success_adjudicated"] is False
    assert adjudication["warm_start_failure_adjudicated"] is False
    assert all(value is False for value in summary["claims_fixed_false"].values())


def test_v215_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v215：" in text
    assert "# v215:" in text
    assert "466/546" in text
    assert "1/13" in text
    assert "algorithm_breakthrough=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 600
        extrema = image.convert("RGB").getextrema()
        assert any(low != high for low, high in extrema)


def test_v215_is_preserved_beneath_the_v217_1_current_headline() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] >= "2026-08-24"
    assert current["v221_low64_exact_rowspace_lift_scientific_decision"] == (
        "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert current["metrics"]["v215_reference_strict_cells_passed"] == 466
    assert current["metrics"]["v215_reference_complete_rigs_passed"] == 1
    assert current["current_decision"]["v215_reference_adequate"] is False
    assert current["current_decision"]["v215_proxy_warm_start_adjudicated"] is False
    assert current["current_decision"]["v215_resource_gate_authorized"] is False


def test_primary_pages_reference_v215_in_both_languages() -> None:
    for relative in (
        "index.html",
        "operator-learning/index.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "blastnet_case5_observation_proxy_warm_replay_v215" in content
        assert "v215" in content
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v215 不是 warm start 失败" in log
    assert "v215 is not a failed warm start" in log


def test_v215_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "checkpoint",
        "19136e49",
        "launch_formal",
    )
    for path in (SUMMARY, RESULT):
        content = path.read_text(encoding="utf-8")
        assert all(token not in content for token in forbidden)
