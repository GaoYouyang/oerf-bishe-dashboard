from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_observation_graph_traversal_v252_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_observation_graph_traversal_v252_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_observation_graph_traversal_v252.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_observation_graph_traversal_v252_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v252_summary_preserves_inconclusive_numeric_gate() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    diagnostic = data["post_open_discrete_diagnostic"]
    adjudication = data["adjudication"]
    assert data["scope"]["cells"] == 429
    assert data["scope"]["truth_available_before_prediction_barrier"] is False
    assert data["mechanism"]["graph_calls_A_At_per_rig"] == [496, 464]
    assert data["mechanism"]["midpoint_control_calls_A_At_per_rig"] == [496, 464]
    assert validation["passed"] is False
    assert validation["scientific_decision"] == (
        "INCONCLUSIVE_INVALID_CASE19_OBSERVATION_GRAPH_TRAVERSAL_V252"
    )
    failed = [check for check in validation["numeric_checks"] if not check["passed"]]
    assert [check["name"] for check in failed] == [
        "field_recomputation_within_tolerance",
        "metric_recomputation_within_tolerance",
        "residual_recomputation_within_tolerance",
    ]
    assert diagnostic["scientific_performance_result"] is False
    assert diagnostic["graph_absolute_cells"] == 427
    assert diagnostic["graph_absolute_complete_rigs"] == 11
    assert diagnostic["midpoint_control_absolute_cells"] == 429
    assert diagnostic["midpoint_control_absolute_complete_rigs"] == 13
    assert diagnostic["k16_reference_absolute_cells"] == 417
    assert diagnostic["k16_reference_absolute_complete_rigs"] == 9
    assert adjudication["route_action"] == (
        "CLOSE_OBSERVATION_GRAPH_TRAVERSAL_WITHOUT_SCIENTIFIC_PASS_OR_FAIL_CLAIM"
    )
    assert adjudication["scientific_pass_claimed"] is False
    assert adjudication["scientific_fail_claimed"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v252_result_is_bilingual_and_does_not_upgrade_discrete_diagnostics() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v252：" in text and "# v252:" in text
    for token in ("1.1990e-7", "8.6125e-8", "8.1713e-6", "427/429", "429/429", "9.0909%"):
        assert token in text
    assert "不是通过独立验证的科学性能结果" in text
    assert "not independently validated scientific performance results" in text
    assert "不把 v252 升格为科学 FAIL" in text
    assert "without upgrading v252 to a scientific FAIL" in text
    assert "algorithm_breakthrough=false" in text


def test_v252_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v252_is_preserved_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_CASE19_K1_SET_SUBSPACE_MATCHED_ACCURACY_V254"
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert metrics["v252_numeric_checks_failed"] == 3
    assert metrics["v252_graph_absolute_cells"] == 427
    assert metrics["v252_midpoint_absolute_cells"] == 429
    assert decision["v252_independent_validation_passed"] is False
    assert decision["v252_scientific_result_inconclusive"] is True
    assert decision["v252_observation_graph_traversal_closed"] is True
    assert decision["v252_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_k1_set_subspace_v254.png"
    )
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_observation_graph_traversal_v252" in text
        assert "427/429" in text and "429/429" in text and "INCONCLUSIVE" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    home = HOME.read_text(encoding="utf-8")
    assert "v252 Case 19：三项数值一致性越线" in home
    assert "v252 Case 19: three numeric agreement checks fail" in home
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v252" in log
    assert "post-open discrete" in log


def test_v252_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)
