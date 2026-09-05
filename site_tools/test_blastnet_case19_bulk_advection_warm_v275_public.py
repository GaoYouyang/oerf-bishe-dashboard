from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_bulk_advection_warm_v275_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_bulk_advection_warm_v275_result_2026-08-28.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_bulk_advection_warm_v275.png"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LOG = ROOT / "docs/operator_3d_learning_log.md"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_v275_public_summary_preserves_inconclusive_boundary() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    execution = data["execution"]
    diagnostic = data["diagnostic_results_not_scientific_pass"]
    adjudication = data["adjudication"]
    assert execution["independent_checks_passed"] == 26
    assert execution["independent_checks_total"] == 31
    assert execution["first_independent_attempt_scientific_result"] is False
    assert execution["repaired_predictor_reads_only_sealed_deployment_visible_inputs"] is True
    assert diagnostic["primary_absolute_cells"] == 428
    assert diagnostic["reference_absolute_cells"] == 428
    assert diagnostic["primary_matched_cells"] == 13
    assert diagnostic["primary_matched_complete_rigs"] == 0
    assert adjudication["scientific_decision"] == (
        "INCONCLUSIVE_INVALID_CASE19_BULK_ADVECTION_WARM_V275"
    )
    assert adjudication["algorithm_breakthrough"] is False
    assert data["cost_accounting"]["effective_exact_call_reduction_established"] is False


def test_v275_result_is_bilingual_and_fail_closed() -> None:
    text = RESULT.read_text(encoding="utf-8")
    for token in ("26/31", "428/429", "13/429", "0/13", "1.5673e-6"):
        assert token in text
    assert "# v275：" in text and "# v275:" in text
    assert "不事后放宽容差" in text
    assert "no post-hoc tolerance relaxation" in text
    assert "algorithm_breakthrough=false" in text


def test_v275_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 60_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1000


def test_v275_remains_visible_on_public_surfaces() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == (
        "PASS_INDEPENDENT_LINEAR_SOURCE_BUDGET_V280"
    )
    assert current["metrics"]["v275_independent_checks_passed"] == 26
    assert current["current_decision"]["v275_algorithm_breakthrough"] is False
    for page in (FOCUS, HOME, DAILY, LOG):
        text = page.read_text(encoding="utf-8")
        assert "v275" in text
        assert "26/31" in text
    assert "blastnet_case19_bulk_advection_warm_v275" in FOCUS.read_text(encoding="utf-8")


def test_v275_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "/private/tmp/",
        "private_results",
        "source_commit",
        "protocol_sha256",
        "validator_commit",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)
