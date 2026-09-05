from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_krylov_history_joint_reorthogonalization_v269_1_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_krylov_history_joint_reorthogonalization_v269_1_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_krylov_history_joint_reorthogonalization_v269_1.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PAGES = (
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
)
LOG = ROOT / "docs/operator_3d_learning_log.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v269_1_summary_is_inconclusive_and_diagnostic_only() -> None:
    data = _summary()
    independent = data["independent_validation"]
    diagnostic = data["diagnostic_results"]
    assert independent["passed"] is False
    assert (independent["checks_passed"], independent["checks_total"]) == (28, 29)
    assert independent["blocking_check"] == "formal_independent_observation_exact"
    assert independent["maximum_observation_normalized_difference"] < 1e-15
    assert diagnostic["authoritative_scientific_result"] is False
    assert diagnostic["primary_absolute_pass_cells"] == 13
    assert diagnostic["primary_matched_pass_cells"] == 0
    assert diagnostic["matched_failure_counts"]["observation"] == 13


def test_v269_1_adjudication_closes_without_rerun_or_claims() -> None:
    data = _summary()
    decision = data["adjudication"]
    assert decision["status"].startswith("INCONCLUSIVE_INVALID_CASE19")
    assert decision["fixed_cached_history_joint_solve_closed"] is True
    assert decision["rerun_or_tolerance_relaxation_authorized"] is False
    assert decision["diagnostic_counts_may_not_be_promoted_to_formal_pass_or_fail"] is True
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v269_1_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v269.1：" in text and "# v269.1:" in text
    assert "21/21" in text and "28/29" in text and "5.20e-16" in text
    assert "0/13" in text and "diagnostic only" in text
    assert "algorithm_breakthrough=false" in text


def test_v269_1_figure_is_public_and_readable() -> None:
    assert FIGURE.exists() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2000
        assert image.height >= 900


def test_v269_1_remains_historical_after_v270() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["headline"].startswith("v283")
    assert current["metrics"]["v269_1_independent_checks_passed"] == 28
    assert current["current_decision"]["v269_1_fixed_history_route_closed"] is True
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        assert "krylov_history_joint_reorthogonalization_v269_1" in text
    assert "## 2026-08-27：v269.1" in LOG.read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-27"') == 1
    assert "v268" in daily


def test_v269_1_public_artifacts_exclude_private_execution_details() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (SUMMARY, RESULT, *PAGES))
    forbidden = (
        "/Users/",
        "/Volumes/",
        "source-private",
        "FORMAL_EXIT_CODE",
        "VALIDATOR_EXIT_CODE",
        "private-source-identity",
        "e669b092",
    )
    assert all(token not in text for token in forbidden)
