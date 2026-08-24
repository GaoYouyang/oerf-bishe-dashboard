"""Public evidence checks for the sealed v221 negative adjudication."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_exact_rowspace_lift_v221_public_summary.json"
RESULT = ROOT / "docs/blastnet_case2_case5_low64_exact_rowspace_lift_v221_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case2_case5_low64_exact_rowspace_lift_v221.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v221_summary_preserves_negative_scientific_decision() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["independent_validation"]["status"] == "PASS_INDEPENDENT_RECOMPUTATION_LOW64_EXACT_ROWSPACE_LIFT_V221"
    assert payload["independent_validation"]["scientific_decision"] == "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    assert payload["independent_validation"]["checks_passed"] == 32
    assert payload["independent_validation"]["checks_failed"] == 0


def test_v221_failure_and_equal_cost_control_are_not_hidden() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    case5 = payload["outcomes"]["case5"]
    case2 = payload["outcomes"]["case2"]
    assert case5["matched_strict_safe"] == 0
    assert case2["matched_strict_safe"] == 0
    assert case5["complete_rigs_passed"] == 0
    assert case2["complete_rigs_passed"] == 0
    assert case5["controls"]["direct_low64_k11_complete_rigs"] == 13
    assert payload["primary"]["online_exact_calls"] == {"A": 12, "AT": 11, "total": 23}
    assert payload["adjudication"]["exact_rowspace_lift_route_retained"] is False


def test_v221_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v221：" in text
    assert "# v221:" in text
    assert "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221" in text
    assert "0/546" in text
    assert "0/715" in text
    assert "algorithm_breakthrough=false" in text


def test_v221_figure_is_rendered() -> None:
    assert FIGURE.is_file()
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 650
        assert image.mode == "RGB"


def test_v221_current_pages_and_learning_log_are_synchronized() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["v221_low64_exact_rowspace_lift_scientific_decision"] == "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    assert current["current_decision"]["v221_exact_rowspace_lift_route_closed"] is True
    assert current["current_decision"]["v221_algorithm_breakthrough"] is False
    for page in [ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html"]:
        content = page.read_text(encoding="utf-8")
        assert "v221" in content
        assert "0/13" in content
        assert "algorithm_breakthrough=false" in content
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v221 精确行空间 lift" in log
    assert "v221 exact row-space lift" in log
