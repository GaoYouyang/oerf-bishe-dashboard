"""Public evidence checks for the sealed v220.2 inconclusive adjudication."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_observable_fallback_v220_2_public_summary.json"
RESULT = ROOT / "docs/blastnet_case2_case5_observable_fallback_v220_2_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case2_case5_observable_fallback_v220_2.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v220_2_summary_preserves_inconclusive_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = payload["independent_validation"]
    assert validation["scientific_decision"] == "INCONCLUSIVE_INVALID_OBSERVABLE_FALLBACK_V220_2"
    assert validation["nominal_decision_exact"] is True
    assert validation["failed_checks"]["formal_field_relative_difference"]["observed"] > 1e-8
    assert validation["failed_checks"]["camera_permutation_field_relative_difference"]["observed"] > 1e-8
    assert payload["adjudication"]["tolerance_changed_after_results"] is False


def test_v220_2_nominal_cross_condition_failure_is_not_hidden() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    nominal = payload["nominal_recomputation"]
    assert nominal["case5"]["complete_rigs_passed"] == 13
    assert nominal["case2"]["complete_rigs_passed"] == 0
    assert nominal["case2"]["matched_cells_passed"] == 629
    assert payload["adjudication"]["observable_fallback_route_retained"] is False
    assert payload["claims_fixed_false"]["algorithm_breakthrough"] is False
    assert payload["claims_fixed_false"]["resource_speedup"] is False


def test_v220_2_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v220.2：" in text
    assert "# v220.2:" in text
    assert "INCONCLUSIVE_INVALID_OBSERVABLE_FALLBACK_V220_2" in text
    assert "0/13" in text
    assert "1.50948e-8" in text
    assert "algorithm_breakthrough=false" in text


def test_v220_2_figure_is_rendered() -> None:
    assert FIGURE.is_file()
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 650
        assert image.mode == "RGB"


def test_v220_2_historical_pages_and_learning_log_remain_synchronized() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["v220_2_observable_fallback_scientific_decision"] == "INCONCLUSIVE_INVALID_OBSERVABLE_FALLBACK_V220_2"
    assert current["current_decision"]["v220_2_observable_fallback_route_closed"] is True
    assert current["current_decision"]["v220_2_algorithm_breakthrough"] is False
    for page in [ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html"]:
        content = page.read_text(encoding="utf-8")
        assert "v220.2" in content
        assert "0/13" in content
        assert "algorithm_breakthrough=false" in content
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v220.2 不放宽数值门" in log
    assert "v220.2 does not loosen" in log
