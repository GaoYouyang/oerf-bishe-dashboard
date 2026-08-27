from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_haar_irls_null_line_attribution_v272_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_haar_irls_null_line_attribution_v272_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_haar_irls_null_line_attribution_v272.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PAGES = (
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
)
LOG = ROOT / "docs/operator_3d_learning_log.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v272_preserves_mixed_fail_closed_adjudication() -> None:
    data = _summary()
    result = data["formal_result"]
    audit = data["independent_validation"]
    assert result["observation_null_lines_within_limit"] == 13
    assert result["independent_endpoints_with_strict_inward_descent"] == 13
    assert result["formal_endpoints_with_strict_inward_descent"] == 0
    assert (audit["checks_passed"], audit["checks_total"]) == (14, 14)
    assert audit["discrete_decision_exact_match"] is True
    assert data["adjudication"]["status"] == "MIXED_OR_NEAR_FLAT_CASE19_HAAR_IRLS_NULL_LINE_V272"


def test_v272_keeps_all_success_claims_false() -> None:
    data = _summary()
    assert data["adjudication"]["fixed_haar_irls_reference_reopened"] is False
    assert data["adjudication"]["rerun_relaxation_or_tuning_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v272_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v272：" in text and "# v272:" in text
    assert "13/13" in text and "0/13" in text and "14/14" in text
    assert "MIXED_OR_NEAR_FLAT" in text
    assert "algorithm_breakthrough=false" in text


def test_v272_figure_is_public_and_readable() -> None:
    assert FIGURE.exists() and FIGURE.stat().st_size > 45_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2000
        assert image.height >= 900


def test_v272_remains_historical_after_v273() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["headline"].startswith("v275")
    assert current["current_decision"]["v272_independent_validation_passed"] is True
    assert current["public_evidence"]["figure"].endswith("v275.png")
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        assert "blastnet_case19_haar_irls_null_line_attribution_v272" in text
        assert "blastnet_case19_loro_fusion_reference_v274_2" in text
    assert "## 2026-08-27：v272" in LOG.read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-27"') == 1


def test_v272_public_artifacts_exclude_private_execution_details() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (SUMMARY, RESULT, *PAGES))
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "source-private",
        "FORMAL_EXIT_CODE",
        "VALIDATION_EXIT_CODE",
    )
    assert all(token not in text for token in forbidden)
