from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_haar_irls_reference_v271_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_haar_irls_reference_v271_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_haar_irls_reference_v271.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PAGES = (ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html")
LOG = ROOT / "docs/operator_3d_learning_log.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v271_preserves_fail_closed_independent_adjudication() -> None:
    data = _summary()
    audit = data["independent_validation"]
    assert data["formal_validation"]["absolute_pass_cells"] == 13
    assert audit["independent_absolute_pass_cells"] == 13
    assert audit["passed"] is False
    assert (audit["checks_passed"], audit["checks_total"]) == (24, 31)
    assert audit["maximum_field_relative_difference"] > audit["cross_implementation_limit"]
    assert audit["maximum_metric_absolute_difference"] > audit["cross_implementation_limit"]
    assert audit["maximum_normalized_residual_difference"] < audit["cross_implementation_limit"]


def test_v271_closes_fixed_reference_without_claims() -> None:
    data = _summary()
    decision = data["adjudication"]
    assert decision["status"] == "INCONCLUSIVE_INVALID_CASE19_HAAR_IRLS_REFERENCE_V271"
    assert decision["fixed_haar_irls_reference_advanced"] is False
    assert decision["rerun_or_tolerance_relaxation_authorized"] is False
    assert decision["solver_round_epsilon_or_regularization_tuning_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v271_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v271：" in text and "# v271:" in text
    assert "13/13" in text and "24/31" in text and "2e-5" in text
    assert "假阳性" in text and "false positive" in text
    assert "algorithm_breakthrough=false" in text


def test_v271_figure_is_public_and_readable() -> None:
    assert FIGURE.exists() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2000
        assert image.height >= 900


def test_v271_remains_historical() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["headline"].startswith("v280")
    assert current["current_decision"]["v271_independent_validation_passed"] is False
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        assert "blastnet_case19_loro_fusion_reference_v274_2" in text
        assert "blastnet_case19_haar_irls_reference_v271" in text
        assert "blastnet_case19_geometry_normal_sgs_v270" in text
    assert "## 2026-08-27：v271" in LOG.read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-27"') == 1


def test_v271_public_artifacts_exclude_private_execution_details() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (SUMMARY, RESULT, *PAGES))
    forbidden = ("/Users/", "/Volumes/", "private_results", "source-private", "FORMAL_EXIT_CODE", "VALIDATOR_EXIT_CODE")
    assert all(token not in text for token in forbidden)
