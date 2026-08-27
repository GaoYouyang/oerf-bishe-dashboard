from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_geometry_normal_sgs_v270_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_geometry_normal_sgs_v270_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_geometry_normal_sgs_v270.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PAGES = (
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
)
LOG = ROOT / "docs/operator_3d_learning_log.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v270_summary_preserves_reference_first_adjudication() -> None:
    data = _summary()
    reference = data["reference_adequacy"]
    diagnostic = data["diagnostic_results"]
    assert data["formal_validation"] == {"passed": True, "checks_passed": 21, "checks_total": 21}
    assert data["independent_validation"]["passed"] is True
    assert (data["independent_validation"]["checks_passed"], data["independent_validation"]["checks_total"]) == (32, 32)
    assert reference["absolute_pass_cells"] == 12
    assert reference["failing_metric"] == "interior_gradient"
    assert reference["failing_value"] > reference["frozen_limit"]
    assert diagnostic["authoritative_mechanism_adjudication"] is False
    assert diagnostic["primary_absolute_pass_cells"] == 0
    assert diagnostic["primary_matched_pass_cells"] == 0


def test_v270_adjudication_blocks_claims_and_retuning() -> None:
    data = _summary()
    decision = data["adjudication"]
    assert decision["status"].startswith("INCONCLUSIVE_REFERENCE_INADEQUATE_CASE19")
    assert decision["reference_inadequacy_is_authoritative"] is True
    assert decision["primary_counts_are_diagnostic_only"] is True
    assert decision["retuning_or_deeper_same_reference_authorized"] is False
    assert decision["full_sequence_gate_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v270_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v270：" in text and "# v270:" in text
    assert "21/21" in text and "32/32" in text and "12/13" in text
    assert "0.758223" in text and "0.750000" in text
    assert "diagnostic" in text and "algorithm_breakthrough=false" in text


def test_v270_figure_is_public_and_readable() -> None:
    assert FIGURE.exists() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2000
        assert image.height >= 900


def test_v270_remains_historical() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["headline"].startswith("v274.2")
    assert current["metrics"]["v270_independent_checks_passed"] == 32
    assert current["current_decision"]["v270_reference_inadequate"] is True
    assert current["public_evidence"]["figure"].endswith("v274_2.png")
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        assert "blastnet_case19_geometry_normal_sgs_v270" in text
    assert "## 2026-08-27：v270" in LOG.read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-27"') == 1
    assert "v269.1" in daily


def test_v270_public_artifacts_exclude_private_execution_details() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (SUMMARY, RESULT, *PAGES))
    forbidden = (
        "/Users/",
        "/Volumes/",
        "source-private",
        "FORMAL_EXIT_CODE",
        "VALIDATOR_EXIT_CODE",
        "6c28b372",
    )
    assert all(token not in text for token in forbidden)
