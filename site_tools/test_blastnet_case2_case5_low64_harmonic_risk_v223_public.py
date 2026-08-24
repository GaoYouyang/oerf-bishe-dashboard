"""Public evidence checks for the sealed v223 scalar-overlap diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_harmonic_risk_v223_public_summary.json"
RESULT = ROOT / "docs/blastnet_case2_case5_low64_harmonic_risk_v223_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case2_case5_low64_harmonic_risk_v223.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v223_counts_and_post_open_role() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    scope = payload["scope"]
    assert (scope["cells"], scope["safe_cells"], scope["unsafe_cells"]) == (1261, 1064, 197)
    assert scope["diagnostic_type"] == "post-open mechanism-capacity diagnostic"
    assert scope["deployment_policy_established"] is False
    assert scope["truth_used_for_safety_labels"] is True


def test_v223_primary_and_control_overlap() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_harmonic_observability"]
    control = payload["cheap_fit_residual_control"]
    assert primary["safe_minimum"] < primary["unsafe_maximum"]
    assert primary["strict_separation_margin"] < 0
    assert control["unsafe_minimum"] < control["safe_maximum"]
    assert control["strict_separation_margin"] < 0
    assert primary["threshold_exists"] is False
    assert control["threshold_exists"] is False


def test_v223_independent_validation_and_claim_limits() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = payload["independent_validation"]
    assert validation["status"] == "PASS_INDEPENDENT_RECOMPUTATION_LOW64_HARMONIC_RISK_V223"
    assert validation["scientific_decision"] == "FAIL_LOW64_HARMONIC_RISK_OVERLAP_V223"
    assert validation["maximum_formal_independent_feature_difference"] < 2e-14
    assert validation["end_to_end_physics_independence_proven"] is False
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v223_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v223：" in text and "# v223:" in text
    assert "FAIL_LOW64_HARMONIC_RISK_OVERLAP_V223" in text
    assert "1064" in text and "197" in text
    assert "-0.233392" in text and "-0.194502" in text
    assert "algorithm_breakthrough=false" in text


def test_v223_figure_is_rendered() -> None:
    assert FIGURE.is_file()
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 650
        assert image.mode == "RGB"


def test_v223_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_LOW64_HARMONIC_RISK_OVERLAP_V223"
    assert current["current_decision"]["v223_one_dimensional_harmonic_risk_route_closed"] is True
    assert current["current_decision"]["v223_algorithm_breakthrough"] is False
    for page in [ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html"]:
        content = page.read_text(encoding="utf-8")
        assert "v223" in content
        assert "-0.233392" in content
        assert "algorithm_breakthrough=false" in content
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v223 一维可观测调和风险" in log
    assert "v223 one-dimensional observable harmonic risk" in log
