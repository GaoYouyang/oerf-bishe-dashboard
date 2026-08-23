"""Public evidence checks for the independently sealed v213 result."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case5_source_weighted_observability_v213_public_summary.json"
)
RESULT = (
    ROOT
    / "docs/blastnet_case5_source_weighted_observability_v213_result_2026-08-24.md"
)
FIGURE = (
    ROOT
    / "assets/figures/blastnet_case5_source_weighted_observability_v213.png"
)
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_v213_summary_preserves_the_strict_positive_mechanism_decision() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["scientific_decision"] == (
        "PASS_ACTUAL_SOURCE_ALIGNMENT_STRICTLY_SEPARATES_CASE5_REFERENCE_V213"
    )
    primary = summary["fixed_primary"]
    assert primary["comparison_count"] == 169
    assert primary["virtual_strictly_greater_count"] == 169
    assert primary["tie_count"] == 0
    assert primary["strict_family_separation"] is True
    assert primary["strict_gap_virtual_min_minus_supplied_max"] > 0.28
    assert summary["source_blind_control"]["virtual_strictly_greater_count"] == 167
    assert summary["source_blind_control"]["strict_family_separation"] is False


def test_v213_summary_preserves_truth_aware_scope_and_false_claims() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    scope = summary["scope"]
    assert scope["opened_density_frames"] == 42
    assert scope["geometry_rows"] == 39
    assert scope["fixed_low_modes"] == 64
    assert scope["trainable_parameters"] == 0
    assert scope["density_reads"] == 42
    assert scope["observation_reads"] == 0
    assert scope["reconstruction_reads"] == 0
    assert summary["independent_validation"]["all_checks_passed"] is True
    assert summary["independent_validation"]["check_count"] == 19
    assert all(value is False for value in summary["claims_fixed_false"].values())


def test_v213_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v213：" in text
    assert "# v213:" in text
    assert "169/169" in text
    assert "algorithm_breakthrough=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 600
        extrema = image.convert("RGB").getextrema()
        assert any(low != high for low, high in extrema)


def test_v213_remains_preserved_as_parent_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-24"
    assert current["metrics"]["v213_primary_comparison_count"] == 169
    assert current["metrics"]["v213_primary_strictly_greater_count"] == 169
    assert current["current_decision"]["v213_truth_aware_mechanism_supported"] is True
    assert current["current_decision"]["v213_deployable_proxy_established"] is False
    assert current["current_decision"]["v213_predictor_authorized"] is False


def test_primary_pages_reference_v213_in_both_languages() -> None:
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "blastnet_case5_source_weighted_observability_v213" in content
        assert "v213" in content


def test_v213_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "checkpoint",
        "release_75aaf419",
        "f200f294",
        "446d6987",
    )
    for path in (SUMMARY, RESULT):
        content = path.read_text(encoding="utf-8")
        assert all(token not in content for token in forbidden)
