"""Public evidence checks for the independently sealed v214 result."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case5_observation_spectral_proxy_v214_public_summary.json"
RESULT = ROOT / "docs/blastnet_case5_observation_spectral_proxy_v214_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case5_observation_spectral_proxy_v214.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v214_summary_preserves_the_strict_positive_proxy_decision() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["scientific_decision"] == (
        "PASS_OBSERVATION_ONLY_SPECTRAL_ALIGNMENT_PROXY_STRICTLY_SEPARATES_CASE5_REFERENCE_V214"
    )
    primary = summary["fixed_primary"]
    assert primary["comparison_count"] == 169
    assert primary["virtual_strictly_greater_count"] == 169
    assert primary["tie_count"] == 0
    assert primary["strict_family_separation"] is True
    assert primary["strict_gap_virtual_min_minus_supplied_max"] > 0.39
    assert summary["source_blind_control"]["virtual_strictly_greater_count"] == 167
    assert summary["source_blind_control"]["strict_family_separation"] is False


def test_v214_summary_preserves_proxy_barrier_cost_and_false_claims() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    scope = summary["scope"]
    assert scope["geometry_frame_observations"] == 1638
    assert scope["geometry_response_probe_forward_equivalents"] == 2496
    assert scope["synthetic_observation_generation_A_equivalents"] == 1638
    assert scope["logical_proxy_calls_after_observation_and_geometry_cache"] == {
        "A": 0,
        "AT": 0,
    }
    assert scope["trainable_parameters"] == 0
    assert summary["independent_validation"]["all_checks_passed"] is True
    assert summary["independent_validation"]["check_count"] == 19
    assert all(value is False for value in summary["claims_fixed_false"].values())


def test_v214_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v214：" in text
    assert "# v214:" in text
    assert "169/169" in text
    assert "algorithm_breakthrough=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 600
        extrema = image.convert("RGB").getextrema()
        assert any(low != high for low, high in extrema)


def test_v214_is_the_current_public_headline() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-24"
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_OBSERVATION_SPECTRAL_PROXY_V214"
    )
    assert current["scientific_status"] == (
        "PASS_OBSERVATION_ONLY_SPECTRAL_ALIGNMENT_PROXY_STRICTLY_SEPARATES_CASE5_REFERENCE_V214"
    )
    assert current["metrics"]["v214_primary_comparison_count"] == 169
    assert current["metrics"]["v214_primary_strictly_greater_count"] == 169
    assert current["current_decision"]["v214_observation_proxy_supported"] is True
    assert current["current_decision"]["v214_deployable_warm_start_established"] is False
    assert current["current_decision"]["v214_resource_gate_authorized"] is False


def test_primary_pages_reference_v214_in_both_languages() -> None:
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "blastnet_case5_observation_spectral_proxy_v214" in content
        assert "v214" in content
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    assert "v214 观测谱代理已独立封存" in focus
    assert "v214 observation spectral proxy independently sealed" in focus
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v214 用当前二维观测复现谱对齐判决" in log
    assert "v214 moves one step closer to deployment" in log


def test_v214_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "checkpoint",
        "release_75aaf419",
        "764137b0",
    )
    for path in (SUMMARY, RESULT):
        content = path.read_text(encoding="utf-8")
        assert all(token not in content for token in forbidden)
