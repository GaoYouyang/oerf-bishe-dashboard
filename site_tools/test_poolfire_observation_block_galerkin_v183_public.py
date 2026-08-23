from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_observation_block_galerkin_v183_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v183_public_summary_preserves_scientific_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "FAIL_OBSERVATION_BLOCK_GALERKIN_V183"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_OBSERVATION_BLOCK_GALERKIN_V183"
    assert payload["independent_recomputation"]["checks_passed"] == 46
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_v183_block_structure_improves_but_does_not_pass() -> None:
    payload = _payload()
    five = payload["five_camera_primary_k1"]
    nine = payload["all_nine_primary_k1"]
    parent = payload["v182_parent_comparison"]
    gate = payload["absolute_gate"]["observation_p90_max"]
    assert five["global_p90"]["observation_relative_l2"] < parent["five_camera_observation_p90"]
    assert nine["global_p90"]["observation_relative_l2"] < parent["all_nine_observation_p90"]
    assert five["global_p90"]["observation_relative_l2"] > gate
    assert nine["global_p90"]["observation_relative_l2"] > gate
    assert five["strict_cells_safe"] == 1
    assert nine["strict_cells_safe"] == 37
    assert five["passed"] is False
    assert nine["passed"] is False


def test_v183_public_assets_and_bilingual_claims_exist() -> None:
    result = (ROOT / "docs/poolfire_observation_block_galerkin_v183_result_2026-08-21.md").read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    assert "v183：" in result and "# v183:" in result
    assert "FAIL_OBSERVATION_BLOCK_GALERKIN_V183" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert "v183" in daily and "v183" in focus
    figure = ROOT / "assets/figures/poolfire_observation_block_galerkin_v183.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v183_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_observation_block_galerkin_v183_result_2026-08-21.md",
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ]
    forbidden = ("/Users/", "private_results", "SCIENTIFIC_DECISION_V183.json")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_current_evidence_preserves_v183_after_v184_supersedes_latest_status() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["scientific_status"] == "PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    assert current["v183_observation_block_galerkin_scientific_decision"] == (
        "FAIL_OBSERVATION_BLOCK_GALERKIN_V183"
    )
    assert current["v183_observation_block_galerkin_independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_OBSERVATION_BLOCK_GALERKIN_V183"
    )
    assert current["metrics"]["v183_five_primary_k1_strict_safe_count"] == 1
    assert current["metrics"]["v183_all_nine_primary_k1_strict_safe_count"] == 37
    assert current["current_decision"]["v183_camera_component_block_galerkin_family_closed"] is True
    assert current["current_decision"]["v183_exact_call_reduction_established"] is False
    assert current["current_decision"]["v183_algorithm_breakthrough"] is False
