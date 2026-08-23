from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/real_bost_whole_field_time_isolated_selector_v172_public_summary.json"
)
RESULT = (
    ROOT / "docs/real_bost_whole_field_time_isolated_selector_v172_result_2026-08-21.md"
)
FIGURE = ROOT / "assets/figures/real_bost_whole_field_time_isolated_selector_v172.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_the_scientific_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == (
        "PASS_WHOLE_FIELD_TIME_ISOLATED_GEOMETRY_SELECTOR_HEADROOM_V172"
    )
    assert payload["outer_split"]["fold_count"] == 468
    assert payload["outer_split"]["heldout_axes_available_to_prediction"] is False
    assert payload["selector"]["strict_cell_safe_count"] == 468
    assert payload["selector"]["whole_calibrations_safe_count"] == 13
    assert payload["selector"]["whole_field_models_safe_count"] == 9
    assert payload["selector"]["time_strata_passed"] == 4
    assert payload["controls"]["fit_static"]["strict_cell_safe_count"] == 323
    assert payload["controls"]["v169_fixed_geometry"]["strict_cell_safe_count"] == 192
    assert payload["independent_recomputation"]["check_count"] == 22
    assert (
        payload["independent_recomputation"][
            "heldout_axis_mutation_target_maximum_absolute_difference"
        ]
        == 0.0
    )
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert (
        payload["claim_limits"][
            "full_observation_geometry_warm_initializer_established"
        ]
        is False
    )
    assert payload["claim_limits"]["resource_speedup"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v172：几何选择器通过整场与时间三重隔离" in text
    assert "# v172: the geometry selector passes whole-field and time isolation" in text
    assert "468/468" in text
    assert "9/9" in text
    assert "22/22" in text
    assert "algorithm_breakthrough=false" in text
    assert "没有完成 observation-only warm initializer" in text
    assert (
        "does not yet establish the complete observation-only warm initializer" in text
    )


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_preserves_v172_and_points_to_v181() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == (
        "PASS_OBSERVATION_ONLY_SPECTRAL_ALIGNMENT_PROXY_STRICTLY_SEPARATES_CASE5_REFERENCE_V214"
    )
    assert payload["metrics"]["v172_primary_strict_safe_count"] == 468
    assert payload["metrics"]["v172_primary_complete_fields_passed"] == 9
    assert payload["metrics"]["v172_independent_check_count"] == 22
    assert "v214 observation-only proxy field" in payload["next_scientific_gate_en"]
    assert "real-BOST gates remain closed" in payload["next_scientific_gate_en"]
    assert "真实 BOST 门仍关闭" in payload["next_scientific_gate_zh"]
    assert "真实 BOST" in payload["next_scientific_gate_zh"]


def test_primary_pages_reference_v172_in_both_languages() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily, home):
        assert "v172" in text
    assert "整场与时间三重隔离" in operator
    assert "whole-field and time isolation" in operator
    assert (
        "real_bost_whole_field_time_isolated_selector_v172_result_2026-08-21.md"
        in operator
    )
    assert (
        daily.count("PASS_WHOLE_FIELD_TIME_ISOLATED_GEOMETRY_SELECTOR_HEADROOM_V172")
        == 1
    )


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "9ef8813f",
        "candidate_metrics.npy",
        "independent_candidate_metrics.npy",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
