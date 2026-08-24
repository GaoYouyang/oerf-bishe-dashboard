from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_potential_affine_observability_v185_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v185_public_summary_preserves_scientific_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "PASS_POTENTIAL_AFFINE_K1_CAPACITY_V185"
    assert payload["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_AFFINE_OBSERVABILITY_V185"
    )
    assert payload["independent_recomputation"]["checks_passed"] == 32
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["deployable_algorithm"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_v185_primary_k1_passes_both_camera_arms() -> None:
    payload = _payload()
    for camera_arm in ("five_camera", "all_nine"):
        result = payload["primary_k1"][camera_arm]
        assert result["passed"] is True
        assert result["strict_cells_safe"] == 52
        assert result["complete_calibrations_safe"] == 13
        assert result["complete_frames_safe"] == 4
        assert result["field_p90"] <= 0.5
        assert result["gradient_p90"] <= 0.75
        assert result["observation_p90"] <= 0.2
    assert payload["mechanism_diagnostics"]["retained_affine_rank_minimum"] == 1009
    assert payload["mechanism_diagnostics"]["retained_affine_rank_maximum"] == 1009


def test_v185_blocking_control_and_cost_boundary_remain_visible() -> None:
    payload = _payload()
    control = payload["blocking_control"]
    assert control["five_camera_k0_strict_cells_safe"] == 0
    assert control["five_camera_k1_strict_cells_safe"] == 0
    assert control["all_nine_k0_strict_cells_safe"] == 0
    assert control["all_nine_k1_strict_cells_safe"] == 0
    assert payload["cost_boundary"]["potential_transform_rhs_per_sensor_setup"] == 1013
    assert payload["cost_boundary"]["exact_call_reduction_established"] is False
    assert payload["cost_boundary"]["wall_or_rss_measured"] is False


def test_v185_public_assets_and_bilingual_claims_exist() -> None:
    result = (
        ROOT / "docs/poolfire_potential_affine_observability_v185_result_2026-08-22.md"
    ).read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    assert "v185：" in result and "# v185:" in result
    assert "PASS_POTENTIAL_AFFINE_K1_CAPACITY_V185" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert "v185" in daily and "v185" in focus
    figure = ROOT / "assets/figures/poolfire_potential_affine_observability_v185.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v185_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_potential_affine_observability_v185_result_2026-08-22.md",
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ]
    forbidden = (
        "/" + "Users" + "/",
        "private" + "_results",
    )
    private_receipt = re.compile(r"SCIENTIFIC_DECISION_[A-Z0-9_]+\.json")
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in forbidden), path
        assert private_receipt.search(content) is None, path


def test_current_evidence_retains_v185_after_v186_1_advances_the_gate() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["scientific_status"] == "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert current["metrics"]["v185_five_primary_k1_strict_safe_count"] == 52
    assert current["metrics"]["v185_all_nine_primary_k1_strict_safe_count"] == 52
    assert current["current_decision"]["v185_information_capacity_passed"] is True
    assert current["current_decision"]["v185_compact_predictor_authorized"] is True
    assert current["current_decision"]["v185_exact_call_reduction_established"] is False
    assert current["current_decision"]["v185_algorithm_breakthrough"] is False
