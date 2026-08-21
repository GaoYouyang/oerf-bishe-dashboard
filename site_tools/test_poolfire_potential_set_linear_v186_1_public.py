from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_potential_set_linear_v186_1_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v186_1_public_summary_preserves_negative_scientific_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "FAIL_POTENTIAL_SET_LINEAR_V186_1_1"
    assert payload["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_SET_LINEAR_V186_1_1"
    )
    assert payload["independent_recomputation"]["checks_passed"] == 44
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["real_bost"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_v186_1_primary_fails_both_sensor_arms_before_and_after_k1() -> None:
    payload = _payload()
    for stage in ("primary_k0", "primary_k1"):
        for arm in ("five_camera", "all_nine"):
            assert payload[stage][arm]["passed"] is False
            assert payload[stage][arm]["strict_cells_safe"] < 52
            assert payload[stage][arm]["complete_frames_safe"] == 0
    assert payload["primary_k1"]["five_camera"]["strict_cells_safe"] == 39
    assert payload["primary_k1"]["all_nine"]["strict_cells_safe"] == 25
    assert max(payload["primary_k1"]["five_camera"]["observation_p90_by_time"]) > 0.2
    assert min(payload["primary_k1"]["all_nine"]["observation_p90_by_time"]) > 0.2


def test_v186_1_controls_cost_and_closure_remain_visible() -> None:
    payload = _payload()
    controls = payload["blocking_controls"]
    assert controls["geometry_blind_dct12_k1_five_strict_cells_safe"] == 0
    assert controls["one_direction_potential_coordinate_k1_all_nine_strict_cells_safe"] == 0
    assert controls["fit_mean_k1_five_strict_cells_safe"] == 0
    assert payload["cost_boundary"]["logical_online_k1_exact_vector_forward"] == 2
    assert payload["cost_boundary"]["logical_online_k1_exact_vector_adjoint"] == 1
    assert payload["cost_boundary"]["exact_call_reduction_established"] is False
    assert "Close the current DCT12" in payload["route_action"]


def test_v186_1_public_assets_and_bilingual_claims_exist() -> None:
    result = (
        ROOT / "docs/poolfire_potential_set_linear_v186_1_result_2026-08-22.md"
    ).read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    assert "v186.1：" in result and "# v186.1:" in result
    assert "FAIL_POTENTIAL_SET_LINEAR_V186_1_1" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert "v186.1" in daily and "v186.1" in focus
    figure = ROOT / "assets/figures/poolfire_potential_set_linear_v186_1.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v186_1_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_potential_set_linear_v186_1_result_2026-08-22.md",
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


def test_current_evidence_points_to_v186_1_without_overclaiming() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["scientific_status"] == "FAIL_POTENTIAL_SET_LINEAR_V186_1_1"
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_SET_LINEAR_V186_1_1"
    )
    assert current["metrics"]["v186_1_primary_k1_five_strict_safe_count"] == 39
    assert current["metrics"]["v186_1_primary_k1_all_nine_strict_safe_count"] == 25
    assert current["current_decision"]["v186_1_shared_linear_representation_closed"] is True
    assert current["current_decision"]["v186_1_predictor_training_authorized"] is False
    assert current["current_decision"]["v186_1_algorithm_breakthrough"] is False
