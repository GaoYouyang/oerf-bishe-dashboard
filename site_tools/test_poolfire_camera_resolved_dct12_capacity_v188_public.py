from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_camera_resolved_dct12_capacity_v188_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v188_summary_preserves_negative_scientific_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "FAIL_CAMERA_RESOLVED_DCT12_CAPACITY_V188"
    assert payload["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_CAMERA_RESOLVED_DCT12_V188_1"
    )
    assert payload["independent_recomputation"]["checks_passed"] == 44
    assert payload["independent_recomputation"]["checks_total"] == 44
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_v188_camera_resolved_dct12_fails_both_sensor_arms() -> None:
    payload = _payload()
    for stage in ("primary_k0", "primary_k1"):
        for arm in ("five_camera", "all_nine"):
            assert payload[stage][arm]["passed"] is False
            assert payload[stage][arm]["complete_time_strata_safe"] == 0
    assert payload["primary_k1"]["five_camera"]["strict_cells_safe"] == 2
    assert payload["primary_k1"]["all_nine"]["strict_cells_safe"] == 0
    assert payload["primary_k1"]["five_camera"]["observation_p90"] > 0.2
    assert payload["primary_k1"]["all_nine"]["field_p90"] > 0.5


def test_v188_attribution_is_quantified_without_overclaiming() -> None:
    payload = _payload()
    comparison = payload["comparison_to_v187_1"]
    assert payload["mechanism"]["cross_camera_pooling"] is False
    assert comparison["five_camera_numerically_unchanged"] is True
    assert comparison["all_nine_field_p90_reduction_fraction"] > 0.66
    assert comparison["all_nine_gradient_p90_reduction_fraction"] > 0.70
    assert comparison["all_nine_observation_p90_reduction_fraction"] > 0.66
    assert "Close DCT12 truncation" in payload["route_action"]
    assert "dense per-camera" in payload["next_eligible_question"]


def test_v188_public_assets_and_bilingual_claims_exist() -> None:
    result = (
        ROOT / "docs/poolfire_camera_resolved_dct12_capacity_v188_result_2026-08-22.md"
    ).read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v188：" in result and "# v188:" in result
    assert "FAIL_CAMERA_RESOLVED_DCT12_CAPACITY_V188" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert all("v188" in content for content in (daily, focus, root, log))
    figure = ROOT / "assets/figures/poolfire_camera_resolved_dct12_capacity_v188.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v188_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_camera_resolved_dct12_capacity_v188_result_2026-08-22.md",
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ]
    forbidden = ("/" + "Users" + "/", "private" + "_results")
    private_receipt = re.compile(r"SCIENTIFIC_DECISION_[A-Z0-9_]+\.json")
    commit_hash = re.compile(r"\b[0-9a-f]{40}\b")
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in forbidden), path
        assert private_receipt.search(content) is None, path
        assert commit_hash.search(content) is None, path


def test_current_evidence_points_to_v188_without_overclaiming() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["scientific_status"] == "FAIL_SIGNED_LINE_CANCELLATION_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V212"
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_SIGNED_LINE_CANCELLATION_ATTRIBUTION_V212"
    )
    assert current["metrics"]["v188_primary_k1_five_strict_safe_count"] == 2
    assert current["metrics"]["v188_primary_k1_all_nine_strict_safe_count"] == 0
    assert current["current_decision"]["v188_camera_pooling_is_sole_bottleneck"] is False
    assert current["current_decision"]["v188_dct12_representation_closed"] is True
    assert current["current_decision"]["v188_algorithm_breakthrough"] is False
