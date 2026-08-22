from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_geometry_qdeim_coreset_capacity_v190_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v190_summary_preserves_negative_decision_and_claim_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "FAIL_GEOMETRY_QDEIM1280_CORESET_CAPACITY_V190"
    assert payload["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_GEOMETRY_QDEIM1280_CORESET_V190_1"
    )
    assert payload["independent_recomputation"]["checks_passed"] == 59
    assert payload["independent_recomputation"]["checks_total"] == 59
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["deployable_compact_algorithm"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_v190_fixed_1280_subset_fails_both_k1_sensor_arms() -> None:
    payload = _payload()
    mechanism = payload["mechanism"]
    assert mechanism["selected_feature_count"] == 1280
    assert mechanism["five_camera_coordinate_reduction_fraction"] > 0.55
    assert mechanism["all_nine_coordinate_reduction_fraction"] > 0.75
    for arm in ("five_camera", "all_nine"):
        result = payload["primary_k1"][arm]
        assert result["passed"] is False
        assert result["strict_cells_safe"] < 52
        assert result["complete_calibrations_safe"] < 13
        assert result["complete_time_strata_safe"] == 0
    assert payload["primary_k1"]["five_camera"]["gradient_p90"] > 0.75
    assert payload["primary_k1"]["all_nine"]["observation_p90"] > 0.2


def test_v190_rank_retention_does_not_get_relabelled_as_capacity() -> None:
    payload = _payload()
    diagnostics = payload["conditioning_diagnostics"]
    comparison = payload["comparison_to_v189"]
    assert diagnostics["five_camera_retained_rank"] == 1009
    assert diagnostics["all_nine_retained_rank"] == 1009
    assert diagnostics["selected_condition_minimum"] > diagnostics["full_dct_condition_minimum"]
    assert diagnostics["selected_condition_maximum"] > diagnostics["full_dct_condition_maximum"]
    assert comparison["v189_five_camera_k1_strict_safe"] == 52
    assert comparison["v189_all_nine_k1_strict_safe"] == 52
    assert comparison["v190_five_camera_k1_strict_safe"] == 35
    assert comparison["v190_all_nine_k1_strict_safe"] == 30
    assert payload["cost_boundary"]["coordinate_count_reduced"] is True
    assert payload["cost_boundary"]["wall_or_rss_measured"] is False


def test_v190_public_assets_and_bilingual_claims_exist() -> None:
    result = (
        ROOT / "docs/poolfire_geometry_qdeim_coreset_capacity_v190_result_2026-08-22.md"
    ).read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v190：" in result and "# v190:" in result
    assert "FAIL_GEOMETRY_QDEIM1280_CORESET_CAPACITY_V190" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert all("v190" in content for content in (daily, focus, root, log))
    figure = ROOT / "assets/figures/poolfire_geometry_qdeim_coreset_capacity_v190.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v190_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_geometry_qdeim_coreset_capacity_v190_result_2026-08-22.md",
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


def test_current_evidence_preserves_v190_historical_boundary() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["metrics"]["v190_primary_k1_five_strict_safe_count"] == 35
    assert current["metrics"]["v190_primary_k1_all_nine_strict_safe_count"] == 30
    assert current["current_decision"]["v190_fixed_qdeim1280_family_closed"] is True
    assert current["current_decision"]["v190_algorithm_breakthrough"] is False
