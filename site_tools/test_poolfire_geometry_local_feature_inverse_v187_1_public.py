from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_geometry_local_feature_inverse_v187_1_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v187_1_summary_preserves_negative_scientific_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "FAIL_GEOMETRY_LOCAL_FEATURE_CAPACITY_V187_1"
    assert payload["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_GEOMETRY_LOCAL_FEATURE_INVERSE_V187_1"
    )
    assert payload["independent_recomputation"]["checks_passed"] == 40
    assert payload["independent_recomputation"]["checks_total"] == 40
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["real_bost"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_v187_1_geometry_local_inverse_fails_both_sensor_arms() -> None:
    payload = _payload()
    for stage in ("primary_k0", "primary_k1"):
        for arm in ("five_camera", "all_nine"):
            assert payload[stage][arm]["passed"] is False
            assert payload[stage][arm]["strict_cells_safe"] < 52
            assert payload[stage][arm]["complete_time_strata_safe"] == 0
    assert payload["primary_k1"]["five_camera"]["strict_cells_safe"] == 2
    assert payload["primary_k1"]["all_nine"]["strict_cells_safe"] == 0
    assert payload["primary_k1"]["five_camera"]["observation_p90"] > 0.2
    assert payload["primary_k1"]["all_nine"]["field_p90"] > 2.0


def test_v187_1_attribution_and_cost_boundary_are_explicit() -> None:
    payload = _payload()
    assert payload["mechanism"]["shared_cross_geometry_weights"] is False
    assert payload["conditioning_diagnostics"]["retained_rank_minimum"] == 715
    assert payload["conditioning_diagnostics"]["condition_maximum"] > 6.0e7
    assert payload["cost_boundary"]["dense_setup_response_matrices_required"] == 26
    assert payload["cost_boundary"]["exact_call_reduction_established"] is False
    assert "Close both shared and setup-local inverses" in payload["route_action"]
    assert "camera-resolved-versus-pooled" in payload["next_eligible_question"]


def test_v187_1_public_assets_and_bilingual_claims_exist() -> None:
    result = (
        ROOT / "docs/poolfire_geometry_local_feature_inverse_v187_1_result_2026-08-22.md"
    ).read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v187.1：" in result and "# v187.1:" in result
    assert "FAIL_GEOMETRY_LOCAL_FEATURE_CAPACITY_V187_1" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert all("v187.1" in content for content in (daily, focus, root, log))
    figure = ROOT / "assets/figures/poolfire_geometry_local_feature_inverse_v187_1.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v187_1_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_geometry_local_feature_inverse_v187_1_result_2026-08-22.md",
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

    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    section = log.split("## 2026-08-22：v187.1", 1)[1]
    assert not any(token in section for token in forbidden)
    assert private_receipt.search(section) is None
    assert commit_hash.search(section) is None


def test_current_evidence_preserves_v187_1_history_without_overclaiming() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["v187_1_geometry_local_feature_inverse_scientific_decision"] == (
        "FAIL_GEOMETRY_LOCAL_FEATURE_CAPACITY_V187_1"
    )
    assert current["v187_1_geometry_local_feature_inverse_independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_GEOMETRY_LOCAL_FEATURE_INVERSE_V187_1"
    )
    assert current["metrics"]["v187_1_primary_k1_five_strict_safe_count"] == 2
    assert current["metrics"]["v187_1_primary_k1_all_nine_strict_safe_count"] == 0
    assert current["current_decision"]["v187_1_pooled_feature_map_closed"] is True
    assert current["current_decision"]["v187_1_dense_v185_capacity_remains_valid"] is True
    assert current["current_decision"]["v187_1_predictor_training_authorized"] is False
    assert current["current_decision"]["v187_1_algorithm_breakthrough"] is False
