from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_dense_camera_resolved_dct_capacity_v189_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v189_summary_preserves_positive_attribution_and_claim_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "PASS_DCT12_TRUNCATION_ROOT_CAUSE_V189"
    assert payload["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_DENSE_CAMERA_RESOLVED_DCT_V189_1"
    )
    assert payload["independent_recomputation"]["checks_passed"] == 50
    assert payload["independent_recomputation"]["checks_total"] == 50
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["deployable_compact_algorithm"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_v189_full_dct_restores_both_k1_sensor_arms() -> None:
    payload = _payload()
    assert payload["mechanism"]["modes_per_camera"] == 575
    assert payload["mechanism"]["five_camera_feature_count"] == 2875
    assert payload["mechanism"]["all_nine_feature_count"] == 5175
    for arm in ("five_camera", "all_nine"):
        result = payload["primary_k1"][arm]
        assert result["passed"] is True
        assert result["strict_cells_safe"] == 52
        assert result["complete_calibrations_safe"] == 13
        assert result["complete_time_strata_safe"] == 4
        assert result["field_p90"] <= 0.5
        assert result["gradient_p90"] <= 0.75
        assert result["observation_p90"] <= 0.2


def test_v189_attribution_matches_v185_without_relabeling_deployment() -> None:
    payload = _payload()
    comparison = payload["comparison_to_v188"]
    equivalence = payload["v185_dense_equivalence"]
    assert comparison["five_camera_k1_strict_safe_before"] == 2
    assert comparison["five_camera_k1_strict_safe_after"] == 52
    assert comparison["all_nine_k1_strict_safe_before"] == 0
    assert comparison["all_nine_k1_strict_safe_after"] == 52
    assert equivalence["passed"] is True
    assert equivalence["candidate_field_maximum_cell_relative_error"] < 1e-12
    assert payload["cost_boundary"]["compact_storage_established"] is False
    assert payload["cost_boundary"]["wall_or_rss_measured"] is False


def test_v189_public_assets_and_bilingual_claims_exist() -> None:
    result = (
        ROOT / "docs/poolfire_dense_camera_resolved_dct_capacity_v189_result_2026-08-22.md"
    ).read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v189：" in result and "# v189:" in result
    assert "PASS_DCT12_TRUNCATION_ROOT_CAUSE_V189" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert all("v189" in content for content in (daily, focus, root, log))
    figure = ROOT / "assets/figures/poolfire_dense_camera_resolved_dct_capacity_v189.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v189_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_dense_camera_resolved_dct_capacity_v189_result_2026-08-22.md",
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


def test_current_evidence_points_to_v189_without_overclaiming() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["scientific_status"] == "PASS_OBSERVATION_ONLY_SPECTRAL_ALIGNMENT_PROXY_STRICTLY_SEPARATES_CASE5_REFERENCE_V214"
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_OBSERVATION_SPECTRAL_PROXY_V214"
    )
    assert current["metrics"]["v189_primary_k1_five_strict_safe_count"] == 52
    assert current["metrics"]["v189_primary_k1_all_nine_strict_safe_count"] == 52
    assert current["current_decision"]["v189_dct12_truncation_root_cause_supported"] is True
    assert current["current_decision"]["v189_compact_representation_established"] is False
    assert current["current_decision"]["v189_algorithm_breakthrough"] is False
