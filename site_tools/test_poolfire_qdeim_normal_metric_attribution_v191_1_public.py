from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_qdeim_normal_metric_attribution_v191_1_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v191_1_summary_preserves_attribution_and_claim_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == (
        "PASS_OBSERVATION_ACTIVATED_NORMAL_METRIC_DISTORTION_ATTRIBUTION_V191_1"
    )
    assert payload["original_independent_status"].startswith("INCONCLUSIVE_")
    assert payload["repair_status"] == "PASS_NUMERIC_COMPARATOR_REPAIR_V191_1"
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["deployable_algorithm"] is False
    assert payload["claim_limits"]["observation_adaptive_mechanism_constructed"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_v191_1_mixed_setups_rule_out_one_geometry_only_scalar() -> None:
    payload = _payload()
    outcomes = payload["setup_outcomes"]
    assert outcomes["mixed_setups_total"] == 21
    assert outcomes["mixed_setups_total_denominator"] == 26
    assert outcomes["five_camera"]["mixed_pass_fail_setups"] == 10
    assert outcomes["all_nine"]["mixed_pass_fail_setups"] == 11
    assert len(outcomes["five_camera"]["setup_frame_pass_counts"]) == 13
    assert len(outcomes["all_nine"]["setup_frame_pass_counts"]) == 13


def test_v191_1_normal_metric_diagnostics_support_attribution() -> None:
    diagnostics = _payload()["normal_metric_diagnostics"]
    assert diagnostics["every_failed_cell_coordinate_delta_relative_above_1e_8"] is True
    assert diagnostics["every_failed_cell_full_normal_defect_above_1e_8"] is True
    assert diagnostics["every_mixed_setup_within_setup_coordinate_delta_spread_above_1e_8"] is True
    assert diagnostics["discarded_delta_energy_fraction_p50"]["five_camera"] > 0.9
    assert diagnostics["discarded_delta_energy_fraction_p50"]["all_nine"] > 0.9
    assert diagnostics["selected_stationarity_independent_maximum"] < 1e-7


def test_v191_1_repair_is_static_and_original_inconclusive_is_preserved() -> None:
    independent = _payload()["independent_recomputation"]
    assert independent["original_output_preserved"] is True
    assert independent["original_scientific_arrays_read_once"] is True
    assert independent["repair_changed_physics_or_scientific_arrays"] is False
    assert independent["repair_checks_passed"] == 15
    assert independent["repair_checks_total"] == 15
    assert independent["selected_stationarity_max_absolute_difference"] < 1e-8
    assert independent["energy_identity_max_absolute_difference"] < 1e-10


def test_v191_1_public_assets_and_bilingual_claims_exist() -> None:
    result = (ROOT / "docs/poolfire_qdeim_normal_metric_attribution_v191_1_result_2026-08-22.md").read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v191.1：" in result and "# v191.1:" in result
    assert "PASS_OBSERVATION_ACTIVATED_NORMAL_METRIC_DISTORTION_ATTRIBUTION_V191_1" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert all("v191.1" in content for content in (daily, focus, root, log))
    figure = ROOT / "assets/figures/poolfire_qdeim_normal_metric_attribution_v191_1.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v191_1_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_qdeim_normal_metric_attribution_v191_1_result_2026-08-22.md",
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ]
    forbidden = ("/" + "Users" + "/", "private" + "_results", "private" + "_worktrees")
    private_receipt = re.compile(r"SCIENTIFIC_DECISION_[A-Z0-9_]+\.json")
    commit_hash = re.compile(r"\b[0-9a-f]{40}\b")
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in forbidden), path
        assert private_receipt.search(content) is None, path
        assert commit_hash.search(content) is None, path


def test_current_evidence_points_to_v191_1_without_overclaiming() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["scientific_status"] == (
        "FAIL_LOCAL_RAY_COVERAGE_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V211"
    )
    assert current["engineering_status"] == "PASS_INDEPENDENT_RECOMPUTATION_LOCAL_RAY_COVERAGE_ATTRIBUTION_V211"
    assert current["metrics"]["v191_mixed_setup_count"] == 21
    assert current["metrics"]["v191_total_setup_count"] == 26
    assert current["current_decision"]["v191_observation_activated_metric_attribution_passed"] is True
    assert current["current_decision"]["v191_observation_adaptive_mechanism_constructed"] is False
    assert current["current_decision"]["v191_algorithm_breakthrough"] is False
