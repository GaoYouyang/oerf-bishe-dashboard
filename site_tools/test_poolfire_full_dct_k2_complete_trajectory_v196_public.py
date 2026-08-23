from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_full_dct_k2_complete_trajectory_v196_public_summary.json"


def load_summary() -> dict[str, object]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v196_dense_k2_passes_but_reference_is_inadequate() -> None:
    payload = load_summary()
    arms = payload["arms"]
    dense = arms["full_dct_k2"]
    reference = arms["zero_cgls_k4_reference"]
    for sensor in ("five_camera", "all_nine"):
        assert dense[sensor]["strict_safe_cells"] == 1313
        assert dense[sensor]["complete_calibration_groups_passed"] == 13
        assert dense[sensor]["passed"] is True
        assert reference[sensor]["strict_safe_cells"] == 0
        assert reference[sensor]["complete_calibration_groups_passed"] == 0
        assert reference[sensor]["passed"] is False
    assert payload["scientific_decision"] == "INCONCLUSIVE_REFERENCE_ZERO_K4_INADEQUATE_V196"


def test_v196_independent_recomputation_and_call_boundary() -> None:
    payload = load_summary()
    independent = payload["independent_recomputation"]
    assert independent["checks_passed"] == independent["checks_total"] == 23
    assert independent["maximum_metric_absolute_difference"] == 0.0
    assert independent["maximum_summary_absolute_difference"] == 0.0
    assert independent["maximum_camera_permutation_relative_error"] <= 1e-15
    assert independent["end_to_end_physics_independence_proven"] is False
    assert payload["arms"]["full_dct_k2"]["logical_exact_A"] == 3
    assert payload["arms"]["full_dct_k2"]["logical_exact_AT"] == 2
    assert payload["cost_boundary"]["geometry_cache_forward_equivalents"] == 26 * 1009
    assert payload["cost_boundary"]["exact_call_reduction_established"] is False


def test_v196_1_reference_identity_audit_corrects_the_interpretation() -> None:
    payload = load_summary()
    audit = payload["reference_identity_audit_v196_1"]
    assert audit["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_REFERENCE_IDENTITY_AUDIT_V196_1"
    )
    assert audit["scientific_conclusion"] == (
        "PROTOCOL_REFERENCE_GATE_PREDETERMINED_INCONCLUSIVE_V196_1"
    )
    assert audit["overlapping_metric_count"] == 156
    assert audit["maximum_overlap_absolute_difference"] <= 6e-17
    assert audit["retained_field_pairs_bitwise_identical"] == 2
    assert audit["v176_reference_strict_safe_cells"] == 0
    assert audit["v176_reference_total_cells"] == 52
    assert audit["v175_reference_strict_safe_cells"] == 0
    assert audit["v175_reference_total_cells"] == 468
    assert audit["v196_reference_gate_predetermined_inconclusive_before_freeze"] is True
    assert audit["v196_original_numerics_and_decision_preserved"] is True
    assert audit["prospective_comparative_headroom_test"] is False


def test_v196_public_assets_and_bilingual_copy_exist() -> None:
    result = (ROOT / "docs/poolfire_full_dct_k2_complete_trajectory_v196_result_2026-08-22.md").read_text(
        encoding="utf-8"
    )
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    for content in (result, daily, log):
        assert "v196" in content
        assert "v196.1" in content
    assert "v196" in focus
    assert "v197" in focus
    assert "v198" in root
    assert "v197" in root
    assert "# v196：" in result and "# v196:" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    figure = ROOT / "assets/figures/poolfire_full_dct_k2_complete_trajectory_v196.png"
    assert figure.exists() and figure.stat().st_size > 20_000


def test_v196_current_evidence_and_claim_boundary() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["scientific_status"] == "PASS_ALL_NINE_DENSE_REPRESENTATION_CALL_HEADROOM_V204"
    assert current["engineering_status"] == "PASS_INDEPENDENT_RECOMPUTATION_ALL_NINE_CONTROL_ATTRIBUTION_V204"
    assert current["metrics"]["v196_full_dct_k2_five_safe_cells"] == 1313
    assert current["metrics"]["v196_zero_k4_all_nine_safe_cells"] == 0
    assert current["metrics"]["v196_1_overlap_metric_count"] == 156
    assert current["metrics"]["v196_1_retained_field_pairs_bitwise_identical"] == 2
    assert current["current_decision"]["v196_algorithm_breakthrough"] is False
    assert current["current_decision"]["v196_reference_identity_audit_required"] is False
    assert current["current_decision"]["v196_1_reference_identity_audit_completed"] is True
    assert current["current_decision"]["v196_1_reference_gate_predetermined_inconclusive_before_freeze"] is True
    assert current["current_decision"]["v196_1_prospective_comparative_headroom_test"] is False


def test_v196_public_files_exclude_private_execution_details() -> None:
    public_files = [
        SUMMARY,
        ROOT / "docs/poolfire_full_dct_k2_complete_trajectory_v196_result_2026-08-22.md",
        ROOT / "operator-learning/current-evidence.json",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
        ROOT / "index.html",
    ]
    forbidden = [
        "private_results",
        "private_worktrees",
        "/Users/",
        "OPENED receipt",
        "3af9cf37",
        "4f27707e",
        "63cef926",
    ]
    for path in public_files:
        content = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, (path, token)
