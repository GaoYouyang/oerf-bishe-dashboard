from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_result_unopened_condition_v176_public_summary.json"
RESULT = ROOT / "docs/poolfire_result_unopened_condition_v176_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/poolfire_result_unopened_condition_v176.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_the_negative_scientific_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "FAIL_RESULT_UNOPENED_POOLFIRE_CONDITION_PARITY_V176"
    assert payload["primary"]["strict_safe_cells"] == 0
    assert payload["primary"]["strict_safe_total"] == 52
    assert payload["primary"]["complete_calibrations_passed"] == 0
    assert payload["primary"]["frame_strata_passed"] == 0
    assert payload["primary"]["own_reference_joint_harm_count"] == 52
    assert payload["primary"]["own_reference_severe_harm_count"] == 50
    assert payload["reference_adequacy_diagnostic"]["primary_same_subset_k4_strict_safe_cells"] == 0
    assert payload["reference_adequacy_diagnostic"]["all_four_policy_k4_reference_strict_safe_counts"] == [0, 0, 0, 0]
    assert payload["controls"]["controls_that_passed"] == []
    assert payload["independent_recomputation"]["check_count"] == 35
    assert payload["execution_disclosure"]["repair_changed_only_residual_audit_storage"] is True
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["resource_speedup"] is False
    assert payload["claim_limits"]["real_bost"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_result_note_is_bilingual_and_discloses_recovery_without_overclaiming() -> None:
    text = RESULT.read_text()
    assert "# v176：一次结果未开工况" in text
    assert "# v176: one result-unopened condition" in text
    assert "FAIL_RESULT_UNOPENED_POOLFIRE_CONDITION_PARITY_V176" in text
    assert "0/52" in text
    assert "52/52" in text
    assert "35/35" in text
    assert "存储修复" in text
    assert "storage-only repair" in text
    assert "algorithm_breakthrough=false" in text
    assert "不关闭整个 C 路线" in text
    assert "does not close the full C route" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_points_to_v176_and_closes_the_transfer() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == "FAIL_RESULT_UNOPENED_POOLFIRE_CONDITION_PARITY_V176"
    assert payload["metrics"]["v176_primary_strict_safe_count"] == 0
    assert payload["metrics"]["v176_primary_strict_safe_total"] == 52
    assert payload["metrics"]["v176_primary_k4_strict_safe_count"] == 0
    assert payload["metrics"]["v176_independent_check_count"] == 35
    assert payload["current_decision"]["v176_current_minimal_shared_selector_transfer_closed"] is True
    assert payload["current_decision"]["v176_resource_gate_authorized"] is False
    assert payload["current_decision"]["v176_gpu_rental_authorized"] is False


def test_primary_pages_reference_v176_in_both_languages() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily, home):
        assert "v176" in text
        assert "FAIL_RESULT_UNOPENED_POOLFIRE_CONDITION_PARITY_V176" in text
    assert "当前最小共享选择器迁移关闭" in operator
    assert "current minimal shared-selector transfer is closed" in operator
    assert "poolfire_result_unopened_condition_v176.png" in operator
    assert daily.count("FAIL_RESULT_UNOPENED_POOLFIRE_CONDITION_PARITY_V176") == 1


def test_route_metadata_and_cachebuster_advance_to_v176() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    curriculum = (ROOT / "operator-learning/curriculum.js").read_text()
    assert "curriculum.js?v=20260821-v176" in operator
    assert 'version: "2026.08.21-c-v176-result-unopened-condition-negative"' in curriculum
    assert 'updated: "2026-08-21"' in curriculum


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
