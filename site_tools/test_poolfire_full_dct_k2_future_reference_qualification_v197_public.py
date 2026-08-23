from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/poolfire_full_dct_k2_future_reference_qualification_v197_public_summary.json"
)


def load_summary() -> dict[str, object]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v197_future_reference_is_complete_and_non_exchangeable() -> None:
    payload = load_summary()
    assert payload["scientific_decision"] == (
        "PASS_FUTURE_ONLY_FULL_DCT_K2_REFERENCE_QUALIFICATION_V197"
    )
    assert payload["qualification"]["strict_cells_safe"] == 2626
    assert payload["qualification"]["complete_groups_passed"] == 26
    assert payload["qualification"]["call_rows_matching"] == 2626
    assert payload["qualification"]["minimum_strict_cell_margin"] > 0.0
    assert payload["reference_identity"]["non_exchangeable"] is True
    assert payload["reference_identity"]["future_only"] is True


def test_v197_independent_recomputation_and_chronology_boundary() -> None:
    payload = load_summary()
    independent = payload["independent_recomputation"]
    assert independent["checks_passed"] == independent["checks_total"] == 14
    assert independent["formal_independent_maximum_numeric_difference"] == 0.0
    assert independent["independent_arrays_used_before_formal_output_read"] is True
    assert (
        payload["scope"]["future_candidate_results_seen_before_reference_freeze"]
        is False
    )
    assert payload["scope"]["changes_v196_decision"] is False


def test_v197_public_assets_and_bilingual_copy_exist() -> None:
    result = (
        ROOT
        / "docs/poolfire_full_dct_k2_future_reference_qualification_v197_result_2026-08-23.md"
    ).read_text(encoding="utf-8")
    assert "# v197：" in result and "# v197:" in result
    assert "future-only" in result
    figure = (
        ROOT
        / "assets/figures/poolfire_full_dct_k2_future_reference_qualification_v197.png"
    )
    assert figure.exists() and figure.stat().st_size > 20_000


def test_v197_remains_visible_after_v199_becomes_current_state() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["updated"] == "2026-08-23"
    assert current["formal_status"] == (
        "FORMAL_PENDING_INDEPENDENT_GEOMETRY_OBSERVABILITY_ATTRIBUTION_V210"
    )
    assert current["scientific_status"] == (
        "PARTIAL_OVERLAPPING_GEOMETRY_ONLY_OBSERVABILITY_EVIDENCE_V210"
    )
    assert current["v197_future_reference_qualification_formal_status"] == (
        "PASS_FORMAL_FULL_DCT_K2_FUTURE_REFERENCE_QUALIFICATION_V197"
    )
    assert current["v197_future_reference_qualification_scientific_decision"] == (
        "PASS_FUTURE_ONLY_FULL_DCT_K2_REFERENCE_QUALIFICATION_V197"
    )
    assert current["metrics"]["v197_reference_strict_safe_cells"] == 2626
    assert current["metrics"]["v197_reference_complete_groups_passed"] == 26
    assert current["current_decision"]["v197_algorithm_breakthrough"] is False

    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    assert "poolfire_full_dct_k2_future_reference_qualification_v197" in focus
    assert "2626/2626" in focus
    assert "26/26" in focus

    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-23"') == 1
    assert daily.count('id="latest"') == 1


def test_v197_learning_log_keeps_plain_language_boundary() -> None:
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v197 先把未来比较的尺子校准好" in log
    assert "English summary" in log
    assert "不是算法变快了" in log


def test_v197_claim_boundary_remains_false() -> None:
    claims = load_summary()["claims_fixed_false"]
    assert all(value is False for value in claims.values())


def test_v197_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT
        / "docs/poolfire_full_dct_k2_future_reference_qualification_v197_result_2026-08-23.md",
    ]
    forbidden = [
        "private_results",
        "private_worktrees",
        "/Users/",
        "cb6fb8f3",
        "e01895d2",
        "f7e2a571",
    ]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, (path, token)
