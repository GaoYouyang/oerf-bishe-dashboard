from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/poolfire_nine_camera_control_attribution_p14_v204_public_summary.json"
)


def test_v204_public_summary_preserves_control_attribution() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert (
        payload["scientific_decision"]
        == "PASS_ALL_NINE_DENSE_REPRESENTATION_CALL_HEADROOM_V204"
    )
    v203 = payload["v203_physical_information_attribution"]
    assert v203["audited_five_camera_failures"] == 24
    assert v203["five_camera_k2_strict_safe"] == 0
    assert v203["all_nine_camera_k2_strict_safe"] == 24
    v204 = payload["v204_all_nine_complete_trajectory"]
    assert v204["full_dct_k1_parent"]["strict_safe_cells"] == 1313
    assert v204["full_dct_k1_parent"]["complete_groups"] == 13
    assert v204["full_dct_k1_parent"]["logical_online_exact_calls"] == {
        "A": 2,
        "AT": 1,
    }
    assert v204["full_dct_k2_reference"]["logical_online_exact_calls"] == {
        "A": 3,
        "AT": 2,
    }
    assert v204["passing_pure_classical_controls"] == []


def test_v204_public_summary_preserves_claim_boundaries() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert all(value is False for value in payload["claims_fixed_false"].values())
    audit = payload["independent_numeric_audit"]
    assert audit["formal_independent_metric_maximum_absolute"] == 0.0
    assert audit["strict_masks_exact"] is True
    assert audit["passing_arm_roster_exact"] is True


def test_v204_public_assets_and_bilingual_copy_exist() -> None:
    result = (
        ROOT
        / "docs/poolfire_nine_camera_control_attribution_p14_v204_result_2026-08-23.md"
    ).read_text()
    assert "# v203-v204：" in result and "# v203-v204:" in result
    assert "PASS_ALL_NINE_DENSE_REPRESENTATION_CALL_HEADROOM_V204" in result
    assert (
        ROOT
        / "assets/figures/poolfire_nine_camera_control_attribution_p14_v204.png"
    ).is_file()
    for page in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = page.read_text()
        assert "poolfire_nine_camera_control_attribution_p14_v204" in content
    assert "PASS_ALL_NINE_DENSE_REPRESENTATION_CALL_HEADROOM_V204" in (
        ROOT / "operator-learning/index.html"
    ).read_text()


def test_v204_current_evidence_is_preserved_beneath_the_v205_headline() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["v221_low64_exact_rowspace_lift_scientific_decision"] == "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    assert current["v221_low64_exact_rowspace_lift_independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_LOW64_EXACT_ROWSPACE_LIFT_V221"
    assert current["v204_all_nine_control_scientific_decision"] == "PASS_ALL_NINE_DENSE_REPRESENTATION_CALL_HEADROOM_V204"
    assert current["metrics"]["v203_nine_camera_rescued_failures"] == 24
    assert current["metrics"]["v204_full_dct_k1_strict_safe_cells"] == 1313
    assert current["current_decision"]["v204_algorithm_breakthrough"] is False
    assert current["current_decision"]["v204_dense_cache_removal_required"] is True
    assert current["public_evidence"]["result"].startswith("../document_reader.html?doc=docs%2F")
    assert current["public_evidence"]["summary"].startswith("../docs/")
