from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_tgv2_pdhg_reference_p14_v201_public_summary.json"


def test_v201_public_summary_preserves_failed_reference_gate() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "FAIL_TGV2_PDHG_REFERENCE_ADEQUACY_V201"
    tgv2 = payload["fixed_primary"]
    huber = payload["controls"]["huber_pdhg_parent_v200"]
    assert tgv2["five_camera"]["strict_safe_cells"] == 1289
    assert tgv2["five_camera"]["complete_groups_passed"] == 5
    assert tgv2["five_camera"]["failed_cells"] == 24
    assert huber["five_camera"]["strict_safe_cells"] == 1289
    assert huber["five_camera"]["complete_groups_passed"] == 5
    assert tgv2["logical_online_exact_calls"] == {"A": 259, "AT": 258}
    assert payload["fixed_tgv2_reference_closed"] is True
    assert payload["fixed_identity_candidate_closed"] is False


def test_v201_public_attribution_preserves_zero_rescues() -> None:
    attribution = json.loads(SUMMARY.read_text())["mechanism_attribution"]
    assert attribution["observation_error_improved_cells"] == 1313
    assert attribution["observation_error_worsened_cells"] == 0
    assert attribution["shared_failed_cells"] == 24
    assert attribution["rescued_failed_cells"] == 0
    assert attribution["failure_mask_xor_cells"] == 0
    assert attribution["tgv2_failed_cell_metric_counts"] == {
        "field": 4,
        "gradient": 24,
        "observation": 0,
    }


def test_v201_public_claim_boundary_remains_false() -> None:
    claims = json.loads(SUMMARY.read_text())["claims_fixed_false"]
    assert all(value is False for value in claims.values())


def test_v201_public_assets_and_bilingual_copy_exist() -> None:
    result = (ROOT / "docs/poolfire_tgv2_pdhg_reference_p14_v201_result_2026-08-23.md").read_text()
    assert "# v201：" in result and "# v201:" in result
    assert "FAIL_TGV2_PDHG_REFERENCE_ADEQUACY_V201" in result
    assert (ROOT / "assets/figures/poolfire_tgv2_pdhg_reference_p14_v201.png").is_file()
    for page in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = page.read_text()
        assert "poolfire_tgv2_pdhg_reference_p14_v201" in content
    assert "FAIL_TGV2_PDHG_REFERENCE_ADEQUACY_V201" in (
        ROOT / "operator-learning/index.html"
    ).read_text()


def test_v201_current_evidence_remains_historical_under_v204_headline() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert (
        current["v221_low64_exact_rowspace_lift_scientific_decision"]
        == "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert (
        current["v201_tgv2_reference_scientific_decision"]
        == "FAIL_TGV2_PDHG_REFERENCE_ADEQUACY_V201"
    )
    assert (
        current["v201_tgv2_reference_formal_status"]
        == "PASS_FORMAL_POOLFIRE_TGV2_PDHG_REFERENCE_P14_EXECUTION_V201"
    )
    assert (
        current["v201_tgv2_reference_independent_status"]
        == "PASS_INDEPENDENT_RECOMPUTATION_TGV2_PDHG_REFERENCE_P14_V201"
    )
    assert current["metrics"]["v201_tgv2_five_strict_safe_cells"] == 1289
    assert current["metrics"]["v201_tgv2_observation_improved_cells"] == 1313
    assert current["metrics"]["v201_tgv2_rescued_failed_cells"] == 0
    assert current["current_decision"]["v201_tgv2_reference_adequate"] is False
    assert current["current_decision"]["v201_fixed_tgv2_reference_closed"] is True
    assert current["public_evidence"]["result"].endswith(
        "blastnet_case2_case5_low64_algebraic_nullspace_attribution_v222_1_result_2026-08-24.md"
    )
