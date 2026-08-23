from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_fixed_identity_prior_p14_v199_public_summary.json"


def test_v199_public_summary_preserves_reference_inadequacy() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "INCONCLUSIVE_P14_REFERENCE_INADEQUATE_V199"
    primary = payload["methods"]["fixed_identity_k1"]
    parent = payload["methods"]["full_dct_k1_parent"]
    reference = payload["methods"]["full_dct_k2_reference"]
    assert primary["all_nine"]["strict_safe_cells"] == 1313
    assert primary["five_camera"]["strict_safe_cells"] == 1268
    assert parent["five_camera"]["strict_safe_cells"] == 1173
    assert reference["five_camera"]["strict_safe_cells"] == 1213
    assert primary["five_camera"]["complete_groups_passed"] == 3
    assert reference["five_camera"]["complete_groups_passed"] == 0
    assert primary["logical_online_exact_calls"] == {"A": 2, "AT": 1}
    assert reference["logical_online_exact_calls"] == {"A": 3, "AT": 2}


def test_v199_public_claim_boundary_remains_false() -> None:
    claims = json.loads(SUMMARY.read_text())["claims_fixed_false"]
    assert all(value is False for value in claims.values())


def test_v199_public_assets_and_bilingual_copy_exist() -> None:
    result = (ROOT / "docs/poolfire_fixed_identity_prior_p14_v199_result_2026-08-23.md").read_text()
    assert "# v199：" in result and "# v199:" in result
    assert "INCONCLUSIVE_P14_REFERENCE_INADEQUATE_V199" in result
    assert (ROOT / "assets/figures/poolfire_fixed_identity_prior_p14_v199.png").is_file()
    for page in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = page.read_text()
        assert "poolfire_fixed_identity_prior_p14_v199" in content


def test_v199_evidence_remains_preserved_after_v201_becomes_current() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["scientific_status"] == "PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    assert current["formal_status"] == "PASS_FORMAL_POOLFIRE_POTENTIAL_NORMAL_COMPACT_CACHE_P14_V205"
    assert current["engineering_status"] == "PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    assert current["v199_fixed_identity_p14_scientific_decision"] == "INCONCLUSIVE_P14_REFERENCE_INADEQUATE_V199"
    assert current["metrics"]["v199_primary_five_strict_safe_cells"] == 1268
    assert current["metrics"]["v199_reference_five_complete_groups_passed"] == 0
    assert current["current_decision"]["v199_p14_reference_adequate"] is False
    assert current["current_decision"]["v199_exact_call_headroom_interpretable"] is False
    assert current["public_evidence"]["result"].endswith(
        "poolfire_potential_normal_compact_cache_p14_v205_result_2026-08-23.md"
    )
