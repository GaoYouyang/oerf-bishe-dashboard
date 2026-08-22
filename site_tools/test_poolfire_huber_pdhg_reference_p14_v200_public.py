from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_huber_pdhg_reference_p14_v200_public_summary.json"


def test_v200_public_summary_preserves_failed_reference_gate() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "FAIL_HUBER_PDHG_REFERENCE_ADEQUACY_V200"
    huber = payload["fixed_primary"]
    k2 = payload["controls"]["full_dct_k2_parent"]
    assert huber["five_camera"]["strict_safe_cells"] == 1289
    assert huber["five_camera"]["complete_groups_passed"] == 5
    assert huber["five_camera"]["failed_cells"] == 24
    assert huber["five_camera"]["failed_groups"] == 8
    assert k2["five_camera"]["strict_safe_cells"] == 1213
    assert k2["five_camera"]["complete_groups_passed"] == 0
    assert huber["logical_online_exact_calls"] == {"A": 131, "AT": 130}
    assert payload["fixed_huber_reference_closed"] is True
    assert payload["fixed_identity_candidate_closed"] is False


def test_v200_public_claim_boundary_remains_false() -> None:
    claims = json.loads(SUMMARY.read_text())["claims_fixed_false"]
    assert all(value is False for value in claims.values())


def test_v200_public_assets_and_bilingual_copy_exist() -> None:
    result = (ROOT / "docs/poolfire_huber_pdhg_reference_p14_v200_result_2026-08-23.md").read_text()
    assert "# v200：" in result and "# v200:" in result
    assert "FAIL_HUBER_PDHG_REFERENCE_ADEQUACY_V200" in result
    assert (ROOT / "assets/figures/poolfire_huber_pdhg_reference_p14_v200.png").is_file()
    for page in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = page.read_text()
        assert "poolfire_huber_pdhg_reference_p14_v200" in content


def test_v200_current_evidence_is_preserved_as_parent_history() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["metrics"]["v200_huber_five_strict_safe_cells"] == 1289
    assert current["metrics"]["v200_huber_five_complete_groups_passed"] == 5
    assert current["current_decision"]["v200_huber_reference_adequate"] is False
    assert current["current_decision"]["v200_fixed_huber_reference_closed"] is True
    assert SUMMARY.is_file()
    assert (ROOT / "docs/poolfire_huber_pdhg_reference_p14_v200_result_2026-08-23.md").is_file()
