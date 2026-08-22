from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_covariance_gcv_full_dct_v198_public_summary.json"


def test_v198_public_summary_preserves_control_attribution() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "PASS_CHEAPER_CONTROL_EXPLAINS_COVARIANCE_GCV_V198"
    primary = payload["methods"]["empirical_covariance_gcv_full_dct_k1"]
    control = payload["methods"]["identity_gcv_full_dct_k1"]
    parent = payload["methods"]["full_dct_k1_parent"]
    assert primary["strict_safe_cells"] == control["strict_safe_cells"] == 2626
    assert primary["complete_groups_passed"] == control["complete_groups_passed"] == 26
    assert primary["logical_online_exact_calls"] == control["logical_online_exact_calls"] == {"A": 2, "AT": 1}
    assert parent["strict_safe_cells"] == 2623
    assert payload["selected_regularization"]["identity_gcv"]["selected_cells"] == 2626


def test_v198_public_claim_boundary_remains_false() -> None:
    claims = json.loads(SUMMARY.read_text())["claims_fixed_false"]
    assert all(value is False for value in claims.values())


def test_v198_public_assets_and_bilingual_copy_exist() -> None:
    result = (ROOT / "docs/poolfire_covariance_gcv_full_dct_v198_result_2026-08-23.md").read_text()
    assert "# v198：" in result and "# v198:" in result
    assert "identity-GCV" in result
    assert (ROOT / "assets/figures/poolfire_covariance_gcv_full_dct_v198.png").is_file()
    for page in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = page.read_text()
        assert "poolfire_covariance_gcv_full_dct_v198" in content
        assert "PASS_CHEAPER_CONTROL_EXPLAINS_COVARIANCE_GCV_V198" in content


def test_v198_current_evidence_is_the_public_headline() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["scientific_status"] == "PASS_CHEAPER_CONTROL_EXPLAINS_COVARIANCE_GCV_V198"
    assert current["metrics"]["v198_primary_strict_safe_cells"] == 2626
    assert current["metrics"]["v198_identity_control_strict_safe_cells"] == 2626
    assert current["current_decision"]["v198_algorithm_breakthrough"] is False
    assert "identity-prior" in current["next_scientific_gate_en"]


def test_v198_public_files_exclude_private_execution_details() -> None:
    files = (
        SUMMARY,
        ROOT / "docs/poolfire_covariance_gcv_full_dct_v198_result_2026-08-23.md",
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    )
    forbidden = (
        "private_" + "results",
        "private_" + "worktrees",
        "/" + "Users/",
    )
    for path in files:
        content = path.read_text()
        assert not any(token in content for token in forbidden)
