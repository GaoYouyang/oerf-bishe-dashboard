from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_signed_sketch_normal_refinement_v194_public_summary.json"


def load_summary() -> dict[str, object]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v194_primary_fails_and_diagonal_control_is_not_promoted() -> None:
    payload = load_summary()
    safe = payload["strict_safe_cells"]
    assert safe["full_hessian_primary_k1"] == {"five_camera": 0, "all_nine": 0}
    assert safe["diagonal_control_k1"] == {"five_camera": 52, "all_nine": 52}
    assert payload["scientific_decision"] == "FAIL_SIGNED_SKETCH_FULL_NORMAL_REFINEMENT_V194"
    assert "not a successful v194 method" in payload["decision_boundary_en"]


def test_v194_independent_recomputation_and_claim_boundary() -> None:
    payload = load_summary()
    independent = payload["independent_recomputation"]
    assert independent["checks_passed"] == independent["checks_total"] == 17
    assert independent["maximum_numeric_array_relative_difference"] <= 1e-8
    assert independent["camera_permutation_feature_relative_error"] == 0.0
    assert independent["camera_permutation_response_relative_error"] == 0.0
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v194_public_assets_and_bilingual_copy_exist() -> None:
    result = (ROOT / "docs/poolfire_signed_sketch_normal_refinement_v194_result_2026-08-22.md").read_text(
        encoding="utf-8"
    )
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    for content in (result, daily, focus, log):
        assert "v194" in content
    assert "v196" in root
    assert "# v194：" in result and "# v194:" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    figure = ROOT / "assets/figures/poolfire_signed_sketch_normal_refinement_v194.png"
    assert figure.exists() and figure.stat().st_size > 20_000


def test_v194_public_files_exclude_private_execution_details() -> None:
    public_files = [
        SUMMARY,
        ROOT / "docs/poolfire_signed_sketch_normal_refinement_v194_result_2026-08-22.md",
        ROOT / "operator-learning/current-evidence.json",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
        ROOT / "index.html",
    ]
    forbidden = [
        "private_results",
        "private_worktrees",
        "b804b056",
        "82546cd5",
        "d07c33ee",
        "/Users/",
        "OPENED receipt",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, (path, token)
