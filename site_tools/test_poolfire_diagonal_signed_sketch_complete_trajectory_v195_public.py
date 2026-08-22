from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_diagonal_signed_sketch_complete_trajectory_v195_public_summary.json"


def load_summary() -> dict[str, object]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v195_primary_fails_complete_trajectory_and_closes_route() -> None:
    payload = load_summary()
    primary = payload["primary_diagonal_k1"]
    assert primary["five_camera"]["strict_safe_cells"] == 987
    assert primary["five_camera"]["complete_calibration_groups_passed"] == 0
    assert primary["all_nine"]["strict_safe_cells"] == 1234
    assert primary["all_nine"]["complete_calibration_groups_passed"] == 3
    assert payload["scientific_decision"] == "FAIL_DIAGONAL_SIGNED_SKETCH_COMPLETE_TRAJECTORY_V195_2"
    assert "p14" in payload["decision_boundary_en"]


def test_v195_full_dct_is_stronger_but_not_promoted() -> None:
    payload = load_summary()
    full_dct = payload["controls"]["full_dct_k1"]
    assert full_dct["five_camera"]["strict_safe_cells"] == 1310
    assert full_dct["five_camera"]["complete_calibration_groups_passed"] == 12
    assert full_dct["five_camera"]["passed"] is False
    assert full_dct["all_nine"]["passed"] is True
    assert "cannot replace the primary post hoc" in payload["decision_boundary_en"]


def test_v195_independent_recomputation_and_claim_boundary() -> None:
    payload = load_summary()
    independent = payload["independent_recomputation"]
    assert independent["checks_passed"] == independent["checks_total"] == 27
    assert independent["maximum_coordinate_relative_error"] <= 1e-12
    assert independent["maximum_metric_absolute_error"] <= 1e-12
    assert independent["maximum_camera_ray_permutation_error"] == 0.0
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v195_public_assets_and_bilingual_copy_exist() -> None:
    result = (ROOT / "docs/poolfire_diagonal_signed_sketch_complete_trajectory_v195_result_2026-08-22.md").read_text(
        encoding="utf-8"
    )
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    for content in (result, daily, focus, log):
        assert "v195" in content
    assert "v196" in root
    assert "# v195.2：" in result and "# v195.2:" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    figure = ROOT / "assets/figures/poolfire_diagonal_signed_sketch_complete_trajectory_v195.png"
    assert figure.exists() and figure.stat().st_size > 20_000


def test_v195_public_files_exclude_private_execution_details() -> None:
    public_files = [
        SUMMARY,
        ROOT / "docs/poolfire_diagonal_signed_sketch_complete_trajectory_v195_result_2026-08-22.md",
        ROOT / "operator-learning/current-evidence.json",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
        ROOT / "index.html",
    ]
    forbidden = [
        "private_results",
        "private_worktrees",
        "b114bd5a",
        "bccafafa",
        "daac3259",
        "/Users/",
        "OPENED receipt",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, (path, token)
