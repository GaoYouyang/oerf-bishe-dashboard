from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_geometry_voxel_block_jacobi_v248_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_geometry_voxel_block_jacobi_v248_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_geometry_voxel_block_jacobi_v248.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_geometry_voxel_block_jacobi_v248_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v248_summary_preserves_inconclusive_frame_zero_decision() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    execution = data["execution"]
    agreement = data["formal_independent_agreement"]
    diagnostic = data["diagnostic_only_not_an_admissible_performance_verdict"]
    adjudication = data["adjudication"]
    assert data["scope"]["cells"] == 13
    assert data["scope"]["trainable_parameters"] == 0
    assert data["mechanism"]["block_shape_zyx"] == [4, 2, 2]
    assert execution["formal_validity_checks_passed"] == 21
    assert execution["independent_checks_passed"] == 26
    assert execution["independent_checks_total"] == 28
    assert execution["numeric_tolerance_relaxed"] is False
    assert agreement["observation_absolute_maximum"] > 0.0
    assert agreement["observation_agreement_passed"] is False
    assert diagnostic["primary_strict_safe_cells"] == 0
    assert diagnostic["diagonal_jacobi_control_strict_safe_cells"] == 12
    assert adjudication["scientific_decision"] == (
        "INCONCLUSIVE_INVALID_CASE19_GEOMETRY_VOXEL_BLOCK_JACOBI_FRAME_ZERO_V248"
    )
    assert adjudication["full_sequence_authorized"] is False
    assert adjudication["mechanism_operationally_retired"] is True
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v248_result_is_bilingual_and_keeps_claim_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v248：" in text and "# v248:" in text
    for token in ("21/21", "26/28", "1.33e-15", "0/13", "12/13", "16A+16A^T"):
        assert token in text
    assert "完整 429 单元序列不运行" in text
    assert "full 429-cell sequence does not run" in text
    assert "algorithm_breakthrough=false" in text


def test_v248_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v248_is_latest_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == (
        "INCONCLUSIVE_INVALID_CASE19_GEOMETRY_VOXEL_BLOCK_JACOBI_FRAME_ZERO_V248"
    )
    assert current["current_decision"]["v248_independent_validation_passed"] is False
    assert current["current_decision"]["v248_full_sequence_authorized"] is False
    assert current["current_decision"]["v248_mechanism_operationally_retired"] is True
    assert current["current_decision"]["v248_algorithm_breakthrough"] is False
    assert current["metrics"]["v248_independent_checks_passed"] == 26
    assert current["metrics"]["v248_independent_checks_total"] == 28
    assert current["metrics"]["v248_primary_strict_safe_cells"] == 0
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_geometry_voxel_block_jacobi_v248.png"
    )
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_geometry_voxel_block_jacobi_v248" in text
        assert "26/28" in text and "1.33e-15" in text and "INCONCLUSIVE" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v248 固定三维体素块" in log
    assert "v248 fixed 3D voxel blocks" in log


def test_v248_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)
