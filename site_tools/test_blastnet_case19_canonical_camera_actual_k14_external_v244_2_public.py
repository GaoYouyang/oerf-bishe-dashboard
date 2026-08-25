from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_canonical_camera_actual_k14_external_v244_2_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_canonical_camera_actual_k14_external_v244_2_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_canonical_camera_actual_k14_external_v244_2.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_canonical_camera_actual_k14_external_v244_2_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v244_2_summary_preserves_the_inconclusive_decision() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    diagnostic = data["post_open_diagnostic_only"]
    assert data["scope"]["cells"] == 429
    assert data["input_engineering"]["payload_objects"] == 37
    assert validation["checks_passed"] == 26
    assert validation["checks_total"] == 29
    assert validation["maximum_differences"]["residual_relative"] > 1e-8
    assert validation["maximum_differences"]["metric_absolute"] > 1e-8
    assert diagnostic["k16_reference"]["absolute_complete_rigs_passed"] == 9
    assert diagnostic["primary_warm_k14"]["absolute_complete_rigs_passed"] == 12
    assert diagnostic["primary_warm_k14"]["matched_complete_rigs_passed"] == 13
    assert data["adjudication"]["prospective_confirmation"] is False
    assert data["adjudication"]["fresh_resource_gate_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v244_2_result_is_bilingual_and_does_not_upgrade_diagnostics() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v244.2：" in text and "# v244.2:" in text
    for token in ("26/29", "9/13", "12/13", "13/13", "1.1060e-7", "1.0219e-8"):
        assert token in text
    assert "开封后诊断，不是替代判决" in text
    assert "Post-open diagnostic only" in text
    assert "algorithm_breakthrough=false" in text


def test_v244_2_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 1100


def test_v244_2_remains_as_historical_evidence_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["current_decision"]["v244_2_prospective_confirmation"] is False
    assert current["current_decision"]["v244_2_fresh_resource_gate_authorized"] is False
    assert current["current_decision"]["v244_2_algorithm_breakthrough"] is False
    assert current["metrics"]["v244_2_reference_absolute_complete_rigs_passed"] == 9
    assert current["metrics"]["v244_2_primary_absolute_complete_rigs_passed"] == 12
    assert current["v244_2_scientific_decision"] == (
        "INCONCLUSIVE_INVALID_CASE19_CANONICAL_CAMERA_ACTUAL_K14_EXTERNAL_V244_2"
    )
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_canonical_camera_actual_k14_external_v244_2" in text
        assert "26/29" in text and "9/13" in text and "12/13" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    daily = DAILY.read_text(encoding="utf-8")
    assert 'data-i18n-zh="51 天" data-i18n-en="51 days"' in daily
    assert "2026-07-06 to 2026-08-25" in daily
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v244.2 Case 19" in log
    assert "one-shot Case 19" in log


def test_v244_2_public_artifacts_exclude_private_execution_details() -> None:
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
