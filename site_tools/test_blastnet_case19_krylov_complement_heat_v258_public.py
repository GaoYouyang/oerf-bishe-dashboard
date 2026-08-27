from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_krylov_complement_heat_v258_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_krylov_complement_heat_v258_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_krylov_complement_heat_v258.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_krylov_complement_heat_v258_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v258_summary_records_valid_independent_negative_result() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    primary = data["worst_implementation_envelope"]["primary"]
    assert data["scope"]["cells"] == 13
    assert data["scope"]["full_sequence_run"] is False
    assert data["scope"]["truth_available_before_prediction_barrier"] is False
    assert validation["completed"] is True
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 47
    assert primary["absolute_cells"] == 13
    assert primary["matched_cells"] == 0
    assert primary["matched_ratio_p90_higher"][3] > 1.05
    assert primary["matched_ratio_worst"][3] > 1.05
    assert data["adjudication"]["scientific_fail_claimed"] is True
    assert data["adjudication"]["full_sequence_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v258_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v258：" in text and "# v258:" in text
    for token in ("47/47", "13/13", "0/13", "1.226", "1.367", "15A+14A^T"):
        assert token in text
    assert "不是有效 exact-call 减少" in text
    assert "not effective exact-call reduction" in text
    assert "algorithm_breakthrough=false" in text


def test_v258_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v258_is_preserved_after_v259_becomes_latest() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert current["scientific_status"] == "INCONCLUSIVE_INVALID_CASE19_KRYLOV_HISTORY_JOINT_REORTHOGONALIZATION_FRAME_ZERO_V269_1"
    assert metrics["v258_independent_checks_passed"] == 47
    assert metrics["v258_independent_checks_total"] == 47
    assert metrics["v258_primary_absolute_cells"] == 13
    assert metrics["v258_primary_matched_cells"] == 0
    assert decision["v258_independent_validation_passed"] is True
    assert decision["v258_scientific_result_failed"] is True
    assert decision["v258_full_sequence_authorized"] is False
    assert decision["v258_effective_exact_call_reduction_established"] is False
    assert decision["v258_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith("blastnet_case19_krylov_history_joint_reorthogonalization_v269_1.png")
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_krylov_complement_heat_v258" in text
        assert "47/47" in text and "0/13" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert "v258" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v258_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)
