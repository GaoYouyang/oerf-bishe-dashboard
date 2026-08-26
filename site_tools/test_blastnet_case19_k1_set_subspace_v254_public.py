from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_k1_set_subspace_v254_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_k1_set_subspace_v254_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_k1_set_subspace_v254.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_k1_set_subspace_v254_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v254_summary_records_valid_negative_whole_sequence_result() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = data["results"]["primary"]
    validation = data["independent_validation"]
    assert data["scope"]["cells"] == 429
    assert data["scope"]["physical_replay_completed"] is True
    assert data["scope"]["truth_available_before_prediction_barrier"] is False
    assert data["mechanism"]["frame_set_permutation_equivariant"] is True
    assert primary["absolute_safe_cells"] == 383
    assert primary["absolute_safe_rigs"] == 6
    assert primary["matched_safe_cells"] == 0
    assert primary["matched_safe_rigs"] == 0
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 29
    assert validation["scientific_decision"] == "FAIL_CASE19_K1_SET_SUBSPACE_MATCHED_ACCURACY_V254"
    assert data["adjudication"]["route_action"] == "CLOSE_UNORDERED_NORMALIZED_K1_PAIR_RANK16_SUBSPACE"
    assert data["adjudication"]["resource_gate_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v254_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v254：" in text and "# v254:" in text
    for token in ("383/429", "6/13", "0/429", "0/13", "29/29", "6.25%"):
        assert token in text
    assert "不是有效减调用" in text
    assert "rather than effective call reduction" in text
    assert "algorithm_breakthrough=false" in text


def test_v254_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 3000
        assert image.height >= 1100


def test_v254_is_retained_after_v255_becomes_latest() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert current["scientific_status"] == "INCONCLUSIVE_INVALID_CASE19_GALERKIN_PYRAMID_FRAME_ZERO_V256"
    assert metrics["v254_primary_absolute_safe_cells"] == 383
    assert metrics["v254_primary_absolute_safe_rigs"] == 6
    assert metrics["v254_primary_matched_safe_cells"] == 0
    assert metrics["v254_primary_matched_safe_rigs"] == 0
    assert decision["v254_independent_validation_passed"] is True
    assert decision["v254_k1_set_subspace_closed"] is True
    assert decision["v254_resource_gate_authorized"] is False
    assert decision["v254_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith("blastnet_case19_galerkin_pyramid_v256.png")
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_k1_set_subspace_v254" in text
        assert "383/429" in text and "0/429" in text and "29/29" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert "v254" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v254_public_artifacts_exclude_private_execution_details() -> None:
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
