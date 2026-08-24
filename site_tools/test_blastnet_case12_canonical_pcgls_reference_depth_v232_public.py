from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case12_canonical_pcgls_reference_depth_v232_public_summary.json"
RESULT = ROOT / "docs/blastnet_case12_canonical_pcgls_reference_depth_v232_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case12_canonical_pcgls_reference_depth_v232.png"


def test_v232_summary_preserves_inconclusive_reference_adjudication() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["execution"]["formal_cells_completed"] == 598
    assert data["execution"]["independent_cells_completed"] == 598
    assert all(value == 0.0 for value in data["canonicalization"].values())
    numerical = data["numerical_adjudication"]
    assert numerical["scientific_decision"] == (
        "INCONCLUSIVE_INVALID_CASE12_CANONICAL_PCGLS_REFERENCE_DEPTH_V232_1"
    )
    assert numerical["k16_field_relative_maximum"] < numerical["field_relative_tolerance"]
    assert numerical["first_field_failure_relative_difference"] > numerical["field_relative_tolerance"]
    assert numerical["first_field_failure_depth"] == 17
    assert data["provisional_accuracy_only"]["k17_released_as_reference"] is False
    assert data["adjudication"]["selected_depth"] is None
    assert data["adjudication"]["current_canonical_pcgls_reference_shell_closed"]
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v232_result_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v232/v232.1：" in text and "# v232/v232.1:" in text
    for token in ("598/598", "selected_depth=null", "1.67429e-8", "2.24977e-16"):
        assert token in text
    assert "不能作为合格 reference" in text
    assert "cannot serve as an adequate reference" in text
    assert "algorithm_breakthrough=false" in text


def test_v232_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v232_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["scientific_status"] == (
        "INCONCLUSIVE_INVALID_CASE12_CANONICAL_PCGLS_REFERENCE_DEPTH_V232_1"
    )
    assert current["metrics"]["v232_1_first_field_failure_depth"] == 17
    assert current["current_decision"]["v232_1_selected_depth"] is None
    assert current["current_decision"]["v232_1_reference_depth_adjudicated"] is False
    assert current["current_decision"]["v232_1_algorithm_breakthrough"] is False
    assert "stable" in current["next_scientific_gate"].lower()
    assert "reference_depth_v232" in current["public_evidence"]["result"]
    for relative in (
        "index.html",
        "operator-learning/index.html",
        "operator-learning/daily-progress.html",
    ):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v232.1" in content
        assert "blastnet_case12_canonical_pcgls_reference_depth_v232.png" in content
        assert "1.674" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v232.1 把问题收窄" in log
    assert "v232.1 narrows the blocker" in log


def test_v232_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
    )
    assert all(token not in text for token in forbidden)
