from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case12_pcgls_reference_depth_v231_public_summary.json"
RESULT = ROOT / "docs/blastnet_case12_pcgls_reference_depth_v231_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case12_pcgls_reference_depth_v231.png"


def test_v231_summary_preserves_inconclusive_numerical_adjudication() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["execution"]["formal_cells_completed"] == 598
    assert data["execution"]["independent_cells_completed"] == 598
    assert data["execution"]["formal_depth_checkpoints_per_cell"] == 64
    numerical = data["numerical_adjudication"]
    assert numerical["scientific_decision"] == "INCONCLUSIVE_INVALID_CASE12_PCGLS_REFERENCE_DEPTH_V231"
    assert numerical["formal_camera_permutation_field_relative_maximum"] > numerical["camera_permutation_tolerance"]
    assert numerical["independent_camera_permutation_field_relative_maximum"] > numerical["camera_permutation_tolerance"]
    assert data["adjudication"]["selected_depth"] is None
    assert data["adjudication"]["adequate_depth_through_k64_proven"] is False
    assert data["adjudication"]["no_adequate_depth_through_k64_proven"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v231_result_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v231：" in text and "# v231:" in text
    for token in ("598 x 64", "selected_depth=null", "1.08496e-2", "8.35e-14"):
        assert token in text
    assert "不是\u201c已经证明 K64 以内没有合格 reference\u201d" in text
    assert "does not show that no adequate reference exists through K64" in text
    assert "algorithm_breakthrough=false" in text


def test_v231_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v231_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["scientific_status"] == "INCONCLUSIVE_INVALID_CASE12_PCGLS_REFERENCE_DEPTH_V231"
    assert current["metrics"]["v231_case12_cells_completed"] == 598
    assert current["metrics"]["v231_depth_states_per_cell"] == 64
    assert current["current_decision"]["v231_selected_depth"] is None
    assert current["current_decision"]["v231_reference_depth_adjudicated"] is False
    assert current["current_decision"]["v231_algorithm_breakthrough"] is False
    assert "canonical" in current["next_scientific_gate"].lower()
    assert "reference_depth_v231" in current["public_evidence"]["result"]
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v231" in content
        assert "blastnet_case12_pcgls_reference_depth_v231.png" in content
        assert "598 x 64" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v231 把 K1-K64 全部算完" in log
    assert "v231 completes every K1-K64" in log


def test_v231_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = ("/Users/", "private_results", "private_worktrees", "source_commit", "sha256", "checkpoint.pt")
    assert all(token not in text for token in forbidden)
