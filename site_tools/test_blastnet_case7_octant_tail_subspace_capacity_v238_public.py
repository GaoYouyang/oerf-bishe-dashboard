from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_octant_tail_subspace_capacity_v238_public_summary.json"
RESULT = ROOT / "docs/blastnet_case7_octant_tail_subspace_capacity_v238_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case7_octant_tail_subspace_capacity_v238.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v238_summary_records_the_independent_negative() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["new_condition_opened"] is False
    assert data["question"]["total_dimension"] == 64
    assert data["question"]["global_control_dimension"] == 64
    assert data["results"]["octant_rank8_primary"]["complete_rigs_passed"] == 0
    assert data["comparison"]["octant_all_frame_p90_worse_on_rigs"] == 13
    assert data["comparison"]["octant_late_frame_p90_worse_on_rigs"] == 13
    assert data["independent_validation"]["checks_passed"] == 12
    assert data["adjudication"]["scientific_decision"] == (
        "FAIL_CASE7_OCTANT_TAIL_SUBSPACE_CAPACITY_V238"
    )
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v238_result_is_bilingual_and_keeps_the_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v238：" in text and "# v238:" in text
    for token in ("0/13", "0.751069", "0.833760", "504x1024", "12/12"):
        assert token in text
    assert "开封后" in text and "post-open" in text
    assert "algorithm_breakthrough=false" in text


def test_v238_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 1100


def test_v238_remains_synchronized_as_historical_parent_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "PASS_LINEAR_VIRTUAL_PIXEL_INTERFACE_V282"
    assert current["current_decision"]["v238_independent_validation_passed"] is True
    assert current["current_decision"]["v238_algorithm_breakthrough"] is False
    assert current["metrics"]["v238_octant_complete_rigs_passed"] == 0
    assert current["metrics"]["v238_octant_global_p90_higher"] > (
        current["metrics"]["v238_global_rank64_global_p90_higher"]
    )

    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "FAIL_CASE7_OCTANT_TAIL_SUBSPACE_CAPACITY_V238" in text or (
            "blastnet_case7_octant_tail_subspace_capacity_v238" in text
        )
        assert "0.751069" in text and "0.833760" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text

    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert log.index("## 2026-08-25：v241") < log.index(
        "## 2026-08-25：v240"
    ) < log.index(
        "## 2026-08-25：v237.2"
    ) < log.index(
        "## 2026-08-25：v239"
    ) < log.index("## 2026-08-25：v238") < log.index("## 2026-08-25：v236")
    assert "fixed spatial locality" in log


def test_v238_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "6cf54968",
    )
    assert all(token not in text for token in forbidden)
