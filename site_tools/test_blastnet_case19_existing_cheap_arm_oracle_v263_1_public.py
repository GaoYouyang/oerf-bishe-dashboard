from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_existing_cheap_arm_oracle_v263_1_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_existing_cheap_arm_oracle_v263_1_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_existing_cheap_arm_oracle_v263_1.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PRIMARY_PAGES = (
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
)


def test_v263_1_summary_records_the_oracle_no_go() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["candidate_arms"] == 9
    assert data["scope"]["rigs"] == 13
    assert data["scope"]["new_exact_calls_A"] == 0
    assert data["scope"]["new_exact_calls_At"] == 0
    assert data["oracle"]["joint_pass_rigs"] == 0
    assert all(arm["matched_pass_rigs"] == 0 for arm in data["arms"])
    validation = data["independent_validation"]
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 19
    assert validation["maximum_numeric_array_absolute_difference"] == 0.0
    assert data["adjudication"]["selector_over_these_nine_arms_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v263_1_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v263.1：" in text and "# v263.1:" in text
    for token in ("19/19", "0/13", "1.06082", "1.06693", "1.06876"):
        assert token in text
    assert "不关闭整条 C 路线" in text
    assert "does not close the entire C route" in text
    assert "algorithm_breakthrough=false" in text


def test_v263_1_figure_is_public_and_readable() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v263_1_remains_historical_after_v264_on_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["updated"] == "2026-08-27"
    assert current["scientific_status"].endswith("V267")
    assert current["metrics"]["v263_1_oracle_joint_pass_rigs"] == 0
    assert current["current_decision"]["v263_1_selector_over_nine_arms_authorized"] is False
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_two_color_additive_schwarz_v267.png"
    )
    for page in PRIMARY_PAGES:
        text = page.read_text(encoding="utf-8")
        assert "v263.1" in text
        assert "19/19" in text and "0/13" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text


def test_v263_1_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "protocol_sha256",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)
