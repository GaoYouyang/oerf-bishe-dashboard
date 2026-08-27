from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_tail_subspace_attribution_v236_public_summary.json"
RESULT = ROOT / "docs/blastnet_case7_tail_subspace_attribution_v236_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case7_tail_subspace_attribution_v236.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v236_summary_records_the_independent_negative() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["new_condition_opened"] is False
    assert data["scope"]["prospective_or_external_result"] is False
    assert data["question"]["unique_primary_rank"] == 64
    assert data["results"]["loro_rank64_primary"]["complete_rigs_passed"] == 0
    assert data["results"]["loro_rank64_primary"]["global_p90_higher"] > 0.7
    assert data["independent_validation"]["complete_rigs_recomputed"] == 13
    assert data["adjudication"]["scientific_decision"] == (
        "FAIL_CASE7_LORO_TAIL_SUBSPACE_CAPACITY_V236"
    )
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v236_result_is_bilingual_and_keeps_the_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v236：" in text and "# v236:" in text
    for token in ("0/13", "0.731692", "0.805609", "6.66e-15", "504x8192"):
        assert token in text
    assert "开封后" in text and "post-open" in text
    assert "algorithm_breakthrough=false" in text


def test_v236_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2000
        assert image.height >= 900


def test_v236_remains_as_historical_parent_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "MIXED_OR_NEAR_FLAT_CASE19_HAAR_IRLS_NULL_LINE_V272"
    assert current["current_decision"]["v236_independent_validation_passed"] is True
    assert current["current_decision"]["v236_algorithm_breakthrough"] is False
    assert current["metrics"]["v236_loro_rank64_complete_rigs_passed"] == 0
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_haar_irls_null_line_attribution_v272.png"
    )

    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "FAIL_CASE7_LORO_TAIL_SUBSPACE_CAPACITY_V236" in text or (
            "blastnet_case7_tail_subspace_attribution_v236" in text
        )
        assert "0.731692" in text and "0.805609" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text

    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert log.index("## 2026-08-25：v241") < log.index(
        "## 2026-08-25：v240"
    ) < log.index(
        "## 2026-08-25：v237.2"
    ) < log.index(
        "## 2026-08-25：v239"
    ) < log.index("## 2026-08-25：v238") < log.index(
        "## 2026-08-25：v236"
    ) < log.index("## 2026-08-25：v235/v235.1")
    assert "post-open mechanism attribution" in log


def test_v236_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "4ba9a3b9",
    )
    assert all(token not in text for token in forbidden)
