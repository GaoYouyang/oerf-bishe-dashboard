from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case7_jacobi_canonical_tail_subspace_capacity_v239_public_summary.json"
RESULT = ROOT / "docs/blastnet_case7_jacobi_canonical_tail_subspace_capacity_v239_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case7_jacobi_canonical_tail_subspace_capacity_v239.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


def test_v239_summary_records_the_independent_negative() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["scope"]["new_condition_opened"] is False
    assert data["question"]["rank"] == 64
    assert data["question"]["global_control_dimension"] == 64
    assert data["results"]["jacobi_canonical_rank64_primary"]["complete_rigs_passed"] == 0
    assert data["comparison"]["jacobi_late_frame_p90_worse_on_rigs"] == 13
    assert data["comparison"]["jacobi_minus_global_p50"] < 0
    assert data["comparison"]["jacobi_minus_global_p90_higher"] > 0
    assert data["comparison"]["jacobi_minus_global_worst"] > 0
    assert data["independent_validation"]["checks_passed"] == 18
    assert data["adjudication"]["scientific_decision"] == (
        "FAIL_CASE7_JACOBI_CANONICAL_TAIL_SUBSPACE_CAPACITY_V239"
    )
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v239_result_is_bilingual_and_keeps_the_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v239/v239.2：" in text and "# v239/v239.2:" in text
    for token in ("0/13", "0.734855", "0.813573", "504x8192", "18/18"):
        assert token in text
    assert "开封" in text and "post-open" in text
    assert "before any independent SVD" in text
    assert "algorithm_breakthrough=false" in text


def test_v239_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 1100


def test_v239_remains_synchronized_as_historical_public_evidence() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["v239_scientific_decision"] == (
        "FAIL_CASE7_JACOBI_CANONICAL_TAIL_SUBSPACE_CAPACITY_V239"
    )
    assert current["current_decision"]["v239_independent_validation_passed"] is True
    assert current["current_decision"]["v239_algorithm_breakthrough"] is False
    assert current["metrics"]["v239_jacobi_complete_rigs_passed"] == 0
    assert current["metrics"]["v239_jacobi_global_p90_higher"] > (
        current["metrics"]["v239_global_rank64_global_p90_higher"]
    )

    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "FAIL_CASE7_JACOBI_CANONICAL_TAIL_SUBSPACE_CAPACITY_V239" in text or (
            "blastnet_case7_jacobi_canonical_tail_subspace_capacity_v239" in text
        )
        assert "0.734855" in text and "0.813573" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text

    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert log.index("v237.2 确认") < log.index("v239")
    assert log.index("## 2026-08-25：v239") < log.index("## 2026-08-25：v238")
    assert "symmetric geometry-Jacobi" in log


def test_v239_public_artifacts_do_not_expose_private_execution_details() -> None:
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
