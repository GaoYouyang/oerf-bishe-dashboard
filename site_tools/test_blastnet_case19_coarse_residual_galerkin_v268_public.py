from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_coarse_residual_galerkin_v268_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_coarse_residual_galerkin_v268_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_coarse_residual_galerkin_v268.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PAGES = (
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
)
LOG = ROOT / "docs/operator_3d_learning_log.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v268_summary_records_observation_only_matched_failure() -> None:
    data = _summary()
    results = data["results"]
    assert results["primary_absolute_pass_cells"] == 13
    assert results["primary_matched_pass_cells"] == 7
    assert results["matched_failure_counts"] == {
        "field": 0,
        "full_gradient": 0,
        "interior_gradient": 0,
        "observation": 6,
    }
    assert results["matched_ratio_p90_higher"][3] > 1.05
    assert max(results["matched_ratio_p90_higher"][:3]) < 1.05


def test_v268_summary_records_controls_cost_and_independence() -> None:
    data = _summary()
    assert data["results"]["same_work_full_row_control_pass_cells"] == 13
    assert data["results"]["cheaper_single_half_control_pass_cells"] == 13
    assert data["cost_ledger"]["primary_calls_A"] == 16
    assert data["cost_ledger"]["primary_calls_At"] == 15
    assert data["independent_validation"]["checks_passed"] == 25
    assert data["independent_validation"]["checks_total"] == 25
    assert data["independent_validation"]["maximum_camera_permutation_difference"] == 0
    assert data["independent_validation"]["end_to_end_physics_independence_proven"] is False


def test_v268_adjudication_closes_only_the_fixed_mechanism() -> None:
    data = _summary()
    decision = data["adjudication"]
    assert decision["status"] == "FAIL_CASE19_COARSE_RESIDUAL_GALERKIN_FRAME_ZERO_V268_4"
    assert decision["fixed_one_step_coarse_residual_route_closed"] is True
    assert decision["primary_matched_gate_passed"] is False
    assert decision["full_sequence_gate_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v268_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v268：" in text and "# v268:" in text
    assert "13/13" in text and "7/13" in text and "25/25" in text
    assert "post-open" in text and "algorithm_breakthrough=false" in text
    assert "完整序列结果" in text and "not a complete-sequence result" in text


def test_v268_figure_is_public_and_readable() -> None:
    assert FIGURE.exists() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2000
        assert image.height >= 900


def test_v268_remains_historical_after_v269_1() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["headline"].startswith("v270")
    assert current["metrics"]["v268_primary_matched_pass_cells"] == 7
    assert current["current_decision"]["v268_fixed_one_step_coarse_route_closed"] is True
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        assert "blastnet_case19_coarse_residual_galerkin_v268" in text
    assert "## 2026-08-27：v268" in LOG.read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-27"') == 1
    assert "v267" in daily


def test_v268_public_artifacts_exclude_private_execution_details() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (SUMMARY, RESULT, *PAGES)
    )
    forbidden = (
        "/Users/",
        "/Volumes/",
        "source-private",
        "FORMAL_EXIT_CODE",
        "VALIDATION_EXIT_CODE",
        "private-source-identity",
    )
    assert all(token not in text for token in forbidden)
