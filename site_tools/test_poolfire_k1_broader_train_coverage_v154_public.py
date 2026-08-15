from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_broader_train_coverage_v154_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_broader_train_coverage_v154_result_2026-08-16.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_broader_train_coverage_v154.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_summary_preserves_broader_coverage_failure_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    outcome = payload["gate_outcome"]
    assert payload["formal_status"] == "PASS_FORMAL_BROADER_TRAIN_COVERAGE_EXECUTION_V154"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_BROADER_TRAIN_COVERAGE_V154"
    assert payload["scientific_decision"] == "FAIL_BROADER_TRAIN_COVERAGE_V154"
    assert payload["sample_count"] == 7400
    assert payload["active_camera_rows"] == 61050
    assert payload["supported_camera_rows"] == 53157
    assert outcome["aggregate_trajectory_pass_count"] == 7
    assert outcome["aggregate_trajectory_count"] == 10
    assert payload["trajectory_support"]["p45_size05"] == 0.1678951678951679
    assert payload["trajectory_support"]["p58_size03"] == 0.7762489762489763
    assert payload["trajectory_support"]["p58_size05"] == 0.8712530712530713
    assert payload["independent_recomputation"]["check_count"] == 20
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_public_surfaces_are_bilingual_and_point_to_v154() -> None:
    required = [
        "poolfire_k1_broader_train_coverage_v154",
        "FAIL_BROADER_TRAIN_COVERAGE_V154",
        "87.07%",
        "data-i18n-zh",
        "data-i18n-en",
    ]
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{needle} missing from {surface.name}"
    daily = SURFACES[2].read_text(encoding="utf-8")
    assert "42 天" in daily
    assert "Day 42" in daily


def test_current_evidence_closes_current_predictor_and_gpu() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v154_sample_count"] == 7400
    assert metrics["v154_global_supported_fraction"] == 0.8707125307125307
    assert metrics["v154_aggregate_trajectory_pass_count"] == 7
    assert metrics["v154_p45s05_supported_fraction"] == 0.1678951678951679
    assert decision["v154_broader_train_coverage_gate_passed"] is False
    assert decision["v154_current_cross_trajectory_predictor_route_closed"] is True
    assert decision["v154_predictor_training_authorized"] is False
    assert decision["v154_gpu_rental_authorized"] is False
    assert decision["algorithm_breakthrough"] is False
    assert "broader or real" in payload["next_scientific_gate_en"]


def test_result_states_independent_failure_and_claim_limits() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "FAIL_BROADER_TRAIN_COVERAGE_V154" in text
    assert "20/20" in text
    assert "87.07%" in text
    assert "16.79%" in text
    assert "algorithm_breakthrough=false" in text
    assert "GPU" in text


def test_public_artifacts_do_not_disclose_private_execution_details() -> None:
    forbidden = ["/Users/", "private_results", "private_worktrees"]
    for artifact in [SUMMARY, RESULT, EVIDENCE, *SURFACES]:
        text = artifact.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"private token {needle!r} leaked into {artifact.name}"


def test_figure_is_large_nonblank_png() -> None:
    with Image.open(FIGURE) as image:
        assert image.format == "PNG"
        assert image.width >= 2000
        assert image.height >= 800
        assert image.getbbox() is not None
