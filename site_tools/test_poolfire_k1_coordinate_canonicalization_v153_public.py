from __future__ import annotations

import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_coordinate_canonicalization_v153_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_coordinate_canonicalization_v153_result_2026-08-16.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_coordinate_canonicalization_v153.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_summary_preserves_negative_coordinate_support_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    p33 = payload["p33_size01_camera_support"]
    assert payload["formal_status"] == "PASS_FORMAL_COORDINATE_CANONICALIZATION_EXECUTION_V153"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_COORDINATE_CANONICALIZATION_V153"
    assert payload["scientific_decision"] == "FAIL_TARGET_FREE_MONOTONE_COORDINATE_SUPPORT_V153"
    assert payload["sample_count"] == 4440
    assert payload["active_camera_rows"] == 36630
    assert p33["v152_raw"]["5"] == 0.8367567567567568
    assert p33["v153_monotone"]["5"] == 0.7113513513513513
    assert p33["v153_monotone"]["7"] == 0.8293436293436294
    assert payload["monotone_trajectory_support"]["p45_size05"] == 0.076003276003276
    assert payload["frozen_gate_outcome"]["previously_passing_strata_preserved"] is False
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_public_surfaces_are_bilingual_and_point_to_v153() -> None:
    historical_required = [
        "FAIL_TARGET_FREE_MONOTONE_COORDINATE_SUPPORT_V153",
        "71.14%",
    ]
    bilingual_required = [
        "data-i18n-zh",
        "data-i18n-en",
    ]
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in bilingual_required:
            assert needle in text, f"{needle} missing from {surface.name}"
    operator_text = SURFACES[1].read_text(encoding="utf-8")
    for needle in historical_required:
        assert needle in operator_text, f"{needle} missing from {SURFACES[1].name}"
    assert "poolfire_k1_coordinate_canonicalization_v153" in SURFACES[2].read_text(encoding="utf-8")
    assert "poolfire_k1_support_root_cause_v155" in SURFACES[0].read_text(encoding="utf-8")
    daily = SURFACES[2].read_text(encoding="utf-8")
    assert "46 天" in daily
    assert "Day 44" in daily


def test_current_evidence_closes_predictor_and_gpu() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v153_p33s01_five_camera_monotone_supported_fraction"] == 0.7113513513513513
    assert metrics["v153_p45_monotone_supported_fraction"] == 0.076003276003276
    assert decision["v153_coordinate_canonicalization_gate_passed"] is False
    assert decision["v153_prior_passed_strata_preserved"] is False
    assert decision["v153_current_cross_trajectory_predictor_route_closed"] is True
    assert decision["v153_predictor_training_authorized"] is False
    assert decision["v153_gpu_rental_authorized"] is False
    assert decision["algorithm_breakthrough"] is False
    assert payload["v153_coordinate_canonicalization_scientific_decision"] == "FAIL_TARGET_FREE_MONOTONE_COORDINATE_SUPPORT_V153"
    assert payload["v154_broader_train_coverage_scientific_decision"] == "FAIL_BROADER_TRAIN_COVERAGE_V154"


def test_result_states_independent_failure_and_claim_limits() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "FAIL_TARGET_FREE_MONOTONE_COORDINATE_SUPPORT_V153" in text
    assert "15/15" in text
    assert "71.14%" in text
    assert "7.60%" in text
    assert "algorithm_breakthrough=false" in text
    assert "gpu_rental_authorized=false" in text


def test_public_artifacts_do_not_disclose_private_execution_details() -> None:
    forbidden = ["/Users/", "private_results", "private_worktrees"]
    for artifact in [SUMMARY, RESULT, EVIDENCE, *SURFACES]:
        text = artifact.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"private token {needle!r} leaked into {artifact.name}"


def test_figure_is_large_nonblank_png() -> None:
    raw = FIGURE.read_bytes()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", raw[16:24])
    assert width >= 2000
    assert height >= 800
    assert len(raw) >= 50_000
