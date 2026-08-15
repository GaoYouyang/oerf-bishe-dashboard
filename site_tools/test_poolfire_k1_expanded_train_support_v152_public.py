from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_expanded_train_support_v152_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_expanded_train_support_v152_result_2026-08-15.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_expanded_train_support_v152.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_summary_preserves_expanded_train_failure_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_raw_camera_state"]
    assert payload["formal_status"] == "PASS_FORMAL_EXPANDED_TRAIN_SUPPORT_EXECUTION_V152"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_EXPANDED_TRAIN_SUPPORT_V152"
    assert payload["scientific_decision"] == "FAIL_P33_SAME_POWER_MUTUAL_SUPPORT_V152"
    assert payload["added_sample_count"] == 740
    assert payload["combined_sample_count"] == 4440
    assert payload["active_camera_rows"] == 36630
    assert primary["p33_size01_rows_rescued_by_added_trajectory"] == 265
    assert primary["p33_size01_expanded_by_camera_count"]["5"] == 0.8367567567567568
    assert primary["p33_size03_heldout_by_camera_count"]["5"] == 0.9567567567567568
    assert primary["all_p33_camera_strata_passed"] is False
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_public_surfaces_are_bilingual_and_point_to_v152() -> None:
    required = [
        "poolfire_k1_expanded_train_support_v152",
        "FAIL_P33_SAME_POWER_MUTUAL_SUPPORT_V152",
        "83.68%",
        "data-i18n-zh",
        "data-i18n-en",
    ]
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{needle} missing from {surface.name}"


def test_current_evidence_closes_predictor_replay_and_gpu() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v152_added_sample_count"] == 740
    assert metrics["v152_active_camera_row_count"] == 36630
    assert metrics["v152_p33s01_five_camera_expanded_supported_fraction"] == 0.8367567567567568
    assert metrics["v152_p33s03_five_camera_supported_fraction"] == 0.9567567567567568
    assert metrics["v152_p33s01_rows_rescued"] == 265
    assert decision["v152_expanded_train_support_gate_passed"] is False
    assert decision["v152_predictor_training_authorized"] is False
    assert decision["v152_physical_replay_authorized"] is False
    assert decision["v152_gpu_rental_authorized"] is False
    assert decision["algorithm_breakthrough"] is False
    assert "canonicalization" in payload["next_scientific_gate_en"]


def test_result_states_independent_failure_and_claim_limits() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "FAIL_P33_SAME_POWER_MUTUAL_SUPPORT_V152" in text
    assert "17/17" in text
    assert "The scientific decision is" in text
    assert "does not fit a predictor" in text
    assert "algorithm_breakthrough=false" in text
    assert "gpu_rental_authorized=false" in text


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
