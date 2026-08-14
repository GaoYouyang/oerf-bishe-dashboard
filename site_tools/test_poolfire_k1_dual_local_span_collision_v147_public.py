from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_local_span_collision_v147_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_dual_local_span_collision_v147_result_2026-08-15.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_dual_local_span_collision_v147.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [ROOT / "index.html", ROOT / "operator-learning/index.html"]


def test_summary_preserves_truth_aware_negative_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_LOCAL_SPAN_CAPACITY_V147"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_LOCAL_SPAN_COLLISION_V147"
    assert payload["methods"]["cross_span32"]["sentinel_pass_count"] == 14
    assert payload["methods"]["within_span32"]["sentinel_pass_count"] == 18
    assert payload["methods"]["cross_span32"]["trajectory_pass_count"] == 1
    assert payload["methods"]["within_span32"]["trajectory_pass_count"] == 2
    assert payload["evaluation"]["stage_b_full_trajectory_run"] is False
    assert payload["relative_neighborhood_conflict"]["exact_feature_collision_proven"] is False
    assert payload["route_action"]["gpu_rental_authorized"] is False
    assert all(value is False for value in payload["claim_boundary"].values())


def test_public_surfaces_are_bilingual_and_point_to_v147() -> None:
    required = ["poolfire_k1_dual_local_span_collision_v147", "FAIL_LOCAL_SPAN_CAPACITY_V147", "data-i18n-zh", "data-i18n-en"]
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{needle} missing from {surface.name}"


def test_current_evidence_keeps_training_and_gpu_closed() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v147_cross_span32_pass_count"] == 14
    assert metrics["v147_within_span32_pass_count"] == 18
    assert metrics["v147_cross_span32_trajectory_pass_count"] == 1
    assert metrics["v147_within_span32_trajectory_pass_count"] == 2
    assert decision["v147_sample_level_direction_local_span_k_le_32_closed"] is True
    assert decision["v147_neural_training_authorized"] is False
    assert decision["gpu_rental_recommended_now"] is False
    assert decision["algorithm_breakthrough"] is False


def test_result_distinguishes_relative_conflict_from_exact_collision() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "相对邻域冲突" in text
    assert "不是精确 feature collision" in text
    assert "not proof of an exact feature collision" in text
    assert "gpu_rental_authorized=false" in text


def test_figure_is_large_nonblank_png() -> None:
    with Image.open(FIGURE) as image:
        assert image.format == "PNG"
        assert image.width >= 2200
        assert image.height >= 1000
        assert image.getbbox() is not None
