from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_detector_krylov_capacity_v148_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_dual_detector_krylov_capacity_v148_result_2026-08-15.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_dual_detector_krylov_capacity_v148.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html"]


def test_summary_preserves_capacity_only_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "HEADROOM_BLOCK_DETECTOR_KRYLOV4_V148"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_DETECTOR_KRYLOV_CAPACITY_V148"
    assert payload["methods"]["block_krylov4"]["sentinel_pass_count"] == 20
    assert payload["methods"]["block_krylov4"]["trajectory_pass_count"] == 5
    assert payload["methods"]["global_krylov4"]["trajectory_pass_count"] == 3
    assert payload["route_action"]["gpu_rental_authorized"] is False
    assert payload["claim_boundary"]["deployable_algorithm"] is False
    assert payload["claim_boundary"]["algorithm_breakthrough"] is False


def test_public_surfaces_are_bilingual_and_point_to_v148() -> None:
    required = ["poolfire_k1_dual_detector_krylov_capacity_v148", "HEADROOM_BLOCK_DETECTOR_KRYLOV4_V148", "data-i18n-zh", "data-i18n-en"]
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{needle} missing from {surface.name}"


def test_current_evidence_keeps_gpu_and_algorithm_claims_closed() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v148_block_krylov4_pass_count"] == 20
    assert metrics["v148_block_krylov4_trajectory_pass_count"] == 5
    assert metrics["v148_global_krylov4_trajectory_pass_count"] == 3
    assert decision["v148_groupwise_spectral_capacity_headroom"] is True
    assert decision["v148_observation_only_predictor_passed"] is False
    assert decision["gpu_rental_recommended_now"] is False
    assert decision["algorithm_breakthrough"] is False


def test_result_distinguishes_oracle_capacity_from_algorithm() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "真值 oracle 容量" in text
    assert "不是已经可部署的算法" in text
    assert "genuine mechanism-capacity headroom" in text
    assert "gpu_rental_authorized=false" in text


def test_figure_is_large_nonblank_png() -> None:
    with Image.open(FIGURE) as image:
        assert image.format == "PNG"
        assert image.width >= 2200
        assert image.height >= 1000
        assert image.getbbox() is not None
