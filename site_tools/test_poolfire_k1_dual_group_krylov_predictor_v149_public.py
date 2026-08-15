from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_dual_group_krylov_predictor_v149_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_dual_group_krylov_predictor_v149_result_2026-08-15.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_dual_group_krylov_predictor_v149.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_summary_preserves_inconclusive_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["formal_scientific_decision"] == "FAIL_OBSERVATION_ONLY_GROUP_KRYLOV_PREDICTOR_V149"
    assert payload["independent_status"] == "INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_GROUP_KRYLOV_PREDICTOR_V149"
    assert payload["methods"]["oracle_block_krylov4"]["formal_cell_pass_count"] == 3700
    assert payload["methods"]["linear_set_ridge"]["formal_cell_pass_count"] == 3089
    assert payload["methods"]["linear_set_ridge"]["trajectory_pass_count"] == 0
    assert payload["route_action"]["current_group_coordinate_predictor_family_closed"] is True
    assert payload["route_action"]["gpu_rental_authorized"] is False
    assert payload["claim_boundary"]["independently_validated_formal_negative"] is False
    assert payload["claim_boundary"]["algorithm_breakthrough"] is False


def test_public_surfaces_are_bilingual_and_point_to_v149() -> None:
    required = [
        "poolfire_k1_dual_group_krylov_predictor_v149",
        "INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_GROUP_KRYLOV_PREDICTOR_V149",
        "data-i18n-zh",
        "data-i18n-en",
    ]
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{needle} missing from {surface.name}"


def test_current_evidence_closes_predictor_gpu_and_algorithm_claims() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v149_oracle_cell_pass_count"] == 3700
    assert metrics["v149_linear_cell_pass_count"] == 3089
    assert metrics["v149_linear_trajectory_pass_count"] == 0
    assert decision["v149_current_group_coordinate_predictor_family_closed"] is True
    assert decision["v149_formal_negative_independently_validated"] is False
    assert decision["gpu_rental_recommended_now"] is False
    assert decision["algorithm_breakthrough"] is False


def test_result_never_converts_inconclusive_into_success_or_impossibility() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "必须保持" in text
    assert "INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_GROUP_KRYLOV_PREDICTOR_V149" in text
    assert "不能被改写成“数学上证明预测不可能”" in text
    assert "neither an algorithmic success nor a proof of impossibility" in text


def test_figure_is_large_nonblank_png() -> None:
    with Image.open(FIGURE) as image:
        assert image.format == "PNG"
        assert image.width >= 2200
        assert image.height >= 1000
        assert image.getbbox() is not None
