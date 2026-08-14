from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "docs/poolfire_k1_dual_direction_conditioned_identifiability_v146_"
    "public_summary.json"
)
RESULT = ROOT / (
    "docs/poolfire_k1_dual_direction_conditioned_identifiability_v146_"
    "result_2026-08-15.md"
)
FIGURE = ROOT / (
    "assets/figures/poolfire_k1_dual_direction_conditioned_"
    "identifiability_v146.png"
)
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
]


def test_summary_preserves_stage_a_failure_and_unrun_stage_b() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_DIRECTION_CONDITIONED_IDENTIFIABILITY_V146"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_DIRECTION_CONDITIONED_V146"
    assert payload["evaluation"]["stage_a_sentinel_count"] == 20
    assert payload["evaluation"]["stage_b_full_roster_run"] is False
    assert payload["route_action"]["stage_b_full_roster_authorized"] is False
    assert payload["route_action"]["gpu_rental_authorized"] is False
    assert payload["methods"]["cross_local_only"]["sentinel_pass_count"] == 1
    assert payload["methods"]["within_local_only"]["sentinel_pass_count"] == 9
    assert all(method["all_pass"] is False for method in payload["methods"].values())
    assert payload["independent_recomputation"]["maximum_float_array_absolute_difference"] <= 1e-15
    assert all(value is False for value in payload["claim_boundary"].values())


def test_public_surfaces_are_bilingual_and_point_to_v146() -> None:
    required = [
        "poolfire_k1_dual_direction_conditioned_identifiability_v146",
        "FAIL_DIRECTION_CONDITIONED_IDENTIFIABILITY_V146",
        "data-i18n-zh",
        "data-i18n-en",
    ]
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{needle} missing from {surface.name}"


def test_current_evidence_keeps_gpu_and_breakthrough_closed() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v146_cross_local_pass_count"] == 1
    assert metrics["v146_within_local_pass_count"] == 9
    assert metrics["v146_stage_b_full_roster_run"] is False
    assert decision["v146_hard_count_direction_conditioned_neighbor_family_closed"] is True
    assert decision["v146_neural_training_authorized"] is False
    assert decision["gpu_rental_recommended_now"] is False
    assert decision["algorithm_breakthrough"] is False


def test_result_explicitly_rejects_zero_over_3700_wording() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "准确结论不是 `0/3700`" in text
    assert "must not be reported as `0/3700`" in text
    assert "algorithm_breakthrough=false" in text


def test_figure_is_large_nonblank_png() -> None:
    with Image.open(FIGURE) as image:
        assert image.format == "PNG"
        assert image.width >= 2200
        assert image.height >= 1000
        assert image.getbbox() is not None
