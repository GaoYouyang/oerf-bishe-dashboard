from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_signed_spatial_support_v151_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_signed_spatial_support_v151_result_2026-08-15.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_signed_spatial_support_v151.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_summary_preserves_signed_spatial_failure_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["formal_status"] == "PASS_FORMAL_SIGNED_SPATIAL_SUPPORT_EXECUTION_V151"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_SIGNED_SPATIAL_SUPPORT_V151"
    assert payload["scientific_decision"] == "FAIL_SIGNED_SPATIAL_CROSS_TRAJECTORY_SUPPORT_V151"
    assert payload["active_group_rows"] == 60654
    assert payload["baseline"]["global_supported_fraction"] == 0.8443466218221387
    assert payload["signed_spatial_peer_state"]["global_supported_fraction"] == 0.6741847198865697
    assert payload["signed_spatial_peer_state"]["component_strata_passing"] == 16
    assert payload["signed_spatial_peer_state"]["camera_strata_passing"] == 8
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_public_surfaces_are_bilingual_and_point_to_v151() -> None:
    focus_page = SURFACES[1].read_text(encoding="utf-8")
    assert "FAIL_SIGNED_SPATIAL_CROSS_TRAJECTORY_SUPPORT_V151" in focus_page
    assert "67.42%" in focus_page
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in ["data-i18n-zh", "data-i18n-en"]:
            assert needle in text, f"{needle} missing from {surface.name}"


def test_current_evidence_closes_state_predictor_and_gpu() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v151_active_group_row_count"] == 60654
    assert metrics["v151_signed_global_supported_fraction"] == 0.6741847198865697
    assert metrics["v151_signed_component_strata_passing"] == 16
    assert metrics["v151_signed_camera_strata_passing"] == 8
    assert decision["v151_signed_spatial_peer_state_closed"] is True
    assert decision["v151_predictor_training_authorized"] is False
    assert decision["v151_physical_replay_authorized"] is False
    assert decision["v151_gpu_rental_authorized"] is False
    assert decision["algorithm_breakthrough"] is False
    assert payload["v151_signed_spatial_support_formal_status"] == "PASS_FORMAL_SIGNED_SPATIAL_SUPPORT_EXECUTION_V151"
    assert payload["v151_signed_spatial_support_independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_SIGNED_SPATIAL_SUPPORT_V151"


def test_result_states_independent_failure_and_claim_limits() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "FAIL_SIGNED_SPATIAL_CROSS_TRAJECTORY_SUPPORT_V151" in text
    assert "第二实现" in text
    assert "The scientific decision is" in text
    assert "fits no target model" in text
    assert "algorithm_breakthrough=false" in text
    assert "gpu_rental_authorized=false" in text


def test_public_artifacts_do_not_disclose_private_execution_details() -> None:
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
    ]
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
