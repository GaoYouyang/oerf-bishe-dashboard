from __future__ import annotations

import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_support_root_cause_v155_public_summary.json"
RESULT = ROOT / "docs/poolfire_k1_support_root_cause_v155_result_2026-08-17.md"
FIGURE = ROOT / "assets/figures/poolfire_k1_support_root_cause_v155.png"
EVIDENCE = ROOT / "operator-learning/current-evidence.json"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_summary_preserves_mixed_support_root_cause_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    shares = payload["unsupported_aggregate_group_share"]
    assert payload["formal_status"] == "PASS_FORMAL_POSTOPEN_SUPPORT_ROOT_CAUSE_V155"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_SUPPORT_ROOT_CAUSE_V155"
    assert payload["scientific_decision"] == "ROOT_CAUSE_MIXED_SUPPORT_GAP_V155"
    assert payload["active_camera_rows"] == 61050
    assert payload["failed_trajectory_unsupported_rows"] == {
        "p45_size05": 5080,
        "p58_size03": 1366,
        "p58_size05": 786,
    }
    assert shares["p45_size05"]["deployment_visible_state"] > 0.71
    assert shares["p58_size03"]["reported_geometry"] > 0.38
    assert shares["p58_size05"]["reported_geometry"] > 0.38
    assert payload["independent_recomputation"]["all_scientific_checks_passed"] is True
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_public_surfaces_are_bilingual_and_point_to_v155() -> None:
    required = [
        "poolfire_k1_support_root_cause_v155",
        "ROOT_CAUSE_MIXED_SUPPORT_GAP_V155",
        "71.91%",
        "data-i18n-zh",
        "data-i18n-en",
    ]
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{needle} missing from {surface.name}"
    daily = SURFACES[2].read_text(encoding="utf-8")
    assert "47 天" in daily
    assert "Day 44" in daily


def test_current_evidence_preserves_route_closure() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    decision = payload["current_decision"]
    assert metrics["v155_p45s05_state_share"] == 0.7191090781221176
    assert metrics["v155_p58s03_geometry_share"] == 0.3866431535401504
    assert metrics["v155_p58s05_geometry_share"] == 0.38586017656260674
    assert decision["v155_mixed_support_gap_confirmed"] is True
    assert decision["v155_geometry_only_explanation_supported"] is False
    assert decision["v155_temporal_only_explanation_supported"] is False
    assert decision["v155_predictor_training_authorized"] is False
    assert decision["v155_gpu_rental_authorized"] is False
    assert decision["algorithm_breakthrough"] is False


def test_result_states_independent_attribution_and_claim_limits() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "ROOT_CAUSE_MIXED_SUPPORT_GAP_V155" in text
    assert "71.91%" in text
    assert "38.66%" in text
    assert "1.78e-15" in text
    assert "algorithm_breakthrough=false" in text
    assert "GPU" in text


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
