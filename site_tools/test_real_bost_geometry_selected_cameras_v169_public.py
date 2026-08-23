from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_geometry_selected_cameras_v169_public_summary.json"
RESULT = ROOT / "docs/real_bost_geometry_selected_cameras_v169_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/real_bost_geometry_selected_cameras_v169.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v169_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_GEOMETRY_SELECTED_CAMERAS_V169"
    assert payload["primary"]["strata_passed"] == 8
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    failed = [row for row in payload["primary"]["strata"] if not row["passed"]]
    assert [(row["time"], row["camera_count"]) for row in failed] == [
        (0.0, 5),
        (0.25, 5),
        (0.75, 5),
        (1.0, 5),
    ]
    assert failed[2]["gradient_p90"] == 0.8959141946922606
    assert (
        payload["selection_audit"][
            "five_camera_subsets_different_from_previous_fixed_roster"
        ]
        == 13
    )
    assert payload["selection_audit"]["five_camera_unique_selected_subsets"] == 10
    assert payload["execution"]["independent_validity_checks"] == 27
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "纯几何相机选择反而放大" in text
    assert "geometry-only camera selection amplifies" in text
    assert "13,299" in text
    assert "FAIL_GEOMETRY_SELECTED_CAMERAS_V169" in text
    assert "algorithm_breakthrough=false" in text


def test_public_surfaces_keep_v169_as_traceable_parent_evidence() -> None:
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        assert "v169" in text
        assert "algorithm_breakthrough" in text
    for surface in SURFACES[:2]:
        text = surface.read_text(encoding="utf-8")
        assert "real_bost_geometry_selected_cameras_v169_result_2026-08-21.md" in text
        assert "FAIL_GEOMETRY_SELECTED_CAMERAS_V169" in text
        assert "8/12" in text
    assert "0.895914" in SURFACES[1].read_text(encoding="utf-8")


def test_current_evidence_preserves_v169_but_points_to_v181() -> None:
    payload = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert payload["v169_geometry_selected_cameras_scientific_decision"] == (
        "FAIL_GEOMETRY_SELECTED_CAMERAS_V169"
    )
    assert payload["current_decision"]["v169_geometry_selected_cameras_passed"] is False
    assert payload["current_decision"]["v169_predictor_training_authorized"] is False
    assert (
        payload["engineering_status"]
        == "PASS_INDEPENDENT_RECOMPUTATION_ZERO_CGLS_REFERENCE_ADEQUACY_V208"
    )
    assert (
        payload["formal_status"]
            == "FORMAL_PENDING_INDEPENDENT_ZERO_CGLS_REFERENCE_ADEQUACY_V208"
    )
    assert payload["scientific_status"] == (
        "INCONCLUSIVE_CASE5_REFERENCE_REMAINS_INADEQUATE_AT_ZERO_CGLS_K16_V208"
    )


def test_public_figure_is_large_and_nonblank() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.convert("RGB").getbbox() is not None


def test_public_payload_does_not_expose_private_execution_material() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = [
        "/Users/",
        "private_results",
        "private_data",
        "model_tree_seal",
        "calibration_tree_seal",
        "checkpoint",
        "cameraData",
        "hexplane",
    ]
    for value in forbidden:
        assert value not in text
