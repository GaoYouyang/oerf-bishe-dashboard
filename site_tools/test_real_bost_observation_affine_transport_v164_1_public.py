from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/real_bost_observation_affine_transport_v164_1_public_summary.json"
)
RESULT = (
    ROOT / "docs/real_bost_observation_affine_transport_v164_1_result_2026-08-20.md"
)
FIGURE = ROOT / "assets/figures/real_bost_observation_affine_transport_v164_1.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v164_1_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_OBSERVATION_AFFINE_TRANSPORT_V164_1"
    assert payload["primary"]["strata_passed"] == 10
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    failed = [row for row in payload["primary"]["strata"] if not row["passed"]]
    assert [(row["time"], row["camera_count"]) for row in failed] == [
        (0.75, 5),
        (1.0, 5),
    ]
    assert failed[0]["gradient_p90"] == 0.7885312942827567
    assert failed[1]["gradient_worst"] == 1.1576682405159193
    assert payload["controls"]["no_transport_centered_h1"]["strata_passed"] == 10
    assert payload["execution"]["independent_validity_checks"] == 39
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "全局仿射输运没有救回" in text
    assert "global affine transport does not repair" in text
    assert "13A+1A^T" in text
    assert "FAIL_OBSERVATION_AFFINE_TRANSPORT_V164_1" in text
    assert "algorithm_breakthrough=false" in text


def test_public_surfaces_expose_the_same_verdict() -> None:
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        assert "FAIL_OBSERVATION_AFFINE_TRANSPORT_V164_1" in text
        assert "10/12" in text
        assert "0.788531" in text
        assert "algorithm_breakthrough" in text


def test_public_figure_is_large_and_nonblank() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1180)
        assert image.convert("RGB").getbbox() is not None


def test_public_payload_does_not_expose_private_execution_material() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = [
        "/Users/",
        "private_results",
        "private_data",
        "27184255",
        "model_tree_seal",
        "calibration_tree_seal",
    ]
    for value in forbidden:
        assert value not in text
