from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_geometry_anisotropic_h1_temporal_v161_public_summary.json"
RESULT = ROOT / "docs/real_bost_geometry_anisotropic_h1_temporal_v161_result_2026-08-19.md"
FIGURE = ROOT / "assets/figures/real_bost_geometry_anisotropic_h1_temporal_v161.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v161_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_GEOMETRY_ANISOTROPIC_H1_TEMPORAL_V161"
    assert payload["primary"]["multiplier"] == 0.03
    assert payload["primary"]["strata_passed"] == 11
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    failed = [row for row in payload["primary"]["strata"] if not row["passed"]]
    assert failed == [
        {
            "time": 0.75,
            "camera_count": 5,
            "field_p90": 0.41790457829893674,
            "gradient_p90": 0.7681965846720447,
            "observation_p90": 0.11942435577776343,
            "passed": False,
        }
    ]
    assert payload["frozen_h1_control"]["primary_minus_h1_failed_gradient_p90"] > 0
    assert payload["execution"]["independent_validity_checks"] == 19
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "几何各向异性保住 11/12 分层" in text
    assert "geometry-derived diagonal anisotropy does not repair" in text
    assert "0.768197" in text
    assert "FAIL_GEOMETRY_ANISOTROPIC_H1_TEMPORAL_V161" in text


def test_public_surfaces_point_to_v161_in_both_languages() -> None:
    for path in SURFACES:
        text = path.read_text(encoding="utf-8")
        assert "real_bost_geometry_anisotropic_h1_temporal_v161" in text
        assert "FAIL_GEOMETRY_ANISOTROPIC_H1_TEMPORAL_V161" in text
        assert "data-i18n-zh" in text
        assert "data-i18n-en" in text


def test_public_figure_is_nonblank_and_wide() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1180)
        extrema = image.convert("RGB").getextrema()
        assert all(low < high for low, high in extrema)
