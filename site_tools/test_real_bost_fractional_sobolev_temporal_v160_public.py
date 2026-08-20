from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_fractional_sobolev_temporal_v160_public_summary.json"
RESULT = ROOT / "docs/real_bost_fractional_sobolev_temporal_v160_result_2026-08-19.md"
FIGURE = ROOT / "assets/figures/real_bost_fractional_sobolev_temporal_v160.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v160_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_FRACTIONAL_SOBOLEV_TEMPORAL_V160"
    assert payload["primary"]["fractional_order"] == 0.5
    assert payload["primary"]["strata_passed"] == 8
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    five_camera = [
        row for row in payload["primary"]["strata"] if row["camera_count"] == 5
    ]
    assert len(five_camera) == 4
    assert all(not row["passed"] for row in five_camera)
    assert all(row["gradient_p90"] > 0.75 for row in five_camera)
    assert payload["execution"]["independent_validity_checks"] == 19
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "放松高频惩罚没有救回五相机梯度" in text
    assert "weaker high-frequency attenuation does not repair" in text
    assert "0.809636" in text
    assert "FAIL_FRACTIONAL_SOBOLEV_TEMPORAL_V160" in text


def test_public_surfaces_point_to_v160_in_both_languages() -> None:
    texts = [path.read_text(encoding="utf-8") for path in SURFACES]
    assert any("real_bost_fractional_sobolev_temporal_v160" in text for text in texts)
    assert any("FAIL_FRACTIONAL_SOBOLEV_TEMPORAL_V160" in text for text in texts)
    for text in texts:
        assert "data-i18n-zh" in text
        assert "data-i18n-en" in text


def test_public_figure_is_nonblank_and_wide() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1180)
        extrema = image.convert("RGB").getextrema()
        assert all(low < high for low, high in extrema)
