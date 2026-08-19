from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_virtual_temporal_reference_transfer_v159_public_summary.json"
RESULT = ROOT / "docs/real_bost_virtual_temporal_reference_transfer_v159_result_2026-08-18.md"
FIGURE = ROOT / "assets/figures/real_bost_virtual_temporal_reference_transfer_v159.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v159_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_TEMPORAL_REFERENCE_TRANSFER_V159_1"
    assert payload["primary"]["strata_passed"] == 11
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    assert payload["failed_stratum"]["time"] == 0.75
    assert payload["failed_stratum"]["camera_count"] == 5
    assert payload["failed_stratum"]["failed_metric"] == "gradient_p90"
    assert payload["failed_stratum"]["value"] > payload["failed_stratum"]["threshold"]
    assert payload["independent_recomputation"]["check_count"] == 17
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "虚拟时序配对可由代码生成" in text
    assert "virtual temporal pairings can be generated in code" in text
    assert "0.758639" in text
    assert "FAIL_TEMPORAL_REFERENCE_TRANSFER_V159_1" in text


def test_public_surfaces_point_to_v159_in_both_languages() -> None:
    for path in SURFACES[1:]:
        text = path.read_text(encoding="utf-8")
        assert "FAIL_TEMPORAL_REFERENCE_TRANSFER_V159_1" in text
        assert "data-i18n-zh" in text
        assert "data-i18n-en" in text
    daily = SURFACES[2].read_text(encoding="utf-8")
    assert "real_bost_virtual_temporal_reference_transfer_v159" in daily
    homepage = SURFACES[0].read_text(encoding="utf-8")
    assert "real_bost_fractional_sobolev_temporal_v160" in homepage
    assert "FAIL_FRACTIONAL_SOBOLEV_TEMPORAL_V160" in homepage


def test_public_figure_is_nonblank_and_wide() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1180)
        extrema = image.convert("RGB").getextrema()
        assert all(low < high for low, high in extrema)
