from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_calibrated_proxy_spectral_smoothness_v158_public_summary.json"
RESULT = ROOT / "docs/real_bost_calibrated_proxy_spectral_smoothness_v158_result_2026-08-17.md"
FIGURE = ROOT / "assets/figures/real_bost_calibrated_proxy_spectral_smoothness_v158.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v158_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_SPECTRAL_SMOOTHNESS_REFERENCE_V158"
    assert payload["independent_recomputation"]["check_count"] == 18
    assert payload["primary"]["by_camera_count"]["5"]["passed"] is False
    assert payload["primary"]["by_camera_count"]["7"]["passed"] is True
    assert payload["primary"]["by_camera_count"]["9"]["passed"] is True
    assert payload["fixed_lambda_diagnostics"]["can_replace_primary"] is False
    assert payload["fixed_lambda_diagnostics"]["multipliers_passing_all_absolute_camera_gates"] == [0.03, 0.1]
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "五相机场尾部仍未过门" in text
    assert "the five-camera field tail still fails" in text
    assert "diagnostic-only" in text
    assert "FAIL_SPECTRAL_SMOOTHNESS_REFERENCE_V158" in text


def test_public_surfaces_point_to_v158_in_both_languages() -> None:
    for path in SURFACES:
        text = path.read_text(encoding="utf-8")
        assert "real_bost_calibrated_proxy_spectral_smoothness_v158" in text
        assert "FAIL_SPECTRAL_SMOOTHNESS_REFERENCE_V158" in text
        assert "data-i18n-zh" in text
        assert "data-i18n-en" in text


def test_public_figure_is_nonblank_and_wide() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1040)
        extrema = image.convert("RGB").getextrema()
        assert all(low < high for low, high in extrema)
