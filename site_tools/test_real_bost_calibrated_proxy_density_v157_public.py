from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_calibrated_proxy_density_v157_public_summary.json"
RESULT = ROOT / "docs/real_bost_calibrated_proxy_density_v157_result_2026-08-17.md"
FIGURE = ROOT / "assets/figures/real_bost_calibrated_proxy_density_v157.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_scientific_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_REFERENCE_ADEQUACY_V157"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_PRIVATE_CALIBRATED_PROXY_DENSITY_V157"
    assert payload["primary"]["by_camera_count"]["5"]["passed"] is False
    assert payload["primary"]["by_camera_count"]["7"]["passed"] is False
    assert payload["primary"]["by_camera_count"]["9"]["passed"] is True
    assert payload["dct1024_oracle_capacity"]["passed"] is True
    assert payload["claim_limits"]["paired_experimental_2d_projections_used"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["real_bost"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "九相机受控代理过参考门" in text
    assert "the nine-camera controlled proxy clears the reference gate" in text
    assert "not 117 independent real experiments" in text
    assert "FAIL_REFERENCE_ADEQUACY_V157" in text


def test_public_surfaces_point_to_v157_in_both_languages() -> None:
    for path in SURFACES:
        text = path.read_text(encoding="utf-8")
        assert "real_bost_calibrated_proxy_density_v157" in text
        assert "FAIL_REFERENCE_ADEQUACY_V157" in text
        assert "data-i18n-zh" in text
        assert "data-i18n-en" in text


def test_public_figure_is_nonblank_and_wide() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1040)
        extrema = image.convert("RGB").getextrema()
        assert all(low < high for low, high in extrema)
