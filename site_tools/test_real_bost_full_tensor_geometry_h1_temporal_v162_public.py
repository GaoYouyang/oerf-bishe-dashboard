from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_full_tensor_geometry_h1_temporal_v162_public_summary.json"
RESULT = ROOT / "docs/real_bost_full_tensor_geometry_h1_temporal_v162_result_2026-08-20.md"
FIGURE = ROOT / "assets/figures/real_bost_full_tensor_geometry_h1_temporal_v162.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v162_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_FULL_TENSOR_GEOMETRY_H1_TEMPORAL_V162"
    assert payload["primary"]["multiplier"] == 0.03
    assert payload["primary"]["strata_passed"] == 11
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    failed = [row for row in payload["primary"]["strata"] if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["time"] == 0.75 and failed[0]["camera_count"] == 5
    assert failed[0]["gradient_p90"] == 0.7510347942945529
    comparison = payload["failed_stratum_comparison"]
    assert comparison["full_tensor_minus_gate"] > 0
    assert comparison["full_tensor_minus_isotropic_h1"] < 0
    assert comparison["full_tensor_minus_diagonal_h1"] < 0
    assert payload["execution"]["independent_validity_checks"] == 21
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "全张量几何耦合确实改善尾部" in text
    assert "full-tensor geometry coupling improves" in text
    assert "0.751035" in text
    assert "FAIL_FULL_TENSOR_GEOMETRY_H1_TEMPORAL_V162" in text


def test_public_surfaces_point_to_v162_in_both_languages() -> None:
    for path in SURFACES:
        text = path.read_text(encoding="utf-8")
        assert "real_bost_full_tensor_geometry_h1_temporal_v162" in text
        assert "FAIL_FULL_TENSOR_GEOMETRY_H1_TEMPORAL_V162" in text
        assert "data-i18n-zh" in text
        assert "data-i18n-en" in text


def test_public_figure_is_nonblank_and_wide() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1180)
        extrema = image.convert("RGB").getextrema()
        assert all(low < high for low, high in extrema)
