from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_temporal_innovation_l2_v163_public_summary.json"
RESULT = ROOT / "docs/real_bost_temporal_innovation_l2_v163_result_2026-08-20.md"
FIGURE = ROOT / "assets/figures/real_bost_temporal_innovation_l2_v163.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v163_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert payload["scientific_decision"] == "FAIL_TEMPORAL_INNOVATION_L2_V163"
    assert payload["primary"]["strata_passed"] == 7
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    failed = [row for row in payload["primary"]["strata"] if not row["passed"]]
    assert len(failed) == 5
    worst = next(
        row for row in failed if row["time"] == 1.0 and row["camera_count"] == 5
    )
    assert worst["gradient_worst"] == 1.4335357105154565
    assert payload["same_scale_static_l2_control"]["strata_passed"] == 1
    assert payload["execution"]["independent_validity_checks"] == 28
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "单向时序系数持续在稀疏视角放大梯度尾部" in text
    assert "one-sided temporal coefficient persistence amplifies" in text
    assert "0.939342 / 1.358706" in text
    assert "FAIL_TEMPORAL_INNOVATION_L2_V163" in text


def test_public_surfaces_point_to_v163_in_both_languages() -> None:
    texts = [path.read_text(encoding="utf-8") for path in SURFACES]
    assert any("real_bost_temporal_innovation_l2_v163" in text for text in texts)
    for path in SURFACES:
        text = path.read_text(encoding="utf-8")
        assert "FAIL_TEMPORAL_INNOVATION_L2_V163" in text
        assert "data-i18n-zh" in text
        assert "data-i18n-en" in text

    assert any("7/12" in text for text in texts)


def test_current_evidence_preserves_independent_v163() -> None:
    payload = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert (
        payload["v163_temporal_innovation_l2_scientific_decision"]
        == "FAIL_TEMPORAL_INNOVATION_L2_V163"
    )
    assert payload["current_decision"]["v163_independently_recomputed"] is True
    assert payload["current_decision"]["v163_temporal_innovation_l2_passed"] is False
    assert payload["current_decision"]["algorithm_breakthrough"] is False


def test_public_figure_is_nonblank_and_wide() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1180)
        extrema = image.convert("RGB").getextrema()
        assert all(low < high for low, high in extrema)
