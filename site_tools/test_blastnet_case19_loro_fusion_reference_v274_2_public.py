from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_loro_fusion_reference_v274_2_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_loro_fusion_reference_v274_2_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_loro_fusion_reference_v274_2.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PAGES = (ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html")
LOG = ROOT / "docs/operator_3d_learning_log.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v274_2_is_inconclusive_and_closes_only_the_fixed_reference() -> None:
    data = _summary()
    audit = data["independent_validation"]
    decision = data["adjudication"]
    assert (audit["validity_checks_passed"], audit["validity_checks_total"]) == (27, 32)
    assert audit["maximum_formal_independent_field_relative_difference"] > audit["field_projection_and_residual_tolerance"]
    assert audit["maximum_formal_independent_target_projection_relative_difference"] > audit["field_projection_and_residual_tolerance"]
    assert audit["maximum_formal_independent_target_residual_relative_difference"] > audit["field_projection_and_residual_tolerance"]
    assert decision["status"] == "INCONCLUSIVE_CASE19_LORO_FUSION_REFERENCE_V274_2"
    assert decision["fixed_loro_k16_reference_route_closed"] is True
    assert decision["entire_c_route_closed"] is False


def test_v274_2_diagnostic_accuracy_is_not_promoted() -> None:
    data = _summary()
    diagnostic = data["accuracy_diagnostic_not_a_scientific_result"]
    assert diagnostic["primary_strict_cells"] == "429/429"
    assert diagnostic["primary_complete_rigs"] == "13/13"
    assert diagnostic["scientific_accuracy_interpretation_allowed"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v274_2_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v274.2：" in text and "# v274.2:" in text
    assert "15 项有效性检查通过 14 项" in text
    assert "32 项检查通过 27 项" in text
    assert "不能解释为科学通过" in text
    assert "algorithm_breakthrough=false" in text


def test_v274_2_figure_is_public_and_readable() -> None:
    assert FIGURE.exists() and FIGURE.stat().st_size > 45_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2000
        assert image.height >= 900


def test_v274_2_remains_historical_after_v275_and_daily_card_is_unique() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["headline"].startswith("v278.1")
    assert current["current_decision"]["v274_2_independent_validation_passed"] is False
    for path in PAGES:
        assert "blastnet_case19_loro_fusion_reference_v274_2" in path.read_text(encoding="utf-8")
    assert "## 2026-08-27：v274.2" in LOG.read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-27"') == 1


def test_v274_2_public_artifacts_exclude_private_execution_details() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (SUMMARY, RESULT, *PAGES))
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "source-private",
        "FORMAL_EXIT_CODE",
        "VALIDATION_EXIT_CODE",
        "private_commit",
        "private_hash",
    )
    assert all(token not in text for token in forbidden)
    assert re.search(r"\b[0-9a-f]{40}\b", text) is None
