from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_projection_potential_warm_v184_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v184_public_summary_preserves_scientific_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "FAIL_PROJECTION_POTENTIAL_WARM_V184"
    assert payload["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_PROJECTION_POTENTIAL_WARM_V184"
    )
    assert payload["independent_recomputation"]["checks_passed"] == 50
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["exact_call_reduction"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_v184_integrability_does_not_hide_failed_inverse_lift() -> None:
    payload = _payload()
    assert payload["mechanism_diagnostics"][
        "minimum_detector_gradient_explained_energy"
    ] > 0.88
    for arm in ("five_camera_primary_k1", "all_nine_primary_k1"):
        result = payload[arm]
        assert result["passed"] is False
        assert result["strict_cells_safe"] == 0
        assert result["global_p90"]["field_relative_l2"] > 0.5
        assert result["global_p90"]["gradient_relative_l2"] > 0.75
        assert result["global_p90"]["observation_relative_l2"] > 0.2


def test_v184_public_assets_and_bilingual_claims_exist() -> None:
    result = (
        ROOT / "docs/poolfire_projection_potential_warm_v184_result_2026-08-22.md"
    ).read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(
        encoding="utf-8"
    )
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    assert "v184：" in result and "# v184:" in result
    assert "FAIL_PROJECTION_POTENTIAL_WARM_V184" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert "v184" in daily and "v184" in focus
    figure = ROOT / "assets/figures/poolfire_projection_potential_warm_v184.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v184_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_projection_potential_warm_v184_result_2026-08-22.md",
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ]
    forbidden = (
        "/Users/",
        "private_results",
        "SCIENTIFIC_DECISION_V184.json",
        "8061dc33",
        "f33dba3945",
    )
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in forbidden), path


def test_current_evidence_preserves_v184_as_historical_negative() -> None:
    current = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert current["v184_projection_potential_scientific_decision"] == (
        "FAIL_PROJECTION_POTENTIAL_WARM_V184"
    )
    assert current["v184_projection_potential_independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_PROJECTION_POTENTIAL_WARM_V184"
    )
    assert current["metrics"]["v184_five_primary_k1_strict_safe_count"] == 0
    assert current["metrics"]["v184_all_nine_primary_k1_strict_safe_count"] == 0
    assert current["current_decision"]["v184_projection_potential_family_closed"] is True
    assert current["current_decision"]["v184_exact_call_reduction_established"] is False
    assert current["current_decision"]["v184_algorithm_breakthrough"] is False
