from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/real_bost_equal_cost_selector_attribution_v174_public_summary.json"
)
RESULT = (
    ROOT / "docs/real_bost_equal_cost_selector_attribution_v174_result_2026-08-21.md"
)
FIGURE = ROOT / "assets/figures/real_bost_equal_cost_selector_attribution_v174.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_the_scientific_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "PASS_POSTOPEN_SELECTOR_ONLY_HEADROOM_V174"
    assert payload["policies"]["v172_selector"]["strict_safe_cells"] == 468
    assert payload["policies"]["v172_selector"]["complete_pass"] is True
    assert payload["policies"]["fit_static"]["strict_safe_cells"] == 323
    assert payload["policies"]["v169_low_mode_d_opt"]["strict_safe_cells"] == 192
    assert payload["policies"]["ray_axis_maximin"]["strict_safe_cells"] == 455
    assert payload["controls_that_passed"] == []
    assert payload["evaluation"]["candidate_exact_forward_calls"] == 1
    assert payload["evaluation"]["candidate_exact_adjoint_calls"] == 1
    assert payload["independent_recomputation"]["check_count"] == 27
    assert (
        payload["claim_limits"]["selector_only_headroom_established_on_opened_proxy"]
        is True
    )
    assert payload["claim_limits"]["deployable_learned_selector_established"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["resource_speedup"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v174：同成本归因" in text
    assert "# v174: equal-cost attribution" in text
    assert "PASS_POSTOPEN_SELECTOR_ONLY_HEADROOM_V174" in text
    assert "468/468" in text
    assert "455/468" in text
    assert "27/27" in text
    assert "algorithm_breakthrough=false" in text
    assert "不是一个已经完成训练和部署验证的算法" in text
    assert "not yet a trained and deployment-validated algorithm" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_retains_v174_after_v181_advances_the_gate() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == "PASS_FUTURE_ONLY_FULL_DCT_K2_REFERENCE_QUALIFICATION_V197"
    assert (
        payload["v174_equal_cost_selector_scientific_decision"]
        == "PASS_POSTOPEN_SELECTOR_ONLY_HEADROOM_V174"
    )
    assert payload["metrics"]["v174_primary_strict_safe_count"] == 468
    assert payload["metrics"]["v174_ray_axis_maximin_strict_safe_count"] == 455
    assert payload["metrics"]["v174_independent_check_count"] == 27
    assert "non-exchangeable v197 reference" in payload["next_scientific_gate_en"]
    assert "物理上不同的候选合同" in payload["next_scientific_gate_zh"]


def test_primary_pages_retain_v174_as_parent_evidence() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily, home):
        assert "v174" in text
    assert "v174：同成本归因确认选择器开发 headroom" in operator
    assert (
        "v174: equal-cost attribution establishes selector development headroom"
        in operator
    )
    assert (
        "real_bost_equal_cost_selector_attribution_v174_result_2026-08-21.md" in daily
    )
    assert daily.count("PASS_POSTOPEN_SELECTOR_ONLY_HEADROOM_V174") == 1


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "0d47d97f",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
