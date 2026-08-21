from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_minimal_shared_selector_v175_public_summary.json"
RESULT = ROOT / "docs/real_bost_minimal_shared_selector_v175_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/real_bost_minimal_shared_selector_v175.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_the_scientific_boundary() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "PASS_MINIMAL_SHARED_SELECTOR_HEADROOM_V175"
    assert payload["policies"]["minimal_shared_gram_ridge"]["strict_safe_cells"] == 468
    assert payload["policies"]["minimal_shared_gram_ridge"]["complete_pass"] is True
    assert payload["policies"]["fit_static"]["strict_safe_cells"] == 328
    assert payload["policies"]["v169_low_mode_d_opt"]["strict_safe_cells"] == 192
    assert payload["policies"]["ray_axis_maximin"]["strict_safe_cells"] == 455
    assert payload["controls_that_passed"] == []
    assert payload["evaluation"]["candidate_exact_forward_calls"] == 1
    assert payload["evaluation"]["candidate_exact_adjoint_calls"] == 1
    assert payload["independent_recomputation"]["check_count"] == 31
    assert payload["selector"]["trainable_parameters_max"] == 357
    assert payload["selector"]["outer_fold_count"] == 117
    assert payload["selector"]["one_subset_shared_across_all_four_times"] is True
    assert payload["claim_limits"]["minimal_shared_cpu_selector_validated_on_opened_proxy"] is True
    assert payload["claim_limits"]["fresh_external_generalization"] is False
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["resource_speedup"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v175：最小共享 CPU 相机选择器" in text
    assert "# v175: a minimal shared CPU camera selector" in text
    assert "PASS_MINIMAL_SHARED_SELECTOR_HEADROOM_V175" in text
    assert "468/468" in text
    assert "455/468" in text
    assert "31/31" in text
    assert "algorithm_breakthrough=false" in text
    assert "这仍是 post-open 受控代理证据" in text
    assert "opened controlled proxy" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_retains_v175_as_historical_parent_evidence() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["v175_minimal_shared_selector_scientific_decision"] == "PASS_MINIMAL_SHARED_SELECTOR_HEADROOM_V175"
    assert payload["metrics"]["v175_primary_strict_safe_count"] == 468
    assert payload["metrics"]["v175_ray_axis_maximin_strict_safe_count"] == 455
    assert payload["metrics"]["v175_independent_check_count"] == 31
    assert payload["scientific_status"] == "FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177"
    assert "low-depth Zero-CGLS field-reference shell" in payload["next_scientific_gate_en"]
    assert "低深度 Zero-CGLS 场参考壳" in payload["next_scientific_gate_zh"]


def test_primary_pages_reference_v175_in_both_languages() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily, home):
        assert "v175" in text
    assert "最小共享 CPU 选择器" in operator
    assert "minimal shared CPU selector" in operator
    assert "real_bost_minimal_shared_selector_v175_result_2026-08-21.md" in operator
    assert daily.count("PASS_MINIMAL_SHARED_SELECTOR_HEADROOM_V175") == 1


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "18db431d",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
