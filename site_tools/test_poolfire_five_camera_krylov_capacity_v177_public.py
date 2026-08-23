from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_five_camera_krylov_capacity_v177_public_summary.json"
RESULT = ROOT / "docs/poolfire_five_camera_krylov_capacity_v177_result_2026-08-21.md"
FIGURE = ROOT / "assets/figures/poolfire_five_camera_krylov_capacity_v177.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_public_summary_preserves_the_capacity_failure_and_attribution() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert (
        payload["formal_status"]
        == "PASS_FORMAL_POOLFIRE_KRYLOV_CAPACITY_EXECUTION_V177"
    )
    assert (
        payload["independent_status"]
        == "PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_KRYLOV_CAPACITY_V177"
    )
    assert (
        payload["scientific_decision"]
        == "FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177"
    )
    k4 = payload["five_camera_zero_cgls_k4"]
    k8 = payload["five_camera_zero_cgls_k8"]
    nine = payload["nine_camera_zero_cgls_k4"]
    assert (k4["strict_safe_candidates"], k4["strict_safe_candidate_total"]) == (
        0,
        6552,
    )
    assert (k8["strict_safe_candidates"], k8["strict_safe_candidate_total"]) == (
        0,
        6552,
    )
    assert (k4["cellwise_safe_cells"], k8["cellwise_safe_cells"]) == (0, 0)
    assert k8["per_metric_cellwise_oracle_pass"] == {
        "field": 0,
        "gradient": 52,
        "observation": 52,
        "total": 52,
    }
    assert nine["strict_safe_cells"] == 0
    assert payload["independent_recomputation"]["check_count"] == 25
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["mathematical_impossibility_proven"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_result_note_is_bilingual_and_does_not_overclaim() -> None:
    text = RESULT.read_text()
    assert "# v177：穷举排除" in text
    assert "# v177: exhaustive capacity" in text
    assert "FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177" in text
    assert "0/6552" in text
    assert "0/52 · 52/52 · 52/52" in text
    assert "25/25" in text
    assert "不关闭整个 C 路线" in text
    assert "does not close the full C route" in text
    assert "algorithm_breakthrough=false" in text


def test_figure_is_nonblank_and_stable_size() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1320)
        assert image.mode == "RGB"
        extrema = ImageStat.Stat(image).extrema
        assert any(high - low > 100 for low, high in extrema)


def test_current_evidence_preserves_v177_after_v181_becomes_current() -> None:
    payload = json.loads(CURRENT.read_text())
    assert payload["scientific_status"] == "INCONCLUSIVE_INVALID_OBSERVATION_PROXY_WARM_REPLAY_V215"
    assert (
        payload["v177_krylov_capacity_scientific_decision"]
        == "FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177"
    )
    assert payload["metrics"]["v177_five_k4_strict_safe_candidate_count"] == 0
    assert payload["metrics"]["v177_five_k8_strict_safe_candidate_count"] == 0
    assert payload["metrics"]["v177_five_k8_field_pass_count"] == 0
    assert payload["metrics"]["v177_five_k8_gradient_pass_count"] == 52
    assert payload["metrics"]["v177_five_k8_observation_pass_count"] == 52
    assert payload["metrics"]["v177_independent_check_count"] == 25
    assert (
        payload["current_decision"]["v177_wrong_subset_explanation_supported"] is False
    )
    assert (
        payload["current_decision"]["v177_low_depth_field_reference_shell_closed"]
        is True
    )
    assert payload["current_decision"]["v177_gpu_rental_authorized"] is False


def test_primary_pages_retain_v177_as_parent_evidence() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    daily = (ROOT / "operator-learning/daily-progress.html").read_text()
    home = (ROOT / "index.html").read_text()
    for text in (operator, daily):
        assert "v177" in text
    for text in (operator, daily):
        assert "v178" in text
        assert "v179" in text
    for text in (operator, daily, home):
        assert "v179" in text
        assert "v181" in text
    for text in (operator, daily):
        assert "FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177" in text
    assert "低深度场参考" in daily
    assert "low-depth field reference" in daily
    assert 'id="poolfire-five-camera-krylov-capacity-v177"' in operator
    assert daily.count("FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177") == 1


def test_route_metadata_records_v177_as_older_release() -> None:
    operator = (ROOT / "operator-learning/index.html").read_text()
    curriculum = (ROOT / "operator-learning/curriculum.js").read_text()
    assert "curriculum.js?v=20260822-v196" in operator
    assert 'previousVersion: "2026.08.22-c-v184-projection-potential-negative"' in curriculum
    assert 'version: "2026.08.21-c-v181-geometry-conditioned-rank16-negative"' in curriculum
    assert 'previousVersion: "2026.08.21-c-v180-compact-adjoint-preconditioner-negative"' in curriculum
    assert 'updated: "2026-08-22"' in curriculum


def test_public_artifacts_contain_no_private_execution_material() -> None:
    paths = [SUMMARY, RESULT, ROOT / "operator-learning/index.html", CURRENT]
    forbidden = [
        "/Users/",
        "private_results",
        "private_worktrees",
        "4035bad9",
    ]
    for path in paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path, token)
