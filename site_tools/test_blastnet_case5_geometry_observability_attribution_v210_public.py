from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case5_geometry_observability_attribution_v210_public_summary.json"
RESULT = ROOT / "docs/blastnet_case5_geometry_observability_attribution_v210_result_2026-08-23.md"
FIGURE = ROOT / "assets/figures/blastnet_case5_geometry_observability_attribution_v210.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_v210_public_summary_preserves_the_strict_partial_decision() -> None:
    payload = json.loads(SUMMARY.read_text())
    primary = payload["fixed_primary"]
    assert payload["scientific_decision"] == (
        "PARTIAL_OVERLAPPING_GEOMETRY_ONLY_OBSERVABILITY_EVIDENCE_V210"
    )
    assert primary["comparison_count"] == 169
    assert primary["virtual_strictly_greater_count"] == 167
    assert primary["tie_count"] == 0
    assert primary["superiority_fraction"] == 167 / 169
    assert primary["strict_family_separation"] is False
    supplied = payload["family_summaries"]["supplied_nine"]["primary_min_median_max"]
    virtual = payload["family_summaries"]["virtual_nine"]["primary_min_median_max"]
    assert virtual[0] < supplied[2]
    assert virtual[1] > supplied[1]


def test_v210_public_summary_preserves_independence_and_scope() -> None:
    payload = json.loads(SUMMARY.read_text())
    validation = payload["independent_validation"]
    assert validation["status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_GEOMETRY_OBSERVABILITY_ATTRIBUTION_V210"
    )
    assert validation["all_checks_passed"] is True
    assert validation["maximum_formal_independent_metric_difference"] < 1e-9
    assert validation["maximum_camera_reversal_difference"] == 0
    scope = payload["scope"]
    assert scope["offline_forward_equivalent_probes"] == 2496
    assert scope["deployment_exact_calls"] == {"A": 0, "AT": 0}
    assert scope["trainable_parameters"] == 0
    for key in ("density_reads", "observation_reads", "reconstruction_reads", "residual_reads", "parent_metric_array_reads"):
        assert scope[key] == 0
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v210_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text()
    assert "# v210：" in text and "# v210:" in text
    assert "167/169" in text and "169/169" in text
    assert "PARTIAL_OVERLAPPING_GEOMETRY_ONLY_OBSERVABILITY_EVIDENCE_V210" in text
    assert "algorithm_breakthrough=false" in text
    assert "real_bost=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 850
        assert image.mode == "RGB"
        assert any(high - low > 100 for low, high in ImageStat.Stat(image).extrema)


def test_v210_is_the_current_public_headline() -> None:
    current = json.loads(CURRENT.read_text())
    assert current["scientific_status"] == (
        "PARTIAL_OVERLAPPING_GEOMETRY_ONLY_OBSERVABILITY_EVIDENCE_V210"
    )
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_GEOMETRY_OBSERVABILITY_ATTRIBUTION_V210"
    )
    assert current["metrics"]["v210_primary_comparison_count"] == 169
    assert current["metrics"]["v210_primary_strictly_greater_count"] == 167
    assert current["current_decision"]["v210_fixed_primary_strictly_separates"] is False
    assert current["current_decision"]["v210_predictor_authorized"] is False
    assert current["public_evidence"]["result"].endswith(
        "blastnet_case5_geometry_observability_attribution_v210_result_2026-08-23.md"
    )


def test_primary_pages_reference_v210_in_both_languages() -> None:
    for path in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = path.read_text()
        assert "blastnet_case5_geometry_observability_attribution_v210" in content
        assert "PARTIAL_OVERLAPPING_GEOMETRY_ONLY_OBSERVABILITY_EVIDENCE_V210" in content
        assert "algorithm_breakthrough=false" in content
    focus = (ROOT / "operator-learning/index.html").read_text()
    assert "当前：v210 几何可观测性归因已独立封存" in focus
    assert "Current: v210 geometry-observability attribution independently sealed" in focus
    assert "当前：v192" not in focus


def test_v210_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden = [
        "formal_commit",
        "validator_commit",
        '"run_id"',
        "protocol_sha256",
        "private_results",
        "private_worktrees",
    ]
    for path in (SUMMARY, RESULT, CURRENT, ROOT / "operator-learning/index.html"):
        content = path.read_text()
        assert re.search(r"/(?:Users|home)/[^/\s]+", content) is None, path
        for token in forbidden:
            assert token not in content, (path, token)
