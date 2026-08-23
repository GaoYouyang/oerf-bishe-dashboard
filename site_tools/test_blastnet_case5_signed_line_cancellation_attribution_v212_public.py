from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/blastnet_case5_signed_line_cancellation_attribution_v212_public_summary.json"
)
RESULT = (
    ROOT
    / "docs/blastnet_case5_signed_line_cancellation_attribution_v212_result_2026-08-23.md"
)
FIGURE = (
    ROOT
    / "assets/figures/blastnet_case5_signed_line_cancellation_attribution_v212.png"
)
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_v212_public_summary_preserves_the_strict_negative_decision() -> None:
    payload = json.loads(SUMMARY.read_text())
    primary = payload["fixed_primary"]
    assert payload["scientific_decision"] == (
        "FAIL_SIGNED_LINE_CANCELLATION_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V212"
    )
    assert primary["comparison_count"] == 169
    assert primary["virtual_strictly_greater_count"] == 7
    assert primary["tie_count"] == 0
    assert primary["superiority_fraction"] == 7 / 169
    assert primary["strict_family_separation"] is False
    supplied = payload["family_summaries"]["supplied_nine"]
    virtual = payload["family_summaries"]["virtual_nine"]
    assert virtual["primary_min_median_max"][1] < supplied["primary_min_median_max"][1]
    assert virtual["mode_ratio_minimum_min_median_max"][1] > supplied[
        "mode_ratio_minimum_min_median_max"
    ][1]


def test_v212_public_summary_preserves_independence_and_zero_science_reads() -> None:
    payload = json.loads(SUMMARY.read_text())
    validation = payload["independent_validation"]
    assert validation["status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_SIGNED_LINE_CANCELLATION_ATTRIBUTION_V212"
    )
    assert validation["all_checks_passed"] is True
    assert validation["check_count"] == 15
    assert validation["maximum_formal_independent_metric_difference"] < 1e-12
    assert validation["maximum_mode_ratio_difference"] < 1e-12
    assert validation["maximum_camera_reversal_difference"] == 0
    scope = payload["scope"]
    assert scope["geometry_rows"] == 39
    assert scope["fixed_sine_modes"] == 64
    assert scope["offline_active_rays"] == 131359
    assert scope["offline_midpoint_ray_samples"] == 8406976
    assert scope["offline_forward_equivalent_probes"] == 0
    assert scope["deployment_exact_calls"] == {"A": 0, "AT": 0}
    assert scope["trainable_parameters"] == 0
    for key in (
        "density_reads",
        "observation_reads",
        "reconstruction_reads",
        "residual_reads",
        "parent_metric_array_reads",
    ):
        assert scope[key] == 0
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v212_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text()
    assert "# v212：" in text and "# v212:" in text
    assert "7/169" in text and "169/169" in text
    assert "FAIL_SIGNED_LINE_CANCELLATION_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V212" in text
    assert "algorithm_breakthrough=false" in text
    assert "real_bost=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 850
        assert image.mode == "RGB"
        assert any(high - low > 100 for low, high in ImageStat.Stat(image).extrema)


def test_v212_remains_historical_after_the_v214_headline() -> None:
    current = json.loads(CURRENT.read_text())
    assert current["scientific_status"] == (
        "INCONCLUSIVE_INVALID_OBSERVATION_PROXY_WARM_REPLAY_V215"
    )
    assert current["public_evidence"]["result"].endswith(
        "blastnet_case5_observation_proxy_warm_replay_v215_result_2026-08-24.md"
    )
    historical = json.loads(SUMMARY.read_text())
    assert historical["scientific_decision"] == (
        "FAIL_SIGNED_LINE_CANCELLATION_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V212"
    )
    assert historical["fixed_primary"]["virtual_strictly_greater_count"] == 7


def test_primary_pages_reference_v212_in_both_languages() -> None:
    for path in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = path.read_text()
        assert "blastnet_case5_signed_line_cancellation_attribution_v212" in content
        assert "7/169" in content
        assert "162" in content
        assert "algorithm_breakthrough=false" in content
    focus = (ROOT / "operator-learning/index.html").read_text()
    assert "v212 Case 5：固定有符号射线相消比" in focus
    assert "v212 Case 5: the fixed signed-line cancellation ratio" in focus


def test_v212_public_artifacts_contain_no_private_execution_material() -> None:
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
