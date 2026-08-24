from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_potential_normal_streaming_resource_p14_v206_public_summary.json"
RESULT = ROOT / "docs/poolfire_potential_normal_streaming_resource_p14_v206_result_2026-08-23.md"
FIGURE = ROOT / "assets/figures/poolfire_potential_normal_streaming_resource_p14_v206.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_v206_summary_preserves_setup_equivalence_and_execution_counts() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "PASS_STREAMING_COMPACT_FRESH_RESOURCE_V206"
    assert payload["independent_resource_status"] == (
        "PASS_INDEPENDENT_ADJUDICATION_STREAMING_COMPACT_FRESH_RESOURCE_V206"
    )
    setup = payload["streamed_setup"]
    assert setup["setup_count"] == 26
    assert setup["equivalence_cell_count"] == 2626
    assert setup["maximum_coordinate_relative_difference_to_formal"] < 1e-9
    audit = payload["fresh_resource_audit"]
    assert audit["reference_workers"] == 39
    assert audit["timed_workers"] == 429
    assert audit["randomized_complete_blocks"] == 143
    assert audit["raw_worker_records"] == 468
    assert audit["maximum_output_relative_difference_streaming_vs_dense_k1"] < 1e-9


def test_v206_summary_preserves_wall_rss_and_exact_call_boundaries() -> None:
    payload = json.loads(SUMMARY.read_text())
    audit = payload["fresh_resource_audit"]
    assert audit["all_resource_thresholds_passed"] is True
    assert audit["all_13_calibration_p50_checks_passed"] is True
    versus_k1 = audit["streaming_vs_dense_k1"]
    versus_k2 = audit["streaming_vs_dense_k2"]
    assert versus_k1["outer_wall"]["p50"] < 0.90
    assert versus_k1["outer_wall"]["p90_higher"] < 1.05
    assert versus_k1["sampled_pipeline_peak_rss"]["p90_higher"] < 1.05
    assert versus_k2["outer_wall"]["p50"] < 0.90
    assert versus_k2["sampled_pipeline_peak_rss"]["p90_higher"] < 1.05
    calls = payload["logical_exact_calls_per_frame"]
    assert calls["streaming_compact_k1"] == {"A": 2, "AT": 2}
    assert calls["dense_full_dct_k1"] == {"A": 2, "AT": 1}
    assert payload["post_open_all_nine_resource_headroom"] is True
    assert payload["global_resource_speedup_claim"] is False
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v206_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text()
    assert "# v206：" in text and "# v206:" in text
    assert "PASS_STREAMING_COMPACT_FRESH_RESOURCE_V206" in text
    assert "algorithm_breakthrough=false" in text
    assert "global_resource_speedup_claim=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 850
        assert image.mode == "RGB"
        assert any(high - low > 100 for low, high in ImageStat.Stat(image).extrema)


def test_v206_remains_preserved_parent_evidence() -> None:
    current = json.loads(CURRENT.read_text())
    assert current["scientific_status"] == (
        "FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_LOW64_EXACT_ROWSPACE_LIFT_V221"
    )
    assert current["metrics"]["v206_outer_wall_vs_dense_k1_p50"] < 0.90
    assert current["metrics"]["v206_pipeline_rss_vs_dense_k1_p90_higher"] < 1.05
    assert current["current_decision"]["v206_post_open_all_nine_resource_headroom"] is True
    assert current["current_decision"]["v206_global_resource_speedup_claim"] is False
    assert current["current_decision"]["v208_case5_external_gate_adjudicated"] is False
    assert current["public_evidence"]["result"].endswith(
        "blastnet_case2_case5_low64_exact_rowspace_lift_v221_result_2026-08-24.md"
    )


def test_primary_pages_preserve_v206_as_bilingual_parent_evidence() -> None:
    for path in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = path.read_text()
        assert "poolfire_potential_normal_streaming_resource_p14_v206" in content
        assert "v206" in content
        assert "data-i18n-zh" in content and "data-i18n-en" in content
        assert "algorithm_breakthrough=false" in content


def test_v206_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden_schema_keys = ["formal_commit", "validator_commit", '"run_id"', "protocol_sha256"]
    for path in (SUMMARY, RESULT, CURRENT, ROOT / "operator-learning/index.html"):
        content = path.read_text()
        assert re.search(r"/(?:Users|home)/[^/\s]+", content) is None, path
        for token in forbidden_schema_keys:
            assert token not in content, (path, token)
