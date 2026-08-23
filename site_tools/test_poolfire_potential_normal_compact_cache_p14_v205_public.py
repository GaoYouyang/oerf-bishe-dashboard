from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_potential_normal_compact_cache_p14_v205_public_summary.json"
RESULT = ROOT / "docs/poolfire_potential_normal_compact_cache_p14_v205_result_2026-08-23.md"
FIGURE = ROOT / "assets/figures/poolfire_potential_normal_compact_cache_p14_v205.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_v205_summary_preserves_compaction_and_equivalence() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["scientific_decision"] == "PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    assert payload["independent_status"] == "PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    representation = payload["representation"]
    assert representation["retained_packed_values"] == 509545
    assert representation["compression_ratio_five_camera"] > 5.69
    assert representation["compression_ratio_all_nine_camera"] > 10.24
    audit = payload["equivalence_audit"]
    assert audit["maximum_coordinate_relative_difference_to_formal"] < 1e-9
    assert audit["maximum_field_relative_difference_to_v199"] < 1e-9
    assert audit["maximum_camera_permutation_coordinate_relative_difference"] < 1e-9


def test_v205_summary_preserves_accuracy_and_cost_boundaries() -> None:
    payload = json.loads(SUMMARY.read_text())
    assert payload["inherited_accuracy"]["all_nine_camera"] == {
        "strict_safe_cells": 1313,
        "cell_count": 1313,
        "complete_groups_passed": 13,
        "complete_group_count": 13,
    }
    assert payload["inherited_accuracy"]["five_camera"]["strict_safe_cells"] == 1268
    assert payload["inherited_accuracy"]["five_camera"]["complete_groups_passed"] == 3
    calls = payload["logical_online_exact_calls"]
    assert calls["compact_k1_total"] == {"A": 2, "AT": 2}
    assert calls["dense_full_dct_k1_parent"] == {"A": 2, "AT": 1}
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v205_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text()
    assert "# v205：" in text and "# v205:" in text
    assert "PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205" in text
    assert "algorithm_breakthrough=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 900
        assert image.mode == "RGB"
        assert any(high - low > 100 for low, high in ImageStat.Stat(image).extrema)


def test_v205_is_the_current_public_headline() -> None:
    current = json.loads(CURRENT.read_text())
    assert current["scientific_status"] == "PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    assert current["engineering_status"] == "PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    assert current["metrics"]["v205_retained_packed_values"] == 509545
    assert current["metrics"]["v205_all_nine_strict_safe_cells"] == 1313
    assert current["current_decision"]["v205_compact_cache_equivalence_passed"] is True
    assert current["current_decision"]["v205_resource_speedup"] is False
    assert current["public_evidence"]["result"].endswith(
        "poolfire_potential_normal_compact_cache_p14_v205_result_2026-08-23.md"
    )


def test_primary_pages_reference_v205_in_both_languages() -> None:
    for path in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = path.read_text()
        assert "poolfire_potential_normal_compact_cache_p14_v205" in content
        assert "PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205" in content
        assert "algorithm_breakthrough=false" in content


def test_v205_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden_schema_keys = ["formal_commit", "validator_commit", '"run_id"']
    for path in (SUMMARY, RESULT, CURRENT, ROOT / "operator-learning/index.html"):
        content = path.read_text()
        assert re.search(r"/(?:Users|home)/[^/\s]+", content) is None, path
        for token in forbidden_schema_keys:
            assert token not in content, (path, token)
