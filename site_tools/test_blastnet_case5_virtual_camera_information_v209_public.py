from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/blastnet_case5_virtual_camera_information_v209_public_summary.json"
)
RESULT = (
    ROOT / "docs/blastnet_case5_virtual_camera_information_v209_result_2026-08-23.md"
)
FIGURE = ROOT / "assets/figures/blastnet_case5_virtual_camera_information_v209.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_v209_public_summary_attributes_geometry_not_cardinality() -> None:
    payload = json.loads(SUMMARY.read_text())
    diagnostic = payload["v209_virtual_ring_diagnostic"]
    assert diagnostic["scientific_decision"] == (
        "PASS_SYNTHETIC_RING_GEOMETRY_NOT_CARDINALITY_RESCUES_CASE5_REFERENCE_V209"
    )
    for arm in diagnostic["arms"].values():
        assert arm["strict_safe_cells"] == arm["strict_total_cells"] == 546
        assert arm["complete_groups_passed"] == arm["complete_groups_total"] == 13
        assert arm["logical_exact_calls"] == {"A": 16, "AT": 16}
        assert arm["group_p90_higher_ranges"]["field_relative_l2"][1] < 0.5
        assert arm["group_p90_higher_ranges"]["gradient_relative_l2"][1] < 0.75
        assert arm["group_p90_higher_ranges"]["observation_relative_l2"][1] < 0.2
    assert payload["claims_fixed_false"]["extra_camera_cardinality_benefit"] is False


def test_v209_public_summary_preserves_residual_closure_and_boundaries() -> None:
    payload = json.loads(SUMMARY.read_text())
    closure = payload["v209_virtual_ring_diagnostic"]["residual_equation_closure"]
    assert closure["all_values_finite"] is True
    assert closure["all_checks_passed"] is True
    assert closure["maximum_independent_vs_formal_residual_over_observation"] < 1e-8
    assert closure["maximum_formal_recursive_residual_closure_over_observation"] < 1e-12
    assert (
        closure["maximum_independent_recursive_residual_closure_over_observation"]
        < 1e-12
    )
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v209_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text()
    assert "# v209：" in text and "# v209:" in text
    assert (
        "PASS_SYNTHETIC_RING_GEOMETRY_NOT_CARDINALITY_RESCUES_CASE5_REFERENCE_V209"
        in text
    )
    assert "algorithm_breakthrough=false" in text
    assert "real_bost=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 850
        assert image.mode == "RGB"
        assert any(high - low > 100 for low, high in ImageStat.Stat(image).extrema)


def test_v209_remains_preserved_as_parent_evidence_below_v210() -> None:
    current = json.loads(CURRENT.read_text())
    assert current["v209_virtual_camera_scientific_decision"] == (
        "PASS_SYNTHETIC_RING_GEOMETRY_NOT_CARDINALITY_RESCUES_CASE5_REFERENCE_V209"
    )
    assert current["v209_residual_closure_adjudication_status"] == (
        "PASS_RESIDUAL_EQUATION_CLOSURE_ADJUDICATION_V209_2"
    )
    assert current["metrics"]["v209_virtual_nine_strict_safe_cells"] == 546
    assert current["metrics"]["v209_virtual_twelve_strict_safe_cells"] == 546
    assert (
        current["current_decision"]["v209_extra_three_camera_cardinality_credited"]
        is False
    )
    assert current["current_decision"]["v209_resource_gate_authorized"] is False
    assert current["scientific_status"] == (
        "FAIL_SIGNED_LINE_CANCELLATION_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V212"
    )


def test_primary_pages_reference_v209_in_both_languages() -> None:
    for path in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = path.read_text()
        assert "blastnet_case5_virtual_camera_information_v209" in content
        assert "algorithm_breakthrough=false" in content
    for path in (ROOT / "index.html", ROOT / "operator-learning/index.html"):
        assert (
            "PASS_SYNTHETIC_RING_GEOMETRY_NOT_CARDINALITY_RESCUES_CASE5_REFERENCE_V209"
            in path.read_text()
        )


def test_v209_public_artifacts_contain_no_private_execution_material() -> None:
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
