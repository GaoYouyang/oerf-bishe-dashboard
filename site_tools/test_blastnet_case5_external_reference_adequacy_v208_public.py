from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case5_external_reference_adequacy_v207_v208_public_summary.json"
RESULT = ROOT / "docs/blastnet_case5_external_reference_adequacy_v207_v208_result_2026-08-23.md"
FIGURE = ROOT / "assets/figures/blastnet_case5_external_reference_adequacy_v208.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"


def test_v208_public_summary_preserves_reference_inadequacy() -> None:
    payload = json.loads(SUMMARY.read_text())
    v207 = payload["v207_external_gate"]
    v208 = payload["v208_post_open_reference_diagnostic"]
    assert v207["scientific_decision"] == "INCONCLUSIVE_BLASTNET_CASE5_REFERENCE_INADEQUATE_V207"
    assert not v207["resource_gate_authorized"]
    assert v208["scientific_decision"] == (
        "INCONCLUSIVE_CASE5_REFERENCE_REMAINS_INADEQUATE_AT_ZERO_CGLS_K16_V208"
    )
    primary = v208["arms"]["K16_primary"]
    assert primary["strict_safe_cells"] == 0
    assert primary["strict_total_cells"] == 546
    assert primary["complete_calibration_groups_passed"] == 0
    assert primary["complete_calibration_groups_total"] == 13
    assert primary["group_p90_higher_ranges"]["field_relative_l2"][0] > 0.5
    assert primary["group_p90_higher_ranges"]["gradient_relative_l2"][1] < 0.75
    assert primary["group_p90_higher_ranges"]["observation_relative_l2"][1] < 0.2


def test_v208_public_summary_preserves_independent_and_claim_boundaries() -> None:
    payload = json.loads(SUMMARY.read_text())
    v208 = payload["v208_post_open_reference_diagnostic"]
    checks = v208["independent_numeric_checks"]
    assert checks["all_checks_passed"] is True
    assert checks["maximum_field_relative_difference"] < 1e-8
    assert checks["maximum_residual_relative_difference"] < 1e-8
    assert v208["arms"]["K4"]["logical_exact_calls"] == {"A": 4, "AT": 4}
    assert v208["arms"]["K8"]["logical_exact_calls"] == {"A": 8, "AT": 8}
    assert v208["arms"]["K16_primary"]["logical_exact_calls"] == {"A": 16, "AT": 16}
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v208_result_and_figure_are_bilingual_and_nonblank() -> None:
    text = RESULT.read_text()
    assert "# v207-v208：" in text and "# v207-v208:" in text
    assert "INCONCLUSIVE_CASE5_REFERENCE_REMAINS_INADEQUATE_AT_ZERO_CGLS_K16_V208" in text
    assert "algorithm_breakthrough=false" in text
    assert "real_bost=false" in text
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 850
        assert image.mode == "RGB"
        assert any(high - low > 100 for low, high in ImageStat.Stat(image).extrema)


def test_v208_is_preserved_as_the_historical_parent() -> None:
    current = json.loads(CURRENT.read_text())
    assert current["v208_reference_scientific_decision"] == (
        "INCONCLUSIVE_CASE5_REFERENCE_REMAINS_INADEQUATE_AT_ZERO_CGLS_K16_V208"
    )
    assert current["v208_reference_independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_ZERO_CGLS_REFERENCE_ADEQUACY_V208"
    )
    assert current["metrics"]["v208_k16_strict_safe_cells"] == 0
    assert current["metrics"]["v208_k16_complete_groups_passed"] == 0
    assert current["current_decision"]["v208_case5_external_gate_adjudicated"] is False
    assert current["current_decision"]["v208_resource_gate_authorized"] is False
    assert current["scientific_status"] == (
        "FAIL_FIXED_LOW64_PROXY_WARM_START_AGAINST_ADEQUATE_PCGLS_REFERENCE_V216"
    )


def test_primary_pages_reference_v208_in_both_languages() -> None:
    for path in (
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ):
        content = path.read_text()
        assert "blastnet_case5_external_reference_adequacy_v207_v208" in content
        assert "v207-v208" in content
        assert "algorithm_breakthrough=false" in content


def test_v208_public_artifacts_contain_no_private_execution_material() -> None:
    forbidden = ["formal_commit", "validator_commit", '"run_id"', "protocol_sha256", "private_results", "private_worktrees"]
    for path in (SUMMARY, RESULT, CURRENT, ROOT / "operator-learning/index.html"):
        content = path.read_text()
        assert re.search(r"/(?:Users|home)/[^/\s]+", content) is None, path
        for token in forbidden:
            assert token not in content, (path, token)
