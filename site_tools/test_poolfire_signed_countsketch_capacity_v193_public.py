from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_signed_countsketch_capacity_v193_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v193_summary_preserves_negative_gate_and_claim_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == "FAIL_SIGNED_COUNTSKETCH_CAPACITY_V193"
    assert payload["formal_status"] == "PASS_FORMAL_POOLFIRE_SIGNED_COUNTSKETCH_CAPACITY_V193"
    assert payload["independent_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_SIGNED_COUNTSKETCH_CAPACITY_V193_1"
    )
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["deployable_algorithm"] is False
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_v193_signed_aggregation_improves_but_does_not_pass() -> None:
    safe = _payload()["strict_safe_cells"]
    assert safe["fixed_geometry_qdeim_v190"] == {"five_camera": 35, "all_nine": 30}
    assert safe["adaptive_selected_columns_v192"] == {"five_camera": 40, "all_nine": 40}
    assert safe["unsigned_bucket_control"] == {"five_camera": 48, "all_nine": 46}
    assert safe["signed_countsketch_primary"] == {"five_camera": 51, "all_nine": 49}
    assert safe["signed_countsketch_primary"] != safe["required"]


def test_v193_failure_modes_and_strata_are_exact() -> None:
    payload = _payload()
    failures = payload["primary_failure_modes"]
    assert failures["five_camera"] == {
        "failed_cells": 1,
        "field_failures": 0,
        "gradient_failures": 1,
        "observation_failures": 0,
    }
    assert failures["all_nine"] == {
        "failed_cells": 3,
        "field_failures": 0,
        "gradient_failures": 0,
        "observation_failures": 3,
    }
    assert payload["primary_frame_safe_cells_out_of_13"] == {
        "five_camera": [12, 13, 13, 13],
        "all_nine": [13, 13, 11, 12],
    }


def test_v193_independent_recomputation_is_strict_and_permutation_equivariant() -> None:
    independent = _payload()["independent_recomputation"]
    assert independent["checks_passed"] == independent["checks_total"] == 19
    assert independent["maximum_regular_array_relative_difference"] < 1e-8
    assert independent["maximum_near_zero_array_absolute_difference"] < 1e-10
    assert independent["camera_permutation_feature_error"] == 0.0
    assert independent["camera_permutation_response_error"] == 0.0
    assert independent["camera_permutation_compact_response_error"] == 0.0
    assert independent["camera_permutation_coordinate_error"] == 0.0
    assert independent["bucket_and_sign_permutation_exact"] is True
    assert independent["measured_refinement_calls_exact"] is True


def test_v193_public_assets_and_bilingual_claims_exist() -> None:
    result = (ROOT / "docs/poolfire_signed_countsketch_capacity_v193_result_2026-08-22.md").read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    curriculum = (ROOT / "operator-learning/curriculum.js").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v193：" in result and "# v193:" in result
    assert "FAIL_SIGNED_COUNTSKETCH_CAPACITY_V193" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert all("v193" in content for content in (daily, focus, log))
    assert "v198" in root
    assert "curriculum.js?v=20260822-v196" in focus
    assert 'version: "2026.08.22-c-v193-signed-countsketch-negative"' in curriculum
    assert 'updated: "2026-08-22"' in curriculum
    figure = ROOT / "assets/figures/poolfire_signed_countsketch_capacity_v193.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v193_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_signed_countsketch_capacity_v193_result_2026-08-22.md",
        ROOT / "index.html",
        ROOT / "operator-learning/index.html",
        ROOT / "operator-learning/daily-progress.html",
    ]
    forbidden = ("/" + "Users" + "/", "private" + "_results", "private" + "_worktrees")
    private_receipt = re.compile(r"SCIENTIFIC_DECISION_[A-Z0-9_]+\.json")
    commit_hash = re.compile(r"\b[0-9a-f]{40}\b")
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in forbidden), path
        assert private_receipt.search(content) is None, path
        assert commit_hash.search(content) is None, path


def test_current_evidence_preserves_v193_after_v195_without_overclaiming() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_LOCAL_RAY_COVERAGE_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V211"
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_LOCAL_RAY_COVERAGE_ATTRIBUTION_V211"
    )
    assert current["metrics"]["v193_primary_five_safe_cells"] == 51
    assert current["metrics"]["v193_primary_all_nine_safe_cells"] == 49
    assert current["current_decision"]["v193_exact_mechanism_closed"] is True
    assert current["current_decision"]["v193_predictor_training_authorized"] is False
    assert current["current_decision"]["v193_algorithm_breakthrough"] is False
