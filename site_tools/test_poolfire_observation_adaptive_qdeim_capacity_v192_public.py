from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_observation_adaptive_qdeim_capacity_v192_public_summary.json"


def _payload() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v192_summary_preserves_negative_gate_and_claim_boundary() -> None:
    payload = _payload()
    assert payload["scientific_decision"] == (
        "FAIL_NORMAL_CONTRIBUTION_OBSERVATION_ADAPTIVE_QDEIM_CAPACITY_V192"
    )
    assert payload["formal_status"].startswith("PASS_FORMAL_")
    assert payload["independent_status"].startswith("PASS_INDEPENDENT_")
    assert payload["claim_limits"]["algorithm_breakthrough"] is False
    assert payload["claim_limits"]["deployable_algorithm"] is False
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False


def test_v192_primary_improves_but_does_not_pass_capacity() -> None:
    safe = _payload()["strict_safe_cells"]
    assert safe["normal_contribution_primary"] == {
        "five_camera": 40,
        "all_nine": 40,
    }
    assert safe["fixed_geometry_qdeim_v190"] == {
        "five_camera": 35,
        "all_nine": 30,
    }
    assert safe["observation_magnitude_control"] == {
        "five_camera": 32,
        "all_nine": 26,
    }
    assert safe["normal_contribution_primary"] != safe["required"]


def test_v192_failure_modes_are_sensor_specific() -> None:
    failures = _payload()["primary_failure_modes"]
    assert failures["five_camera"]["failed_cells"] == 12
    assert failures["five_camera"]["gradient_failures"] == 10
    assert failures["five_camera"]["field_failures"] == 0
    assert failures["all_nine"]["failed_cells"] == 12
    assert failures["all_nine"]["observation_failures"] == 12
    assert failures["all_nine"]["gradient_failures"] == 0


def test_v192_independent_recomputation_is_strict_and_camera_equivariant() -> None:
    independent = _payload()["independent_recomputation"]
    assert independent["checks_passed"] == independent["checks_total"] == 17
    assert independent["maximum_regular_array_relative_difference"] < 1e-8
    assert independent["maximum_near_zero_array_absolute_difference"] < 1e-10
    assert independent["camera_permutation_feature_error"] == 0.0
    assert independent["camera_permutation_response_error"] == 0.0
    assert independent["camera_permutation_selection_exact"] is True


def test_v192_public_assets_and_bilingual_claims_exist() -> None:
    result = (ROOT / "docs/poolfire_observation_adaptive_qdeim_capacity_v192_result_2026-08-22.md").read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    focus = (ROOT / "operator-learning/index.html").read_text(encoding="utf-8")
    root = (ROOT / "index.html").read_text(encoding="utf-8")
    curriculum = (ROOT / "operator-learning/curriculum.js").read_text(encoding="utf-8")
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v192：" in result and "# v192:" in result
    assert "FAIL_NORMAL_CONTRIBUTION_OBSERVATION_ADAPTIVE_QDEIM_CAPACITY_V192" in result
    assert "data-i18n-zh" in daily and "data-i18n-en" in daily
    assert all("v192" in content for content in (daily, focus, root, log))
    assert "curriculum.js?v=20260822-v192-1" in focus
    assert 'version: "2026.08.22-c-v192-observation-adaptive-qdeim-negative"' in curriculum
    assert 'updated: "2026-08-22"' in curriculum
    figure = ROOT / "assets/figures/poolfire_observation_adaptive_qdeim_capacity_v192.png"
    assert figure.is_file()
    with Image.open(figure) as image:
        assert image.size == (2400, 1240)
        assert image.mode == "RGB"
        assert max(ImageStat.Stat(image).var) > 100.0


def test_v192_public_files_exclude_private_execution_details() -> None:
    paths = [
        SUMMARY,
        ROOT / "docs/poolfire_observation_adaptive_qdeim_capacity_v192_result_2026-08-22.md",
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


def test_current_evidence_points_to_v192_without_overclaiming() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["scientific_status"] == (
        "FAIL_NORMAL_CONTRIBUTION_OBSERVATION_ADAPTIVE_QDEIM_CAPACITY_V192"
    )
    assert current["engineering_status"] == (
        "PASS_INDEPENDENT_RECOMPUTATION_OBSERVATION_ADAPTIVE_QDEIM_CAPACITY_V192_1"
    )
    assert current["metrics"]["v192_primary_five_safe_cells"] == 40
    assert current["metrics"]["v192_primary_all_nine_safe_cells"] == 40
    assert current["current_decision"]["v192_exact_mechanism_closed"] is True
    assert current["current_decision"]["v192_predictor_training_authorized"] is False
    assert current["current_decision"]["v192_algorithm_breakthrough"] is False
