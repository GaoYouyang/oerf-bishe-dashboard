"""Public evidence checks for the sealed v224 camera-jackknife diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case2_case5_low64_camera_jackknife_risk_v224_public_summary.json"
RESULT = ROOT / "docs/blastnet_case2_case5_low64_camera_jackknife_risk_v224_result_2026-08-24.md"
FIGURE = ROOT / "assets/figures/blastnet_case2_case5_low64_camera_jackknife_risk_v224.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"
PAGES = [ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html"]


def test_v224_counts_rank_and_post_open_role() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    scope = payload["scope"]
    assert (scope["cells"], scope["safe_cells"], scope["unsafe_cells"]) == (1261, 1064, 197)
    assert scope["diagnostic_type"] == "post-open mechanism-capacity diagnostic"
    assert scope["deployment_policy_established"] is False
    assert scope["observable_features_sealed_before_truth_scores"] is True
    assert payload["adjudication"]["all_leave_one_camera_systems_rank_64"] is True
    assert payload["independent_validation"]["minimum_leave_one_camera_rank"] == 64


def test_v224_primary_and_control_overlap() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_camera_jackknife_instability"]
    control = payload["cheap_max_camera_residual_control"]
    assert primary["safe_maximum"] > primary["unsafe_minimum"]
    assert primary["strict_separation_margin"] == -0.05380905876177532
    assert control["safe_maximum"] > control["unsafe_minimum"]
    assert control["strict_separation_margin"] == -0.1708996853122583
    assert primary["threshold_exists"] is False
    assert control["threshold_exists"] is False
    assert primary["policy_evaluated"] is False
    assert control["policy_evaluated"] is False


def test_v224_independent_validation_and_claim_limits() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = payload["independent_validation"]
    assert validation["status"] == "PASS_INDEPENDENT_RECOMPUTATION_LOW64_CAMERA_JACKKNIFE_RISK_V224"
    assert validation["scientific_decision"] == "FAIL_LOW64_CAMERA_JACKKNIFE_RISK_OVERLAP_V224"
    assert validation["all_required_checks_passed"] is True
    assert validation["maximum_formal_independent_feature_difference"] < 6e-15
    assert validation["maximum_camera_permutation_difference"] < 8e-15
    assert validation["end_to_end_physics_independence_proven"] is False
    assert payload["adjudication"]["worst_camera_jackknife_scalar_route_closed"] is True
    assert payload["adjudication"]["all_multiview_mechanisms_closed"] is False
    assert all(value is False for value in payload["claims_fixed_false"].values())


def test_v224_result_is_bilingual_and_fact_consistent() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v224：" in text and "# v224:" in text
    assert "FAIL_LOW64_CAMERA_JACKKNIFE_RISK_OVERLAP_V224" in text
    assert "1064" in text and "197" in text
    assert "-0.053809" in text and "-0.170900" in text
    assert "algorithm_breakthrough=false" in text
    assert "不证明所有多视角自一致机制不可能" in text


def test_v224_figure_is_rendered() -> None:
    assert FIGURE.is_file()
    with Image.open(FIGURE) as image:
        assert image.width >= 2400
        assert image.height >= 900
        assert image.mode == "RGB"


def test_v224_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert (
        current["v224_low64_camera_jackknife_scientific_decision"]
        == "FAIL_LOW64_CAMERA_JACKKNIFE_RISK_OVERLAP_V224"
    )
    assert current["current_decision"]["v224_scalar_camera_jackknife_route_closed"] is True
    assert current["current_decision"]["v224_all_multiview_mechanisms_closed"] is False
    assert current["current_decision"]["v224_algorithm_breakthrough"] is False
    assert current["metrics"]["v224_jackknife_strict_margin"] == -0.05380905876177532
    assert current["metrics"]["v224_view_residual_strict_margin"] == -0.1708996853122583
    for page in PAGES:
        content = page.read_text(encoding="utf-8")
        assert "v224" in content
        assert "algorithm_breakthrough=false" in content
        assert "blastnet_case2_case5_low64_camera_jackknife_risk_v224.png" in content
    for page in PAGES[:2]:
        content = page.read_text(encoding="utf-8")
        assert "-0.053809" in content
        assert "-0.170900" in content
    log = LEARNING_LOG.read_text(encoding="utf-8")
    assert "v224 逐相机删除稳定度仍重叠" in log
    assert "v224 leave-one-camera-out stability overlaps" in log


def test_v224_public_artifacts_do_not_expose_private_execution_details() -> None:
    public_text = "\n".join(
        [
            SUMMARY.read_text(encoding="utf-8"),
            RESULT.read_text(encoding="utf-8"),
            *(page.read_text(encoding="utf-8") for page in PAGES),
        ]
    )
    forbidden = [
        "/Users/",
        "/private/tmp/",
        "private_results",
        "private_worktrees",
        "formal_c75c72e6",
        "validation_c75c72e6",
        "c75c72e6",
    ]
    for token in forbidden:
        assert token not in public_text
