from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case12_absolute_spectral_reference_v233_public_summary.json"
RESULT = ROOT / "docs/blastnet_case12_absolute_spectral_reference_v233_result_2026-08-25.md"
FIGURE = ROOT / "assets/figures/blastnet_case12_absolute_spectral_reference_v233.png"


def test_v233_summary_separates_numerical_success_from_scientific_failure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["execution"]["formal_cells_completed"] == 598
    assert data["execution"]["independent_cells_completed"] == 598
    assert data["execution"]["independent_checks_passed"] == 17
    assert data["numerical_certificate"]["formal_independent_field_relative_maximum"] < 1e-12
    accuracy = data["absolute_accuracy"]
    assert accuracy["strict_safe_cells"] == 0
    assert accuracy["complete_rigs_passed"] == 0
    assert accuracy["p90_higher"]["observation"] <= accuracy["p90_limits"]["observation"]
    for metric in ("field", "full_gradient", "interior_gradient"):
        assert accuracy["p90_higher"][metric] > accuracy["p90_limits"][metric]
    assert data["adjudication"]["scientific_decision"] == (
        "FAIL_INADEQUATE_CASE12_ABSOLUTE_SPECTRAL_REFERENCE_V233"
    )
    assert data["adjudication"]["fixed_dct1024_reference_closed"]
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v233_result_is_bilingual_and_preserves_the_evidence_boundary() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v233/v233.1：" in text and "# v233/v233.1:" in text
    for token in ("0/598", "0/13", "0.133957", "1.231545", "17/17"):
        assert token in text
    assert "二维投影吻合并不自动意味着三维重建正确" in text
    assert "Agreement in projection space does not establish correct volumetric reconstruction" in text
    assert "algorithm_breakthrough=false" in text


def test_v233_figure_is_rendered() -> None:
    assert FIGURE.is_file() and FIGURE.stat().st_size > 50_000
    with Image.open(FIGURE) as image:
        assert image.width >= 1800
        assert image.height >= 700


def test_v233_current_surfaces_and_log_are_synchronized() -> None:
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8"))
    assert current["v233_scientific_decision"] == "FAIL_INADEQUATE_CASE12_ABSOLUTE_SPECTRAL_REFERENCE_V233"
    assert current["metrics"]["v233_strict_safe_cells"] == 0
    assert current["metrics"]["v233_observation_p90_higher"] == 0.13395703881876667
    assert current["current_decision"]["v233_reference_adequate"] is False
    assert current["current_decision"]["v233_fixed_dct1024_reference_closed"] is True
    assert current["current_decision"]["v233_algorithm_breakthrough"] is False
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "v233" in content
        assert "blastnet_case12_absolute_spectral_reference_v233.png" in content
        assert "0.133" in content
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text(encoding="utf-8")
    assert "v233/v233.1" in log
    assert "projection fit does not certify the 3D field" in log


def test_v233_public_artifacts_do_not_expose_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "sha256",
        "checkpoint.pt",
        "f172ace9",
        "489c79bf",
    )
    assert all(token not in text for token in forbidden)
