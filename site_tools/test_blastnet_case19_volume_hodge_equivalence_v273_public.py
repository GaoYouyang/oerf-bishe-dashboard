from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_volume_hodge_equivalence_v273_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_volume_hodge_equivalence_v273_result_2026-08-27.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_volume_hodge_equivalence_v273.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PAGES = (ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html")
LOG = ROOT / "docs/operator_3d_learning_log.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_v273_closes_only_the_direct_linear_hodge_route() -> None:
    data = _summary()
    factorization = data["operator_factorization"]
    audit = data["independent_validation"]
    adjudication = data["adjudication"]
    assert factorization["identity"] == "A = M D"
    assert factorization["new_exact_calls"] == "0A+0A^T"
    assert factorization["reads_density_truth_observation_residual_field_or_metric_arrays"] is False
    assert (audit["validity_checks_passed"], audit["validity_checks_total"]) == (16, 16)
    assert audit["all_discrete_and_validity_decisions_match"] is True
    assert adjudication["status"] == "FAIL_CASE19_DIRECT_VOLUME_HODGE_IS_POISSON_REPARAMETERIZATION_V273"
    assert adjudication["direct_linear_volume_hodge_route_closed"] is True
    assert adjudication["entire_c_route_closed"] is False


def test_v273_keeps_success_claims_false() -> None:
    data = _summary()
    assert data["adjudication"]["new_reconstruction_or_accuracy_result"] is False
    assert data["adjudication"]["hodge_boundary_stencil_projector_or_laplacian_tuning_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v273_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v273：" in text and "# v273:" in text
    assert "A = M D" in text and "1.75e-12" in text and "16/16" in text
    assert "FAIL_CASE19_DIRECT_VOLUME_HODGE_IS_POISSON_REPARAMETERIZATION_V273" in text
    assert "algorithm_breakthrough=false" in text


def test_v273_figure_is_public_and_readable() -> None:
    assert FIGURE.exists() and FIGURE.stat().st_size > 45_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2000
        assert image.height >= 900


def test_v273_remains_historical_after_v274_2_and_the_daily_card_is_unique() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["headline"].startswith("v275")
    assert current["current_decision"]["v273_independent_validation_passed"] is True
    assert current["public_evidence"]["figure"].endswith("v275.png")
    for path in PAGES:
        assert "blastnet_case19_volume_hodge_equivalence_v273" in path.read_text(encoding="utf-8")
    assert "## 2026-08-27：v273" in LOG.read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-27"') == 1


def test_v273_public_artifacts_exclude_private_execution_details() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (SUMMARY, RESULT, *PAGES))
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "source-private",
        "FORMAL_EXIT_CODE",
        "VALIDATION_EXIT_CODE",
        "private_commit",
        "private_hash",
    )
    assert all(token not in text for token in forbidden)
    assert re.search(r"\b[0-9a-f]{40}\b", text) is None
