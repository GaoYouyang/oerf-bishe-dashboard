from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/real_bost_observation_local_divfree_vortex_v168_public_summary.json"
)
RESULT = (
    ROOT / "docs/real_bost_observation_local_divfree_vortex_v168_result_2026-08-21.md"
)
FIGURE = ROOT / "assets/figures/real_bost_observation_local_divfree_vortex_v168.png"
SURFACES = [
    ROOT / "index.html",
    ROOT / "operator-learning/index.html",
    ROOT / "operator-learning/daily-progress.html",
]


def test_public_summary_keeps_the_v168_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert (
        payload["scientific_decision"] == "FAIL_OBSERVATION_LOCAL_DIVFREE_VORTEX_V168"
    )
    assert payload["primary"]["strata_passed"] == 10
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    failed = [row for row in payload["primary"]["strata"] if not row["passed"]]
    assert [(row["time"], row["camera_count"]) for row in failed] == [
        (0.75, 5),
        (1.0, 5),
    ]
    assert failed[0]["gradient_p90"] == 0.8179899494053545
    assert failed[1]["gradient_worst"] == 1.271908652182564
    assert payload["primary"]["logical_online_calls_non_anchor"] == {"A": 13, "AT": 1}
    assert payload["execution"]["formal_validity_checks"] == 48
    assert payload["execution"]["independent_validity_checks"] == 60
    assert payload["execution"]["first_validation_attempt_outputs_reused"] is False
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert payload["independent_recomputation"]["local_rank"] == 12
    assert payload["independent_recomputation"]["maximum_analytic_divergence"] == 0.0
    assert payload["independent_recomputation"]["minimum_density_factor"] == 1.0
    assert payload["independent_recomputation"]["maximum_density_factor"] == 1.0
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "局部无散旋涡仍未救回" in text
    assert "local divergence-free vortices still do not repair" in text
    assert "13A+1A^T" in text
    assert "FAIL_OBSERVATION_LOCAL_DIVFREE_VORTEX_V168" in text
    assert "algorithm_breakthrough=false" in text


def test_public_surfaces_expose_the_same_verdict() -> None:
    for surface in SURFACES[:2]:
        text = surface.read_text(encoding="utf-8")
        assert "FAIL_OBSERVATION_LOCAL_DIVFREE_VORTEX_V168" in text
        assert "algorithm_breakthrough" in text
        assert "10/12" in text
        assert "0.817990" in text


def test_current_evidence_preserves_v168_as_historical_evidence() -> None:
    payload = json.loads(
        (ROOT / "operator-learning/current-evidence.json").read_text(encoding="utf-8")
    )
    assert payload["v168_observation_local_divfree_vortex_formal_status"].endswith(
        "OBSERVATION_LOCAL_DIVFREE_VORTEX_V168"
    )
    assert payload["v168_observation_local_divfree_vortex_independent_status"].endswith(
        "OBSERVATION_LOCAL_DIVFREE_VORTEX_V168"
    )
    assert payload["v168_observation_local_divfree_vortex_scientific_decision"] == (
        "FAIL_OBSERVATION_LOCAL_DIVFREE_VORTEX_V168"
    )
    assert payload["scientific_status"] == (
        "PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205"
    )


def test_public_figure_is_large_and_nonblank() -> None:
    with Image.open(FIGURE) as image:
        assert image.size == (2520, 1180)
        assert image.convert("RGB").getbbox() is not None


def test_public_payload_does_not_expose_private_execution_material() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = [
        "/Users/",
        "private_results",
        "private_data",
        "model_tree_seal",
        "calibration_tree_seal",
        "checkpoint",
    ]
    for value in forbidden:
        assert value not in text
