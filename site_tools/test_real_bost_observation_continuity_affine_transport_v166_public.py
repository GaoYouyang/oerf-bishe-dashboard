from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/real_bost_observation_continuity_affine_transport_v166_public_summary.json"
)
RESULT = (
    ROOT
    / "docs/real_bost_observation_continuity_affine_transport_v166_result_2026-08-20.md"
)
FIGURE = ROOT / "assets/figures/real_bost_observation_continuity_affine_transport_v166.png"
SURFACES = [ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html"]


def test_public_summary_keeps_the_v166_boundary() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert (
        payload["scientific_decision"]
        == "FAIL_OBSERVATION_CONTINUITY_AFFINE_TRANSPORT_V166"
    )
    assert payload["primary"]["strata_passed"] == 10
    assert payload["primary"]["strata_total"] == 12
    assert payload["primary"]["passed"] is False
    failed = [row for row in payload["primary"]["strata"] if not row["passed"]]
    assert [(row["time"], row["camera_count"]) for row in failed] == [(0.75, 5), (1.0, 5)]
    assert failed[0]["gradient_p90"] == 0.795555563289827
    assert failed[1]["gradient_worst"] == 1.0879873835714176
    assert payload["primary"]["logical_online_calls_non_anchor"] == {
        "A": 13,
        "AT": 1,
    }
    assert payload["execution"]["independent_validity_checks"] == 53
    assert payload["independent_recomputation"]["all_checks_passed"] is True
    assert (
        payload["independent_recomputation"][
            "maximum_density_factor_times_determinant_difference"
        ]
        <= 2e-16
    )
    assert payload["claim_limits"]["predictor_training_authorized"] is False
    assert payload["claim_limits"]["gpu_rental_authorized"] is False
    assert payload["claim_limits"]["real_bost"] is False


def test_public_result_is_substantively_bilingual() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "质量守恒全局仿射输运仍未稳住" in text
    assert "mass-conserving global affine transport still does not stabilize" in text
    assert "13A+1A^T" in text
    assert "FAIL_OBSERVATION_CONTINUITY_AFFINE_TRANSPORT_V166" in text
    assert "algorithm_breakthrough=false" in text


def test_public_surfaces_keep_v166_as_a_traceable_parent() -> None:
    for surface in SURFACES:
        text = surface.read_text(encoding="utf-8")
        assert "FAIL_OBSERVATION_CONTINUITY_AFFINE_TRANSPORT_V166" in text
        assert "10/12" in text
        assert "real_bost_observation_continuity_affine_transport_v166_result_2026-08-20.md" in text
        assert "algorithm_breakthrough" in text

    # The full historical metric remains visible on the two research summaries,
    # while the daily card can stay focused on the current v167 result.
    for surface in SURFACES[:2]:
        assert "0.795556" in surface.read_text(encoding="utf-8")


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
    ]
    for value in forbidden:
        assert value not in text
