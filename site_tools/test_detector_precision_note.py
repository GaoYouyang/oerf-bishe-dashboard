import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
NAME = "poolfire_detector_precision_2026-10-09_public_summary.json"
MARKER = "poolfire-detector-precision-20261009"


def test_result_and_attribution_have_separate_scopes():
    data = json.loads((SITE / "docs" / NAME).read_text())
    assert data["scope"]["cells"] == 99
    assert data["scope"]["camera_counts"] == [5, 7, 9]
    assert data["scope"]["complete_sequence"] is False
    result = data["result"]
    assert result["primary_matched_cells"] == result["diagonal_control_matched_cells"] == 0
    assert result["primary_absolute_sampled_strata"] == 33
    assert result["qualified_reference_matched_cells"] == result["expensive_direct_control_matched_cells"] == 99
    assert result["fixed_recipe_closed"] and result["new_model_fits"] == 0
    attribution = data["post_open_geometry_attribution"]
    assert all(0.82 < value < 0.87 for value in attribution["outside_additive_squared_frobenius_fraction"])
    assert attribution["is_inverse_action_or_CFD_error_bound"] is False
    assert attribution["changes_candidate_verdict"] is False
    assert attribution["independent_max_relative"] < 1e-10
    assert data["independent"]["field_max_relative"] < 1e-10
    assert data["independent"]["metric_max_absolute"] < 1e-10


def test_cost_privacy_and_claim_limits():
    data = json.loads((SITE / "docs" / NAME).read_text())
    cost = data["cost_scope"]
    assert cost["online_each_new_arm"] == {"A": 35, "AT": 35}
    assert cost["geometry_transforms_and_inherited_cost_nonfree"]
    assert cost["mixed_audit_duration_is_not_deployment_latency"]
    assert cost["new_fresh_wall_RSS_tested"] is False
    assert all(value is False for value in data["claims"].values())
    raw = (SITE / "docs" / NAME).read_text()
    assert not any(term in raw for term in ("/Users/", "private_results/", "private_data/", "sha256", "checkpoint"))


def test_bilingual_notes_and_preserved_prior_evidence():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-primal-balanced-coarse-20261009"')]
        assert note.count("data-i18n-zh=") == note.count("data-i18n-en=") == 4
        assert all(term in note for term in ("0/99", "33/33", "82.87%/84.70%/86.20%", NAME))
        assert "not an inverse-action" in note and "not a complete-sequence" in note
        assert "no new speedup" in note
    current = json.loads((SITE / "operator-learning/current-evidence.json").read_text())
    assert current["latest_detector_precision"]["matched_cells"] == 0
    assert current["latest_primal_balanced_coarse"]["matched_cells"] == 0
    assert current["latest_finite_loss_decomposition"]["actual_harm"] == 98
    assert NAME in (SITE / "docs/operator_3d_learning_log.md").read_text()
