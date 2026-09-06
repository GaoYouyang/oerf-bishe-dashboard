"""Verify reference scope, whole-sequence arithmetic and bilingual boundaries."""
import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = "poolfire_fixed512_reference_20260906"


def test_reference_is_not_learned_success():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["latest_reference_qualification"]["status"] == data["status"]
    assert current["latest_prediction"]["matched_cells"] == 367
    assert data["total"] == data["new_nonpilot_samples"] + data["reused_pilot_samples"] == 505
    assert all(data["checks"].values())
    for path in ("formal", "independent"):
        for arm in ("cgls", "jacobi_pcgls"):
            result = data["summaries"][path][arm]
            assert result["passing"] == 505 and result["complete_trajectories"] == 5
            for trajectory in result["trajectories"]:
                assert trajectory["passing"] == trajectory["total"] == 101
                assert all(v["worst"] <= .01 for v in trajectory["tails"].values())
    for key in ("learned_algorithm", "minimum_calls_proven", "resource_speedup", "external_generalization",
                "real_bost", "variable_camera_validated", "independent_test", "previous_cost_curves_requalified"):
        assert data[key] is False
    assert data["new_solver_calls"] == {"A": 2000 * 512, "AT": 2000 * 512}
    assert data["logical_calls_per_endpoint"] == {"A": 512, "AT": 512}


def test_bilingual_notes_and_no_private_identifiers():
    for file in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / file).read_text(), "html.parser")
        notes = soup.select("#fixed512-reference-result")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(word in notes[0][f"data-i18n-{lang}"] for word in ("512", "505/505", "5/5", "1%"))
        assert "not minimum calls" in notes[0]["data-i18n-en"]
        assert "不是最小调用数" in notes[0]["data-i18n-zh"]
    text = (ROOT / f"docs/{STEM}.md").read_text()
    assert "## 中文" in text and "## English" in text and "pilot-informed" in text
    assert "同一离散forward" in text and "same discrete forward" in text
    for suffix in ("md", "json"):
        content = (ROOT / f"docs/{STEM}.{suffix}").read_text()
        assert all(term not in content for term in ("/Users/", "/Volumes/", "private_results", "sha256", ".pt"))
    assert (ROOT / f"assets/figures/{STEM}.png").stat().st_size > 10000


def test_actual_residual_bound_has_narrow_scope():
    public = json.loads((ROOT / f"docs/{STEM}.json").read_text())
    data = public["direction_diagnostic"]
    assert data["cells"] == 505 and data["complete_trajectories"] == 5
    assert .2957 < data["minimum_field_error_lower_bound"] < .2958
    assert data["field_target"] == data["reference_field_error_guarantee"] == .01
    assert data["excludes_only_single_scalar_completion"]
    assert not data["excludes_further_cgls"] and not data["excludes_multidirection_warm_start"]
    assert not data["CFD_truth_parsed"] and not data["learned_speedup"]
    assert len(data["trajectories"]) == 5
    for row in data["trajectories"]:
        assert row["minimum_actual_residual_scalar_bound"] >= data["minimum_field_error_lower_bound"]
        assert 0 < row["median_rayleigh_ratio"] < 1
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#late-direction-diagnostic")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert "29.57%" in notes[0][f"data-i18n-{lang}"]
            assert "CGLS" in notes[0][f"data-i18n-{lang}"]
    text = (ROOT / f"docs/{STEM}.md").read_text()
    assert "lower bounds, not measured-error ranges" in text
    assert "preprocessed gauge-fixed" in text


def test_fsai_sentinel_is_not_complete_sequence_or_learning_success():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["fsai_sentinel"]
    assert data["sentinel_points"] == 5 and data["primary_passing_points"] == 3
    assert data["jacobi_passing_points"] == 0
    assert data["primary_no_worse_all_four_metrics_points"] == 5
    assert data["per_endpoint_exact_calls"] == {"A": 256, "AT": 256}
    assert not any(data[key] for key in ("complete_sequence_verified", "learned_algorithm",
        "resource_speedup", "external_generalization", "full_sequence_launch_authorized"))
    assert len(data["rows"]) == 5
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#fsai-sentinel-result")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(value in notes[0][f"data-i18n-{lang}"] for value in ("3/5", "0/5", "256"))


def test_field81_is_a_failed_necessary_gate_not_505_successes():
    data = json.loads((ROOT / f"docs/{STEM}.json").read_text())["field81_learning"]
    assert data["status"] == "FAIL_FIELD81_NECESSARY_MIDPOINTS"
    assert data["parameters_per_model"] == 81 and data["outer_fits"] == 10
    assert data["predictions_sealed"] == 505 and data["evaluated_midpoints"] == 5
    assert data["primary_passing"] == 0 and data["remaining_refinement_skipped"] == 500
    assert data["per_deployment_exact_calls"] == {"A": 258, "AT": 258}
    assert data["zero_bp_ridge_dominate_primary_all_four_points"] == 5
    assert not any(data[k] for k in ("complete_sequence_verified", "learned_advantage", "resource_speedup", "external_generalization", "real_bost"))
    assert all(row["formal"][2] > .01 and row["independent"][2] > .01 for row in data["rows"])
    for relative in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html"):
        soup = BeautifulSoup((ROOT / relative).read_text(), "html.parser")
        notes = soup.select("#field81-learning-result")
        assert len(notes) == 1
        for lang in ("zh", "en"):
            assert all(value in notes[0][f"data-i18n-{lang}"] for value in ("81", "505", "0/5", "258", "500"))
