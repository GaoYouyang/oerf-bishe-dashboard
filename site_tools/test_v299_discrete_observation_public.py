"""Public regression for the de-identified v299 counterfactual evidence."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "operator-learning/index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING = ROOT / "learning_log.html"
TECHNICAL = ROOT / "docs/operator_3d_learning_log.md"
NOTE = ROOT / "docs/poolfire_v299_discrete_observation_counterfactual_2026-09-30.md"
SUMMARY = ROOT / "docs/poolfire_v299_discrete_observation_counterfactual_2026-09-30.json"


def test_v299_public_summary_matches_frozen_independent_result() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == "POSTOPEN_EXACT_DISCRETE_OBSERVATION_COUNTERFACTUAL"
    assert summary["scope"]["rows"] == 25
    assert summary["scope"]["already_open_trajectories"] == 5
    assert summary["scope"]["new_split_opened"] is False
    assert summary["matched_budget_17A_plus_17AT"]["learned_lower_tie_higher_rows"] == {
        "field": [20, 0, 5],
        "full_gradient": [20, 0, 5],
        "interior_gradient": [19, 0, 6],
        "observation_residual": [20, 0, 5],
    }
    assert summary["independent_recomputation"]["formal_tree_unchanged"] is True
    assert summary["independent_recomputation"]["upstream_trees_unchanged"] is True
    assert summary["limits"][-1] == "real_bost=false"


def test_v299_bilingual_pages_and_notes_preserve_claim_boundaries() -> None:
    files = (PAGE, DAILY, LEARNING, TECHNICAL, NOTE, SUMMARY)
    contents = [path.read_text(encoding="utf-8") for path in files]
    combined = "\n".join(contents).lower()
    for text in contents[:3]:
        assert "v299" in text
        assert "data-i18n-zh" in text
        assert "data-i18n-en" in text
    for path in (PAGE, DAILY, LEARNING):
        text = path.read_text(encoding="utf-8")
        assert "v299-discrete-observation" in text or "v299_discrete_observation_counterfactual" in text
    for value in ("-0.004108", "-0.008820", "-0.003524", "-0.003397", "5.44e-15"):
        assert value in combined
    for boundary in (
        "not causal proof",
        "not a prospective test",
        "does not reverse v284",
        "algorithm_breakthrough=false",
        "paper_success=false",
        "real_bost=false",
    ):
        assert boundary in combined
    for forbidden in (
        "private_results",
        "/users/gaoyouyang",
        "checkpoint_sha256",
        "operator_sha256",
        "run_v299_counterfactual",
    ):
        assert forbidden not in combined


def test_v299_focus_and_document_links_resolve() -> None:
    page = PAGE.read_text(encoding="utf-8")
    daily = DAILY.read_text(encoding="utf-8")
    learning = LEARNING.read_text(encoding="utf-8")
    assert '<section id="v299-discrete-observation"' in page
    assert 'id="research-update-2026-09-30-v299"' in learning
    assert 'data-date="2026-09-30"' in daily
    assert NOTE.is_file() and SUMMARY.is_file()
    assert "20/25" in page and "19/25" in page
    assert "20/25" in daily and "19/25" in daily
