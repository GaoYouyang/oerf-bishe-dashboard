"""Publication checks, not scientific recomputation."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STEM = "poolfire_hybrid_gcv_reference_2026-10-05"


def summary():
    return json.loads((ROOT / "docs" / f"{STEM}_public_summary.json").read_text())


def test_reference_rejection_not_disguised_as_speedup():
    s = summary()
    assert s["scope"]["cells"] == 11 * 101 * 3 == 3333
    assert s["scope"]["scored_rows"] == 3333 * 7
    assert s["scope"]["strata"] == 33 + 3 == 36
    assert s["reference_absolute_strata"] == 36
    assert s["reference_stable_strata"] == 0
    assert not s["reference_qualified"]
    assert s["independent_checks_passed"] == 13
    assert all(v["strata_exceeding_0_001"] == 36 for v in s["stability_changes"].values())
    for key in ("algorithm_breakthrough", "resource_speedup", "external_generalization",
                "real_bost", "paper_success", "new_predictor_training_authorized",
                "lower_call_diagnostics_authoritative"):
        assert s[key] is False


def test_recovery_scope_cost_and_independent_tolerances():
    s = summary()
    assert s["recovery"]["solver_rerun"] is False
    assert s["recovery"]["original_outputs_modified"] is False
    assert s["recovery"]["added_exact_calls"] == {"A": 0, "AT": 0}
    assert not s["recovery"]["historical_exception_log_available"]
    assert s["cost"]["logical_per_sample_at_128"] == {"A": 128, "AT": 128}
    assert len(s["cost"]["not_included_in_logical_ledger"]) == 4
    assert not s["cost"]["fresh_wall_rss_measured"]
    assert max(s["max_per_checkpoint_state_relative_difference"],
               s["max_physical_metric_absolute_difference"],
               s["max_summary_absolute_difference"]) < 1e-8


def test_current_evidence_and_history():
    evidence = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert evidence["updated"] == "2026-10-05"
    assert evidence["latest_hybrid_gcv_reference"] == summary()
    assert evidence["current_decision"]["hybrid_gcv_reference_qualified"] is False
    assert "latest_case2_sarc_v285" in evidence
    for language in ("zh", "en"):
        headline = evidence[f"latest_execution_headline_{language}"]
        assert "36/36" in headline and "0/36" in headline


def test_bilingual_surfaces_links_and_privacy():
    for filename in ("index.html", "operator-learning/index.html",
                     "operator-learning/daily-progress.html", "learning_log.html"):
        text = (ROOT / filename).read_text()
        assert text.count('id="hybrid-gcv-reference-20261005"') == 1
        block = text.split('id="hybrid-gcv-reference-20261005"')[1].split("</section>" if "daily-progress" not in filename else "</article>")[0]
        assert "data-i18n-zh" in block and "data-i18n-en" in block
        assert f"{STEM}.md" in block and f"{STEM}_public_summary.json" in block
    note = (ROOT / "docs" / f"{STEM}.md").read_text()
    assert "## 中文结论" in note and "## English Result" in note
    for text in (note, json.dumps(summary())):
        for forbidden in ("/Users/", "private_results", "sha256", "checkpoint_path", ".pt", "cweiwei@", "PHY2309489"):
            assert forbidden not in text
        assert not re.search(r"\b[0-9a-f]{40,64}\b", text)
    for language in ("zh", "en"):
        assert (ROOT / "assets/figures" / f"{STEM}_{language}.png").stat().st_size > 10000


def test_language_switch_translates_figure_sources():
    js = (ROOT / "assets/site-language.js").read_text()
    assert "'src'" in js
    assert "'[data-i18n-src-zh]'" in js and "'[data-i18n-src-en]'" in js
    for filename in ("index.html", "operator-learning/index.html"):
        text = (ROOT / filename).read_text()
        assert f'{STEM}_zh.png' in text and f'{STEM}_en.png' in text
