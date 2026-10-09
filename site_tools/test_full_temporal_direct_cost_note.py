import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = "poolfire-full-temporal-direct-cost-20261007"
SUMMARY = "poolfire_full_temporal_direct_cost_2026-10-07_public_summary.json"


def data():
    return json.loads((ROOT / "docs" / SUMMARY).read_text())


def test_full_sequence_classical_resource_boundary():
    result = data()
    assert result["decision"] == "NO_JOINT_FULL_TEMPORAL_CACHED_DIRECT_RESOURCE_ADVANTAGE"
    assert result["cells"] == 3333 and result["frames_per_trajectory"] == 101
    assert result["independent_checks_passed"] == result["independent_checks_total"] == 57
    assert result["faster_all_pairs"] and not result["lower_rss_all_pairs"]
    assert result["matched_cells_each_pair"] == {"CGLS128": 3333, "DirectRidge-Warm34": 3333}
    assert result["absolute_strata_each_pair"] == result["complete_matched_strata_each_pair"] == {"CGLS128": 33, "DirectRidge-Warm34": 33}
    assert all(r["wall_ratio"] < 1 and r["rss_ratio"] > 8 for r in result["paired_ratios"])
    assert len(result["timed_repeats"]) == 6 and all(result["all_repeat_fields_bitwise"].values())
    assert all(result[k] is False for k in ("learned_initializer", "algorithm_breakthrough", "resource_speedup", "paper_success", "real_bost", "main_goal_complete"))


def test_old_workload_and_nonfree_costs_remain_disclosed():
    result = data()
    assert result["old99_negative_preserved"] and result["old99_execution_policy_differs"]
    assert result["batch_vs_parallelism_not_causally_isolated"] and result["factor_build_and_output_included"]
    assert result["query_workers"] == 8 and result["query_blas_threads"] == 1
    assert not result["OS_cache_clearing"] and not result["historical_cache_input_generation_included"]
    assert result["inherited_geometry_forward_equivalents"] == 35547
    assert result["independent_replay_calls"] == {"A": 19998, "AT": 0}
    assert result["standalone_calls"]["DirectRidge-Warm34"] == {"A": 35, "AT": 35, "factor_solves": 1}
    text = (ROOT / "docs" / SUMMARY).read_text()
    assert not any(s in text for s in ("/Users/", "private_results/", "private_data/", "sha256", "checkpoint"))


def test_bilingual_notes_and_resource_figures():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        section = BeautifulSoup((ROOT / name).read_text(), "html.parser").select("#" + MARKER)
        assert len(section) == 1
        translated = section[0].select("[data-i18n-zh][data-i18n-en]")
        assert len(translated) == 7
        for lang in ("zh", "en"):
            text = " ".join(n["data-i18n-" + lang] for n in translated)
            for token in ("3,333", "101", "5/7/9", "57/57", "33/33", "135.99", "161.72", "5.675", "0.668", "35A+35AT", "128A+128AT"):
                assert token in text
            image = section[0].select_one("img")
            assert image["data-i18n-alt-" + lang]
            assert (ROOT / name).parent.joinpath(image["data-i18n-src-" + lang]).exists()
        assert (ROOT / name).parent.joinpath(section[0].select_one("a")["href"]).resolve() == ROOT / "docs" / SUMMARY


def test_prior_records_are_retained():
    aggregate = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert aggregate["latest_full_temporal_direct_cost"]["faster_all_pairs"]
    assert not aggregate["latest_full_temporal_direct_cost"]["lower_rss_all_pairs"]
    assert aggregate["latest_full_temporal_direct_control"]["primary_matched_cells"] == 3333
    assert aggregate["latest_full_temporal_reference"]["complete_absolute_strata"] == 33
    assert aggregate["latest_direct_ridge_cost"]["matched_cells_passed"] == 99
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert "57/57独立核验全真" in log and "about 16% less time" in log
    for name in ("index.html", "operator-learning/index.html"):
        text = (ROOT / name).read_text()
        assert text.index("const fullTemporal = evidence.latest_full_temporal_direct_cost;") < text.index("const current = evidence.latest_direct_ridge_cost")
        soup = BeautifulSoup(text, "html.parser")
        historical = soup.find(id="poolfire-full-temporal-direct-cost-20261007")
        assert "3,333/3,333" in historical.get_text() and "135.99" in historical.get_text()
