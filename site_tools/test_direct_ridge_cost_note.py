import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "poolfire_direct_ridge_cost_2026-10-07_public_summary.json"
MARKER = "poolfire-direct-ridge-cost-20261007"
METHODS = ("CGLS128", "DirectRidge-Warm34")


def test_fresh_cost_has_accuracy_identity_and_negative_resource_verdict():
    text = (ROOT / "docs" / SUMMARY).read_text()
    data = json.loads(text)
    assert data["decision"] == "NO_JOINT_CACHED_DIRECT_RIDGE_RESOURCE_ADVANTAGE"
    assert data["independent_checks_passed"] == data["independent_checks_total"] == 52
    assert data["cells"] == 99 and data["absolute_strata_total"] == 33
    assert data["repeats_per_method"] == 3
    assert len(data["all_pairs_matched_cells"]) == len(data["all_pairs_absolute_strata"]) == 3
    assert all(pair == dict.fromkeys(METHODS, 99) for pair in data["all_pairs_matched_cells"])
    assert all(pair == dict.fromkeys(METHODS, 33) for pair in data["all_pairs_absolute_strata"])
    assert all(pair["wall_ratio"] > 1 and pair["rss_ratio"] > 1 for pair in data["paired_ratios"])
    assert not data["faster_all_pairs"] and not data["lower_rss_all_pairs"]
    for name in ("field_relative", "image_relative", "metric_absolute"):
        assert data["independent_maxima"][name] < 1e-7
    assert data["standalone_calls"] == {
        "CGLS128": {"A": 128, "AT": 128, "factor_solves": 0},
        "DirectRidge-Warm34": {"A": 35, "AT": 35, "factor_solves": 1},
    }
    assert data["query_CFD_truth_reads"] == data["query_parent_field_reads"] == data["new_model_fits"] == 0
    for name in ("all_six_states_sealed_before_scoring", "compared_to_both_parent_implementations",
                 "original_query_and_reference_row_identity", "factor_input_only_sorted",
                 "new_factor_build_included", "single_inference_process_peak_RSS",
                 "old_PCGLS_cost_attempt_still_inconclusive", "cached_input_classical_benchmark_only"):
        assert data[name] is True
    for name in ("cold_raw_data_benchmark", "distinct_learned_gain", "full_sequence_authorized",
                 "algorithm_breakthrough", "paper_success", "resource_speedup", "external_generalization",
                 "curved_ray_validated", "real_bost", "gpu_rental_authorized", "main_goal_complete",
                 "OS_page_cache_cleared", "historical_cache_and_observation_generation_included"):
        assert data[name] is False
    assert all(token not in text for token in ("/Users/", "private_results/", "private_data/", "sha256", "checkpoint", "theta"))


def test_resource_notes_are_bilingual_complete_and_linked():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        notes = BeautifulSoup((ROOT / name).read_text(), "html.parser").select("#" + MARKER)
        assert len(notes) == 1
        nodes = notes[0].select("[data-i18n-zh][data-i18n-en]")
        assert len(nodes) == 8
        zh = " ".join(node["data-i18n-zh"] for node in nodes)
        en = " ".join(node["data-i18n-en"] for node in nodes)
        for value in ("99/99", "33/33", "52/52", "22.88", "35.49", "0.68", "5.61"):
            assert value in zh and value in en
        assert all(value in zh for value in ("三帧训练哨兵", "没有速度或内存优势", "不含历史算子", "主目标仍未完成"))
        assert all(value in en for value in ("three-frame train sentinels", "no speed or memory advantage", "excludes historical operator", "main goal remains unmet"))
        base = (ROOT / name).parent
        assert base.joinpath(notes[0].select_one("a")["href"]).resolve() == ROOT / "docs" / SUMMARY
        image = notes[0].select_one("img")
        for language in ("zh", "en"):
            assert base.joinpath(image["data-i18n-src-" + language]).resolve().is_file()
            assert image["data-i18n-alt-" + language]


def test_current_foreground_is_cost_result_without_replacing_old_science():
    evidence = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    latest = evidence["latest_direct_ridge_cost"]
    assert latest["cells"] == latest["matched_cells_passed"] == 99
    assert latest["strata"] == latest["basic_strata_passed"] == 33
    assert "52/52" in latest["headline_zh"] and "52/52" in latest["headline_en"]
    assert MARKER in latest["note"] and not latest["algorithm_breakthrough"]
    assert "latest_direct_ridge_initializer" in evidence
    assert not evidence["latest_direct_ridge_initializer"]["algorithm_breakthrough"]
    for name in ("index.html", "operator-learning/index.html"):
        assert "evidence.latest_direct_ridge_cost || evidence.latest_direct_ridge_initializer" in (ROOT / name).read_text()
