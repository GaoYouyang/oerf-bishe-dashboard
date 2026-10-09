import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
NAME = "poolfire_exact_band_cost_2026-10-07_public_summary.json"
PAGES = ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html")


def test_exact_band_accuracy_and_separate_resource_directions():
    data = json.loads((ROOT / "docs" / NAME).read_text())
    assert data["independent_checks_passed"] == data["independent_checks_total"] == 57
    assert data["numerical_qualification_checks"] == 51
    assert data["cells"] == 3333 and data["complete_strata"] == 33
    assert all(v == 3333 for v in data["matched_cells_each_pair"].values())
    assert all(v == 33 for v in data["complete_matched_strata_each_pair"].values())
    assert all(data["all_repeat_fields_bitwise"].values())
    assert data["faster_all_pairs"] and not data["lower_rss_all_pairs"]
    assert all(r["wall_ratio"] < 1 < r["rss_ratio"] for r in data["paired_ratios"])
    assert data["decision"] == "NO_JOINT_FULL_TEMPORAL_CACHED_DIRECT_RESOURCE_ADVANTAGE"


def test_repeats_cost_and_factor_bytes_are_not_process_memory():
    data = json.loads((ROOT / "docs" / NAME).read_text())
    assert len(data["timed_repeats"]) == 6 and data["repeats_per_method"] == 3
    assert data["factor_payload_bytes"] == 1016644200
    assert abs(data["factor_payload_reduction_fraction"] - (1 - 1016644200 / 1878778224)) < 1e-14
    assert data["factor_payload_not_RSS"] and data["prior_sparse_timing_not_new_causal_pair"]
    assert data["standalone_calls"]["DirectRidge-Warm34"] == {"A": 35, "AT": 35, "factor_solves": 1}
    assert data["query_workers"] == 8 and data["setup_sorting_packing_factors_imports_IO_included"]
    assert not data["historical_geometry_input_generation_included"]
    assert not data["joint_resource_win"] and not data["learned_resource_speedup"]


def test_notes_are_bilingual_and_keep_the_mechanism_unproven():
    for name in PAGES:
        doc = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        note = doc.find(id="poolfire-exact-band-cost-20261007")
        assert note and len(doc.find_all(id="poolfire-exact-band-cost-20261007")) == 1
        for lang in ("zh", "en"):
            nodes = note.select("[data-i18n-" + lang + "]")
            assert len(nodes) >= 6
            text = " ".join(n["data-i18n-" + lang] for n in nodes)
            for token in ("3,333/3,333", "33/33", "57/57", "5/7/9", "152.13", "161.04", "3.678", "0.666", "45.9%", "35A+35AT"):
                assert token in text
            assert ("尚无" if lang == "zh" and "daily" in name or lang == "zh" and name == "learning_log.html" else
                    "待验证" if lang == "zh" else "no scientific result yet" if "daily" in name or name == "learning_log.html" else "untested") in text
        for link in note.find_all("a", href=True):
            assert (ROOT / name).parent.joinpath(link["href"].split("#")[0]).exists()


def test_old_evidence_and_best_sparse_timing_stay_intact():
    current = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert current["latest_exact_band_cost"]["independent_checks_passed"] == 57
    assert current["latest_full_temporal_direct_cost"]["faster_all_pairs"]
    assert current["latest_range_cross_lift"]["lifted_action_probes_passed"] == 0
    for name in PAGES[:2]:
        doc = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        historical = doc.find(id="poolfire-full-temporal-direct-cost-20261007")
        assert "135.99" in historical.get_text()


def test_no_private_exports_and_no_learned_claim():
    data = json.loads((ROOT / "docs" / NAME).read_text())
    assert data["new_model_fits"] == 0
    assert not data["algorithm_breakthrough"] and not data["real_bost"] and not data["main_goal_complete"]
    assert all(not value for key, value in data.items() if key in ("external_generalization", "learned_resource_speedup"))
    assert all(s not in json.dumps(data) for s in ("/Users/", "private_results", "sha256", "checkpoint"))
