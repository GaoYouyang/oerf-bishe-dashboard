import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "poolfire_range_cross_lift_2026-10-07_public_summary.json"
MARKER = "poolfire-range-cross-lift-20261007"


def test_direct_map_and_exact_lift_boundary():
    raw = (ROOT / "docs" / SUMMARY).read_text()
    data = json.loads(raw)
    assert data["decision"] == "DIRECT_OBSERVATION_MAP_EXACT_LIFT_ACTION_FAIL"
    assert data["independent_checks"] == 68 and data["synthetic_measurement_probes"] == 36
    assert data["storage_geometries_passed"] == 3 and data["lifted_action_probes_passed"] == 0
    assert data["raw_map_cannot_replace_lifted_primary"] and data["packed_bytes_not_RSS"]
    assert data["new_CFD_experimental_or_teacher_reads"] == data["new_fits"] == 0
    assert data["tracked_audit_each_implementation"] == {"A": 441, "AT": 351}
    assert data["geometry_only_construction_rhs_columns_each_implementation"] == 48614
    for g in data["geometries"]:
        assert g["storage_gate"] and not g["capacity_gate"]
        assert .04 < g["dense_inverse_storage_ratio"] < .07
        assert g["lifted_worst_field_action_relative_error"] > 40000
        assert g["raw_worst_field_action_relative_error"] > 1
        assert g["BP_worst_field_action_relative_error"] < g["lifted_worst_field_action_relative_error"]
        assert g["exact_target_primal_dual_relative_difference"] < 1e-9
    for k in ("refinement_run", "learned_initializer", "resource_speedup", "algorithm_breakthrough", "real_bost", "main_goal_complete"):
        assert data[k] is False
    assert not any(s in raw for s in ("/Users/", "private_results/", "checkpoint", "sha256"))


def test_all_four_pages_pair_action_not_reconstruction_notes():
    for file in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        soup = BeautifulSoup((ROOT / file).read_text(), "html.parser")
        notes = soup.select("#" + MARKER)
        assert len(notes) == 1
        nodes = notes[0].select("[data-i18n-zh][data-i18n-en]")
        assert len(nodes) == 10
        for lang in ("zh", "en"):
            text = " ".join(n["data-i18n-" + lang] for n in nodes)
            assert all(t in text for t in ("5/7/9", "36", "68/68", "3/3", "0/36", "4.26%/5.14%/6.02%", "1A+1AT", "441A+351AT", "48,614"))
            assert ("不是CFD场或最终重建" if lang == "zh" else "not CFD-field or final-reconstruction") in text
            assert ("不能事后替代唯一primary" if lang == "zh" else "cannot replace the sole lifted primary") in text
        assert len(notes[0].select("tbody tr")) == 3
        assert "42,747.82" in notes[0].get_text() and "4.336" in notes[0].get_text()
        assert (ROOT / file).parent.joinpath(notes[0].select_one("a")["href"]).resolve() == ROOT / "docs" / SUMMARY


def test_previous_full_sequence_and_spectral_records_unchanged():
    record = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert record["latest_spectral_inverse_structure"]["independent_checks"] == 47
    old = record["latest_full_temporal_direct_cost"]
    assert old["cells"] == 3333 and old["faster_all_pairs"] and not old["lower_rss_all_pairs"]
    assert record["latest_range_cross_lift"]["lifted_action_probes_passed"] == 0
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert "68/68独立核验全真" in log and "原映射本身已经" in log
    assert log.index("观测映射降到4%") < log.index("谱块压缩未保住逆作用")
    for file in ("index.html", "operator-learning/index.html"):
        soup = BeautifulSoup((ROOT / file).read_text(), "html.parser")
        historical = soup.find(id="poolfire-full-temporal-direct-cost-20261007")
        assert "3,333/3,333" in historical.get_text() and "135.99" in historical.get_text()
