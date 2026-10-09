import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "poolfire_spectral_inverse_structure_2026-10-07_public_summary.json"
MARKER = "poolfire-spectral-inverse-structure-20261007"


def test_geometry_only_action_audit_boundary():
    raw = (ROOT / "docs" / SUMMARY).read_text()
    result = json.loads(raw)
    assert result["decision"] == "FIXED_SPECTRAL_BLOCK_INVERSE_ACTION_FAIL"
    assert result["operator_geometries"] == 3
    assert result["probes_per_geometry"] == 12
    assert result["independent_checks"] == 47
    assert result["new_observations_CFD_teachers"] == result["new_fits"] == 0
    assert result["accounting_only_successor"] and result["original_counter_failure_preserved"]
    assert not result["formal_inverse_SVD_repeated"]
    assert [g["cameras"] for g in result["geometries"]] == [5, 7, 9]
    for g in result["geometries"]:
        assert g["packed_representation_bytes"] == 378038160
        assert .3365 < g["dense_inverse_storage_ratio"] < .3366
        assert not g["capacity_gate"]
        assert 1 < g["worst_range_action_relative_error"] < g["diagonal_worst_range_action_relative_error"]
        assert 1 < g["worst_projected_action_relative_error"] < g["diagonal_worst_projected_action_relative_error"]
    assert max(result["independent_maxima"].values()) < 1e-8
    for key in ("exact_lift_deployment_demonstrated", "learned_initializer", "algorithm_breakthrough", "resource_speedup", "real_bost", "main_goal_complete"):
        assert result[key] is False
    assert not any(s in raw for s in ("/Users/", "private_results/", "sha256", "checkpoint"))


def test_paired_action_notes_not_final_reconstruction():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        soup = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        sections = soup.select("#" + MARKER)
        assert len(sections) == 1
        note = sections[0]
        nodes = note.select("[data-i18n-zh][data-i18n-en]")
        assert len(nodes) == 9
        for lang in ("zh", "en"):
            text = " ".join(n["data-i18n-" + lang] for n in nodes)
            assert all(t in text for t in ("5/7/9", "12", "47/47", "33.66%", "10%", "1%"))
            assert ("不是CFD场误差" if lang == "zh" else "not CFD-field") in text
            assert ("未重复正式逆求解" if lang == "zh" else "not repeated") in text
        assert len(note.select("tbody tr")) == 3
        assert "1795.82" in note.get_text() and "135.78" in note.get_text()
        assert (ROOT / name).parent.joinpath(note.select_one("a")["href"]).resolve() == ROOT / "docs" / SUMMARY


def test_old_full_sequence_resource_record_remains_primary():
    aggregate = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    old = aggregate["latest_full_temporal_direct_cost"]
    assert old["cells"] == 3333 and old["faster_all_pairs"]
    assert not old["lower_rss_all_pairs"]
    new = aggregate["latest_spectral_inverse_structure"]
    assert new["independent_checks"] == 47 and new["capacity_gates_passed"] == 0
    assert aggregate["latest_full_temporal_reference"]["complete_absolute_strata"] == 33
    log = (ROOT / "docs/operator_3d_learning_log.md").read_text()
    assert log.index("谱块压缩未保住逆作用") < log.index("完整序列经典初值实际更快")
    assert "observation-to-initializer map" in log and "57/57独立核验全真" in log
    for name in ("index.html", "operator-learning/index.html"):
        soup = BeautifulSoup((ROOT / name).read_text(), "html.parser")
        historical = soup.find(id="poolfire-full-temporal-direct-cost-20261007")
        assert "3,333/3,333" in historical.get_text() and "135.99" in historical.get_text()
