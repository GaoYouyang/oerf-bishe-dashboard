import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/blastnet_case2_sarc_postopen_v285_result_2026-09-27.md"


def test_v285_scientific_summary_and_independent_replay():
    evidence = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    result = evidence["latest_case2_sarc_v285"]
    assert evidence["historical_common_stop_v4_header"]["scientific_status"] == "FAIL_LEARNED_INITIALIZER_NO_ROBUST_CALL_ADVANTAGE"
    assert result["status"] == "FAIL_CASE2_SAME_FAMILY_POSTOPEN_TRANSFER_ACCURACY_V285"
    assert result["frames"] == 54 and result["joint_accuracy_passed"] == 0
    assert result["independent_curved_forward_replays"] == 216
    assert result["maximum_metric_absolute_difference"] < 3e-16
    assert result["field_lower_than_direct_k4_frames"] == 54
    assert result["gradient_lower_than_direct_k4_frames"] == 54
    assert result["observation_lower_than_direct_k4_frames"] == 0
    assert abs(result["observation_ratio_vs_direct_k4_median"] - 1.18921) < 5e-6
    assert abs(result["observation_ratio_vs_direct_k4_p90"] - 1.33596) < 5e-6
    assert not result["independent_dataset_family"]
    assert not result["independent_external_generalization"]
    assert not result["real_bost"] and not result["algorithm_breakthrough"]


def test_v285_bilingual_surfaces_and_redacted_report():
    report = REPORT.read_text()
    assert "0/54" in report and "216" in report
    assert "not a new independent dataset family" in report
    assert not any(token in report for token in ("/Users/", "/Volumes/", "private_results", "sha256", ".pt"))
    for rel in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        text = (ROOT / rel).read_text()
        assert "v285" in text
    focus = BeautifulSoup((ROOT / "operator-learning/index.html").read_text(), "html.parser")
    section = focus.select_one("#v285-case2-same-family")
    assert section
    localized = " ".join(
        value
        for node in section.find_all(attrs={"data-i18n-zh": True, "data-i18n-en": True})
        for value in (node["data-i18n-zh"], node["data-i18n-en"])
    )
    assert localized.count("0/54") >= 2
