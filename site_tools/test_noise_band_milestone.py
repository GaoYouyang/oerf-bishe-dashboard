import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).absolute().parents[1]


def data():
    return json.loads((ROOT/"docs/noise_informed_band_2026-10-16_public_summary.json").read_text())


def test_scientific_scope_and_exact_population():
    row = data()
    assert row["decision"]=="PASS_NOISE_INFORMED_BAND_ACTION"
    assert row["queries"]==909 and row["complete_trajectories"]==9 and row["known_cameras"]==5
    assert row["new_training_updates"]==row["CGLS_steps"]==0
    assert row["inherited_prior_parameters"]==225
    assert row["matched"]==[909,894,894,909,801,798,909,909,909]
    assert row["absolute"]==[9]*9 and all(row["gates"].values())
    assert row["independent_checks_pass"] and row["terminal_readback_pass"]
    assert row["maximum_metric_absolute_gap"]<=1e-7
    assert row["maximum_band_image_row_relative"]<=1e-7


def test_counterevidence_not_promoted_or_suppressed():
    row = data()
    assert row["simulator_reported_variance_is_new_side_information"]
    assert row["old_noise_failure_unchanged"] and not row["real_noise_calibration_established"]
    assert row["inherited_independent_physical_fields_disclosed"]
    assert not row["fresh_fused_inference_measured"]
    assert not any(row["claims"].values())
    assert row["joint"][7]<row["joint"][0]<row["joint"][1]
    assert abs(row["joint"][0]-row["joint"][3])<1e-6
    assert row["primary_harms_vs_GMM"][-1]==909
    assert row["global_four_mse"][0][-1]>row["global_four_mse"][1][-1]
    for index in (1,2,3,4,5,6,7):
        assert abs(row["gains_percent"][str(index)]-100*(1-row["joint"][0]/row["joint"][index]))<1e-12


def test_one_substantive_bilingual_milestone_and_privacy():
    for name in ("index.html","operator-learning/index.html","operator-learning/daily-progress.html","learning_log.html"):
        source = (ROOT/name).read_text()
        soup = BeautifulSoup(source,"html.parser")
        sections = soup.select("#noise-band-20261016")
        assert len(sections)==1
        section = sections[0]
        assert "18.14%" in section.get_text() and "GMM0" in section.get_text() and "plug-in" in section.get_text()
        for node in section.select("[data-i18n-zh]"):
            assert node.get_text()==node["data-i18n-zh"] and len(node["data-i18n-en"])>8
        image = section.select_one("img")
        assert image.get("data-i18n-alt-zh") and image.get("data-i18n-alt-en")
        if name in ("index.html","operator-learning/index.html"):
            assert "evidence.latest_noise_band || evidence.latest_potential_native" in source
    summary = (ROOT/"docs/noise_informed_band_2026-10-16_public_summary.json").read_text()
    for token in ("/Users/","/Volumes/","private_results","private_notes","sha256","weights_1024","checkpoint"):
        assert token not in summary
    current = json.loads((ROOT/"operator-learning/current-evidence.json").read_text())["latest_noise_band"]
    assert current["queries"]==current["matched_queries"]==909
    assert current["absolute_strata"]==current["complete_strata"]==9
