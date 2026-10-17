"""Post-open source-block evidence stays distinct from neural and resource success."""

import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
SLUG="paired_source_crossfit_2026-10-17"
ID="paired-source-crossfit-20261017"


def data():
    return json.loads((ROOT/f"docs/{SLUG}_public_summary.json").read_text())


def test_whole_block_roles_and_postopen_boundary():
    p=data()
    assert (p["source_frames"],p["source_blocks"],p["frames_per_source_block"],p["fit_frames_per_fold"])==(808,8,101,707)
    assert p["complete_source_block_exclusion"] and p["all_label_derived_statistics_refit"]
    assert p["post_open_development_population"]
    assert not p["pristine_unopened_test"] and not p["original_held_opened"]
    assert p["all_source_supervised_models_not_used_as_crossfit_controls"]


def test_aggregate_quality_controls_and_tails():
    p=data()
    block=np.asarray(p["block_four_mse"])
    assert block.shape==(8,11,4)
    joint=block[:,:,:2].mean((0,2))
    assert np.allclose(joint,p["global_joint_mse"],rtol=0,atol=1e-14)
    for name,index in (("Single3",2),("Mean4",3),("Range4",4)):
        assert abs(100*(1-joint[0]/joint[index])-p["primary_joint_gain_percent"][name])<1e-11
        assert np.all(block[:,0]<=1.05*block[:,index])
    assert p["matched_counts"][:5]==[808]*5
    assert p["absolute_block_counts"][:5]==[8]*5
    assert "PairReference128" in p["arms"] and "PairJacobi5" in p["arms"]


def test_cost_and_unearned_claims():
    p=data()
    assert p["conditional_model_fit_shared_not_independent_EM"]
    assert p["GMM0_already_matches_all"]
    assert p["cost"]["GMM3"]==dict(A=4,AT=4,full_normal_inverses=2)
    assert p["cost"]["GMM0"]==dict(A=0,AT=1,full_normal_inverses=2)
    assert p["raw_capture_and_row_multiplier"]==2
    assert p["new_neural_updates"]==0
    for key in ("learned_initializer_tested","algorithm_breakthrough","paper_success","external_generalization","resource_speedup","real_bost","curved_ray_validated","full_goal_complete"):
        assert p[key] is False
    assert max(p["independent_physical_maximum"].values())<1e-7


def test_bilingual_views_and_retained_history():
    for file in ("index.html","operator-learning/index.html","operator-learning/daily-progress.html","learning_log.html"):
        soup=BeautifulSoup((ROOT/file).read_text(),"html.parser")
        section=soup.select_one("#"+ID)
        assert len(soup.select("#"+ID))==1
        assert len(section.select("p[data-i18n-zh][data-i18n-en]"))==3
        assert "707" in section.get_text() and "GMM0" in section.get_text()
        assert section.select_one("img")["data-i18n-alt-en"]
        assert soup.select_one("#paired-source-information-20261017")
        assert soup.select_one("#pinhole-camera-deletion-20261017")
    text=(ROOT/"docs/operator_3d_learning_log.md").read_text()
    assert "Paired source-block crossfit" in text and "旧FAIL原样保留" in text


def test_aggregate_privacy_and_plot():
    text=(ROOT/f"docs/{SLUG}_public_summary.json").read_text()
    assert not re.search(r"\b[a-f0-9]{64}\b",text)
    assert not any(s in text for s in ("/Users/","/Volumes/","private_results/","private_notes/","checkpoint"))
    assert (ROOT/f"assets/{SLUG}.png").stat().st_size>10000
