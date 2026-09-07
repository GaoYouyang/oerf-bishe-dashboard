import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_vector_transport75_20260907'


def test_complete_negative_and_actual_training():
    d=json.loads((ROOT/f'docs/{STEM}.json').read_text())
    assert d['status']=='FAIL_OPENED_VECTOR_TRANSPORT75_LOTO'
    assert d['parameter_count']==75 and d['fit_steps']==2600
    assert d['coverage']['frames']==505 and d['coverage']['actual_camera_counts']==[9]
    assert d['criterion']['relative_error_maximum']==.01
    assert d['criterion']['not_old_K4_matched_gate']
    assert d['training']['additional_train_only_scale_per_fold']==1
    assert not d['training']['independent_optimizer_trajectory_repeated']
    assert d['verification']['status']=='PASS_INDEPENDENT_VECTOR_TRANSPORT75_AUDIT'
    assert d['verification']['replayed_endpoints']==5555
    assert d['verification']['max_score_difference']<1e-13
    for r in d['reports']:
        assert len(r['trajectories'])==5
        assert sum(t['passed'] for t in r['trajectories'])==r['passing']
        assert sum(t['passed']==101 for t in r['trajectories'])==r['complete_trajectories']
        assert r['passing']==(505 if r['arm']=='full_direct' else 0)
    controls={r['control']:r for r in d['comparisons']}
    assert controls['fixed_transport']['any_harm_cells']==2
    assert controls['normalized_bp']['any_harm_cells']==0
    assert all(controls[k]['any_harm_cells']==505 for k in ('dual_ridge','direct_field_ridge','rayset369'))
    assert not any(d[k] for k in ('algorithm_breakthrough','paper_success','external_generalization','resource_speedup','real_bost'))
    assert d['fixed_recipe_closed'] and not d['cost']['comparative_wall_rss_completed']


def test_bilingual_pages_and_current_manifest():
    for relative in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup=BeautifulSoup((ROOT/relative).read_text(),'html.parser')
        notes=soup.select('#vector-transport75-result')
        assert len(notes)==1
        for lang in ('zh','en'):
            text=notes[0][f'data-i18n-{lang}']
            assert '503/505' in text and '0/5' in text and '1%' in text
    current=json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert current['latest_vector_transport75']['passing_frames']==0
    assert STEM in current['latest_execution_evidence']['note']
    daily=BeautifulSoup((ROOT/'operator-learning/daily-progress.html').read_text(),'html.parser')
    assert len(daily.select('#latest'))==1
    assert daily.select_one('#latest')['data-date']=='2026-09-07'
    report=(ROOT/f'docs/{STEM}.md').read_text()
    assert '74.61%' in report and 'not independently retrained' in report
    assert (ROOT/f'assets/figures/{STEM}.png').stat().st_size>10000
