import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_world_component_head_20260907'


def test_conditional_failure_and_disclosed_repair():
    d = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    assert d['status']=='FAIL_CONDITIONAL_WORLD_HEAD_LOTO'
    assert d['query_passing']==d['complete_trajectories']==0
    assert d['new_parameter_count']==9 and d['frozen_parent_parameters']==1840
    assert d['unique_frames']==505 and d['overlapping_train_pairs']
    assert not d['fairness'] and not d['cheap_control_explains'] and d['reference_qualified']
    assert sum(r['any_harm_vs_previous'] for r in d['rows'])==405
    assert all(.0007<r['relative_train_improvement']<.0016 for r in d['rows'])
    assert len(d['comparisons'])==18
    assert sum(max(r['worst_harm'])<0 for r in d['comparisons'])==12
    assert d['counts']['logical_online']==dict(A=2,AT=2)
    assert d['verification']['independently_refitted_heads']==15
    assert d['verification']['original_failure_preserved'] and d['verification']['after_failure_reporting_repair']
    assert d['verification']['maxima']['score']<1e-13 and d['verification']['maxima']['coefficients']<1e-10
    assert d['conditional_pre_K1_optimum_only']
    assert not any(d[k] for k in ('intrinsic_capacity_failure_proven','algorithm_breakthrough','paper_success',
        'external_generalization','resource_speedup','real_bost'))


def test_current_bilingual_links_and_privacy():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_execution_evidence']['note']
    for path in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/path).read_text(),'html.parser')
        notes = soup.select('#world-head-result')
        assert len(notes)==1
        for lang in ('zh','en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('1840','505','405','1%'))
    assert (ROOT/f'assets/figures/{STEM}.png').stat().st_size>10000
    for path in (f'docs/{STEM}.md',f'docs/{STEM}.json'):
        value = (ROOT/path).read_text()
        assert not any(v in value for v in ('/Users/','/Volumes/','private_results','sha256','.pt','parameters.json'))
    value = (ROOT/f'docs/{STEM}.md').read_text()
    assert 'conditional optimum' in value and '原失败记录保留' in value
