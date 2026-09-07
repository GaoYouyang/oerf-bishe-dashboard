import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_camera_mix_ablation_20260907'


def test_mixed_not_algorithm_success():
    d = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    assert d['status']=='MIXED_CURRENT_CROSS_CAMERA_EFFECT'
    assert d['full_metric_wins']==[489,390,253,485] and d['self_metric_wins']==[16,115,252,20]
    assert d['full_passing']==d['self_passing']==0 and d['reference_qualified']
    assert d['frozen_parent_parameters']==1840 and d['scalars_per_fold_per_arm']==1
    assert d['verification']['independently_recalibrated_scalars']==10
    assert d['verification']['maxima']['score']<1e-12
    assert d['rows'][2]['effect']['self_metric_wins'][2]==101
    assert d['counts']['logical_online']==dict(A=2,AT=2) and d['counts']['inherited_arms']==19
    assert d['counts']['native_query_endpoints']==505
    assert not any(d[k] for k in ('accuracy_pass','fairness','independently_retrained_architecture',
        'algorithm_breakthrough','paper_success','external_generalization','resource_speedup','real_bost'))


def test_bilingual_links_and_privacy():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_execution_evidence']['note']
    for path in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/path).read_text(),'html.parser')
        notes = soup.select('#camera-mix-result')
        assert len(notes)==1
        for lang in ('zh','en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('489','485','253','252','505','1%'))
    assert (ROOT/f'assets/figures/{STEM}.png').stat().st_size>10000
    for path in (f'docs/{STEM}.md',f'docs/{STEM}.json'):
        value = (ROOT/path).read_text()
        assert not any(v in value for v in ('/Users/','/Volumes/','private_results','sha256','.pt','parameters.json'))
    value = (ROOT/f'docs/{STEM}.md').read_text()
    assert 'fixed-feature counterfactual' in value and '未重训网络' in value
