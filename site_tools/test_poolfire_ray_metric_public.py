import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_ray_metric_tensor_20260907'


def test_scientific_negative_and_retained_learning():
    d = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    assert d['status']=='FAIL_OPENED_RAY_METRIC_TENSOR_LOTO'
    assert d['query_passing']==d['train_passing']==d['complete_trajectories']==0
    assert d['parameter_count']==1840 and d['optimizer_iterations']==[200]*5
    assert d['train_pairs']==2020 and d['unique_frames']==505 and d['overlapping_train_pairs']
    assert not d['fairness'] and not d['cheap_control_explains'] and d['reference_qualified']
    assert sum(r['any_harm_vs_previous'] for r in d['rows'])==459
    assert sum(r['query']['p90'][0]>r['previous']['p90'][0] for r in d['rows'])==4
    assert len(d['comparisons'])==15
    assert sum(max(r['worst_harm'])<0 for r in d['comparisons'])==12
    assert d['counts']['logical_online']==dict(A=2,AT=2)
    assert d['counts']['native_query_endpoints']==2020 and d['counts']['native_train_sentinels']==60
    assert d['verification']['maxima']['score']<1e-13 and d['verification']['maxima']['gradient']<1e-10
    assert not any(d[k] for k in ('stationary_solution_proven','intrinsic_capacity_failure_proven',
        'algorithm_breakthrough','paper_success','external_generalization','resource_speedup','real_bost'))


def test_current_bilingual_links_and_privacy():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_execution_evidence']['note']
    assert STEM in current['latest_ray_metric_tensor']['summary']
    for path in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/path).read_text(),'html.parser')
        notes = soup.select('#ray-metric-result')
        assert len(notes)==1
        for lang in ('zh','en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('1840','505','459','1%'))
    assert (ROOT/f'assets/figures/{STEM}.png').stat().st_size>10000
    for path in (f'docs/{STEM}.md',f'docs/{STEM}.json'):
        text = (ROOT/path).read_text()
        assert not any(v in text for v in ('/Users/','/Volumes/','private_results','sha256','.pt','parameters.json'))
    text = (ROOT/f'docs/{STEM}.md').read_text()
    assert 'not2020 independent' in text and '未重训第二个完整优化器' in text
