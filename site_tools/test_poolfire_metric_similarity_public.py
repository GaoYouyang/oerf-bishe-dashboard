import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_metric_similarity_20260908'


def test_training_diagnostic_not_outer_or_reconstruction_claim():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert data['passing'] == 0 and data['train_pairs'] == 20
    assert data['overlapping_pairs'] and data['metric_has_seen_diagnostic_queries']
    assert data['counts']['field'] == dict(better=17,same=3,worse=0)
    assert data['counts']['weak'] == dict(better=7,same=3,worse=10)
    assert data['counts']['rank_improved'] == 20
    assert data['new_work'] == dict(F=2020,T=2020,A=0,AT=0)
    assert not any(data[k] for k in ('outer_generalization_test','new_predictor_fit',
        'new_CFD_truth_read','predictor_experiment_eligible','algorithm_breakthrough',
        'paper_success','external_generalization','resource_speedup','real_bost'))


def test_latest_bilingual_and_historical_results_retained():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_execution_evidence']['note']
    assert current['latest_full_trajectory_controls']['passing'] == 505
    assert current['latest_teacher_fidelity_warm']['all20_actual_accuracy']
    for rel in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(),'html.parser')
        note = soup.select_one('#metric-similarity-result')
        assert note and soup.select_one('#teacher-fidelity-result') and soup.select_one('#full-control-cost-result')
        assert '0/5' in note['data-i18n-zh'] and '0/5' in note['data-i18n-en']
        if 'daily' in rel:
            assert soup.select_one('#latest #metric-similarity-result')
        if rel == 'index.html':
            assert STEM in soup.select_one('#latestFigure')['src']
            assert 'TRAIN' in soup.select_one('#latestFigureCaption')['data-i18n-en']


def test_redacted_report_limits_and_figure():
    for ext in ('md','json'):
        text = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(s in text for s in ('/Users/','/Volumes/','private_results','sha256','.npy','.pt'))
    text = (ROOT/'docs'/f'{STEM}.md').read_text()
    assert '不是外折测试' in text and 'not outer tests' in text
    assert 'No new ridge is actually run' in text
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
