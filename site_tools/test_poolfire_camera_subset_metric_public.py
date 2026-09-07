import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_camera_subset_metric_20260908'


def test_negative_gate_and_partial_signal_are_distinct():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'FAIL_SUBSET_METRIC_CERTIFIED_ACCURACY'
    assert d['passing_accuracy'] == 2 and d['passing_cost'] == 4
    assert d['condition_cells'] == 10 and d['unique_frames'] == 5
    assert not d['full_subset_evaluation_eligible']
    assert not any(d[k] for k in ('new_training_authorized', 'warm_advantage', 'algorithm_breakthrough',
        'paper_success', 'external_generalization', 'real_bost', 'resource_speedup', 'cached_direct_defeated'))
    for row, actual, certified in zip(d['counts'], (5, 3), (2, 0)):
        learned = row['arms'][2]
        assert learned['arm'] == 'learned'
        assert learned['physical_passes'] == actual and learned['certified_passes'] == certified


def test_current_bilingual_and_historical_evidence():
    d = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in d['latest_execution_evidence']['note']
    assert d['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#camera-subset-metric-result')
        assert '2/10' in note['data-i18n-zh'] and '2/10' in note['data-i18n-en']
        assert soup.select_one('#camera-subset-reference-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #camera-subset-metric-result')


def test_redaction_links_and_figure():
    for ext in ('md', 'json'):
        text = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(s in text for s in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.npy', '.pt'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
