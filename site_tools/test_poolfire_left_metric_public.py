import json
from pathlib import Path

from bs4 import BeautifulSoup
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_left_metric_20260907'


def test_certified_metric_gain_is_not_warm_attribution():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'INCONCLUSIVE_LEFT_METRIC_CONTROL_DOMINANCE'
    assert d['independent_status'] == 'PASS_INDEPENDENT_FACTORED_METRIC_ENDPOINTS'
    assert all(d['checks'].values()) and len(d['checks']) == 8
    assert d['original_verifier_exit_code'] == 1 and d['type_only_repair']
    assert (d['unique_opened_frames'], d['camera_count'], d['paths']) == (5, 9, 2)
    assert d['primary_certified_per_path'] == d['zero_learned_metric_certified'] == [5, 5]
    assert d['fixed_metric_certified'] == [0, 0]
    scores = np.array(d['scores'])
    assert scores.size == 120 and np.isfinite(scores).all() and (scores < .01).all()
    assert d['actual_accuracy_passed'] == 30
    assert d['classic_best_A'] == [871, 880, 866, 932, 782]
    costs = d['costs']
    for path in range(2):
        for fold in range(5):
            q = next(r for r in costs if r['path'] == path and r['fold'] == fold and r['arm'] == 'learned_metric')
            reduction = 1-q['online']['A']/d['classic_best_A'][fold]
            assert .623 <= reduction <= .678
    formal_at = [r['online']['AT'] for r in costs if r['path'] == 0 and r['arm'] == 'learned_metric']
    zero_at = [r['online']['AT'] for r in costs if r['path'] == 0 and r['arm'] == 'zero_learned_metric']
    assert np.array_equal(np.array(formal_at)-np.array(zero_at), [1, 1, 0, 0, 0])
    assert d['zero_projection_charged_but_avoidable'] and d['certificate_not_minimal_accuracy_stop']
    assert d['clean_only'] and d['weighted_objective'] and d['full_direct_remains_stronger_control']
    assert not any(d[k] for k in ('full505_authorized', 'training_authorized', 'new_training',
                                  'algorithm_breakthrough', 'paper_success', 'external_generalization',
                                  'resource_speedup', 'real_bost'))


def test_bilingual_attribution_and_redacted_payload():
    d = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in d['latest_execution_evidence']['note']
    assert d['latest_left_metric']['warm_attribution_confirmed'] is False
    assert d['next_scientific_gate'] == d['next_scientific_gate_en']
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        notes = soup.select('#left-metric-result')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('INCONCLUSIVE', '62.3%-67.7%'))
        if 'daily' in rel:
            assert len(soup.select('#latest')) == 1
            assert soup.select_one('#latest #left-metric-result')
    for ext in ('md', 'json'):
        value = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(v in value for v in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
