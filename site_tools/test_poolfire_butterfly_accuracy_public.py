import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_butterfly_accuracy_20260907'


def test_independent_accuracy_decision_and_retained_controls():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'FAIL_OPENED_BUTTERFLY_FIRST_ACCURACY_LOTO'
    assert d['verification']['status'] == 'PASS_INDEPENDENT_BUTTERFLY_FIRST_ACCURACY_LOTO'
    assert len(d['reports']) == 24 and len(d['fits']) == 10
    assert all(r['passing'] == 0 and r['complete_trajectories'] == 0 for r in d['reports'][:2])
    assert next(r for r in d['reports'] if r['arm'] == 'full_direct')['passing'] == 505
    assert d['comparisons'][0]['any_harm_cells'] == 485 and not d['fairness']
    assert d['budget_limited_not_converged'] and d['closes_fixed_recipe_not_representation_class']
    assert all(r['iterations'] == 200 and not r['optimizer_success'] for r in d['fits'])
    assert d['active_parameters_per_arm'] == [243, 86]
    assert d['prior_query_timing_failure_preserved'] and d['engineering_shape_failure_preserved']
    assert not d['verification']['independent_optimizer_retrained']
    assert d['verification']['independent_query_endpoints'] == 1010
    assert not any(d[k] for k in ('algorithm_breakthrough', 'paper_success', 'resource_speedup', 'real_bost', 'external_generalization'))
    old = json.loads((ROOT/'docs/poolfire_spectral_camera_precision_20260907.json').read_text())
    assert d['reports'][2:] == old['reports']


def test_current_bilingual_claims_and_redaction():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_butterfly_accuracy']['summary']
    assert current['next_scientific_gate'] == current['next_scientific_gate_en']
    assert all('243' in current[f'next_scientific_gate_{lang}'] for lang in ('zh', 'en'))
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        notes = soup.select('#butterfly-accuracy-result')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('243', '86', '485', '0/505', '0/5', '1%', '200'))
    for suffix in ('md', 'json'):
        text = (ROOT/'docs'/f'{STEM}.{suffix}').read_text()
        assert not any(v in text for v in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000


def test_actual_call_ledger_and_unique_samples():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['unique_frames'] == 505 and d['training_sets_overlap']
    assert d['train_frames_per_fold'] == 404 and d['query_frames_per_fold'] == 101
    assert d['online'] == dict(A=2, AT=2)
    calls = sum(r['optimizer_evaluations']+1 for r in d['fits'])
    assert d['formal_offline_calls']['fit_A'] == d['formal_offline_calls']['fit_AT'] == 808*calls
    assert d['independent_additional_offline_calls']['post_query_A'] == 2020
