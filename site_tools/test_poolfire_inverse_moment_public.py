import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_inverse_moment_20260907'


def test_scope_and_energy_units():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'CONSISTENT_INVERSE_NORMAL_ERROR_RETENTION'
    assert d['independent'] == 'PASS_INDEPENDENT_INVERSE_NORMAL_RETENTION'
    assert (d['unique_frames'], d['overlapping_train_pairs'], d['camera_count'], d['implementations']) == (5, 20, 9, 2)
    assert all(v['passing_pairs'] == 20 for v in d['arms'])
    assert d['arms'][0]['ratios_median'] == [0.8367702204222067, 0.06368473022747659, 0.03471882910491024]
    assert d['squared_energy_not_error'] and d['no_causal_iteration_claim'] and d['controls_also_all20']
    assert not any(d[k] for k in ('new_CFD_truth_read', 'training_authorized', 'algorithm_breakthrough',
                                  'paper_success', 'external_generalization', 'resource_speedup', 'real_bost'))
    assert d['offline_actions'] == dict(A=270, AT=220) and d['triangular_solves'] == 260
    assert d['inherited_dense_factors_are_not_free']


def test_bilingual_notes_and_privacy():
    d = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in d['latest_inverse_moment']['summary']
    assert d['next_scientific_gate'] == d['next_scientific_gate_en']
    assert d['latest_functional_stop']['status'] == 'FAIL_FUNCTIONAL_CERTIFIED_WARM_NECESSARY_PILOT'
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        notes = soup.select('#inverse-moment-result')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('6.37%', '83.68%', '20'))
    for suffix in ('md', 'json'):
        text = (ROOT/'docs'/f'{STEM}.{suffix}').read_text()
        assert not any(v in text for v in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
