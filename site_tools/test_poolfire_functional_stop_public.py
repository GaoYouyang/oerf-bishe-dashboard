import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_functional_stop_20260907'


def test_scope_and_independent_counterexamples():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'FAIL_FUNCTIONAL_CERTIFIED_WARM_NECESSARY_PILOT'
    assert d['independent_status'] == 'PASS_INDEPENDENT_FUNCTIONAL_CERTIFICATES_AND_ENDPOINTS'
    assert (d['unique_frames'], d['methods'], d['implementations'], d['endpoint_count']) == (5, 6, 2, 60)
    assert all(r['passing'] == 5 for r in d['reports'])
    assert d['all_endpoints_certified'] and d['parameter_updates'] == 0
    assert d['reports'][0]['costs'][0][3] == dict(A=947, AT=946)
    assert d['reports'][3]['costs'][0][3] == dict(A=932, AT=931)
    assert d['common_counterexamples'] and d['cost_sign_disagreements'] == 10
    assert len(d['inherited_full505_reports']) == 24
    assert not any(d[k] for k in ('full505_authorized', 'algorithm_breakthrough', 'paper_success',
                                  'resource_speedup', 'external_generalization', 'real_bost'))


def test_bilingual_current_claims_and_private_boundary():
    c = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in c['latest_execution_evidence']['note']
    assert c['next_scientific_gate'] == c['next_scientific_gate_en']
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        notes = soup.select('#functional-stop-result')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('1%', '947', '932', '505'))
    for suffix in ('md', 'json'):
        text = (ROOT/'docs'/f'{STEM}.{suffix}').read_text()
        assert not any(v in text for v in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000


def test_setup_work_is_not_hidden_as_zero_calls():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['geometry_setup']['new_Q_setup_forward_equivalent_columns_total'] == 11760
    assert d['geometry_setup']['not_zero_setup_work']
    assert d['total_online_audit_actions'] == dict(A=52363, AT=52303)
    assert sum(d['offline_audit'].values()) == 240
    assert d['interval_arithmetic_proof'] is False
