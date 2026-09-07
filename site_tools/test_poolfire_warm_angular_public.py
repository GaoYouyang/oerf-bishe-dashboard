import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_warm_cost_angular_audit_20260908'


def test_cost_failure_and_uncertainty_are_distinct():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'FAIL_NONLINEAR_RETROSPECTIVE_ACTUAL_COST'
    assert (d['passing'], d['robust_failures'], d['inconclusive']) == (0, 4, 1)
    assert (d['unique_frames'], d['recovered_states'], d['original_trajectories']) == (5, 5920, 20)
    assert d['truth_oracle_costs_not_deployment_stop'] and d['original_certified_verdict_unchanged']
    assert d['points'][1]['status'] == 'INCONCLUSIVE'
    assert all(p['intervals']['first_cost'] == p['intervals']['sustained_cost'] for p in d['points'])
    assert not any(d[k] for k in ('warm_advantage', 'algorithm_breakthrough', 'paper_success',
        'resource_speedup', 'external_generalization', 'real_bost', 'cached_direct_defeated', 'full505_authorized'))
    assert d['angular']['endpoints'] == 60 and d['angular']['descriptors']['cosine_passes'] == 0
    assert 'squared' in d['angular']['ratio_kind'] and not d['angular']['cfd_truth_read']


def test_current_bilingual_and_retained_evidence():
    d = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in d['latest_execution_evidence']['note']
    assert d['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#warm-cost-angular-result')
        assert '0/5' in note['data-i18n-zh'] and '0/5' in note['data-i18n-en']
        assert '不确定' in note['data-i18n-zh'] and 'inconclusive' in note['data-i18n-en']
        assert soup.select_one('#camera-subset-metric-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #warm-cost-angular-result')


def test_redaction_links_and_figure():
    for ext in ('md', 'json'):
        text = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(s in text for s in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.npy', '.pt'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000


def test_residual_reuse_closes_only_this_recipe():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['residual_reuse']
    assert (d['passing'], d['failed'], d['inconclusive']) == (0, 5, 0)
    assert d['status'] == 'FAIL_ONE_SHOT_RESIDUAL_OPERATOR_REUSE'
    assert d['scored_states'] == 7710 and d['unique_frames'] == 5 and d['camera_count'] == 9
    assert d['closes_only_no_refit_reuse'] and d['posterior_stops_not_deployable']
    assert d['logical_total_per_arm'] == dict(A=256, AT=255)
    assert not any(d[k] for k in ('new_fitting', 'parameter_changes', 'new_data', 'full505_authorized',
        'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    for p in d['points']:
        assert p['intervals']['first_cost'] == p['intervals']['sustained_cost']
        primary, zero = [p['intervals']['first_cost'][a] for a in ('residual_nonlinear', 'zero_metric')]
        assert all(a > b for a, b in zip(primary['lower'], zero['upper']))


def test_residual_reuse_bilingual_and_current():
    e = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert e['latest_residual_reuse']['failed'] == 5
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        n = soup.select_one('#residual-reuse-result')
        assert '不否定全部残差学习' in n['data-i18n-zh'] and 'not all residual learning' in n['data-i18n-en']
        assert n.get_text() == n['data-i18n-zh']
        assert soup.select_one('#warm-cost-angular-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #residual-reuse-result')
