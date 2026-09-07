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


def test_error_shape_is_counterfactual_not_speedup():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['error_shape']
    assert d['status'] == 'CONSISTENT_ERROR_DIRECTION_PENALTY' and d['consistent_points'] == 5
    assert (d['scored_states'], d['unique_frames'], d['camera_count']) == (5140, 5, 9)
    assert d['counterfactual_not_deployment'] and d['diagnostic_full_teacher_required'] and d['steps_not_online_cost']
    assert not any(d[k] for k in ('whole_trajectories', 'new_fitting', 'full505_authorized',
        'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    for p in d['points']:
        assert p['intervals']['first'] == p['intervals']['sustained'] and p['consistent']
        b = p['intervals']['first']
        assert b['warm'][0] > b['scaled_cold'][1]
        assert b['restored_warm'][0] > b['cold'][1]
    assert all(.39 < r < .461 for r in d['norm_ratios'])


def test_error_shape_bilingual_and_figure():
    e = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert e['latest_error_shape']['consistent_points'] == 5
    assert 'poolfire_warm_error_shape_20260908' in e['latest_execution_evidence']['figure']
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#error-shape-result')
        assert '离线反事实' in note['data-i18n-zh'] and 'offline counterfactuals' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        if 'daily' in rel:
            assert soup.select_one('#latest #error-shape-result')
    assert (ROOT/'assets/figures/poolfire_warm_error_shape_20260908.png').stat().st_size > 10000


def test_geometry_inverse_probe_cost_and_scope():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['inverse_probe']
    assert data['status'] == 'FAIL_GEOMETRY_INVERSE_PROBE_WARM_PILOT'
    assert data['robust_wins'] == 0 and data['fixed_recipe_closed']
    assert (data['unique_frames'], data['camera_count'], data['scored_states']) == (5, 9, 5140)
    assert data['geometry_requires_inherited_full_factor']
    assert data['posterior_cost_not_deployable_stopping']
    assert data['direct_field_equivalent_saves_one_AT']
    assert not any(data[k] for k in ('new_fitting', 'full_trajectories', 'full505_authorized',
        'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    for point, expected in zip(data['points'], (217, 208, 210, 239, 218)):
        bands = point['costs_A_equals_AT']
        assert bands['first'] == bands['sustained']
        assert bands['first']['inverse_probe'] == [expected, expected]
        assert expected > max(bands['first'][a][1] for a in ('raw_probe', 'zero', 'old_neural'))


def test_geometry_inverse_probe_bilingual_and_preserved_history():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert evidence['latest_inverse_probe']['robust_wins'] == 0
    assert evidence['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#inverse-probe-result')
        assert '0/5' in note['data-i18n-zh'] and '0/5' in note['data-i18n-en']
        assert '505' in note['data-i18n-zh'] and '505' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        assert soup.select_one('#error-shape-result') and soup.select_one('#residual-reuse-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #inverse-probe-result')
