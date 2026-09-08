import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_warm_cost_angular_audit_20260908'


def test_full_amg_common_certificate_and_scoped_costs():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['full_amg_control']
    assert d['status'] == 'PASS_FULL_OPENED_AMG_CERTIFIED_CONTROL'
    assert (d['frames'], d['eligible'], d['passing'], d['complete_trajectories']) == (505, 505, 505, 5)
    assert d['query_paths'] == 1010 and d['camera_count'] == 9
    assert all(d[k] for k in ('common_observation_only_primary_stop', 'conservative_paired_intervals',
        'secondary_uses_ideal_control_stops', 'query_truth_after_prediction_seal',
        'inherited_independently_qualified_hierarchy', 'geometry_and_certificate_setup_nonfree',
        'V_cycle_nonfree', 'full_direct_unbeaten', 'opened_data_not_external'))
    assert not any(d[k] for k in ('new_training', 'algorithm_breakthrough', 'learned_warm_success',
        'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    assert d['cost_stats']['amg_upper']['A'] == dict(min=102, median=119, max=125)
    assert d['cost_stats']['learned_lower']['A'] == dict(min=261, median=297, max=352)
    assert .6013 < d['cost_stats']['savings_lower']['A']['median'] < .6014
    assert .5402 < d['cost_stats']['savings_lower']['A']['min'] < .5403
    assert len(d['trajectories']) == 5 and all(t['passing'] == 101 for t in d['trajectories'])
    assert all(v == 505 for v in d['secondary_passing'].values())
    assert d['post_summary_maximum'] == 0


def test_full_amg_bilingual_latest_and_pilot_preserved():
    e = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert e['latest_full_amg_control']['complete_trajectories'] == 5
    assert e['latest_amg_classical_control']['opened_points'] == 5
    assert 'AMG' in e['headline_en'] and '505' in e['headline_zh']
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#full-amg-control-result')
        assert '505/505' in note['data-i18n-zh'] and '505/505' in note['data-i18n-en']
        assert '停止证书' in note['data-i18n-zh'] and 'stopping certificate' in note['data-i18n-en']
        assert '不免费' in note['data-i18n-zh'] and 'not free' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        assert soup.select_one('#amg-classical-control-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #full-amg-control-result')


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


def test_manufactured_residual_new_fit_and_actual_cost():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['manufactured_residual']
    assert d['status'] == 'FAIL_MANUFACTURED_RESIDUAL_WARM_NECESSARY_PILOT'
    assert (d['passing'], d['failed'], d['inconclusive']) == (0, 5, 0)
    assert (d['new_models'], d['trainable_parameters'], d['steps'], d['cfd_fit_samples']) == (1, 369, 640, 0)
    assert (d['scored_states'], d['unique_cfd_frames'], d['camera_count']) == (10280, 5, 9)
    assert d['downstream_metric_uses_own_fold_cfd_training'] and d['posterior_stops_not_deployable']
    assert d['fixed_recipe_closed'] and not d['full_trajectories']
    assert not any(d[k] for k in ('full505_authorized', 'algorithm_breakthrough', 'paper_success',
        'resource_speedup', 'external_generalization', 'real_bost'))
    for point, low in zip(d['points'], ([154, 153], [150, 149], [143, 142], [178, 177], [139, 138])):
        assert point['intervals']['first_cost'] == point['intervals']['sustained_cost']
        b = point['intervals']['first_cost']
        assert b['manufactured_neural']['lower'] == low
        assert all(a > z for a, z in zip(low, b['zero_metric']['upper']))
        assert all(a >= z for a, z in zip(low, b['prefix_only']['upper']))


def test_manufactured_residual_bilingual_and_preserved_history():
    e = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert e['latest_manufactured_residual']['failed'] == 5
    assert e['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#manufactured-residual-result')
        assert '0/5' in note['data-i18n-zh'] and '0/5' in note['data-i18n-en']
        assert '369' in note['data-i18n-zh'] and '369' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        assert soup.select_one('#inverse-probe-result') and soup.select_one('#error-shape-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #manufactured-residual-result')


def test_frozen_feature_floor_is_only_a_joint_initial_loss_bound():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['manufactured_feature_capacity']
    assert d['status'] == 'FROZEN_FEATURE_FLOOR_DOMINATES_RESIDUAL_ERROR'
    assert (d['examples'], d['independent_rows'], d['readout_directions']) == (37, 74, 17)
    assert d['oracle_not_deployable'] and d['bound_on_initial_joint_loss_only']
    assert d['oracle_not_run_through_refinement'] and d['frozen_hidden_parameters']
    assert not any(d[k] for k in ('new_training', 'new_solver_runs', 'cfd_truth_parsed', 'full_trajectories',
        'full505_authorized', 'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    for summary in d['summaries']:
        assert summary['populations']['synthetic']['floor_dominant'] == 32
        assert summary['populations']['cfd']['floor_dominant'] == 5
        assert .981 < summary['populations']['cfd']['fractions']['min'] < .982
        assert .986 < summary['populations']['cfd']['fractions']['worst'] < .987
    for row in d['cfd_points']:
        assert abs(row['floor']+row['excess']-row['total']) < 1e-9
        assert row['floor'] < row['scalar_floor']
        assert .981 < row['floor_fraction'] < .987


def test_feature_floor_bilingual_scope_and_history():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert evidence['latest_manufactured_feature_capacity']['examples'] == 37
    assert evidence['latest_manufactured_residual']['failed'] == 5
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#manufactured-feature-floor-result')
        assert '联合平方' in note['data-i18n-zh'] and 'joint squared' in note['data-i18n-en']
        assert '完整网络' in note['data-i18n-zh'] and 'whole-network' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        assert soup.select_one('#manufactured-residual-result') and soup.select_one('#inverse-probe-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #manufactured-feature-floor-result')


def test_loss_oracle_actual_cost_not_a_call_optimal_bound():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['loss_oracle_cost']
    assert d['status'] == 'FAIL_FIXED_LOSS_ORACLE_CONDITIONAL_COST'
    assert (d['passing'], d['failed'], d['inconclusive']) == (0, 5, 0)
    assert (d['unique_cfd_frames'], d['new_solver_runs'], d['camera_count']) == (5, 10, 9)
    assert (d['stored_scored_entries'], d['inherited_prefix_entries'], d['computed_state_entries']) == (2570, 170, 2400)
    assert all(d[k] for k in ('loss_oracle_not_call_optimal', 'oracle_not_deployable',
        'conditional_shell_excludes_oracle_preparation', 'oracle_reference_inherited_nonfree',
        'original_observation_line_search_retained', 'posterior_stops_not_deployable', 'fixed_composition_closed'))
    assert not any(d[k] for k in ('new_training', 'trained_model_requalified', 'full_trajectories',
        'full505_authorized', 'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    for p, low in zip(d['points'], ([154, 153], [150, 149], [143, 142], [178, 177], [139, 138])):
        b = p['intervals']['first_cost']
        assert b == p['intervals']['sustained_cost']
        assert b['loss_oracle']['lower'] == low
        assert b['loss_oracle'] == b['manufactured_neural']
        assert all(a > z for a, z in zip(low, b['zero_metric']['upper']))
        assert all(a >= z for a, z in zip(low, b['prefix_only']['upper']))


def test_loss_oracle_public_cost_receipts_and_losses():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['loss_oracle_cost']
    assert d['actual_new_correction_refinement'] == dict(A=2400, AT=2390)
    assert d['inherited_prefix_logical'] == dict(A=160, AT=160)
    assert d['raw_native_audit'] == dict(A=10, AT=10)
    assert d['scoring_work'] == dict(A=2570, native_A=2570, consistency_A=10)
    assert d['metric_work'] == dict(F=1195, FT=1190, T=2380)
    for row in d['initialization_losses']:
        assert row['after_line_search_joint_loss'] >= row['raw_joint_loss']
        assert set(row) == {'point', 'raw_joint_loss', 'after_line_search_joint_loss'}
    assert d['maxima']['raw_joint_loss_reproduces_floor'] < 1e-8


def test_loss_oracle_bilingual_and_preserved_capacity_history():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert evidence['latest_loss_oracle_cost']['failed'] == 5
    assert evidence['latest_manufactured_feature_capacity']['examples'] == 37
    assert evidence['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#loss-oracle-cost-result')
        assert '0/5' in note['data-i18n-zh'] and '0/5' in note['data-i18n-en']
        assert '损失最优不等于调用数最优' in note['data-i18n-zh']
        assert 'loss-optimal is not call-optimal' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        assert soup.select_one('#manufactured-feature-floor-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #loss-oracle-cost-result')


def test_physical_sketch_necessary_failure_not_family_rehabilitation():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['physical_sketch']
    assert data['family_status'] == 'INCONCLUSIVE_P03_PHYSICAL_SKETCH_NUMERIC'
    assert data['necessary_status'] == 'FAIL_SEALED_P03_PRIMARY_NECESSARY_ACCURACY'
    assert data['checkpoints'] == 25 and data['cameras'] == [5, 7, 9, 5, 7]
    assert [r['numerical_valid'] for r in data['arms']] == [25]*7+[8, 25]
    assert [r['accuracy_passing'] for r in data['arms']] == [0, 25, 25, 0, 0, 0, 0, 0, 25]
    assert data['fixed_primary_closed'] and data['geometry_factor_nonfree']
    assert data['direct_control_not_primary_substitution']
    assert not any(data[k] for k in ('full_family_rehabilitated', 'trained_algorithm', 'full_trajectories',
        'resource_speedup', 'algorithm_breakthrough', 'paper_success', 'external_generalization', 'real_bost', 'training_authorized'))


def test_physical_sketch_bilingual_and_retained_scientific_boundaries():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert evidence['latest_physical_sketch']['primary_passing'] == 0
    assert evidence['latest_physical_sketch']['direct_passing'] == 25
    assert evidence['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#physical-sketch-result')
        for lang in ('zh', 'en'):
            assert '0/25' in note['data-i18n-'+lang] and '25/25' in note['data-i18n-'+lang]
        assert '不确定' in note['data-i18n-zh'] and 'inconclusive' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        assert soup.select_one('#loss-oracle-cost-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #physical-sketch-result')


def test_metric_boundary_is_not_a_noise_performance_or_warm_result():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['metric_objective_boundary']
    assert data['status'] == 'COUNTEREXAMPLES_TO_LEARNED_METRIC_OBJECTIVE_PRESERVATION'
    assert data['numerical_valid'] and data['shared_probes'] == 8 and data['learned_models'] == 5
    assert data['shared_probes_not_independent_rows'] and data['ratios_not_reconstruction_errors']
    assert data['geometry_factors_nonfree'] and data['clean505_result_unchanged']
    assert data['new_fitted_parameters'] == data['new_iterative_solver_paths'] == 0
    assert not any(data[k] for k in ('cfd_truth_read', 'cfd_observations_read', 'measured_noise',
        'final_reconstruction_scored', 'learned_warm_advantage', 'training_authorized',
        'resource_speedup', 'algorithm_breakthrough', 'paper_success', 'external_generalization', 'real_bost'))
    for p in (0, 1):
        rows = [r for r in data['summaries'] if r['path'] == p]
        assert [r['preserving'] for r in rows] == [8, 0, 0, 0, 0, 0, 0]
        assert rows[0]['leakage']['worst'] < 2e-13
        assert all(.32 < r['leakage']['p50'] < .35 for r in rows[2:])


def test_metric_boundary_bilingual_and_retains_strong_clean_control():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert evidence['latest_metric_objective_boundary']['learned_pairs'] == 40
    assert not evidence['latest_metric_objective_boundary']['new_performance_claim']
    assert evidence['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#metric-objective-boundary')
        assert '未训练' in note['data-i18n-zh'] and 'untrained' in note['data-i18n-en']
        assert '505' in note['data-i18n-zh'] and '505' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        assert soup.select_one('#physical-sketch-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #metric-objective-boundary')


def test_fixed_handoff_cost_censoring_and_same_reset_control():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['metric_prefix_handoff']
    assert data['status'] == 'FAIL_FIXED_METRIC_PREFIX_HANDOFF_NECESSARY_COST'
    assert data['numerical_valid'] and data['unique_opened_points'] == 5
    assert (data['prefix_steps'], data['suffix_steps'], data['camera_count']) == (16, 256, 9)
    assert data['primary_passing_points'] == 0 and data['reset_passing_points'] == 5
    assert data['first_equals_sustained'] and data['right_censored_not_divergence']
    assert data['crossings_not_deployable'] and data['classical_ordering_still_censored']
    assert data['full_direct_unbeaten'] and data['clean505_preserved'] and data['fixed_16_handoff_closed']
    assert not any(data[k] for k in ('complete_trajectories', 'new_training', 'algorithm_breakthrough',
        'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    for row, reset in zip(data['points'], (142, 138, 132, 169, 128)):
        assert row['handoff'] == dict(lower=274, upper=None)
        assert row['reset'] == dict(lower=reset, upper=reset)
        assert 2 <= reset-row['uninterrupted']['upper'] <= reset-row['uninterrupted']['lower'] <= 3


def test_handoff_bilingual_is_five_points_not_full_trajectories():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert evidence['latest_metric_prefix_handoff']['opened_points'] == 5
    assert not evidence['latest_metric_prefix_handoff']['complete_trajectories']
    assert evidence['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#metric-prefix-handoff-result')
        for lang in ('zh', 'en'):
            assert '273A+273AT' in note['data-i18n-'+lang] and '505' in note['data-i18n-'+lang]
        assert '固定16步' in note['data-i18n-zh'] and 'fixed 16-step' in note['data-i18n-en']
        assert note.get_text() == note['data-i18n-zh']
        assert soup.select_one('#metric-objective-boundary') and soup.select_one('#physical-sketch-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #metric-prefix-handoff-result')


def test_amg_counterexample_is_calls_not_resource_or_warm_success():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())['amg_classical_control']
    assert data['status'] == 'CLASSICAL_AMG_COUNTEREXAMPLE_TO_STRICT_LEARNED_T_ADVANTAGE'
    assert (data['opened_points'], data['primary_passing_points'], data['cameras']) == (5, 5, 9)
    assert data['independent_hierarchy'] and data['numerical_valid']
    assert data['geometry_cache_nonfree'] and data['V_cycle_nonfree']
    assert data['first_equals_sustained'] and data['crossings_not_deployable']
    assert data['clean505_against_old_controls_preserved'] and data['full_direct_unbeaten']
    assert not any(data[k] for k in ('complete_trajectories', 'new_training', 'algorithm_breakthrough',
        'paper_success', 'resource_speedup', 'external_generalization', 'real_bost', 'learned_warm_success'))
    for row, count in zip(data['points'], (71, 70, 73, 70, 75)):
        assert row['amg'] == dict(lower=count, upper=count)
        assert count < row['learned']['lower']


def test_amg_bilingual_retains_old_505_and_handoff_history():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert evidence['latest_amg_classical_control']['amg_earlier_points'] == 5
    assert evidence['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select_one('#amg-classical-control-result')
        assert note.get_text() == note['data-i18n-zh']
        assert '70至75' in note['data-i18n-zh'] and '70 to 75' in note['data-i18n-en']
        assert '505' in note['data-i18n-zh'] and '505' in note['data-i18n-en']
        assert '不免费' in note['data-i18n-zh'] and 'not free' in note['data-i18n-en']
        assert soup.select_one('#metric-prefix-handoff-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #amg-classical-control-result')
