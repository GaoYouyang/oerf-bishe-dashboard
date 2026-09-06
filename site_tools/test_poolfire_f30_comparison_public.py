import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_f30_comparison_20260906'


def test_aggregate_counts_and_cost_boundary():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    arms = {row['arm']: row for row in data['arms']}
    assert data['coverage']['cells'] == 505 and data['coverage']['geometries'] == 1
    assert data['coverage']['camera_counts'] == [9]
    assert (arms['ray_set_k1']['passing_cells'], arms['ray_set_k1']['complete_trajectories']) == (409, 2)
    assert (arms['linear_response_k1']['passing_cells'], arms['linear_response_k1']['complete_trajectories']) == (412, 1)
    assert arms['cgls4']['passing_cells'] == 505 and arms['cgls4']['is_reference']
    assert arms['field_k1']['logical_calls'] == {'A': 2, 'AT': 1}
    assert arms['ray_set_k1']['logical_calls'] == {'A': 2, 'AT': 2}
    for arm in arms.values():
        assert sum(r['passing_cells'] for r in arm['trajectory_results']) == arm['passing_cells']
        assert sum(r['passing_cells'] == 101 for r in arm['trajectory_results']) == arm['complete_trajectories']
        assert len(arm['trajectory_results']) == 5
    assert not data['neural_comparator']['hidden_weights_trained']
    assert not any(data['claims_fixed_false'].values())
    assert not data['cost_boundary']['fresh_wall_rss_comparison_completed']
    assert not data['criterion']['absolute_reference_adequacy_proven']


def test_exact_normal_diagnostic_is_not_prediction_success():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    diagnostic = data['diagnostics']['exact_normal_complement']
    assert diagnostic['cells'] == 25 and diagnostic['frames_per_trajectory'] == 5
    assert diagnostic['cross_at_least_normal_cells'] == 10
    assert diagnostic['cross_median_quarter_gate_trajectories'] == 5
    assert not diagnostic['promotion_gate_passed']
    assert diagnostic['teacher_visible'] and not diagnostic['new_predictor']
    assert not diagnostic['K1_refinement_run']
    assert all(row['cross'] >= .25 for row in diagnostic['per_trajectory'])
    assert diagnostic['per_trajectory'][3]['normal'] > .50
    for path in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT / path).read_text(), 'html.parser')
        note = soup.select_one('#normal-moment-diagnostic')
        assert note and '10/25' in note['data-i18n-zh'] and '10/25' in note['data-i18n-en']


def test_trained_comparator_is_not_an_independent_retraining_or_success_claim():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    trained = data['trained_neural_comparator']
    arm = next(a for a in data['arms'] if a['arm'] == 'trained_ray_set_k1')
    assert len(data['arms']) == 15
    assert (arm['passing_cells'], arm['complete_trajectories']) == (420, 1)
    assert [r['passing_cells'] for r in arm['trajectory_results']] == [101, 89, 90, 49, 91]
    assert trained['trainable_parameters_per_fold'] == 369
    assert trained['additional_train_only_output_scale_per_fold'] == 1
    assert trained['epochs'] == 20 and trained['optimizer_updates'] == 10100
    assert trained['hidden_weights_trained'] and trained['independent_prediction_and_physical_replay']
    assert trained['shared_sealed_parameter_artifacts'] and not trained['independent_optimizer_trajectory_repeated']
    assert trained['dominates_cheaper_ridge_cells'] == 504 and not trained['cheap_control_gate_passed']
    assert trained['fixed_budget_instance_closed']
    assert trained['call_counts']['training_A'] == trained['call_counts']['training_AT'] == 80800
    current = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_prediction'] == trained
    for file in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT / file).read_text(), 'html.parser')
        note = soup.select_one('#trained-ray-set-result')
        assert note and '504/505' in note['data-i18n-zh'] and '504/505' in note['data-i18n-en']
        assert 'not independent retraining' in note['data-i18n-en']


def test_normal_cache_arithmetic_is_not_a_wall_or_reconstruction_claim():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    row = data['diagnostics']['exact_normal_cache']
    assert row['normal_nonzeros'] == 64746812 and row['factor_nonzeros'] == 2960876
    assert row['geometry_rows'] == 29700 and row['independently_verified']
    assert abs(row['normal_nonzeros'] / (2 * row['factor_nonzeros']) - row['arithmetic_terms_ratio']) < 1e-6
    assert row['memory_budget_passed'] and not row['arithmetic_budget_passed']
    assert row['measured_wall_ratio'] is None
    assert not row['truth_or_teacher_read'] and not row['predictor_or_replay_run']
    current = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert current['normal_cache_diagnostic']['measured_wall_ratio'] is None
    assert not current['normal_cache_diagnostic']['cheap_control_screen_passed']
    for path in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        note = BeautifulSoup((ROOT / path).read_text(), 'html.parser').select_one('#normal-cache-diagnostic')
        assert note and '10.93' in note['data-i18n-zh'] and '10.93' in note['data-i18n-en']
        assert 'not measured wall' in note['data-i18n-en']


def test_trained_loss_attribution_does_not_replace_the_predictor():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    diagnostic = data['diagnostics']['trained_loss_attribution']
    assert diagnostic['floor_dominant_query_cells'] == 505
    assert diagnostic['learned_floor_lower_than_random_feature_cells'] == 464
    assert not diagnostic['new_deployed_model'] and not diagnostic['nonlinear_convergence_proven']
    assert not diagnostic['final_K1_or_truth_error_bound'] and not diagnostic['optimizer_retry_authorized']
    assert sum(r['learned_floor_lower_cells'] for r in diagnostic['per_trajectory']) == 464
    assert all(.018 < r['train_objective_removable_fraction'] < .043 for r in diagnostic['per_trajectory'])
    assert all(.81 < r['query_floor_fraction_median'] < .89 for r in diagnostic['per_trajectory'])
    p45 = diagnostic['per_trajectory'][3]
    assert p45['query_train_optimal_head_teacher_loss_mean'] > p45['query_original_teacher_loss_mean']
    assert diagnostic['call_counts']['basis_A'] == diagnostic['call_counts']['basis_AT'] == 42925
    current = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_diagnostic'] == diagnostic
    assert current['latest_prediction']['matched_cells'] == 420
    for file in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        note = BeautifulSoup((ROOT / file).read_text(), 'html.parser').select_one('#trained-loss-attribution')
        assert note and '464/505' in note['data-i18n-zh'] and '464/505' in note['data-i18n-en']


def test_weak_message_not_uniformly_harmless():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    diagnostic = data['diagnostics']['global_context']
    assert diagnostic['field_norm_contribution_max'] < .000245
    assert diagnostic['projected_field_norm_contribution_max'] < .000274
    assert diagnostic['unchanged_joint_classifications'] == 505
    assert not diagnostic['strict_uniform_removability_passed']
    assert diagnostic['ablation_all_four_better_cells'] == 41
    assert diagnostic['parent_all_four_better_cells'] == 12


def test_private_boundary_bilingual_current_and_archives():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    text = (ROOT / f'docs/{STEM}.md').read_text()
    for forbidden in ('/Users/', '/Volumes/', 'private_results', 'sha256', 'cameraData', 'checkpoint', '.pt'):
        assert forbidden not in json.dumps(data) and forbidden not in text
    assert '# PoolFire:' in text and '完整轨迹' in text and 'not 505 independent' in text
    current = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert current['scientific_status'] == data['scientific_status']
    assert current['current_decision']['v283_fixed_reference_closed']
    assert STEM in current['public_evidence']['summary']
    for file in ('index.html', 'operator-learning/index.html'):
        soup = BeautifulSoup((ROOT / file).read_text(), 'html.parser')
        section = soup.select_one('#poolfire-f30-comparison')
        assert section and soup.select_one('#v283-nodal-reference')
        for node in section.select('h2,p[data-i18n-zh],figcaption'):
            assert node.get('data-i18n-zh') and node.get('data-i18n-en')
        assert '409' in soup.select_one('header').get_text()
        image = section.select_one('img')
        assert image.get('data-i18n-alt-zh') and image.get('data-i18n-alt-en')
    daily = BeautifulSoup((ROOT / 'operator-learning/daily-progress.html').read_text(), 'html.parser')
    assert daily.select_one('#latest')['data-date'] == '2026-09-06'
    assert daily.select_one('#day-2026-09-05')
    assert (ROOT / f'assets/figures/{STEM}.png').stat().st_size > 10000
