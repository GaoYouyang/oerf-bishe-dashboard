import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
NAME = 'poolfire_global_review_2026-10-09_public_summary.json'
MARKER = 'poolfire-global-review-20261009'


def test_scope_and_parent_disposition():
    data = json.loads((SITE / 'docs' / NAME).read_text())
    assert data['scope']['cells'] == 99 and data['scope']['sampled_strata'] == 33
    assert data['scope']['camera_counts'] == [5, 7, 9]
    assert data['scope']['complete_sequence'] is data['scope']['complete_trajectory_LOTO'] is False
    result = data['scientific_diagnostic']
    assert result['primary_matched_cells'] == result['global_control_matched_cells'] == result['cold_equal_budget_matched_cells'] == 0
    assert result['primary_absolute_sampled_strata'] == 33
    assert result['reference_matched_cells'] == result['direct_control_matched_cells'] == 99
    assert result['initial_native_execution_resource_invalid'] and not result['initial_native_execution_promoted']
    assert result['original_output_cap_unchanged'] and result['sealed_candidates_not_regenerated']
    assert result['serialization_failure_preserved']
    assert result['primary_strict_wins_each_error_vs_global'] == result['primary_strict_wins_each_error_vs_cold'] == [0]*4
    assert result['paired_comparison_is_post_seal_descriptive'] and result['fixed_recipe_closed']


def test_cost_ratios_and_weak_bound_are_not_success():
    data = json.loads((SITE / 'docs' / NAME).read_text())
    cost = data['cost_scope']
    assert cost['grouped_and_global_online_A_AT_pairs_by_camera'] == [75, 91, 107]
    assert cost['cold_AT_by_camera'] == [74, 90, 106]
    assert cost['setup_orthogonalization_storage_validation_nonfree'] and cost['old_failed_work_not_erased']
    assert not cost['fresh_wall_RSS_tested']
    assert all(x > y for x, y in zip(data['p90_error_ratios_to_reference']['grouped_field'], data['p90_error_ratios_to_reference']['cold_field']))
    attribution = data['previous_independent_weak_gap_attribution']
    assert attribution['constant_gauge_removed'] and attribution['majority_certificates'] == [33]*3
    assert attribution['is_CFD_error_fraction'] is attribution['is_exact_unobservability_proof'] is False
    assert attribution['unique_neural_failure_cause'] is False
    assert all(v is False for v in data['claims'].values())
    assert data['independent']['fresh_rescore_to_original_max_absolute'] == 0


def test_public_privacy_bilingual_notes_and_prior_evidence():
    raw = (SITE / 'docs' / NAME).read_text()
    assert not any(term in raw for term in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        start = text.index(f'id="{MARKER}"')
        section = text[start:text.index('id="poolfire-detector-precision-20261009"', start)]
        assert section.count('data-i18n-zh=') == section.count('data-i18n-en=') == 6
        assert 'data-i18n-alt-zh=' in section and 'data-i18n-alt-en=' in section
        assert all(token in section for token in ('0/99', '33/33', '80.19%/86.73%/89.32%', NAME))
        assert 'post-execution diagnostic' in section and 'resource-invalid' in section
        assert '23.8268/18.1869/16.9640' in section and 'fixed endpoint' in section
        assert 'poolfire-primal-balanced-coarse-20261009' in text
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_global_review']['parent_native_resource_valid'] is False
    assert current['latest_pair_precision']['matched_cells'] == 0
    assert (SITE / 'assets/poolfire_global_review_2026-10-09.png').is_file()


def test_task_capacity_pilot_is_narrow_and_non_deployable():
    data = json.loads((SITE / 'docs' / NAME).read_text())
    pilot = data['task_metric_affine_capacity_pilot']
    assert pilot['cells'] == 3 and pilot['opened_train_trajectories'] == 1 and pilot['frame'] == 0
    assert pilot['camera_counts'] == [5, 7, 9] and pilot['basis_columns'] == [40, 56, 72]
    assert pilot['primary_feasible_cells'] == pilot['global_control_feasible_cells'] == 0
    assert pilot['simultaneous_feasibility_objective_upper_bound'] == 4
    assert all(x > 4 for x in pilot['primary_quadratic_lower_bounds'] + pilot['global_control_quadratic_lower_bounds'])
    assert pilot['truth_visible_offline_oracle_solves'] == 12
    assert pilot['new_A'] == 66 and pilot['new_AT'] == 12
    assert pilot['new_direction_generation'] == pilot['new_refinement_runs'] == pilot['new_predictor_fits'] == 0
    for key in ('actual_CFD_metric_budgets_unchanged', 'teacher_field_identity_not_required',
                'no_basis_truncation_or_new_directions', 'old_algorithm_not_reopened'):
        assert pilot[key] is True
    for key in ('complete_sequence', 'complete_trajectory_LOTO',
                'bound_on_alternative_initialization_then_new_CGLS',
                'universal_direction_family_impossibility', 'predictor_authorized'):
        assert pilot[key] is False
