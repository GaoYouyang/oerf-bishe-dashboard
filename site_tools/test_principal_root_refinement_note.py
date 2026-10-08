import json
from pathlib import Path


SITE = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-principal-root-refinement-20261008'
NAME = 'poolfire_principal_root_refinement_2026-10-08_public_summary.json'


def test_actual_refinement_failure_with_adequate_controls_and_storage():
    d = json.loads((SITE / 'docs' / NAME).read_text())
    assert d['decision'] == 'FAIL_FIXED_PRINCIPAL_ROOT_REFINED_SENTINEL_CLOSED'
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 177
    assert d['sentinel_cells'] == 99 and d['opened_train_trajectories'] == 11
    assert d['frames_per_trajectory'] == 3 and d['sampled_absolute_strata'] == 33
    assert d['camera_counts'] == [5, 7, 9] and d['spatial_degrees_of_freedom'] == 11849
    assert d['reference_adequate'] and d['parent_adequate']
    assert d['matched_cells'][d['primary']] == d['basic_strata'][d['primary']] == 0
    assert d['matched_cells']['NormalFactor-PCGLS34'] == 99
    assert d['basic_strata']['NormalFactor-PCGLS34'] == d['basic_strata']['CGLS128'] == 33
    assert all(v == 0 for v in d['primary_individual_matched'].values())
    for row in d['storage']:
        assert row['compressed_factor_bytes'] == 8446372 and row['storage_gate']
        assert row['storage_ratio'] == row['compressed_factor_bytes'] / row['exact_inverse_factor_bytes']
        assert row['storage_ratio'] < d['storage_ratio_limit']
    assert max(d['independent_maxima'].values()) < 1e-7
    assert d['standalone_calls'] == {'A': 34, 'AT': 34, 'factor_actions': 68}
    assert d['actual_main_calls_each_implementation'] == {'A': 6138, 'AT': 3465}
    assert d['permutation_calls_each_implementation'] == {'A': 1122, 'AT': 1122}
    assert d['full_principal_root_builds_each_implementation'] == 3
    assert d['inherited_geometry_forward_equivalents'] == 35547 and d['geometry_setup_nonfree']
    assert d['construction_geometry_only'] and not d['truth_or_observation_used_for_construction']
    assert d['serialization_only_completion'] and d['failed_serialization_attempt_preserved']
    assert d['native_calculations_repeated_for_serialization'] == 0
    assert d['fixed_principal_root_recipe_closed'] and d['not_a_global_impossibility_certificate']
    assert d['post_refinement_matched_accuracy_tested'] and d['native_permutation_checks'] == 33
    assert all(not d[k] for k in ('learned_initializer_tested', 'algorithm_breakthrough', 'paper_success',
        'resource_speedup', 'fresh_wall_RSS_tested', 'external_generalization', 'real_bost',
        'neural_training_authorized', 'full_sequence_tested', 'main_goal_complete'))


def test_four_bilingual_notes_and_privacy_boundary():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-ray-incidence-packing-20261008"')]
        assert note.count('data-i18n-zh=') >= 5 and note.count('data-i18n-en=') >= 5
        assert all(t in note for t in ('177/177', '0/99', '0/33', '0.752%', '34A+34AT', '35547'))
        assert 'not refute all factorizations' in note and NAME in note
        assert 'data-i18n-alt-en=' in note
    payload = (SITE / 'docs' / NAME).read_text()
    assert not any(t in payload for t in ('/Users/', 'private_results', 'private_data', 'sha256', 'checkpoint'))
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_principal_root_refinement']['matched_cells'] == 0
