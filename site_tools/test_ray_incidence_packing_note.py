import json
from pathlib import Path


SITE = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-ray-incidence-packing-20261008'
NAME = 'poolfire_ray_incidence_packing_2026-10-08_public_summary.json'


def test_independent_fixed_packing_failure_not_inverse_accuracy_failure():
    data = json.loads((SITE / 'docs' / NAME).read_text())
    assert data['decision'] == 'FAIL_FULL_RAY_INCIDENCE_PACKING_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 219
    assert data['camera_counts'] == [5, 7, 9] and data['tree_levels'] == 16
    assert data['spatial_degrees_of_freedom'] == 11849 and data['MB_bytes'] == 1000000
    assert data['new_inverse_builds'] == data['new_target_rhs_solves'] == data['new_trainable_parameters'] == 0
    assert data['CFD_or_actual_observation_reads'] == 0
    assert data['main_calls_each_implementation'] == {'A': 6, 'AT': 6}
    assert data['inherited_geometry_forward_equivalents'] == 35547 and data['geometry_setup_nonfree']
    assert data['connection_patterns_exactly_identical'] and data['native_measurement_row_permutation_pass']
    assert data['synthetic_12_camera_tested'] and not data['native_12_camera_tested']
    for r, expected in zip(data['reports'], (177763052, 220287140, 257152400)):
        assert r['packed_bytes'] == expected and r['control_packed_bytes'] == 8351564
        assert r['packed_bytes'] == r['coefficient_index_bytes'] + r['tree_bytes']
        assert r['storage_ratio'] == r['packed_bytes'] / r['dense_inverse_bytes']
        assert r['storage_ratio'] > data['storage_ratio_limit'] and not r['packing_gate']
        assert r['physical_replay_relative'] < 1e-10
        assert all(level['row_reversal_pattern_identical'] for level in r['levels'])
    assert data['fixed_full_ray_incidence_CSR_recipe_closed'] and data['not_a_global_impossibility_certificate']
    assert all(not data[k] for k in ('inverse_accuracy_tested', 'CFD_matched_accuracy_judgment',
        'learned_initializer_tested', 'neural_training_authorized', 'algorithm_breakthrough',
        'paper_success', 'resource_speedup', 'real_bost', 'main_goal_complete'))


def test_four_bilingual_notes_with_strict_scope_and_no_private_material():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-nonstandard-touching-20261008"')]
        assert note.count('data-i18n-zh=') >= 5 and note.count('data-i18n-en=') >= 5
        assert all(t in note for t in ('219/219', '15.83%', '19.61%', '22.89%', '6A+6AT', '35547'))
        assert 'inverse-action accuracy was not tested' in note
        assert 'not refute all ray-aware' in note and NAME in note and 'data-i18n-alt-en=' in note
    payload = (SITE / 'docs' / NAME).read_text()
    assert not any(t in payload for t in ('/Users/', 'private_results', 'private_data', 'sha256', 'checkpoint'))
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_ray_incidence_packing']['independent_checks_passed'] == 219
    assert not current['latest_ray_incidence_packing']['inverse_accuracy_tested']
