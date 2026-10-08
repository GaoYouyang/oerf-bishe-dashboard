import json
from pathlib import Path


SITE = Path(__file__).resolve().parents[1]
NAME = 'poolfire_zero_support_attribution_2026-10-08_public_summary.json'
MARKER = 'poolfire-zero-support-attribution-20261008'


def test_zero_support_attribution_does_not_claim_full_identifiability():
    d = json.loads((SITE / 'docs' / NAME).read_text())
    assert d['decision'] == 'ZERO_SUPPORT_INSUFFICIENT_TO_EXPLAIN_ALL_PRINCIPAL_ROOT_DISCREPANCIES'
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 17
    assert d['camera_counts'] == [5, 7, 9] and d['nodes'] == 11849
    assert d['sentinels'] == 99 and d['field_decompositions'] == 297 and d['fixed_witnesses'] == 18
    assert all(g['zero_columns'] == g['guaranteed_gauge_free_null_dimension'] == 0 for g in d['geometry'])
    assert all(g['row_reversal_support_identical'] for g in d['geometry'])
    assert all(r['min_fraction'] == r['max_fraction'] == 0 for r in d['methods'].values())
    assert all(v == 0 for v in d['witnesses_by_camera'].values())
    assert max(d['independent_maxima'].values()) < 1e-7
    assert d['actual_calls_each_implementation'] == {'A': 1818, 'AT': 36}
    assert d['factor_actions_each_implementation'] == 54
    assert d['new_solver_iterations'] == d['new_normal_or_eigen_builds'] == d['new_trainable_parameters'] == 0
    assert d['inherited_geometry_forward_equivalents'] == 35547 and d['geometry_setup_nonfree']
    assert d['uncovered_node_explanation_excluded'] and d['old_root_recipe_remains_closed']
    assert d['old_pre_result_format_failure_preserved']
    assert all(not d[k] for k in ('full_nullspace_tested', 'full_rank_established', 'weak_combinations_excluded',
        'preconditioner_range_invariance_established', 'CFD_truth_error_attribution', 'support_amplitude_threshold_used',
        'learned_initializer_tested', 'algorithm_breakthrough', 'paper_success', 'resource_speedup',
        'external_generalization', 'real_bost', 'neural_training_authorized', 'full_sequence_tested', 'main_goal_complete'))


def test_bilingual_notes_exclude_false_zero_leak_success_claim():
    for file in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / file).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-principal-root-refinement-20261008"')]
        assert note.count('data-i18n-zh=') >= 5 and note.count('data-i18n-en=') >= 5
        assert all(v in note for v in ('17/17', '11849', '297', '18', '1818A+36AT', '35547'))
        assert 'not establish full rank' in note and 'zero witnesses are not a success certificate' in note
        assert NAME in note and 'data-i18n-alt-en=' in note
    payload = (SITE / 'docs' / NAME).read_text()
    assert not any(v in payload for v in ('/Users/', 'private_results', 'private_data', 'sha256', 'checkpoint'))
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_zero_support_attribution']['zero_columns_each_camera_count'] == 0
    assert not current['latest_zero_support_attribution']['full_nullspace_tested']
