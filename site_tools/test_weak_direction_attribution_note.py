import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
NAME = 'poolfire_weak_direction_attribution_2026-10-08_public_summary.json'
MARKER = 'poolfire-weak-direction-attribution-20261008'


def test_weak_direction_attribution_preserves_negative_and_scale_scope():
    d = json.loads((SITE / 'docs' / NAME).read_text())
    assert d['decision'] == 'WEAK_DIRECTION_DOMINANCE_NOT_ESTABLISHED_AT_INHERITED_SCALE'
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 17
    assert d['camera_counts'] == [5, 7, 9] and d['nodes'] == 11849
    assert d['sentinels'] == 99 and d['field_decompositions'] == 297 and d['fixed_witnesses'] == 18
    main = d['methods']['PrincipalRootTouch-PCGLS34']
    assert main['at_least_90pct_cells'] == 0
    assert abs(main['score_min'] - .20044780936677614) < 1e-8
    assert abs(main['score_median'] - .5272347024616408) < 1e-8
    assert abs(main['score_max'] - .8600371599872827) < 1e-8
    assert all(v == 0 for v in d['witnesses_by_camera'].values())
    assert max(d['independent_maxima'].values()) < 1e-7
    assert d['actual_calls_each_implementation'] == {'A': 1887, 'AT': 63}
    assert d['normal_builds_each_implementation'] == d['factor_builds_each_implementation'] == 3
    assert d['diagnostic_filter_solves_each_implementation'] == 342
    assert d['new_solver_iterations'] == d['new_trainable_parameters'] == 0
    assert d['inherited_geometry_forward_equivalents'] == 35547 and d['geometry_setup_nonfree']
    assert d['old_root_recipe_remains_closed'] and d['pre_result_entry_failure_preserved']
    assert all(not d[k] for k in ('uniform_dominance_supported', 'score_is_physical_energy_fraction',
        'exact_nullspace_separated', 'full_rank_established', 'new_CFD_truth_read', 'learned_initializer_tested',
        'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost',
        'neural_training_authorized', 'gpu_rental_authorized', 'full_sequence_tested', 'main_goal_complete'))


def test_bilingual_notes_do_not_convert_scores_to_accuracy_or_speed():
    for file in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / file).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-zero-support-attribution-20261008"')]
        assert note.count('data-i18n-zh=') >= 6 and note.count('data-i18n-en=') >= 6
        assert all(v in note for v in ('17/17', '11849', '99', '1887A+63AT', '342', '35547', '0/99'))
        assert 'not a physical-field energy fraction' in note and 'not a learned initializer' in note
        assert NAME in note and 'data-i18n-alt-en=' in note
    payload = (SITE / 'docs' / NAME).read_text()
    assert not any(v in payload for v in ('/Users/', 'private_results', 'private_data', 'sha256', 'checkpoint'))
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert not current['latest_weak_direction_attribution']['uniform_dominance_supported']
    assert not current['latest_weak_direction_attribution']['main_goal_complete']
