import json
from pathlib import Path


SITE = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-nonstandard-touching-20261008'
NAME = 'poolfire_nonstandard_touching_2026-10-08_public_summary.json'


def test_independently_closed_geometry_representation_not_algorithm_success():
    data = json.loads((SITE / 'docs' / NAME).read_text())
    assert data['decision'] == 'FAIL_FIXED_NONSTANDARD_TOUCHING_INVERSE_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 235
    assert data['camera_counts'] == [5, 7, 9] and data['range_probes_total'] == 36
    assert all(not data[k] for k in ('CFD_matched_accuracy_judgment', 'CGLS_compensation_impossibility_claim', 'learned_initializer_tested',
        'neural_training_authorized', 'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'real_bost', 'main_goal_complete'))
    assert data['new_trainable_parameters'] == data['CFD_or_actual_observation_reads'] == 0
    for r in data['reports']:
        assert r['telescoping_relative'] < 1e-7 and r['storage_ratio'] < .1 and not r['capacity_gate']
        assert r['worst_field_inverse_action_error']['primary'] > .01
        assert r['worst_projected_inverse_action_error']['primary'] > .01
    assert data['main_calls_each_implementation'] == {'A': 252, 'AT': 72}
    assert data['full_inverse_scalar_rhs_each_implementation'] == 35547 and data['geometry_targets_nonfree']


def test_four_bilingual_primary_notes_and_private_data_boundary():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-weighted-haar-capacity-20261008"')]
        assert note.count('data-i18n-zh=') >= 5 and note.count('data-i18n-en=') >= 5
        assert '235/235' in note and '0.7436%' in note and 'not refute every coupled' in note
        assert NAME in note and 'data-i18n-alt-en=' in note
    payload = (SITE / 'docs' / NAME).read_text()
    assert not any(t in payload for t in ('/Users/', 'private_results', 'private_data', 'sha256', 'checkpoint'))
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_nonstandard_touching_inverse']['independent_checks_passed'] == 235
