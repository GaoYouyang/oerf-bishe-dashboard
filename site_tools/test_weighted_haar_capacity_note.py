import json
import math
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_weighted_haar_capacity_2026-10-08_public_summary.json'
MARKER = 'poolfire-weighted-haar-capacity-20261008'


def evidence():
    return json.loads((ROOT / 'docs' / NAME).read_text())


def test_geometry_only_capacity_and_independence():
    d = evidence()
    assert d['decision'] == 'FAIL_FIXED_WEIGHTED_HAAR_DIAGONAL_CAPACITY_CLOSED'
    assert d['camera_counts'] == [5, 7, 9] and d['spatial_degrees_of_freedom'] == 11849
    assert d['unrestricted_nonconstant_diagonal_coefficients_per_geometry'] == 11848
    assert d['range_probes_total'] == 36 and d['range_probes_per_camera_count'] == 12
    assert d['CFD_observation_or_field_teacher_reads'] == d['new_trainable_predictor_parameters'] == 0
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 66
    assert d['max_relative_target_difference'] < 1e-7
    assert d['max_relative_diagonal_difference'] < 1e-7
    assert d['max_relative_physical_image_difference'] < 1e-7
    assert d['max_absolute_metric_lower_bound_difference'] < 1e-7


def test_certificate_scope_and_cost():
    d = evidence()
    assert math.isclose(d['necessary_sum_squares_limit'], 12 * d['probe_field_error_limit'] ** 2,
                        rel_tol=0, abs_tol=1e-18)
    for r in d['reports']:
        for values in r['families'].values():
            assert values['certificate'] == 'CERTIFIED_INFEASIBLE_FIXED_DIAGONAL'
            assert values['minimum_relative_field_sum_squares'] > d['necessary_sum_squares_limit'] + 1e-12
            assert math.isclose(values['probe_rms_field_inverse_action_error'] ** 2 * 12,
                                values['minimum_relative_field_sum_squares'])
            assert values['oracle_stationarity'] < 1e-10 and values['quadratic_identity_error'] < 1e-10
    assert d['main_calls_each_implementation'] == {'A': 252, 'AT': 72}
    assert d['row_reversal_calls_each_implementation'] == {'A': 9, 'AT': 9}
    assert d['normal_factor_builds_each_implementation'] == 3 and d['normal_rhs_solves_each_implementation'] == 36
    assert d['geometry_targets_nonfree'] and d['fixed_diagonal_representation_screen_closed']
    for key in ('off_diagonal_or_query_dependent_representation_closed', 'CFD_matched_accuracy_judgment',
                'CGLS_compensation_impossibility_claim', 'learned_initializer_tested', 'algorithm_breakthrough',
                'paper_success', 'resource_speedup', 'fresh_wall_RSS_tested', 'full_sequence_tested',
                'external_generalization', 'real_bost', 'curved_ray_validated', 'main_goal_complete'):
        assert d[key] is False


def test_bilingual_notes_figure_and_history():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        page = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = page.select('#' + MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select('[data-i18n-zh][data-i18n-en]')) == 5
        assert all(token in note.get_text() for token in ('5/7/9', '11849', '11848', '66/66', '0.0012', '252A+72AT'))
        assert (ROOT / name).parent.joinpath(note.find('a')['href']).resolve().is_file()
        image = note.find('img')
        assert image['data-i18n-alt-zh'] and image['data-i18n-alt-en']
        assert (ROOT / name).parent.joinpath(image['src']).resolve().is_file()
        assert page.select('#poolfire-cavity-variance6-20261008')
    aggregate = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert aggregate['latest_cavity_variance6_learner']['independent_checks_passed'] == 46
    assert aggregate['latest_weighted_haar_inverse_capacity']['independent_checks_passed'] == 66
    assert '## 2026-10-08: 完整多尺度表示' in (ROOT / 'docs/operator_3d_learning_log.md').read_text()


def test_privacy():
    payload = (ROOT / 'docs' / NAME).read_text()
    assert not any(token in payload for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
