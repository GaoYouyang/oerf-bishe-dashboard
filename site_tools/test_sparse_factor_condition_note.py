import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_sparse_factor_condition_2026-10-07_public_summary.json'
MARKER = 'poolfire-sparse-factor-condition-20261007'


def evidence():
    return json.loads((ROOT/'docs'/NAME).read_text())


def test_condition_diagnosis_not_accuracy_verdict():
    d = evidence()
    assert d['decision'] == 'PASS_POSTOPEN_GEOMETRY_CONDITION_DIAGNOSTIC'
    assert d['original_accuracy_status'] == 'INCONCLUSIVE' and d['original_primary_queries'] == 0
    assert d['original_primary_not_restarted'] and d['formal_stopped_on_first_control_camera_permutation']
    assert d['CFD_observations_and_truth_reads'] == 0
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 33
    assert d['factor_pattern_matches'] and d['max_relative_difference'] < 1e-7
    assert d['not_unregularized_PCG_eigenvalues_or_accuracy']
    rows = d['geometry_diagnosis']
    assert len(rows) == 6 and {r['camera_count'] for r in rows} == {5, 7, 9}
    assert all(r['normalized_regularized_condition_lower_bound'] > 1e19 for r in rows)
    assert all(abs(r['determinant_geometric_mean']-1) < 1e-7 for r in rows)
    assert all(r['exact_reference_rayleigh_max_error'] < 1e-7 for r in rows)


def test_cost_and_scientific_boundaries():
    d = evidence()
    assert d['diagnostic_each_implementation'] == {
        'full_factorizations': 3, 'quadratic_forward_equivalents': 27, 'direct_quadratic_normal_actions': 18}
    assert d['independent_inverse_weight_rhs'] == 35547 and d['geometry_work_nonfree']
    assert d['uncertified_entry_pruning_recipe_closed']
    for key in ('learned_initializer_tested', 'full_sequence_tested', 'algorithm_breakthrough', 'paper_success',
                'fresh_wall_RSS_tested', 'resource_speedup', 'external_generalization', 'real_bost',
                'curved_ray_validated', 'main_goal_complete'):
        assert d[key] is False


def test_bilingual_notes():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        page = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        notes = page.select('#'+MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select('[data-i18n-zh][data-i18n-en]')) == 5
        assert all(text in note.get_text() for text in ('33/33', '5/7/9', 'inconclusive', '5.50e44', '1.54e31', '3.81e19'))
        assert (ROOT/name).parent.joinpath(note.find('a')['href']).resolve().is_file()


def test_history_and_privacy():
    aggregate = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert aggregate['latest_sparse_factor_condition_diagnostic']['original_accuracy_status'] == 'INCONCLUSIVE'
    assert aggregate['latest_nonlocal_graph_seed']['independent_checks_passed'] == 58
    assert '## 2026-10-07: 稀疏逆因子诊断' in (ROOT/'docs/operator_3d_learning_log.md').read_text()
    assert not any(token in (ROOT/'docs'/NAME).read_text()
                   for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
