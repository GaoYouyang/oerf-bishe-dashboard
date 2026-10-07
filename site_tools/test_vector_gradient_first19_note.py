import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_vector_gradient_first19_2026-10-07_public_summary.json'
MARKER = 'poolfire-vector-gradient-first19-20261007'


def evidence():
    return json.loads((ROOT/'docs'/NAME).read_text())


def test_learning_isolation_and_physics():
    d = evidence()
    assert d['decision'] == 'FAIL_VECTOR_GRADIENT_FIRST19_LOTO_SENTINEL_CLOSED'
    assert d['actual_shared_learning'] and d['parameters_per_fold'] == 19 and d['linear_parameters_per_fold'] == 2
    assert d['outer_folds'] == 11 and d['train_trajectories_per_fold'] == 10
    assert d['held_teacher_reads_by_own_fold'] == 0 and not d['CFD_truth_used_in_training']
    assert d['camera_counts'] == [5, 7, 9] and d['samples'] == 99
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 48
    assert d['extra_physics_checks_passed'] == d['extra_physics_checks_total'] == 5
    assert d['max_relative_state_difference'] < 1e-7 and d['max_absolute_metric_difference'] < 1e-7
    assert d['max_discrete_potential_stationarity'] < 1e-8
    assert d['max_relative_auxiliary_operator_difference'] < 1e-10


def test_accuracy_and_honest_cost():
    d = evidence()
    for method in (d['primary'], d['linear_control']):
        assert d['matched_cells'][method] == 0 and d['basic_strata'][method] == 33
        assert all(d['harm_vs_CGLS35'][method][m] == 99 for m in d['harm_vs_CGLS35'][method])
    assert d['exact_standalone_each_arm'] == {'A': 33, 'AT': 32}
    assert d['auxiliary_gradient_adjoint_products_each_arm'] == 3
    assert d['conservative_AT_equivalent_each_arm'] == 35
    assert d['new_training_each_mode'] == {'A': 298980, 'AT': 247500}
    assert d['sample_updates_each_mode'] == 99000 and d['fixed_local_gradient_recipe_closed']
    for key in ('full_sequence_tested', 'algorithm_breakthrough', 'paper_success', 'fresh_wall_RSS_tested',
                'resource_speedup', 'external_generalization', 'real_bost', 'curved_ray_validated', 'main_goal_complete'):
        assert d[key] is False


def test_bilingual_notes_and_links():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        page = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        notes = page.select('#'+MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select('[data-i18n-zh][data-i18n-en]')) == 5
        assert all(t in note.get_text() for t in ('19', '99', '0/99', '33/33', '48/48', '5/5', '5/7/9', '33A+32AT', '35AT-equivalent'))
        assert (ROOT/name).parent.joinpath(note.find('a')['href']).resolve().is_file()


def test_preserved_history_and_privacy():
    aggregate = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert aggregate['latest_vector_gradient_first19_learner']['matched_cells'] == 0
    assert aggregate['latest_spatial_ray_shared54_learner']['independent_checks_passed'] == 40
    assert aggregate['latest_sparse_factor_condition_diagnostic']['original_accuracy_status'] == 'INCONCLUSIVE'
    assert '## 2026-10-07: 梯度优先学习初值' in (ROOT/'docs/operator_3d_learning_log.md').read_text()
    assert not any(t in (ROOT/'docs'/NAME).read_text() for t in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
