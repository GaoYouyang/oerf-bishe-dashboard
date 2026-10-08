import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_cavity_variance6_2026-10-08_public_summary.json'
MARKER = 'poolfire-cavity-variance6-20261008'


def evidence():
    return json.loads((ROOT / 'docs' / NAME).read_text())


def test_isolation_and_independent_completion():
    data = evidence()
    assert data['decision'] == 'FAIL_CAVITY_VARIANCE6_LOTO_SENTINEL_CLOSED'
    assert data['parameters_per_fold'] == 6 and data['message_sweeps'] == 3
    assert data['outer_folds'] == 11 and data['train_trajectories_per_fold'] == 10
    assert data['held_teacher_reads_by_own_fold'] == 0 and not data['CFD_truth_used_in_training']
    assert data['samples'] == 99 and data['camera_counts'] == [5, 7, 9]
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 46
    assert data['recovery_integrity_checks_passed'] == data['recovery_integrity_checks_total'] == 6
    assert data['max_relative_state_difference'] < 1e-7 and data['max_absolute_metric_difference'] < 1e-7
    assert data['max_query_lift_relative'] < 1e-10 and data['max_query_physical_relative'] < 1e-10
    assert data['corrective_completion_frozen_before_results_read']
    assert not data['scientific_recipe_changed_for_completion']


def test_negative_result_and_full_cost():
    data = evidence()
    assert data['matched_cells'][data['primary']] == data['matched_cells'][data['fixed_control']] == 0
    assert data['basic_strata'][data['primary']] == 13 and data['basic_strata'][data['fixed_control']] == 1
    assert all(value == 99 for value in data['harm_vs_CGLS35'][data['primary']].values())
    assert data['primary_exact_standalone'] == {'A': 27, 'AT': 28}
    assert data['primary_conservative_equivalents'] == {'A': 35, 'AT': 34}
    assert data['control_conservative_equivalents'] == {'A': 35, 'AT': 33}
    assert data['independent_cumulative_main_calls_including_discarded_work'] == {'A': 111960, 'AT': 108702}
    assert data['independent_cumulative_updates_including_discarded_work'] == 572
    assert data['reused_independent_folds'] == 9 and data['new_completion_folds'] == 2
    assert data['discarded_unsealed_updates'] == 22 and data['fixed_cavity_recipe_closed']
    assert data['original_engineering_status'] == 'INCONCLUSIVE_RESOURCE_LIMIT'
    for key in ('original_resource_gate_retroactively_passed', 'fresh_wall_RSS_tested', 'full_sequence_tested',
                'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization',
                'real_bost', 'curved_ray_validated', 'main_goal_complete'):
        assert data[key] is False


def test_bilingual_notes_links_and_figure():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        page = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = page.select('#' + MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select('[data-i18n-zh][data-i18n-en]')) == 5
        assert all(token in note.get_text() for token in ('6', '99', '0/99', '13/33', '1/33', '46/46', '6/6', '5/7/9', '572', '22'))
        assert (ROOT / name).parent.joinpath(note.find('a')['href']).resolve().is_file()
        image = note.find('img')
        assert image['data-i18n-alt-zh'] and image['data-i18n-alt-en']
        assert (ROOT / name).parent.joinpath(image['src']).resolve().is_file()


def test_history_and_privacy():
    aggregate = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert aggregate['latest_cavity_variance6_learner']['matched_cells'] == 0
    assert aggregate['latest_vector_gradient_first19_learner']['independent_checks_passed'] == 48
    assert aggregate['latest_spatial_ray_shared54_learner']['independent_checks_passed'] == 40
    assert '## 2026-10-08: 逐连接消息学习' in (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert not any(token in (ROOT / 'docs' / NAME).read_text() for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
