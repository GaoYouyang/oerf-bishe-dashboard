import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_spatial_ray_shared54_2026-10-07_public_summary.json'
MARKER = 'poolfire-spatial-ray-shared54-20261007'


def evidence():
    return json.loads((ROOT/'docs'/NAME).read_text())


def test_actual_learning_and_trajectory_isolation():
    d = evidence()
    assert d['decision'] == 'FAIL_SPATIAL_RAY_SHARED54_LOTO_SENTINEL_CLOSED'
    assert d['learned_initializer_tested'] and d['parameters_per_fold'] == 54
    assert d['outer_folds'] == 11 and d['train_trajectories_per_fold'] == 10
    assert d['held_teacher_reads_by_own_fold'] == 0 and not d['CFD_truth_used_in_training']
    assert d['camera_counts'] == [5, 7, 9] and d['samples'] == 99
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 40
    assert d['max_relative_model_difference'] < 1e-7 and d['max_relative_state_difference'] < 1e-7
    assert d['max_absolute_metric_difference'] < 1e-7


def test_accuracy_cost_and_boundaries():
    d = evidence()
    for method in ('SpatialRayShared54-Warm33', 'SafeRowRichardson2-Warm33'):
        assert d['matched_cells'][method] == 0 and d['basic_strata'][method] == 33
    for method in ('NormalFactor-PCGLS34', 'DirectRidge-Warm34'):
        assert d['matched_cells'][method] == 99 and d['basic_strata'][method] == 33
    assert d['standalone_each_arm'] == {'A': 35, 'AT': 35}
    assert d['new_training_each_mode'] == {'A': 149490, 'AT': 148500}
    assert d['new_input_cache_each_mode'] == {'A': 0, 'AT': 99}
    assert d['reused_teacher_factory_nonfree'] and d['replay_score_permutation_extra']
    assert d['fixed_spatial_ray_recipe_closed']
    for key in ('full_sequence_tested', 'algorithm_breakthrough', 'paper_success', 'fresh_wall_RSS_tested',
                'resource_speedup', 'external_generalization', 'real_bost', 'curved_ray_validated', 'main_goal_complete'):
        assert d[key] is False


def test_bilingual_notes():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        page = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        notes = page.select('#'+MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select('[data-i18n-zh][data-i18n-en]')) == 5
        assert all(t in note.get_text() for t in ('54', '99', '0/99', '33/33', '40/40', '5/7/9', '35A+35AT'))
        assert (ROOT/name).parent.joinpath(note.find('a')['href']).resolve().is_file()


def test_history_and_privacy():
    aggregate = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert aggregate['latest_spatial_ray_shared54_learner']['matched_cells'] == 0
    assert aggregate['latest_sparse_factor_condition_diagnostic']['original_accuracy_status'] == 'INCONCLUSIVE'
    assert '## 2026-10-07: 共享空间学习器' in (ROOT/'docs/operator_3d_learning_log.md').read_text()
    assert not any(t in (ROOT/'docs'/NAME).read_text()
                   for t in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
