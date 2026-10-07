import json
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_nonlocal_graph_2026-10-07_public_summary.json'
MARKER = 'poolfire-nonlocal-graph-20261007'


def evidence():
    return json.loads((ROOT/'docs'/NAME).read_text())


def test_verified_learned_query_failure():
    d = evidence()
    assert d['decision'] == 'FAIL_GEOMETRY_TEACHER_ODD_GRAPH76_WARM25_CLOSED'
    assert d['cells'] == 99 and d['sampled_three_frame_strata'] == 33
    assert d['camera_counts'] == [5,7,9] and d['frames'] == [0,50,100]
    assert d['matched_cells'][d['primary']] == d['absolute_strata'][d['primary']] == 0
    assert d['matched_cells'][d['control']] == 0 and d['absolute_strata'][d['control']] == 33
    assert d['descriptive_controls']['CGLS35']['all_four_worse'] == 99
    assert d['matched_cells']['NormalFactor-PCGLS34'] == d['matched_cells']['DirectRidge-Warm34'] == 99


def test_training_isolation_cost_and_closure():
    d = evidence()
    assert d['shared_trainable_parameters'] == 76 and d['geometry_synthetic_pairs'] == 96
    assert d['fixed_training_updates'] == 50 and d['CFD_training_observations_and_truth'] == 0
    assert d['all_actual_trajectories_excluded_simultaneously'] and d['geometry_and_training_nonfree']
    assert d['standalone'] == {'A':35,'AT':35,'graph_channel_pairs':9,'CGLS_iterations':25}
    assert d['training_each_implementation'] == {'A':77568,'AT':77568}
    assert d['independent_extra_precision_pairs'] == 384 and d['query_full_factor_solves'] == 0
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 58
    assert d['parameter_max_relative'] < 1e-7 and d['final_field_max_relative'] < 1e-7
    assert d['old_invalid_teacher_attempt_preserved'] and d['metadata_only_summary_adjudication']
    assert d['formal_adopted_bitwise_not_rerun'] and d['original_scientific_recipe_and_gates_unchanged']
    assert d['learned_initializer_tested'] is True
    for key in ('full_sequence_tested','algorithm_breakthrough','paper_success','fresh_wall_RSS_tested',
                'resource_speedup','external_generalization','real_bost','curved_ray_validated','main_goal_complete'):
        assert d[key] is False


def test_bilingual_notes():
    for name in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html','learning_log.html'):
        page = BeautifulSoup((ROOT/name).read_text(),'html.parser')
        notes = page.select('#'+MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select('[data-i18n-zh][data-i18n-en]')) == 6
        assert all(text in note.get_text() for text in ('58/58','0/99','0/33','33/33','5/7/9','99/99','35A+35AT'))
        assert (ROOT/name).parent.joinpath(note.find('a')['href']).resolve().is_file()


def test_history_and_privacy():
    d = evidence()
    aggregate = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert aggregate['latest_nonlocal_graph_seed']['independent_checks_passed'] == 58
    assert aggregate['latest_nonlocal_graph_seed']['absolute_sampled_strata'] == 0
    assert 'latest_observable_band_seed' in aggregate
    assert '## 2026-10-07: 仅几何训练的小型非局部预测器' in (ROOT/'docs/operator_3d_learning_log.md').read_text()
    assert d['metric_max_absolute'] < 1e-7
    assert not any(token in (ROOT/'docs'/NAME).read_text() for token in ('/Users/','private_results/','private_data/','sha256','checkpoint'))
