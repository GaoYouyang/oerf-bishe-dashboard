import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summaries():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    archive = json.loads((ROOT/'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    return current['latest_signed_cross_ray']['bp_coordinate_field_initializer'], archive['bp_coordinate_field_initializer']


def test_same_scientific_summary_and_scope():
    a, b = summaries()
    assert a == b
    assert a['decision'] == 'FAIL_BP_COORDINATE_FIELD_NECESSARY_GATE'
    assert a['queries'] == 9 and a['held_trajectory_folds_tested'] == 1
    assert a['held_pairs_excluded'] == 303 and a['legal_training_pairs'] == 3030
    assert a['hidden_neural_updates'] == 1600
    assert a['primary_matched'] == 0 and a['primary_absolute_strata'] == 3


def test_independence_and_resource_boundaries():
    a, _ = summaries()
    assert a['shared_frozen_weights'] and not a['independent_retraining']
    assert a['query_actions'] == {'A': 35, 'AT': 35}
    assert a['cheaper_analytic_control_matched'] == 9
    assert a['primary_vs_controls']['CGLS35']['joint_nonharm'] == 9
    assert a['primary_vs_controls']['FoldMeanRaw-Warm34']['joint_nonharm'] == 6
    for flag in ('complete_sequence', 'new_geometry', 'whole_pipeline_RSS_test', 'learned_call_reduction',
                 'algorithm_breakthrough', 'paper_success', 'resource_speedup',
                 'external_generalization', 'real_bost', 'main_goal_complete'):
        assert not a[flag]


def test_four_bilingual_surfaces_and_doc():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (ROOT/name).read_text()
        assert text.count('id="bp-coordinate-field"') == 1
        paragraph = text.split('id="bp-coordinate-field"', 1)[1].split('</p>', 1)[0]
        assert 'data-i18n-zh=' in paragraph and 'data-i18n-en=' in paragraph
        assert '0/9' in paragraph and '27.51%' in paragraph and '35A+35AT' in paragraph
        assert 'bp_coordinate_field_2026-10-11.md' in text
    document = (ROOT/'docs/bp_coordinate_field_2026-10-11.md').read_text()
    assert '## 中文' in document and '## English' in document
    for token in ('/Users/', 'private_results/', '.npz', 'checkpoint', 'sha256'):
        assert token not in document
