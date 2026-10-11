import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def summary():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    archived = json.loads((ROOT/'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    a, b = current['latest_signed_cross_ray']['frozen_neural_residual_hybrid'], archived['frozen_neural_residual_hybrid']
    assert a == b
    return a


def test_quality_gain_and_literal_failed_gate():
    a = summary()
    assert a['decision'] == 'FAIL_FROZEN_NEURAL_RESIDUAL_HYBRID'
    assert a['primary_matched'] == 9 and a['primary_absolute_strata'] == 3
    assert a['mean_joint_nonharm'] == 5 and a['mean_observation_only_harms'] == 4
    assert abs(a['field_ratio_vs_mean']-.8904944141005996) < 1e-10
    assert abs(a['field_ratio_vs_zero']-.6948973389846305) < 1e-10


def test_cost_independence_and_scope():
    a = summary()
    assert a['queries'] == 9 and a['held_folds_tested'] == 1 and a['family_already_opened']
    assert a['new_training_updates'] == 0 and a['inherited_updates'] == 1600
    assert a['primary_actions'] == {'A': 6, 'AT': 6} and a['cheaper_zero_actions'] == {'A': 5, 'AT': 5}
    assert a['cheaper_zero_matched'] == 9 and a['full_factor_nonfree']
    assert a['shared_frozen_weights'] and not a['independent_retraining']
    assert a['independent_checks'] == 18 and a['terminal_checks'] == 68
    for flag in ('whole_pipeline_RSS_test', 'complete_sequence', 'new_geometry', 'learned_call_reduction',
                 'resource_speedup', 'algorithm_breakthrough', 'paper_success', 'external_generalization',
                 'real_bost', 'main_goal_complete'):
        assert not a[flag]


def test_bilingual_visible_evidence_and_figure():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (ROOT/name).read_text()
        assert text.count('id="frozen-neural-hybrid"') == 1
        paragraph = text.split('id="frozen-neural-hybrid"', 1)[1].split('</p>', 1)[0]
        assert 'data-i18n-zh=' in paragraph and 'data-i18n-en=' in paragraph
        assert '30.51%' in paragraph and '10.95%' in paragraph and '5/9' in paragraph and 'FAIL' in paragraph
    image = Image.open(ROOT/'docs/frozen_neural_hybrid_2026-10-11.png')
    assert image.size == (1800, 619)
    doc = (ROOT/'docs/bp_coordinate_field_2026-10-11.md').read_text()
    assert '30.51%' in doc and '10.95%' in doc and '6A+6AT' in doc and '5A+5AT' in doc
    for token in ('/Users/', 'private_results/', '.npz', 'checkpoint', 'sha256'):
        assert token not in doc
