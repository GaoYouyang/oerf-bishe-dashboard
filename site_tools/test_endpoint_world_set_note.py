import json
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-endpoint-world-set-20261006'
SUMMARY = 'poolfire_endpoint_world_set_2026-10-06_public_summary.json'


def report():
    return json.loads((ROOT / 'docs' / SUMMARY).read_text())


def test_endpoint_science_is_failed_recipe_not_failed_bost():
    data = report()
    assert data['decision'] == 'FAIL_WORLD_SET52_ENDPOINT_LOTO16_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 25
    assert data['shared_parameters'] == 52 and data['outer_folds'] == 11
    assert data['fixed_full_batch_updates_per_fold'] == 16
    assert data['basic_strata_passed'] == data['basic_strata_total'] == 33
    assert data['strict_four_metric_matches'] == 0 and data['matched_cells_total'] == 99
    assert data['reference_adequate_for_sentinel_strata'] and data['fixed_recipe_closed']
    assert data['held_truth_or_teacher_reads_for_fit'] == 0
    assert data['all_predictions_sealed_before_held_truth']
    assert data['independent_maxima']['fit_relative'] < 1e-12
    assert data['independent_maxima']['metric_absolute'] < 1e-12
    for field in ('full_sequence_authorized', 'neural_operator_comparison_completed',
                  'fresh_deployment_benchmark', 'algorithm_breakthrough', 'paper_success',
                  'resource_speedup', 'external_generalization', 'curved_ray_validated',
                  'real_bost', 'loss_location_causal_ablation', 'closure_fix_refits_or_changes_arrays'):
        assert data[field] is False


def test_endpoint_comparisons_keep_tiny_gain_and_strong_control_harm():
    data = report()
    controls = data['cheap_control_comparisons']
    assert controls['CGLS']['all_four_no_worse'] == 0
    assert controls['CGLS']['all_four_worse'] == 99
    assert controls['PCGLS']['all_four_no_worse'] == 2
    assert controls['PCGLS']['all_four_worse'] == 43 and controls['PCGLS']['mixed'] == 54
    assert controls['UnscaledBP-Warm']['all_four_no_worse'] == 98
    assert controls['UnscaledBP-Warm']['all_four_worse'] == 1
    field = 'field_rel_l2_gauge_centered'
    observation = 'observation_rel_l2'
    assert 1.002 < controls['CGLS']['ratios'][field]['median'] < 1.003
    assert 1.02 < controls['CGLS']['ratios'][observation]['median'] < 1.021
    assert .99998 < controls['UnscaledBP-Warm']['ratios'][field]['median'] < .99999
    assert .99988 < controls['UnscaledBP-Warm']['ratios'][observation]['median'] < .99990
    assert data['descriptive_max_difference'] < 1e-10
    assert data['descriptive_new_fits'] == data['descriptive_new_A'] == data['descriptive_new_AT'] == 0
    for count in (5, 7, 9):
        budgets = data['online_calls_by_camera_count'][str(count)]
        assert budgets['learned'] == {'A': 99 - count, 'AT': 98}
        assert budgets['CGLS'] == {'A': 99 - count, 'AT': 99 - count}
        assert budgets['UnscaledBP-Warm'] == {'A': 98 - count, 'AT': 98 - count}
        assert data['camera_count_comparisons'][str(count)]['CGLS']['all_four_worse'] == 33
    assert data['training_calls_each_implementation'] == {'A': 2914560, 'AT': 2914560}
    assert data['actual_audit_calls_each_implementation'] == {'A': 2970891, 'AT': 2969604}


def test_endpoint_notes_are_bilingual_and_private_safe():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    assert all(marker not in text for marker in ('/Users/', 'private_data/', 'private_results/', 'sha256', 'checkpoint'))
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        pairs = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) >= 6
        zh = ' '.join(p['data-i18n-zh'] for p in pairs)
        en = ' '.join(p['data-i18n-en'] for p in pairs)
        assert all(x in zh for x in ('不是完整序列', '不是有效加速', '不否定全部学习初值', '0/99', '98/99'))
        assert all(x in en for x in ('not full sequences', 'not effective acceleration', 'does not refute all learned initializers', '0/99', '98/99'))
        assert notes[0].select_one('a')['href'].endswith(SUMMARY)
    with Image.open(ROOT / 'assets/figures/poolfire_endpoint_world_set_20261006.png') as image:
        assert image.size == (1600, 500)
