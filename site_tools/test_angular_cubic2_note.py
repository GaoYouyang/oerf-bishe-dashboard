import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-angular-cubic2-20261006'
SUMMARY = 'poolfire_angular_cubic2_2026-10-06_public_summary.json'


def test_geometry_signal_is_not_classical_or_cost_success():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'FAIL_ANGULAR_TENSOR_CUBIC2_LOTO_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 20
    assert data['basic_absolute_strata_passed'] == data['basic_absolute_strata_total'] == 33
    assert data['strict_matches'] == data['geometry_erased_strict_matches'] == 0
    assert data['strict_total'] == 99
    assert data['reference_adequate_for_sentinels'] and data['recipe_closed']
    assert data['two_parameters_per_fold'] and data['small_linear_fits_each_implementation'] == 22
    plain = data['descriptive_controls']['CGLS']
    assert plain['all_four_no_worse'] == 0 and plain['all_four_harm'] == 99
    assert 1.0024 < plain['median_ratios']['field_rel_l2_gauge_centered'] < 1.0025
    assert 1.0210 < plain['median_ratios']['observation_rel_l2'] < 1.0211
    erased = data['descriptive_controls']['GeometryErasedCubic2-Warm']
    assert erased['all_four_no_worse'] == 63 and erased['all_four_harm'] == 31
    assert .99988 < erased['median_ratios']['field_rel_l2_gauge_centered'] < .99989
    assert .99866 < erased['median_ratios']['observation_rel_l2'] < .99868
    assert data['standalone_query_calls'] == {str(n): {'A': 99-n, 'AT': 98} for n in (5, 7, 9)}
    assert data['ordinary_CGLS_query_calls'] == {str(n): {'A': 99-n, 'AT': 99-n} for n in (5, 7, 9)}
    assert data['new_actual_main_audit_calls_each_implementation'] == {'A': 21681, 'AT': 18909}
    for key in ('neural_training', 'old_neural_comparator_refit',
                'held_truth_targets_or_statistics_in_fit_or_prediction',
                'full_sequence_authorized', 'all_geometry_learning_refuted',
                'unique_failure_cause_proven', 'fresh_deployment_benchmark',
                'algorithm_breakthrough', 'paper_success', 'resource_speedup',
                'external_generalization', 'curved_ray_validated', 'real_bost',
                'gpu_rental_authorized'):
        assert data[key] is False
    assert all(token not in text for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_geometry_note_has_bilingual_scope_and_readable_figure():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        pairs = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) == 7
        zh = ' '.join(p['data-i18n-zh'] for p in pairs)
        en = ' '.join(p['data-i18n-en'] for p in pairs)
        assert all(term in zh for term in ('三帧训练哨兵', '不是完整序列', '20/20', '0/99', '99/99', '不是有效加速', '不否定全部几何学习'))
        assert all(term in en for term in ('three-frame train sentinels', 'not full sequences', '20 independent checks', '0/99', 'not effective acceleration', 'does not refute all geometry-aware learning'))
        assert notes[0].select_one('a')['href'].endswith(SUMMARY)
        image = notes[0].select_one('img')
        assert image['width'] == '1320' and image['height'] == '560'
        assert image['data-i18n-alt-zh'] and image['data-i18n-alt-en']
        assert (ROOT / name).parent.joinpath(image['src']).resolve().is_file()
