import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-query-weight-ablation-20261007'
SUMMARY = 'poolfire_query_weight_ablation_2026-10-07_public_summary.json'


def test_query_weight_ablation_is_not_a_rescued_algorithm():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'QUERY_WEIGHTS_UNIFORM_BENEFIT_IN_THIS_ABLATION'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 15
    assert data['original_all_four_no_worse_than_removed'] == 99
    assert data['original_CGLS_gain_cells'] == 97
    assert data['original_CGLS_gain_cells_retained_by_removed'] == 88
    assert set(data['basic_strata'].values()) == {33}
    assert set(data['strict_matches'].values()) == {0} and data['strict_total'] == 99
    assert data['new_fits'] == data['new_teacher_construction'] == 0
    assert data['original_model_recipe_closed']
    assert data['amplitude_and_refinement_still_use_current_observation']
    original = data['descriptive_controls']['KernelBank90-Warm33']
    assert original['all_four_harm'] == 99 and original['all_four_no_worse'] == 0
    assert 1.0059 < original['median_ratios']['field_rel_l2_gauge_centered'] < 1.0061
    assert 1.0326 < original['median_ratios']['observation_rel_l2'] < 1.0327
    plain = data['descriptive_controls']['CGLS35']
    assert (plain['all_four_no_worse'], plain['all_four_harm'], plain['mixed']) == (88, 2, 9)
    assert data['standalone_calls_original_and_removed'] == {'A': 35, 'AT': 34}
    assert data['new_actual_main_audit_calls_each_implementation'] == {'A': 6039, 'AT': 3465}
    assert data['new_permutation_calls_each_implementation'] == {'A': 1155, 'AT': 1122}
    for key in ('observation_blind', 'best_separately_fitted_global_prior_tested',
                'full_sequence_authorized', 'unique_failure_cause_proven',
                'nonlocality_necessity_proven', 'fresh_deployment_benchmark',
                'algorithm_breakthrough', 'paper_success', 'resource_speedup',
                'external_generalization', 'curved_ray_validated', 'real_bost',
                'neural_training_authorized', 'gpu_rental_authorized'):
        assert data[key] is False
    assert all(token not in text for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_query_weight_note_preserves_bilingual_limits():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        pairs = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) == 8
        zh = ' '.join(p['data-i18n-zh'] for p in pairs)
        en = ' '.join(p['data-i18n-en'] for p in pairs)
        assert all(term in zh for term in ('没有重训或新标签', '三帧训练哨兵', '不是完整序列', '88', '97', '0/99', '不能叫完全不读观测', '不是有效加速'))
        assert all(term in en for term in ('no new fits or labels', 'three-frame train sentinels', 'not full sequences', 'not observation-blind', 'not effective acceleration'))
        assert notes[0].select_one('a')['href'].endswith(SUMMARY)
        image = notes[0].select_one('img')
        assert image['data-i18n-alt-zh'] and image['data-i18n-alt-en']
        assert image['width'] == '1320' and image['height'] == '560'
        assert (ROOT / name).parent.joinpath(image['src']).resolve().is_file()


def test_direction_diagnostic_is_not_an_amplitude_tuned_solver_claim():
    data = json.loads((ROOT / 'docs' / SUMMARY).read_text())['direction_diagnostic']
    assert data['decision'] == 'QUERY_CONDITIONING_CHANGES_SEED_DIRECTION_IN_ALL_SENTINELS'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 11
    assert data['changed_seed_directions'] == data['cells'] == 99
    assert .08628 < data['seed_relative_noncollinearity']['median'] < .08629
    assert .02394 < data['seed_relative_noncollinearity']['min'] < .02395
    assert .16754 < data['seed_relative_noncollinearity']['max'] < .16755
    assert data['new_A'] == data['new_AT'] == data['new_fits'] == data['new_CFD_reads'] == 0
    assert data['all_amplitude_tuned_CGLS_paths_ruled_out'] is False
    assert data['finite_reference_distances_are_CFD_errors'] is False
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        note = soup.select_one('#' + MARKER + ' [data-direction-diagnostic]')
        assert note is not None
        assert '8.63%' in note['data-i18n-zh'] and '8.63%' in note['data-i18n-en']
        assert '0/99' in note['data-i18n-zh'] and '0/99' in note['data-i18n-en']
