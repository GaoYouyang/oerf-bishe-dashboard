import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = 'poolfire_direct_ridge_learning_2026-10-07_public_summary.json'
MARKER = 'poolfire-direct-ridge-learning-20261007'


def test_ridge_accuracy_pass_is_not_distinct_learned_success():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'NO_DISTINCT_LEARNED_GAIN_DIRECT_RIDGE_CONTROL'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 117
    for name in ('DirectRidgeDual2-Warm33', 'DirectRidge-Warm34', 'NormalFactor-PCGLS34'):
        assert data['strict_matches'][name] == 99 and data['basic_strata'][name] == 33
    assert data['reference_adequate'] and all(value == 99 for value in data['primary_individual_matches'].values())
    assert data['parameters_per_fold'] == 2 and data['unlearned_control_parameters'] == 0
    assert data['fit_cells_per_fold'] == 90 and data['held_teacher_reads'] == data['candidate_generation_CFD_truth_reads'] == 0
    assert data['standalone_calls_each_arm'] == {'A': 35, 'AT': 35, 'factor_solves': 1}
    assert data['full_normal_PCGLS_control_calls'] == {'A': 34, 'AT': 34, 'factor_solves': 34}
    assert data['geometry_setup_and_factor_solves_nonfree'] and data['unlearned_control_retained']
    control = data['descriptive_controls']['DirectRidge-Warm34']
    assert control['all_four_harm'] == 68 and control['all_four_no_worse'] == 0
    assert all(value > 1 for value in control['median_ratios'].values())
    assert data['fixed_learned_gain_claim_closed']
    for name in ('distinct_learned_eligibility', 'broader_C_route_disproved', 'full_sequence_authorized',
                 'fresh_deployment_benchmark', 'algorithm_breakthrough', 'paper_success', 'resource_speedup',
                 'external_generalization', 'curved_ray_validated', 'real_bost', 'gpu_rental_authorized', 'main_goal_complete'):
        assert data[name] is False
    assert all(term not in text for term in ('/Users/', 'private_results/', 'sha256', 'checkpoint', 'theta'))


def test_ridge_result_notes_are_bilingual_and_privacy_safe():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        nodes = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(nodes) == 7
        zh = ' '.join(node['data-i18n-zh'] for node in nodes)
        en = ' '.join(node['data-i18n-en'] for node in nodes)
        assert all(term in zh for term in ('99/99', '33/33', '117/117', '68/99', '主目标仍未完成', '没有速度或内存优势'))
        assert all(term in en for term in ('99/99', '33/33', '117/117', '68/99', 'main goal remains unmet', 'no speed or memory advantage'))
        base = (ROOT / name).parent
        assert base.joinpath(notes[0].select_one('a')['href']).resolve() == ROOT / 'docs' / SUMMARY
        image = notes[0].select_one('img')
        for lang in ('zh', 'en'):
            assert base.joinpath(image['data-i18n-src-' + lang]).resolve().is_file()
            assert image['data-i18n-alt-' + lang]


def test_current_headline_prioritizes_new_ridge_result():
    current = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())['latest_direct_ridge_initializer']
    assert current['cells'] == current['matched_cells_passed'] == 99
    assert current['strata'] == current['basic_strata_passed'] == 33
    assert not current['algorithm_breakthrough'] and MARKER in current['note']
    for name in ('index.html', 'operator-learning/index.html'):
        assert 'evidence.latest_direct_ridge_initializer || evidence.latest_native_classical_control' in (ROOT / name).read_text()
