import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-balanced-multisecant-20261007'
SUMMARY = 'poolfire_balanced_multisecant_2026-10-07_public_summary.json'


def test_structured_learning_is_not_matched_success():
    text = (ROOT / 'docs' / SUMMARY).read_text(); data = json.loads(text)
    assert data['decision'] == 'FAIL_TRAINED_BALANCED_MULTISECANT30_SENTINEL_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 31
    primary, control = 'TrainedSecant30-Warm34', 'GeometrySecant30-Warm34'
    assert data['matched_cells'][primary] == data['matched_cells'][control] == 0
    assert data['absolute_strata'][primary] == data['absolute_strata'][control] == 33
    assert list(data['individual_matched'][primary].values()) == [32, 19, 16, 0]
    assert data['post_seal_descriptive_primary_comparison'][control]['all_four_better'] == 97
    assert data['matched_cells']['CGLS128'] == data['matched_cells']['DirectRidge-Warm34'] == 99
    assert data['trajectories'] == 11 and data['frames_each'] == 3 and data['native_camera_counts'] == [5, 7, 9]
    assert data['training_responses_per_model'] == 30 and data['primary_models'] == 33
    assert data['standalone_each'] == {'A': 35, 'AT': 35}
    assert data['formal_main'] == {'A': 10491, 'AT': 9288}
    assert data['independent_main'] == {'A': 11778, 'AT': 9288}
    assert data['permutation_each_mode'] == {'A': 5280, 'AT': 4290}
    assert data['fit_held_response_reads'] == data['fit_CFD_truth_reads'] == data['new_teacher_construction'] == 0
    assert data['maximum_secant_defect'] < 1.1e-13
    assert data['teacher_factors_and_training_nonfree'] and data['cheap_improvement_is_not_matched_success']
    assert data['payload_is_not_total_RSS'] and data['fit_known_geometries_only']
    assert all(data[k] is False for k in ('neural_model', 'algorithm_breakthrough', 'paper_success', 'resource_speedup',
        'whole_sequence_evaluated', 'external_generalization', 'real_bost', 'main_goal_complete'))
    assert not any(token in text for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_four_bilingual_notes_keep_the_two_gates_separate():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        nodes = BeautifulSoup((ROOT / name).read_text(), 'html.parser').select('#' + MARKER)
        assert len(nodes) == 1
        translations = nodes[0].select('[data-i18n-zh][data-i18n-en]'); assert len(translations) == 6
        for language in ('zh', 'en'):
            text = ' '.join(node['data-i18n-' + language] for node in translations)
            for token in ('31/31', '0/99', '33/33', '97/99', '27.79%', '32.63%', '0.00913', '0.00309', '35A+35AT'):
                assert token in text
        assert '主目标仍未完成' in nodes[0].get_text()
        assert (ROOT / name).parent.joinpath(nodes[0].select_one('a')['href']).resolve() == ROOT / 'docs' / SUMMARY


def test_old_authority_records_survive():
    data = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    new = data['latest_balanced_multisecant']
    assert new['primary_matched_cells'] == 0 and new['primary_absolute_strata'] == 33
    assert new['primary_all_four_better_than_geometry_control'] == 97 and new['structured_data_fitted_inverse']
    assert not new['neural_model']
    assert data['latest_normal_krylov_budget']['observation_certified_infeasible'] == 99
    assert data['latest_direct_ridge_cost']['matched_cells_passed'] == 99
    assert data['latest_neural_sparse_dual_factor']['primary_matched_cells'] == 0


def test_log_explains_useful_information_without_goal_substitution():
    text = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert '31/31独立检查全真' in text and '97/99样本' in text
    assert '0.00913' in text and '主目标仍未完成' in text
    assert 'useful directional information, not matched success' in text
    assert 'whole goal remains active and unmet' in text
