import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-learned-dual-pcg-20261007'
SUMMARY = 'poolfire_learned_dual_pcg_2026-10-07_public_summary.json'


def test_iterative_learning_improvement_is_not_joint_accuracy():
    text = (ROOT / 'docs' / SUMMARY).read_text(); data = json.loads(text)
    assert data['decision'] == 'FAIL_FROZEN_TRAINED_DUAL_PCG30_WARM4_SENTINEL_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 35
    primary = 'TrainedDualPCG30-Warm4'
    controls = ('PlainDualCG30-Warm4', 'DiagonalDualPCG30-Warm4', 'GeometryDualPCG30-Warm4')
    assert data['matched_cells'][primary] == 0 and data['absolute_strata'][primary] == 33
    assert list(data['individual_matched'][primary].values()) == [71, 54, 56, 0]
    assert data['primary_first_three_joint_matches'] == 53
    assert list(data['primary_camera_individual_matched']['9'].values()) == [10, 0, 0, 0]
    for control in controls:
        assert data['matched_cells'][control] == 0
        assert data['post_seal_descriptive_primary_comparison'][control]['all_four_better'] == 99
    assert data['matched_cells']['CGLS128'] == data['matched_cells']['DirectRidge-Warm34'] == 99
    assert data['trajectories'] == 11 and data['frames_each'] == 3 and data['native_camera_counts'] == [5, 7, 9]
    assert data['standalone_each'] == {'A': 35, 'AT': 35}
    assert data['formal_main'] == {'A': 16335, 'AT': 15048}
    assert data['independent_main'] == {'A': 18018, 'AT': 15048}
    assert data['permutation_each_mode'] == {'A': 6600, 'AT': 6600}
    assert data['formal_query_small_solves_including_permutations'] == 15840
    assert data['formal_permutation_model_secant_audit_extra_small_solves'] == 3960
    assert data['new_fits'] == data['new_teacher_construction'] == data['held_teacher_reads'] == 0
    assert data['teacher_factors_and_training_nonfree'] and data['mixed_audit_wall_RSS_not_deployment']
    assert data['previous_one_shot_recipe_remains_closed'] and data['current_fixed_iterative_recipe_closed_without_tuning']
    assert max(data['maximum_permutation_relative'].values()) < 1.3e-13
    assert all(data[k] is False for k in ('neural_model', 'algorithm_breakthrough', 'paper_success', 'resource_speedup',
        'whole_sequence_evaluated', 'external_generalization', 'real_bost', 'main_goal_complete'))
    assert not any(token in text for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_four_bilingual_notes_preserve_observation_and_camera_deficits():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        nodes = BeautifulSoup((ROOT / name).read_text(), 'html.parser').select('#' + MARKER)
        assert len(nodes) == 1
        translations = nodes[0].select('[data-i18n-zh][data-i18n-en]'); assert len(translations) == 6
        for language in ('zh', 'en'):
            text = ' '.join(node['data-i18n-' + language] for node in translations)
            for token in ('35/35', '0/99', '33/33', '71/99', '54/99', '56/99', '0/33', '26.05%', '26.41%',
                '0.01051', '0.00309', '0.00913', '35A+35AT'):
                assert token in text
        assert '主目标仍未完成' in nodes[0].get_text()
        assert (ROOT / name).parent.joinpath(nodes[0].select_one('a')['href']).resolve() == ROOT / 'docs' / SUMMARY


def test_previous_scientific_records_are_not_replaced():
    data = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    new = data['latest_learned_dual_pcg']
    assert new['primary_individual_matched'] == [71, 54, 56, 0] and new['primary_matched_cells'] == 0
    assert new['primary_all_four_better_than_geometry_control'] == 99 and new['new_model_fits'] == 0
    assert data['latest_balanced_multisecant']['primary_matched_cells'] == 0
    assert data['latest_balanced_multisecant']['primary_all_four_better_than_geometry_control'] == 97
    assert data['latest_normal_krylov_budget']['observation_certified_infeasible'] == 99
    assert data['latest_direct_ridge_cost']['matched_cells_passed'] == 99


def test_log_discloses_tradeoff_without_goal_substitution():
    text = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert '35/35独立检查全真' in text and '前三项联合匹配53/99' in text
    assert '0.01051' in text and '九相机两项梯度均为0/33' in text
    assert '主目标仍未完成' in text and 'the whole goal remains active and unmet' in text
