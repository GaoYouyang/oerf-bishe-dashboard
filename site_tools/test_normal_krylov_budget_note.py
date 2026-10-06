import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-normal-krylov-budget-20261007'
SUMMARY = 'poolfire_normal_krylov_budget_2026-10-07_public_summary.json'


def test_necessary_capacity_is_not_an_algorithm_or_all_calls_impossibility():
    text = (ROOT / 'docs' / SUMMARY).read_text(); data = json.loads(text)
    assert data['decision'] == 'FAIL_COMPLETE_NORMAL_KRYLOV35_OBSERVATION_CAPACITY'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 23
    assert data['cells'] == data['observation_certified_infeasible'] == 99
    assert data['observation_feasible_witnesses'] == data['observation_unresolved'] == 0
    assert data['trajectories'] == 11 and data['frames_each'] == 3 and data['native_camera_counts'] == [5, 7, 9]
    assert data['minimum_to_reference_ratio']['min'] > 3.47
    assert max(abs(v) for v in data['oracle_improvement_over_CGLS35'].values()) < 5e-16
    assert data['scope_is_not_any35_call_algorithm'] and data['outside_energy_not_useful_direction_evidence']
    for group in data['camera_strata'].values():
        assert group['OBSERVATION_INFEASIBLE_WITHIN_VALIDATED_KRYLOV35'] == 33
    assert all(data[k] == 0 for k in ('trainable_parameters', 'new_CFD_reads', 'new_model_fits', 'new_teacher_construction'))
    assert data['standalone_basis'] == {'A': 35, 'AT': 35} and data['extra_replay_A'] == 2
    assert data['each_mode_main'] == {'A': 3663, 'AT': 3465}
    assert data['each_mode_permutation'] == {'A': 1155, 'AT': 1155}
    assert data['basis_solve_and_storage_nonfree'] and data['new_query_states_sealed_before_parent_arrays']
    assert data['seed_outside_fraction']['NeuralDualFactor58-Warm34']['median'] > data['seed_outside_fraction']['DirectRidge-Warm34']['median']
    assert all(data[k] is False for k in ('learned_initializer', 'algorithm_breakthrough', 'paper_success', 'resource_speedup',
        'full_four_metric_feasibility', 'whole_sequence_evaluated', 'external_generalization', 'real_bost', 'main_goal_complete'))
    assert not any(token in text for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_all_four_notes_have_paired_scope_and_numbers():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        nodes = BeautifulSoup((ROOT / name).read_text(), 'html.parser').select('#' + MARKER)
        assert len(nodes) == 1
        translations = nodes[0].select('[data-i18n-zh][data-i18n-en]'); assert len(translations) == 6
        for language in ('zh', 'en'):
            text = ' '.join(node['data-i18n-' + language] for node in translations)
            for token in ('23/23', '0/99', '99/99', '4.31', '19.16%', '50.22%', '22.85%', '35A+35AT'):
                assert token in text
        assert '不是所有35次调用算法' in nodes[0].get_text()
        assert '主目标仍未完成' in nodes[0].get_text()
        assert (ROOT / name).parent.joinpath(nodes[0].select_one('a')['href']).resolve() == ROOT / 'docs' / SUMMARY


def test_old_classic_and_learned_decisions_are_not_overwritten():
    data = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    new = data['latest_normal_krylov_budget']
    assert new['observation_feasible'] == 0 and new['observation_certified_infeasible'] == 99
    assert new['not_any35_call_impossibility'] and not new['full_four_metric_result']
    assert data['latest_direct_ridge_cost']['matched_cells_passed'] == 99
    assert data['latest_neural_sparse_dual_factor']['primary_matched_cells'] == 0
    assert data['latest_inverse_action_span']['seed_identity_certified_infeasible'] == 99


def test_log_records_cost_and_one_way_implication():
    text = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert '23/23独立检查全真' in text and '不是所有35次调用' in text
    assert 'not arbitrary 35-call algorithms' in text and 'necessary observation gate' in text
    assert 'whole goal remains active and unmet' in text
