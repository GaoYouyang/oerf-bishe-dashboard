import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-neural-sparse-dual-factor-20261007'
SUMMARY = 'poolfire_neural_sparse_dual_factor_2026-10-07_public_summary.json'


def test_actual_learning_does_not_imply_matched_accuracy_or_resources():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'FAIL_NEURAL_SPARSE_DUAL_FACTOR58_SENTINEL_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 30
    assert data['cells'] == 99 and data['trajectories'] == 11 and data['frames_each'] == 3
    assert data['trainable_parameters'] == 58 and data['two_independent_full_fits']
    for method in ('NeuralDualFactor58-Warm34', 'DualDiagonal-Warm34'):
        assert data['matched_cells'][method] == 0 and data['absolute_strata'][method] == 33
        assert all(v == 0 for v in data['individual_matched'][method].values())
    for method in ('CGLS128', 'DirectRidge-Warm34', 'NormalFactor-PCGLS34'):
        assert data['matched_cells'][method] == 99 and data['absolute_strata'][method] == 33
    assert data['factor_and_transpose_total_bytes'] == 18047824
    assert sum(data['factor_and_transpose_payload_bytes'].values()) == data['factor_and_transpose_total_bytes']
    assert data['factor_payload_is_not_total_RSS'] and data['fit_is_transductive_on_known_geometry']
    assert all(data[k] == 0 for k in ('fit_observation_reads', 'fit_teacher_field_reads', 'fit_CFD_truth_reads'))
    assert data['standalone_each'] == {'A': 35, 'AT': 35} and data['geometry_and_training_costs_nonfree']
    assert data['operator_loss_not_reconstruction_success']
    assert data['first_and_last_preupdate_fit_loss'][0] > data['first_and_last_preupdate_fit_loss'][1]
    for m in data['median_final_errors']['NeuralDualFactor58-Warm34']:
        assert data['median_final_errors']['NeuralDualFactor58-Warm34'][m] > data['median_final_errors']['DualDiagonal-Warm34'][m]
    assert all(data[k] is False for k in ('algorithm_breakthrough', 'paper_success', 'resource_speedup',
        'full_sequence_authorized', 'whole_sequence_evaluated', 'external_generalization', 'real_bost', 'main_goal_complete'))
    assert all(token not in text for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_notes_distinguish_absolute_and_strict_gates_in_both_languages():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        nodes = BeautifulSoup((ROOT / name).read_text(), 'html.parser').select('#' + MARKER)
        assert len(nodes) == 1
        translations = nodes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(translations) == 6
        for language in ('zh', 'en'):
            text = ' '.join(node['data-i18n-' + language] for node in translations)
            for token in ('30/30', '0/99', '33/33', '99/99', '17.21MiB', '35A+35AT', '33.13%', '32.63%', '26.41%'):
                assert token in text
        assert '主目标仍未完成' in nodes[0].get_text()
        assert (ROOT / name).parent.joinpath(nodes[0].select_one('a')['href']).resolve() == ROOT / 'docs' / SUMMARY


def test_new_failure_does_not_overwrite_classical_or_previous_seed_capacity():
    data = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    new = data['latest_neural_sparse_dual_factor']
    assert new['primary_matched_cells'] == 0 and new['primary_absolute_strata'] == 33
    assert new['parameters'] == 58 and new['known_geometry_operator_fit_only']
    assert data['latest_direct_ridge_cost']['matched_cells_passed'] == 99
    assert data['latest_inverse_action_span']['seed_identity_certified_infeasible'] == 99


def test_log_keeps_known_geometry_and_total_cost_boundaries():
    text = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert '30/30独立检查通过' in text and '严格同精度却为0/99' in text
    assert 'transductive known-geometry learning' in text and 'not total RSS or a speedup' in text
    assert 'whole goal remains active and unmet' in text
