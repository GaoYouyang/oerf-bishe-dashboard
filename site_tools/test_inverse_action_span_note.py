import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-inverse-action-span-20261007'
SUMMARY = 'poolfire_inverse_action_span_2026-10-07_public_summary.json'


def test_projection_certificates_are_not_final_accuracy_or_learning():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'FAIL_DIRECT_INVERSE_TRAIN_SPAN_SEED_CAPACITY'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 16
    assert data['cells'] == 99 and data['trajectories'] == 11 and data['frames_each'] == 3
    assert data['held_target_visible_to_offline_oracle']
    assert data['whole_held_trajectory_excluded_from_directions']
    assert data['seed_reproduction_limit'] == .01
    assert data['same_count_training_directions'] == 30
    for arm in ('DirectSeed30', 'BP30', 'FiniteTeacher30'):
        assert data['aggregate'][arm]['CERTIFIED_INFEASIBLE'] == 99
        assert data['aggregate'][arm]['CERTIFIED_FEASIBLE'] == 0
        assert data['aggregate'][arm]['UNRESOLVED_TRADEOFF'] == 0
        for count in ('5', '7', '9'):
            assert data['by_camera'][count][arm]['CERTIFIED_INFEASIBLE'] == 33
    assert data['unchanged_parent_matched_cells'] == 99
    assert all(data[name] == 0 for name in ('new_A', 'new_AT', 'new_fits', 'new_CFD_reads', 'new_teacher_construction'))
    assert all(data[name] is False for name in ('fresh_physical_replay', 'CFD_matched_accuracy_evaluated',
        'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'full_sequence_authorized',
        'external_generalization', 'real_bost', 'neural_training_authorized', 'main_goal_complete'))
    assert data['does_not_refute_other_initializers_or_subsequent_refinement']
    assert all(token not in text for token in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_notes_keep_offline_and_seed_identity_boundaries_bilingually():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        nodes = BeautifulSoup((ROOT / name).read_text(), 'html.parser').select('#' + MARKER)
        assert len(nodes) == 1
        translated = nodes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(translated) == 6
        zh = ' '.join(node['data-i18n-zh'] for node in translated)
        en = ' '.join(node['data-i18n-en'] for node in translated)
        for value in ('16/16', '1%', '41.16%', '55.92%', '60.21%', '45.95%', '0A+0AT', '99/99'):
            assert value in zh and value in en
        assert '不是部署预测器' in zh and 'not a deployment predictor' in en
        assert '初值重现门不是CFD最终误差门' in zh and 'seed identity is not the final CFD-error gate' in en
        assert '不扩充这套库' in zh and 'without enlarging this bank' in en
        assert '真实BOST仍未证明' in zh and 'real BOST remain unproved' in en
        assert (ROOT / name).parent.joinpath(nodes[0].select_one('a')['href']).resolve() == ROOT / 'docs' / SUMMARY


def test_current_record_does_not_replace_qualified_reconstruction_counts():
    data = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    audit = data['latest_inverse_action_span']
    assert audit['seed_identity_feasible'] == 0 and audit['seed_identity_certified_infeasible'] == 99
    assert audit['offline_target_visible_only'] and audit['not_a_CFD_matched_accuracy_result']
    assert data['latest_direct_ridge_cost']['matched_cells_passed'] == 99
    assert data['latest_direct_ridge_initializer']['matched_cells_passed'] == 99


def test_plain_language_log_contains_both_scope_boundaries():
    text = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    recent = text.split('## 2026-10-07：同精度经典初值更慢且内存更高', 1)[0]
    assert '99/99有不可行下界证书' in recent and 'one-percent seed reproduction' in recent
    assert '经典最终重建99/99同精度' in recent and 'capacity evidence, not deployment prediction' in recent
