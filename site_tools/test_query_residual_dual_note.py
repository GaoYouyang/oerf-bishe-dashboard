import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-query-residual-dual-20261007'
SUMMARY = 'poolfire_query_residual_dual_2026-10-07_public_summary.json'


def test_actual_small_neural_fit_is_not_algorithm_success():
    text = (ROOT/'docs'/SUMMARY).read_text(); data = json.loads(text); primary = 'QueryResidualDual28-Warm33'
    assert data['decision'] == 'FAIL_QUERY_RESIDUAL_DUAL28_SENTINEL_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 38
    assert data['learned_weights_per_model'] == 28 and data['train_only_scale_statistics_per_model'] == 1
    assert data['outer_models'] == 11 and data['fit_audit_including_reverse_refit'] == 12
    assert data['training_cells_each_fold'] == 90 and data['actual_neural_fit']
    assert data['matched_cells'][primary] == 0 and data['absolute_strata'][primary] == 33
    assert list(data['individual_matched'][primary].values()) == [0,0,0,0]
    assert data['matched_cells']['ResidualJacobi-Warm33'] == 0
    assert data['matched_cells']['CGLS128'] == data['matched_cells']['DirectRidge-Warm34'] == 99
    assert data['post_seal_descriptive_primary_comparison']['DualDiagonal-Warm34']['all_four_better'] == 94
    assert data['post_seal_descriptive_primary_comparison']['TrainedDualPCG30-Warm4']['all_four_worse'] == 92
    assert data['training_loss']['final_median'] < data['training_loss']['initial_median']
    assert data['standalone_each'] == {'A':35,'AT':35}
    assert data['formal_main'] == {'A':8811,'AT':7128} and data['independent_main'] == {'A':10692,'AT':7128}
    assert data['permutation_each_mode'] == {'A':2277,'AT':2277}
    assert data['held_teacher_reads'] == data['new_teacher_construction'] == 0
    assert data['teacher_training_and_cache_nonfree'] and data['mixed_audit_wall_RSS_not_deployment']
    assert all(data[k] is False for k in ('algorithm_breakthrough','paper_success','resource_speedup',
        'whole_sequence_evaluated','external_generalization','real_bost','main_goal_complete'))
    assert not any(token in text for token in ('/Users/','private_results/','private_data/','sha256','checkpoint','theta'))


def test_bilingual_notes_disclose_the_small_increment_and_joint_failure():
    for name in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html','learning_log.html'):
        nodes = BeautifulSoup((ROOT/name).read_text(),'html.parser').select('#'+MARKER); assert len(nodes) == 1
        translations = nodes[0].select('[data-i18n-zh][data-i18n-en]'); assert len(translations) == 6
        for language in ('zh','en'):
            text = ' '.join(n['data-i18n-'+language] for n in translations)
            for token in ('38/38','33/33','0/99','94/99','92/99','32.59%','26.41%','0.01355','0.00309','0.99258','35A+35AT'):
                assert token in text
        assert '主目标仍未完成' in nodes[0].get_text()
        assert (ROOT/name).parent.joinpath(nodes[0].select_one('a')['href']).resolve() == ROOT/'docs'/SUMMARY


def test_prior_evidence_and_inconclusive_status_are_preserved():
    data = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    added = data['latest_query_residual_dual']
    assert added['actual_neural_model'] and added['learned_weights'] == 28
    assert added['primary_matched_cells'] == 0 and added['primary_individual_matched'] == [0,0,0,0]
    assert data['latest_learned_dual_pcg']['primary_individual_matched'] == [71,54,56,0]
    assert data['latest_direct_ridge_cost']['matched_cells_passed'] == 99
    public = json.loads((ROOT/'docs'/SUMMARY).read_text())
    assert public['old_recipes_and_inconclusive_attribution_preserved']


def test_plain_language_log_does_not_claim_causal_proof_or_goal_completion():
    text = (ROOT/'docs/operator_3d_learning_log.md').read_text()
    assert '38/38独立核验全真' in text and '当前证据只否定这套具体配方' in text
    assert '92/99' in text and '0.01355' in text and '0.99258' in text
    assert 'not a causal nonlocality' in text and 'the whole goal remains active and unmet' in text
