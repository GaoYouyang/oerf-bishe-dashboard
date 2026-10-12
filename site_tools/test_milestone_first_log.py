import json
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('operator-learning/daily-progress.html', 'learning_log.html')


def test_milestones_visible_and_history_preserved():
    for name in PAGES:
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        digest = soup.select_one('main > #milestones')
        archive = soup.select_one('main > details#experiment-history')
        assert digest and archive and not archive.has_attr('open')
        assert digest.find_previous_sibling() is None
        assert not digest.find_parent('details')
        assert archive.select_one('#conditional-null-prior')
        assert archive.select_one('#compact-conditional-prior')
        assert archive.select_one('#frozen-neural-hybrid')
        for node in digest.select('[data-i18n-zh]'):
            assert node.get_text() == node['data-i18n-zh']
            assert node.get('data-i18n-en')
        text = digest.get_text()
        assert all(token in text for token in ('27.36%', '17.06%', '3333/3333', '33/33', '0.244/0.247', '1.73'))
        assert '真实BOST' in text and '经典' in text
        assert soup.select_one('script[src$="assets/site-language.js"]')


def test_new_digest_links_resolve_locally():
    for name in PAGES:
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        for anchor in soup.select('#milestones a[href]'):
            url = urlsplit(anchor['href'])
            path = (ROOT / name).parent / unquote(url.path)
            assert path.resolve().is_file()
            doc = parse_qs(url.query).get('doc')
            if doc:
                assert (ROOT / doc[0]).is_file()


def test_digest_values_are_existing_verified_evidence():
    evidence = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    prior = evidence['conditional_null_prior']
    assert prior['primary']['matched_queries'] == 1111
    assert prior['primary']['matched_trajectories'] == 11
    assert prior['complete_held_trajectory_excluded_from_each_model']
    assert prior['native_camera_counts'] == [5]
    assert not any(prior['claims'].values())
    for control, gain in (('QualifiedRidge-Warm4', 27.36), ('MeanNull-RidgeWarm4', 17.06)):
        ratio = prior['paired_control_comparison'][control]['median_field_error_ratio']
        assert round(100 * (1 - ratio), 2) == gain
    classic = evidence['analytic_cholesky_cardinality']
    assert classic['matched_queries'] == 3333 and classic['matched_strata'] == 33
    assert classic['native_camera_counts'] == [5, 7, 9]


def test_markdown_leads_with_scoped_milestones_without_deleting_history():
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert log.index('## 阶段摘要') < log.index('## 完整研究归档')
    assert '27.36%' in log and '17.06%' in log
    assert '峰值RSS仍约1.73倍' in log
    assert 'joint resource gate still fails' in log
    assert '固定压缩配方' in log
    assert '## 2026-10-10: 完整留轨迹图网络' in log
    assert '0/3333' in log


def test_source_information_signal_has_its_own_scope_and_keeps_counterevidence():
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    digest, archive = log.split('## 完整研究归档', 1)
    evidence = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    audit = evidence['conditional_information_control_audit']
    assert round(100 * (1 - audit['rbf_mean_objective_ratio']), 2) == 39.30
    assert audit['inner_source_queries'] == 1010 and audit['fit_sources_per_fold'] == 909
    assert audit['rbf_nonharm_source_groups'] == audit['source_groups'] == 10
    assert audit['reference_eligible'] and not audit['primary_eligible']
    assert audit['decision'] == 'FAIL_FIXED_AFFINE_CONDITIONAL_INFORMATION'
    assert audit['first_outer_held_reads'] == audit['new_neural_updates'] == 0
    assert not audit['physical_endpoint_scored'] and not any(audit['claims'].values())
    assert '39.30%' in digest and 'inner-source' in digest
    assert '不是新的神经重建' in digest and 'not new neural reconstruction' in digest
    assert '34.29%' in archive and '56%–58%' in archive
    assert '3.206' in archive and '不证明非线性是唯一原因' in archive


def test_effective_graph_source_fit_preserves_endpoint_and_strong_control_limits():
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    digest, archive = log.split('## 完整研究归档', 1)
    evidence = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    fit = evidence['physical_graph_source_fit_audit']
    assert fit['necessary_source_fit_pass']
    assert fit['sources'] == 1010 and fit['mean_and_zero_joint_nonharm_groups'] == 10
    assert round(100*(1-fit['neural_mean_joint_ratio']), 2) == 55.58
    assert round(100*(1-fit['neural_zero_joint_ratio']), 2) == 70.00
    assert fit['neural_RBF_source_joint_ratio'] > 1 and fit['neural_mean_source_observation_MSE_ratio'] > 1
    assert fit['RBF_control_in_sample_here'] and fit['source_fit_is_not_four_error_matching']
    assert not fit['CGLS_endpoints_evaluated'] and not fit['complete_outer_trajectory_evaluated']
    assert fit['first_outer_reads'] == 0 and not any(fit['claims'].values())
    assert '55.58%' in digest and '70.00%' in digest
    assert '未超过强条件核' in digest and 'not complete-held reconstruction' in digest
    assert '2.01910' in archive and '1.99882' in archive


def test_complete_held_graph_gain_keeps_strict_observation_and_cost_boundary():
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    digest, archive = log.split('## 完整研究归档', 1)
    evidence = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    held = evidence['physical_graph_first_outer']
    assert held['decision'] == 'FAIL_PHYSICAL_GRAPH_FIRST_OUTER_QUALITY'
    assert held['queries'] == held['four_error_reference_matched'] == 101
    assert held['complete_held_trajectories'] == 1 and held['new_train_updates'] == 0
    assert held['source_statistics_train_only'] and held['reference_adequate']
    assert held['RBF_spatial_nonharm_each_metric'] == 101
    for metric, gain in (('field',35.99),('full_gradient',36.52),('interior_gradient',32.98)):
        assert round(100*(1-held['RBF_median_ratios'][metric]),2) == gain
        assert str(gain) + '%' in digest
    assert held['RBF_joint_nonharm'] == 96 and held['RBF_observation_harm_count'] == 5
    assert held['raw_corrected_mean_joint_nonharm'] == 0 and held['raw_corrected_mean_observation_harm_count'] == 101
    assert held['raw_corrected_mean_observation_median_ratio'] > 1
    assert held['standalone_neural_actions']['A'] > held['standalone_cached_control_actions']['A']
    assert held['literal_recipe_closed'] and not held['further_folds_authorized']
    assert held['independent_checks_pass'] == 31 and held['terminal_summary_pass']
    assert not any(held['claims'].values())
    assert '严格联合门仍未通过' in digest and 'strict joint gate still fails' in digest
    assert '1.461007' in archive and '7.70995e-6' in archive
    assert '源拟合本身不是完整留出重建' in digest


def test_local_joint_signal_keeps_reference_first_and_source_only_scope():
    evidence = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    value = evidence['local_conditional_statistical_signal']
    assert value['decision'] == 'INCONCLUSIVE_PATCH_SOURCE_CONTROL'
    assert value['source_queries'] == 1010 and value['source_groups'] == 10
    assert value['fit_fields_per_fold'] == 909 and value['query_fields_per_fold'] == 101
    assert value['query_excluded_before_sampling_whitening_fit']
    assert value['query_truth_time_labels_not_predictor_inputs']
    assert not value['retained_source_bank_at_inference']
    for name,gain in (('mean',51.84),('single_gaussian',64.33),('RBF',35.91)):
        assert round(100*(1-value['primary_joint_ratios'][name]),2) == gain
    assert round(100*(1-value['primary_mean_observation_MSE_ratio']),2) == 84.05
    assert value['groups_meeting_candidate_numeric_predicates'] == 10
    assert not value['reference_eligible'] and not value['primary_eligible']
    assert not value['numeric_predicates_override_reference_first']
    assert value['reference_failed_groups'] == 1
    assert value['reference_failed_group_joint_RBF_mean'][0] > value['reference_failed_group_joint_RBF_mean'][1]
    assert value['joint_nonharm_frames']['RBF'] == 997
    assert value['observation_nonharm_frames_each_control'] == 1010
    assert value['independent_reconstruction_checks'] == 33 and value['terminal_readback_checks'] == 48
    assert value['first_outer_reads'] == value['new_neural_updates'] == 0
    assert not value['first_outer_authorized'] and not value['CGLS_endpoints_evaluated']
    assert value['shared_EM_fit_not_independent_optimizer'] and not any(value['claims'].values())
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    digest,archive = log.split('## 完整研究归档',1)
    assert all(token in digest for token in ('51.84%','64.33%','35.91%','84.05%','997/1010'))
    assert 'authority remains inconclusive' in digest.lower()
    assert 'source-LOTO' in digest and '最终CGLS' in digest
    assert 'INCONCLUSIVE_PATCH_SOURCE_CONTROL' in archive
    assert 'EM fitting is shared' in archive and '11120A+2030AT' in archive


def test_followup_likelihood_does_not_overwrite_statistical_signal_authority():
    evidence = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    parent = evidence['local_conditional_statistical_signal']
    follow = parent['consolidated_neural_followup']
    assert parent['decision'] == 'INCONCLUSIVE_PATCH_SOURCE_CONTROL'
    assert follow['decision'] == 'FAIL_GEOMETRY_PATCH_CONDITIONAL_DENSITY_SOURCE_GATE'
    assert follow['actual_training_and_independent_physics'] and follow['source_likelihood_pass']
    assert follow['source_loto_all_gate_groups'] == 0
    assert all(ratio > 1 for ratio in follow['joint_mean_RBF_GMM_ratios'])
    assert follow['training_conditional_mean_MSE_primary_range'][0] > follow['training_conditional_mean_MSE_GMM_range'][1]
    assert follow['literal_recipe_closed'] and not follow['outer_authorized']
    assert follow['training_mean_audit_post_open_not_new_gate']
    assert follow['prior_GMM_inconclusive_and_numeric_gains_preserved']
    assert follow['likelihood_fit_not_field_value'] and not follow['new_breakthrough_claim']
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert '后续神经检验合并边界' in log and 'Consolidated neural follow-up' in log


def test_structured_pilot_spatial_gain_is_not_a_complete_four_metric_result():
    evidence = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    parent = evidence['local_conditional_statistical_signal']
    pilot = parent['structured_covariance_pilot']
    assert parent['decision'] == 'INCONCLUSIVE_PATCH_SOURCE_CONTROL'
    assert pilot['decision'] == 'FAIL_CONDITIONAL_COVARIANCE_CONSENSUS_PILOT'
    assert pilot['independent_valid'] and pilot['independent_checks'] == 41
    assert pilot['queries'] == 30 and pilot['snapshots_per_source_group'] == 3
    assert pilot['training_fields_per_source_fold'] == 909 and pilot['full_query_group_excluded_each_fit']
    assert pilot['native_known_cameras'] == 5
    assert all(value < 1 for value in pilot['joint_GMM_diagonal_ratios'])
    assert pilot['observation_GMM_ratio'] > 1 and pilot['four_metric_GMM_diagonal_counts'] == [5, 27]
    assert pilot['spatial_three_metric_GMM_nonharm_count'] == 30
    assert pilot['literal_recipe_closed'] and pilot['old_GMM_authority_unchanged']
    assert not pilot['complete_trajectory_result'] and not pilot['new_neural_training']
    assert not pilot['learned_or_old_GMM_outer_authorized'] and not pilot['new_breakthrough_claim']
    assert pilot['full_setup_and_local_solve_nonfree']
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert '结构化跟进合并边界' in log and 'Consolidated structured follow-up' in log


def test_retrieval_training_mean_is_not_loto_or_physical_breakthrough():
    evidence = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    parent = evidence['local_conditional_statistical_signal']
    gate = parent['consolidated_retrieval_mean_gate']
    assert parent['decision'] == 'INCONCLUSIVE_PATCH_SOURCE_CONTROL'
    assert gate['decision'] == 'FAIL_LOCAL_RETRIEVAL_SOURCE_MEAN_GATE'
    assert gate['independent_valid'] and gate['independent_checks'] == 30
    assert gate['training_fields'] == 909 and gate['fixed_source_patches'] == 29088
    assert gate['parameters'] == 1312 and gate['actual_updates'] == 512
    assert gate['source_in_sample_only'] and gate['neural_weights_fit_all_nine_source_training_groups']
    assert gate['entire_query_trajectory_excluded_from_retrieval_bank']
    assert gate['neural_distance_uniform_GMM_ratios'][0] < .95
    assert gate['neural_distance_uniform_GMM_ratios'][1] < .95
    assert gate['neural_distance_uniform_GMM_ratios'][2] > 1.05
    assert gate['source_groups_passing_strong_control_fidelity'] == 0
    assert gate['literal_recipe_closed'] and gate['old_GMM_authority_unchanged']
    assert gate['shared_weights_not_independent_optimizer'] and gate['source_value_bank_and_search_nonfree']
    assert not gate['complete_trajectory_prediction'] and not gate['native_four_metric_result']
    assert not gate['physical_or_outer_expansion_authorized'] and not gate['new_breakthrough_claim']
    capacity = gate['post_open_capacity_followup']
    assert capacity['decision'] == 'FAIL_PILOT_POSITIVE_VALUE_SUPPORT'
    assert capacity['queries'] == 72 and capacity['source_groups'] == 9
    assert capacity['independent_valid'] and capacity['independent_checks'] == 22
    assert capacity['target_visible_diagnostic_only']
    assert capacity['aggregate_convex_MSE_interval'][1] < capacity['same_pilot_GMM_MSE']
    assert capacity['adequate_groups'] == 7 and capacity['insufficient_groups'] == 2
    assert capacity['signed_capacity_is_not_predictability_or_stability']
    assert not capacity['scorer_retraining_or_physical_outer_authorized'] and not capacity['new_breakthrough_claim']
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    digest, archive = log.split('## 完整研究归档', 1)
    assert '检索均值小门合并边界' in archive and 'Consolidated retrieval-mean boundary' in archive
    assert '1.88757' in archive and 'retrieval-mean' not in digest
