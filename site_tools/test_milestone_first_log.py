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
