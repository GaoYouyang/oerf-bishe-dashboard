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
