import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def test_signed_ray_pair_public_numbers_and_boundaries():
    path = ROOT/'docs/poolfire_signed_ray_pair_2026-10-06_public_summary.json'
    s = json.loads(path.read_text())
    assert s['independent_checks_passed'] == s['independent_checks_total'] == 26
    assert s['shared_trainable_parameter_count'] == 3
    assert s['strict_four_metric_matched_cells'] == 0 and s['cells_total'] == 99
    assert s['basic_absolute_strata_passed'] == s['basic_absolute_strata_total'] == 33
    assert s['same_budget_plain_cgls_comparison']['all_four_no_worse_cells'] == 28
    assert s['same_budget_plain_cgls_comparison']['all_four_harm_cells'] == 50
    assert s['same_budget_linear_graph_comparison']['all_four_no_worse_cells'] == 40
    assert s['same_budget_linear_graph_comparison']['all_four_harm_cells'] == 54
    assert s['primary_standalone_calls'] == {'A':99, 'AT':99}
    assert 'shared frozen' in s['independence_boundary']
    for key in ('algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'):
        assert s[key] is False
    for marker in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'theta'):
        assert marker not in path.read_text()


def test_signed_ray_pair_four_bilingual_pages():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        sections = soup.select('#poolfire-signed-ray-pair-section-20261006')
        assert len(sections) == 1
        pairs = sections[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) >= 4 and all(n['data-i18n-zh'] and n['data-i18n-en'] for n in pairs)
        for text in ('0/99', '28/99', '50/99', '40/99', '54/99'):
            assert text in sections[0].get_text()
        assert sections[0].find('a')['href'].endswith('poolfire_signed_ray_pair_2026-10-06_public_summary.json')
