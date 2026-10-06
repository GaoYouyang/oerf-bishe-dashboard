import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def test_modal_response_public_numbers_and_boundaries():
    path = ROOT / 'docs/poolfire_modal_response_2026-10-06_public_summary.json'
    summary = json.loads(path.read_text())
    assert summary['independent_checks_passed'] == summary['independent_checks_total'] == 20
    assert summary['strict_four_metric_matched_cells'] == 0
    assert summary['same_budget_plain_cgls_all_four_better_cells'] == summary['cells_total'] == 99
    assert summary['basic_absolute_strata_passed'] == summary['basic_absolute_strata_total'] == 33
    assert summary['primary_standalone_calls'] == {'A': 99, 'AT': 99}
    attribution = summary['post_closed_tail_attribution']
    assert 0.00076 < attribution['field_captured_energy_percent_p50'] < 0.00078
    assert 1.01 < attribution['image_captured_energy_percent_p50'] < 1.02
    assert 'not total CFD' in attribution['denominator']
    for name in ('algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'):
        assert summary[name] is False
    for marker in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'theta', 'seed617'):
        assert marker not in path.read_text()


def test_modal_response_four_bilingual_pages():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        sections = soup.select('#poolfire-modal-response-section-20261006')
        assert len(sections) == 1
        paired = sections[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(paired) >= 4
        for node in paired:
            assert node['data-i18n-zh'].strip() and node['data-i18n-en'].strip()
        assert '0/99' in sections[0].get_text()
        assert '0.00077%' in sections[0].get_text()
        assert sections[0].find('a')['href'].endswith('poolfire_modal_response_2026-10-06_public_summary.json')
