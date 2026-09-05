import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'real_bost_fold_discrepancy_v281'


def test_redacted_result_and_failure_gate():
    data = json.loads((ROOT/f'docs/{STEM}_public_summary.json').read_text())
    assert data['scientific_decision'] == 'FAIL_LOMO_DISCREPANCY_SENTINEL_V281'
    assert data['coverage']['cells'] == 108 and data['coverage']['rigs'] == 1
    assert data['adjudication']['passed_strata'] == {arm: 0 for arm in data['arms']}
    assert not any(data['claims_fixed_false'].values())
    assert len(data['summaries']) == 36
    for row in data['summaries']:
        assert row['count'] == 9 and row['passed'] is False
        assert all(v['p90'] == v['worst'] for v in row['metrics'].values())
    text = json.dumps(data)
    for banned in ('/Users/', '/Volumes/', '.cpu.pt', 'cameraData', 'checkpoint', 'training_indices'):
        assert banned not in text


def test_current_manifest_and_figure():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert current['scientific_status'] == 'PASS_LINEAR_VIRTUAL_PIXEL_INTERFACE_V282'
    assert current['current_decision']['v281_fixed_estimator_closed'] is True
    assert current['metrics']['v281_cells'] == 108
    assert (ROOT/f'assets/figures/{STEM}.png').stat().st_size > 10000
    home = BeautifulSoup((ROOT/'index.html').read_text(), 'html.parser')
    assert STEM in home.select_one('#v281-fold-discrepancy img')['src']
    assert home.select_one('#v281-fold-discrepancy figcaption') is not None


def test_bilingual_sections_and_single_daily_date():
    for path in ('index.html', 'operator-learning/index.html'):
        soup = BeautifulSoup((ROOT/path).read_text(), 'html.parser')
        section = soup.select_one('#v281-fold-discrepancy')
        assert section and '0/12' in section.get_text()
        for node in section.select('h2, p[data-i18n-zh], figcaption'):
            assert node.get('data-i18n-zh') and node.get('data-i18n-en')
        assert soup.select_one('#v280-source-budget') is not None
    daily = BeautifulSoup((ROOT/'operator-learning/daily-progress.html').read_text(), 'html.parser')
    assert len(daily.select('[data-date="2026-09-05"]')) == 1
    assert 'v281' in daily.select_one('#latest').get_text()
    note = (ROOT/f'docs/{STEM}_result_2026-09-05.md').read_text()
    assert '# v281: cross-model' in note and '0/12' in note
