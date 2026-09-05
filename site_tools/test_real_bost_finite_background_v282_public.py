import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'real_bost_finite_background_v282'


def test_optical_scope_not_algorithm_claim():
    data = json.loads((ROOT/f'docs/{STEM}_public_summary.json').read_text())
    assert data['scientific_decision'] == 'PASS_LINEAR_VIRTUAL_PIXEL_INTERFACE_V282'
    assert data['coverage']['maps'] == 468 and data['coverage']['diagnostic_cells'] == 1404
    assert data['coverage']['calibration_files'] == 13 and data['coverage']['distinct_pixel_operators'] == 11
    assert data['coverage']['rays_per_camera'] == 64
    assert not any(data['claims_fixed_false'].values())
    assert data['cost']['offline_only'] and not data['cost']['is_reconstruction_cost']
    assert len(data['summaries']) == 12 and all(s['count'] == 117 for s in data['summaries'])
    assert max(s['thin_relative']['worst'] for s in data['summaries']) < .06
    for private in ('/Users/', '/Volumes/', '.cpu.pt', 'cameraData', 'sha256', 'checkpoint'):
        assert private not in json.dumps(data)


def test_current_figure_and_bilingual_scope():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert current['scientific_status'] == 'PASS_LINEAR_VIRTUAL_PIXEL_INTERFACE_V282'
    assert current['metrics']['v282_maps'] == 468
    assert current['current_decision']['v281_fixed_estimator_closed']
    assert not current['current_decision']['v282_predictor_training_authorized']
    for path in ('index.html', 'operator-learning/index.html'):
        soup = BeautifulSoup((ROOT/path).read_text(), 'html.parser')
        section = soup.select_one('#v282-finite-background')
        assert section and '不是重建或加速' in section.get_text()
        for node in section.select('h2, p[data-i18n-zh], figcaption'):
            assert node.get('data-i18n-zh') and node.get('data-i18n-en')
        assert soup.select_one('#v281-fold-discrepancy')
    home = BeautifulSoup((ROOT/'index.html').read_text(), 'html.parser')
    assert STEM in home.select_one('#latestFigure')['src']
    assert '不是重建误差' in home.select_one('#latestFigureCaption').get_text()


def test_one_daily_date_and_equivalent_report():
    daily = BeautifulSoup((ROOT/'operator-learning/daily-progress.html').read_text(), 'html.parser')
    assert len(daily.select('[data-date="2026-09-05"]')) == 1
    assert 'v282' in daily.select_one('#latest').get_text()
    report = (ROOT/f'docs/{STEM}_result_2026-09-05.md').read_text()
    assert '# v282: finite-background' in report and 'not reconstruction' in report
    assert '8×8' in report and '十一种' in report
    assert (ROOT/f'assets/figures/{STEM}.png').stat().st_size > 10000
