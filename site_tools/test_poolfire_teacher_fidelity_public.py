import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_teacher_fidelity_warm_20260908'


def test_actual_accuracy_is_not_certificate_failure():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert data['status'] == 'FAIL_TEACHER_FIDELITY_WARM_NECESSARY_COST_GATE'
    assert data['path_passing'] == [0, 0] and data['actual_accuracy_passing'] == [[5, 5], [5, 5]]
    assert data['opened_midpoint_pilot'] and not data['full505_expansion_authorized']
    assert all(max(p['actual_error_max']) < .01 and p['certificate_max'][2] > .01 for p in data['points'])
    assert data['cheaper_direct_field_equivalence'] and not data['dual_lift_specific_advantage']
    assert not any(data[k] for k in ('query_truth_used_in_fit', 'query_truth_used_in_stopping',
        'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))


def test_current_bilingual_preserves_full_roster_result():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_execution_evidence']['note']
    assert current['latest_teacher_fidelity_warm']['passing'] == 0
    assert current['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        notes = soup.select('#teacher-fidelity-result')
        assert len(notes) == 1 and soup.select_one('#full-control-cost-result')
        for language in ('zh', 'en'):
            assert '0/5' in notes[0][f'data-i18n-{language}']
            assert '20' in notes[0][f'data-i18n-{language}']
        if 'daily' in rel:
            assert len(soup.select('#latest')) == 1 and soup.select_one('#latest #teacher-fidelity-result')
        if rel == 'index.html':
            assert STEM in soup.select_one('#latestFigure')['src']
            assert 'certificate' in soup.select_one('#latestFigureCaption')['data-i18n-en']
            assert STEM in soup.select_one('#latestEvidenceLink')['href']


def test_public_report_privacy_and_scope():
    for ext in ('json', 'md'):
        text = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(token in text for token in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.npy', '.pt'))
    report = (ROOT/'docs'/f'{STEM}.md').read_text()
    assert '不是完整轨迹尾部' in report and 'not a complete-trajectory tail result' in report
    assert '不是重建失败' in report and 'not reconstruction' in report
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
