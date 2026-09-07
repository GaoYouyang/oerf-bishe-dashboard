import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_full_control_cost_20260908'


def test_complete_controls_and_censoring():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'PASS_FULL_ROSTER_CONTROL_COST'
    assert (d['passing'], d['complete_trajectories'], d['control_curves'], d['paired_comparisons']) == (505, 5, 4040, 2020)
    assert all(r['passing'] == 505 and r['complete_trajectories'] == 5 and r['control_ahead'] == 0 for r in d['arms'])
    assert {r['arm']: r['censored_paths'] for r in d['arms']} == dict(zero_cgls=0, normalized_bp=0, dual_ridge=0, fixed_metric=1008)
    assert d['zero_initialization'] and d['oracle_truth_controls'] and d['retrospective_opened_data']
    assert d['query_truth_free_primary_stop'] and d['metric_and_setup_nonfree'] and d['full_direct_not_beaten']
    assert not any(d[k] for k in ('new_training', 'warm_advantage', 'resource_speedup', 'external_generalization', 'real_bost', 'algorithm_breakthrough', 'paper_success'))


def test_current_bilingual_and_history():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_execution_evidence']['note']
    assert current['formal_status'] == 'PASS_FULL_ROSTER_CONTROL_COST'
    assert current['latest_full_trajectory_controls']['passing'] == 505
    assert not current['latest_full_trajectory_controls']['warm_attribution_confirmed']
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        notes = soup.select('#full-control-cost-result')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            assert '505/505' in notes[0][f'data-i18n-{lang}']
            assert '1008/1010' in notes[0][f'data-i18n-{lang}']
        assert soup.select_one('#full-metric-cost-result') and soup.select_one('#left-metric-result')
        if 'daily' in rel:
            assert len(soup.select('#latest')) == 1 and soup.select_one('#latest #full-control-cost-result')


def test_redacted_report_and_figure():
    for ext in ('md', 'json'):
        content = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(token in content for token in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    report = (ROOT/'docs'/f'{STEM}.md').read_text()
    assert '不是暖启动贡献或实测提速' in report and 'not warm-start attribution or measured speedup' in report
    assert '>=42.11%' in report and '1008/1010' in report
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
