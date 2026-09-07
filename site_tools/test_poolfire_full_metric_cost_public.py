import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_full_metric_cost_20260908'


def test_complete_trajectory_counts_and_limits():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'PASS_FULL_TRAJECTORY_METRIC_COST'
    assert (d['passing'], d['complete_trajectories'], d['censored_oracle_paths']) == (505, 5, 0)
    assert all(r['passing'] == 101 and r['complete_pass'] for r in d['trajectories'])
    assert abs(d['min_A_reduction']-(1-342/345)) < 1e-15
    assert d['zero_initialization'] and d['oracle_truth_comparator'] and d['retrospective_opened_data']
    assert d['query_truth_free_primary_stop'] and d['metric_and_setup_nonfree'] and d['full_direct_not_beaten']
    assert not any(d[k] for k in ('new_training','warm_advantage','resource_speedup','external_generalization','real_bost','algorithm_breakthrough','paper_success'))


def test_current_and_bilingual_history():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_full_trajectory_metric']['summary']
    assert current['latest_full_trajectory_metric']['passing'] == 505
    assert current['latest_left_metric']['warm_attribution_confirmed'] is False
    for rel in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        note = soup.select('#full-metric-cost-result')
        assert len(note) == 1
        assert all('22.64%' in note[0][f'data-i18n-{lang}'] and '0.87%' in note[0][f'data-i18n-{lang}'] for lang in ('zh','en'))
        assert soup.select_one('#left-metric-result')
        if 'daily' in rel:
            assert len(soup.select('#latest')) == 1 and soup.select_one('#day-20260908-full-metric #full-metric-cost-result')


def test_redacted_report_and_figure():
    for ext in ('md','json'):
        text = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(v in text for v in ('/Users/','/Volumes/','private_results','sha256','.pt','parameters.json'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
    report = (ROOT/'docs'/f'{STEM}.md').read_text()
    assert '不是暖初始化贡献证明' in report and 'not proof of warm-initializer contribution' in report
    assert '342A/341AT' in report and '345A/345AT' in report
