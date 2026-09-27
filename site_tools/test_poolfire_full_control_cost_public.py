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
    assert STEM in current['latest_full_trajectory_controls']['summary']
    assert current['formal_status'] == 'PASS_V285_FORMAL_AND_INDEPENDENT_RECOMPUTATION'
    assert current['scientific_status'] == 'FAIL_CASE2_SAME_FAMILY_POSTOPEN_TRANSFER_ACCURACY_V285'
    assert current['next_scientific_gate'] == current['next_scientific_gate_en']
    assert current['next_scientific_gate_zh']
    assert current['latest_solver_in_loop_loto_v284']['status'] == 'FAIL_SOLVER_IN_LOOP_LOTO_STRICT_COST'
    aggregate = current['latest_solver_in_loop_loto_v284']['posthoc_aggregate_cost_vs_dual_ridge']
    assert not aggregate['preregistered'] and not aggregate['changes_primary_verdict']
    assert aggregate['candidate_calls_per_path'] == {'A': 53050, 'AT': 52545}
    assert aggregate['dual_ridge_calls_per_path'] == {'A': 57133, 'AT': 56628}
    assert aggregate['saved_calls_per_path'] == {'A': 4083, 'AT': 4083}
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
            assert len(soup.select('#latest')) == 1 and soup.select_one('#day-20260908-full-controls #full-control-cost-result')


def test_redacted_report_and_figure():
    for ext in ('md', 'json'):
        content = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(token in content for token in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    report = (ROOT/'docs'/f'{STEM}.md').read_text()
    assert '不是暖启动贡献或实测提速' in report and 'not warm-start attribution or measured speedup' in report
    assert '>=42.11%' in report and '1008/1010' in report
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
