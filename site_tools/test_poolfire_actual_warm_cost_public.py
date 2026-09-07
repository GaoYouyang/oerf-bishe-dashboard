import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_actual_warm_cost_20260908'


def test_interval_decision_and_limited_scope():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['passing'] == 0 and d['robust_failures'] == 5 and d['inconclusive'] == 0
    assert d['recovered_states'] == 2950 and d['common_horizon'] == 256
    for p in d['points']:
        assert p['intervals']['first_cost'] == p['intervals']['sustained_cost']
        bounds = p['intervals']['first_cost']
        assert bounds['full_teacher'][0] > min(bounds[a][1] for a in ('zero_metric','k4_teacher'))
    assert d['truth_oracle_costs_not_deployment_stop'] and d['retrospective_not_prospective']
    assert d['old_exact_count_inconclusive_retained'] and d['all_original_path_identities']
    assert not any(d[k] for k in ('model_or_teacher_fit','caps_or_certificates_changed',
        'new_conditions_opened','full505_authorized','algorithm_breakthrough','paper_success',
        'resource_speedup','external_generalization','real_bost','dual_specific_advantage'))
    assert d['recovered_solver_work'] == dict(A=2950,AT=2950)
    assert d['scoring_offline'] == dict(score_A=2950,native_A=2950,consistency_A=10)


def test_bilingual_latest_and_historical_evidence_retained():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_execution_evidence']['note']
    assert current['latest_full_trajectory_controls']['passing'] == 505
    assert current['latest_teacher_fidelity_warm']['all20_actual_accuracy']
    assert current['latest_metric_similarity']['passing'] == 0
    for rel in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(),'html.parser')
        note = soup.select_one('#actual-warm-cost-result')
        assert note and '0/5' in note['data-i18n-zh'] and '0/5' in note['data-i18n-en']
        assert soup.select_one('#metric-similarity-result') and soup.select_one('#teacher-fidelity-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #actual-warm-cost-result')
        if rel == 'index.html':
            assert STEM in soup.select_one('#latestFigure')['src']
            assert 'truth-oracle' in soup.select_one('#latestFigureCaption')['data-i18n-en']


def test_redaction_and_figure():
    for ext in ('md','json'):
        text = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(s in text for s in ('/Users/','/Volumes/','private_results','sha256','.npy','.pt'))
    text = (ROOT/'docs'/f'{STEM}.md').read_text()
    assert '不是可部署停机规则' in text and 'not a deployable stopping rule' in text
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
