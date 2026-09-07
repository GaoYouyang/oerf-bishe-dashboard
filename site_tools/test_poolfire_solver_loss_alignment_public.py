import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_solver_loss_alignment_20260908'


def test_negative_diagnostic_not_algorithm_claim():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['conflict_folds'] == 0 and d['folds'] == 5
    assert d['overlapping_train_pairs'] == 20 and d['directional_scalar_checks'] == 240
    assert d['old_rescored_states']+d['new_real_tangent_states'] == 1200
    assert len(d['cosines']) == 5 and all(0 < c < 1 for c in d['cosines'])
    assert d['retrospective'] and d['both_positive_steps_reduce_both_losses']
    assert not any(d[k] for k in ('parent_eligibility_restored','new_fit','CFD_truth_read','outer_test',
        'new_model_authorized','algorithm_breakthrough','paper_success','resource_speedup','external_generalization','real_bost'))


def test_paired_language_and_prior_evidence():
    data = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in data['latest_solver_loss_alignment']['summary']
    assert data['latest_full_trajectory_controls']['passing'] == 505
    for rel in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(),'html.parser')
        note = soup.select_one('#solver-loss-alignment-result')
        assert '0/5' in note['data-i18n-zh'] and '0/5' in note['data-i18n-en']
        assert 'TRAIN' in note['data-i18n-zh'] and 'TRAIN' in note['data-i18n-en']
        assert soup.select_one('#actual-warm-cost-result')
        if 'daily' in rel:
            assert soup.select_one('#day-20260908-solver-loss-alignment #solver-loss-alignment-result')


def test_redacted_report_and_figure():
    for ext in ('md','json'):
        text = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(s in text for s in ('/Users/','/Volumes/','private_results','sha256','.npy','.pt'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
