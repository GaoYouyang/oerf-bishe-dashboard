import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_camera_subset_reference_20260908'


def test_reference_pass_not_learned_claim():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'PASS_FIXED_CAMERA_SUBSET_DIRECT_REFERENCE' and d['valid']
    assert d['passing'] == d['total'] == 1010 and d['unique_frames'] == 505
    assert d['complete_count_trajectories'] == 10 and all(d['checks'].values())
    assert sorted(r['camera_count'] for r in d['camera_counts']) == [5,7]
    assert all(r['passing'] == 505 and r['complete_trajectories'] == 5 for r in d['camera_counts'])
    assert not any(d[k] for k in ('new_training_authorized','learned_parameters_consumed','warm_advantage',
        'algorithm_breakthrough','paper_success','resource_speedup','external_generalization','real_bost'))
    assert d['primary_field_only_online'] == dict(A=0,AT=1,triangular_solve=2)


def test_bilingual_current_and_historical_evidence():
    d = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in d['latest_execution_evidence']['note']
    assert d['latest_full_trajectory_controls']['passing'] == 505
    assert d['latest_solver_loss_alignment']['conflict_folds'] == 0
    for rel in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(),'html.parser')
        note = soup.select_one('#camera-subset-reference-result')
        assert '505/505' in note['data-i18n-zh'] and '505/505' in note['data-i18n-en']
        assert soup.select_one('#solver-loss-alignment-result') and soup.select_one('#actual-warm-cost-result')
        if 'daily' in rel:
            assert soup.select_one('#latest #camera-subset-reference-result')


def test_redaction_and_figure():
    for ext in ('md','json'):
        text = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(s in text for s in ('/Users/','/Volumes/','private_results','sha256','.npy','.pt'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
