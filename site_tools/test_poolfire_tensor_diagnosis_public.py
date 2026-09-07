import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_tensor_train_diagnosis_20260907'


def test_diagnosis_and_limits():
    d = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    assert d['status'] == 'TRAIN_ERROR_AND_NONSTATIONARITY_PRESENT'
    assert d['train_passing'] == d['query_passing'] == d['optimizer_steps'] == 0
    assert d['train_pairs'] == 2020 and d['unique_frames'] == d['query_pairs'] == 505
    assert d['overlapping_train_pairs'] and d['probes_not_deployed']
    assert d['certified_folds'] == 5 and d['diagnostic_probes_per_fold'] == 2
    assert d['verification']['maximum_score_difference'] < 1e-13
    assert d['counts']['inherited_baseline_arms'] == 11
    assert d['counts']['inherited_query_endpoints'] == 505
    assert d['counts']['hypothetical_endpoint_online'] == {'A':2, 'AT':2}
    for row in d['rows']:
        assert row['train_post']['passing'] == row['query_post']['passing'] == 0
        assert .00023 < row['relative_train_objective_descent'] < .00030
        assert row['derivative_relative_difference'] < 4e-5 and row['descent_certified']
        assert .25 < row['train_post']['p90'][0] < .30
    assert not any(d[k] for k in ('intrinsic_capacity_failure_proven', 'more_training_guarantees_target',
        'new_learner_authorized', 'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'real_bost'))


def test_bilingual_and_privacy():
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in current['latest_tensor_train_diagnosis']['summary']
    assert current['latest_tensor_train_diagnosis']['optimizer_steps'] == 0
    for relative in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/relative).read_text(), 'html.parser')
        notes = soup.select('#tensor-train-diagnosis')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            text = notes[0][f'data-i18n-{lang}']
            assert '2020' in text and '1%' in text and '0.023%' in text
    text = (ROOT/f'docs/{STEM}.md').read_text()
    assert '25.74%' in text and 'not2020 independent' in text and '微扰' in text
    for name in (f'docs/{STEM}.md', f'docs/{STEM}.json'):
        value = (ROOT/name).read_text()
        assert not any(x in value for x in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    assert (ROOT/f'assets/figures/{STEM}.png').stat().st_size > 10000
