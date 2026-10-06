import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')


@pytest.mark.parametrize('name', PAGES)
def test_attribution_is_bilingual_and_qualified(name):
    soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
    for identifier in ('poolfire-learned-gain-attribution-20261006', 'poolfire-learned-gain-attribution-boundary-20261006'):
        node = soup.find(id=identifier)
        assert node and node.get('data-i18n-zh') and node.get('data-i18n-en')
    text = soup.find(id='poolfire-learned-gain-attribution-20261006')['data-i18n-en']
    assert all(token in text for token in ('0.87%', 'single PCGLS', 'whole PCGLS', '14', '1.0122', 'not CFD-truth'))
    boundary = soup.find(id='poolfire-learned-gain-attribution-boundary-20261006')['data-i18n-en']
    assert all(token in boundary for token in ('no fitting', '0/99', 'not a combined algorithm', 'reopen'))


def test_aggregate_does_not_reopen_the_recipe():
    doc = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    entry = doc['latest_sealed_learned_gain_attribution']
    assert entry['cells'] == 99 and entry['separate_derivations_agree']
    assert entry['old_recipe_still_closed'] and entry['finite_reference_gap_is_not_cfd_truth_error']
    assert entry['not_independence_from_whole_pcgls_search_space']
    assert entry['all_four_finite_reference_gap_nonworse_pcgls_cells'] == 66
    assert entry['all_four_archived_truth_error_nonworse_pcgls_cells'] == 65
    assert entry['nine_camera_observation_reference_gap_nonworse_pcgls_cells'] == 14
    assert entry['original_matched_cells_passed'] == 0
    for key in ('new_A', 'new_AT', 'new_fits', 'new_cfd_reads'):
        assert entry[key] == 0
    for key in ('new_predictor_authorized', 'algorithm_breakthrough', 'paper_success',
                'resource_speedup', 'external_generalization', 'real_bost'):
        assert entry[key] is False
    for forbidden in ('/Users/', 'private_results/', 'sha256', 'checkpoint', 'weights'):
        assert forbidden not in json.dumps(entry)
    assert doc['latest_whole_observation_learning']['fixed_recipe_closed']
