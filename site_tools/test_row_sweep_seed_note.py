import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_row_sweep_seed_2026-10-07_public_summary.json'
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')


def test_row_action_small_gain_is_not_reference_matching():
    data = json.loads((ROOT / 'docs' / NAME).read_text())
    assert data['cells'] == 99 and data['frames'] == [0, 50, 100]
    assert data['camera_counts'] == [5, 7, 9] and not data['full_sequence_tested']
    assert data['matched_cells'][data['primary']] == 0
    assert data['absolute_strata'][data['primary']] == 33 and data['reference_adequate']
    assert data['descriptive_controls']['CGLS35']['all_four_no_worse'] == 99
    assert all(0 < value < .08 for value in data['median_reduction_vs_CGLS35'].values())
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 32


def test_row_action_cost_and_scope_are_explicit():
    data = json.loads((ROOT / 'docs' / NAME).read_text())
    assert data['production_callbacks_per_cell'] == {'A': 33, 'AT': 33}
    assert data['row_arithmetic_upper_equivalents'] == {'A': 2, 'AT': 2}
    assert data['total_upper_equivalents'] == {'A': 35, 'AT': 35}
    assert data['row_arithmetic_is_nonfree'] and data['independent_block_gram_work_is_nonfree']
    assert data['full_normal_factors'] == data['new_fit_parameters'] == 0
    assert all(not data[k] for k in ('algorithm_breakthrough', 'resource_speedup', 'real_bost', 'main_goal_complete', 'learned_initializer_tested'))
    assert data['recipe_closed_without_sweep_or_depth_tuning']
    assert all(s not in json.dumps(data) for s in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))


def test_row_action_notes_are_bilingual_and_links_exist():
    for name in PAGES:
        doc = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = doc.find_all(id='poolfire-row-sweep-seed-20261007')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            nodes = notes[0].select('[data-i18n-' + lang + ']')
            assert len(nodes) >= 7
            text = ' '.join(node['data-i18n-' + lang] for node in nodes)
            assert all(token in text for token in ('99', '0/99', '33/33', '32/32', '5/7/9', '7.62%', '33A+33AT', '35A+35AT'))
        for link in notes[0].find_all('a', href=True):
            assert (ROOT / name).parent.joinpath(link['href'].split('#')[0]).exists()


def test_row_action_keeps_prior_classical_cost_result():
    current = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_row_sweep_seed']['matched_cells'] == 0
    assert current['latest_exact_band_cost']['independent_checks_passed'] == 57
    assert current['latest_full_temporal_direct_cost']['faster_all_pairs']
