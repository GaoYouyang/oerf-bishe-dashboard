import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_conditional_inverse_2026-10-07_public_summary.json'
MARKER = 'poolfire-conditional-inverse-20261007'


def evidence():
    return json.loads((ROOT / 'docs' / NAME).read_text())


def test_independent_result_and_scope():
    d = evidence()
    assert d['decision'] == 'FAIL_FIXED_CONDITIONAL_FSAI32_WARM34_CLOSED' and d['reference_adequate']
    assert d['cells'] == 99 and d['sampled_three_frame_strata'] == 33
    assert d['frames'] == [0, 50, 100] and d['camera_counts'] == [5, 7, 9]
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 36
    assert d['matched_cells'][d['primary']] == 0 and d['absolute_strata'][d['primary']] == 33


def test_controls_and_new_parameters():
    d = evidence()
    assert d['descriptive_controls']['NeuralDualFactor58-Warm34']['all_four_better'] == 77
    assert d['descriptive_controls']['CGLS35']['all_four_better'] == 0
    assert d['descriptive_controls']['CGLS35']['all_four_worse'] == 72
    assert d['shared_trainable_parameters'] == d['full_factor_builds'] == 0


def test_cost_and_claim_boundaries():
    d = evidence()
    assert d['standalone'] == {'A': 35, 'AT': 35, 'sparse_factor_multiplies': 2}
    assert d['local_principal_solves'] == 48614 and d['paired_factor_payload_bytes'] == 18047824
    assert d['factor_payload_not_RSS'] and d['local_Gram_solves_and_inherited_graph_nonfree']
    for k in ('learned_initializer_tested', 'full_sequence_tested', 'algorithm_breakthrough', 'fresh_wall_RSS_tested', 'resource_speedup', 'external_generalization', 'real_bost', 'curved_ray_validated', 'main_goal_complete'):
        assert d[k] is False


def test_bilingual_notes_and_summary_links():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        page = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = page.select('#' + MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select('[data-i18n-zh][data-i18n-en]')) == 6
        assert all(n['data-i18n-zh'] and n['data-i18n-en'] for n in note.select('[data-i18n-zh]'))
        assert all(t in note.get_text() for t in ('36/36', '0/99', '33/33', '5/7/9', '77/99', '35A+35AT', '17.21MiB'))
        assert (ROOT / name).parent.joinpath(note.find('a')['href']).resolve().is_file()


def test_aggregate_log_and_privacy():
    aggregate = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    n = aggregate['latest_conditional_inverse_seed']
    assert n['matched_cells'] == 0 and n['independent_checks_passed'] == 36
    assert 'latest_detector_riesz_seed' in aggregate
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert '## 2026-10-07: 局部解析逆因子优于学习因子对照' in log and NAME in log
    payload = (ROOT / 'docs' / NAME).read_text()
    assert not any(w in payload for w in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
