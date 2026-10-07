import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_detector_riesz_2026-10-07_public_summary.json'
MARKER = 'poolfire-detector-riesz-20261007'


def evidence():
    return json.loads((ROOT / 'docs' / NAME).read_text())


def test_independent_negative_and_finite_scope():
    d = evidence()
    assert d['decision'] == 'FAIL_FIXED_DETECTOR_RIESZ_BAND9_CAPACITY_CLOSED'
    assert d['reference_adequate'] and d['cells'] == 99 and d['sampled_three_frame_strata'] == 33
    assert d['frames'] == [0, 50, 100] and d['camera_counts'] == [5, 7, 9]
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 34
    assert d['matched_cells'][d['primary']] == 0 and d['absolute_strata'][d['primary']] == 33


def test_capacity_is_not_training_or_resource_success():
    d = evidence()
    assert d['shared_trainable_parameters'] == 0 and d['teacher_coefficients_per_query'] == 9
    for key in ('full_sequence_tested', 'learned_initializer_tested', 'fresh_wall_RSS_tested', 'algorithm_breakthrough', 'resource_speedup', 'external_generalization', 'real_bost', 'curved_ray_validated', 'main_goal_complete'):
        assert d[key] is False
    assert d['fixed_recipe_closed_without_band_or_depth_tuning']


def test_costs_and_finite_numerical_closure():
    d = evidence()
    assert d['primary_callbacks'] == {'A': 35, 'AT': 35}
    assert d['capacity_direction_AT_per_query'] == 9 and d['FFT_SVD_reference_construction_nonfree']
    assert d['full_normal_factors'] == 0
    assert d['independent_maxima']['new_states/states'] < 1e-7
    assert d['independent_maxima']['metrics_absolute'] < 1e-7


def test_four_bilingual_notes_and_links():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        page = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = page.select('#' + MARKER)
        assert len(notes) == 1
        note = notes[0]
        assert len(note.select('[data-i18n-zh][data-i18n-en]')) == 6
        assert all(n['data-i18n-zh'] and n['data-i18n-en'] for n in note.select('[data-i18n-zh]'))
        assert all(t in note.get_text() for t in ('34/34', '0/99', '33/33', '5/7/9', '35A+35AT'))
        href = note.find('a')['href']
        assert (ROOT / name).parent.joinpath(href).resolve().is_file()


def test_aggregate_and_log():
    d = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    n = d['latest_detector_riesz_seed']
    assert n['matched_cells'] == 0 and n['independent_checks_passed'] == 34
    assert 'latest_row_sweep_seed' in d
    log = (ROOT / 'docs/operator_3d_learning_log.md').read_text()
    assert log.startswith('## 2026-10-07: 探测器频带初值没有达到高精度参考')
    assert NAME in log


def test_privacy_and_no_native_archives():
    payload = (ROOT / 'docs' / NAME).read_text()
    assert not any(s in payload for s in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
    assert evidence()['role'].startswith('Fixed detector-frequency representation')
