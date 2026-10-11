import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def records():
    a = json.loads((ROOT/'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    b = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())['latest_signed_cross_ray']
    return a, b


def test_same_aggregate_and_preserved_positive():
    a, b = records()
    row = a['empirical_raw_covariance']
    assert row == b['empirical_raw_covariance']
    assert row['decision'] == 'FAIL_EMPIRICAL_RAW_COVARIANCE_SOURCE_POSTERIOR'
    assert row['queries'] == 99 and row['native_camera_counts'] == [5, 7, 9]
    assert row['primary']['matched'] == row['primary']['matched_strata'] == 0
    assert row['primary']['absolute_strata'] == 33
    assert row['metric_matches'] == dict(field=60, full_gradient=59, interior_gradient=56, observation=0)
    assert row['independent_checks'] == 23 and row['all_independent_checks_pass']
    assert row['complete_held_trajectory_excluded'] and row['source_rows_per_fold'] == 1010
    assert a['conditional_null_prior'] == b['conditional_null_prior']
    assert a['conditional_null_prior']['primary']['matched_queries'] == 1111
    assert a['conditional_null_noise'] == b['conditional_null_noise']


def test_cost_and_scope_are_not_upgraded():
    row = records()[0]['empirical_raw_covariance']
    assert row['standalone_actions'] == dict(A=35, AT=34)
    assert row['source_inverse_actions'] == 1 and row['source_triangular_rhs_solves'] == 2
    assert row['full_measurement_or_world_inverse_actions'] == 0
    assert row['cached_cost_eligible'] and not row['matched_accuracy_resource_result']
    assert row['source_fit_and_geometry_setup_nonfree'] and not row['fresh_whole_pipeline_wall_rss']
    assert row['literal_recipe_closed'] and not row['full_sequence'] and row['neural_updates'] == 0
    assert not any(row['claims'].values())
    for token in ('/Users/', '/Volumes/', 'private_results', 'sha256', 'checkpoint'):
        assert token not in json.dumps(row)


def test_four_bilingual_surfaces():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        paragraphs = soup.find_all(id='empirical-raw-covariance')
        assert len(paragraphs) == 1
        p = paragraphs[0]
        assert p.get_text() == p['data-i18n-zh']
        for lang in ('zh', 'en'):
            assert '0/99' in p['data-i18n-'+lang] and '33/33' in p['data-i18n-'+lang]
            assert '1111/1111' in p['data-i18n-'+lang] and '56/66' in p['data-i18n-'+lang]
        assert soup.find(id='conditional-null-prior') and soup.find(id='conditional-null-noise')
