import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def records():
    summary = json.loads((ROOT / 'docs/segmented_cross_ray_learning_2026-10-10_public_summary.json').read_text())
    current = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())['latest_signed_cross_ray']
    return summary, current


def test_aggregate_and_mirror():
    summary, current = records()
    row = summary['compact_conditional_prior']
    assert row == current['compact_conditional_prior']
    assert row['decision'] == 'FAIL_COMPACT_CONDITIONAL_PRIOR'
    assert row['primary']['matched'] == 20 and row['primary']['matched_strata'] == 4
    assert row['primary']['absolute_strata'] == 11
    assert row['linear_control']['matched'] == 28
    assert row['independent_checks'] == 21 and row['all_independent_checks_pass']
    assert row['full_sequence'] is False and row['complete_held_trajectory_excluded']
    assert summary['conditional_null_prior'] == current['conditional_null_prior']
    assert summary['conditional_null_prior']['primary']['matched_queries'] == 1111
    assert summary['conditional_null_noise'] == current['conditional_null_noise']


def test_cost_and_claim_boundaries():
    row = records()[0]['compact_conditional_prior']
    assert row['predictor_payload_ratio'] == row['predictor_packed_bytes'] / row['original_predictor_packed_bytes']
    assert row['local_cached_latency_ratio'] == row['local_cached_batch_seconds']['paired'] / row['local_cached_batch_seconds']['kernel']
    assert row['shared_full_factor_excluded_from_payloads']
    assert row['standalone_actions'] == {'A': 5, 'AT': 5}
    assert row['full_inverse_applications'] == 1 and row['triangular_rhs_solves'] == 2
    assert not any(row['claims'].values())
    assert row['literal_recipe_closed'] and row['control_not_promoted_posthoc']
    assert not row['matched_accuracy_resource_result'] and not row['fresh_whole_pipeline_wall_rss']
    text = json.dumps(row)
    for token in ('/Users/', '/Volumes/', 'private_results', 'sha256', 'checkpoint', 'W.npy'):
        assert token not in text


def test_four_bilingual_surfaces():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        block = soup.find_all(id='compact-conditional-prior')
        assert len(block) == 1
        assert block[0].get_text() == block[0]['data-i18n-zh']
        for lang in ('zh', 'en'):
            assert '20/33' in block[0]['data-i18n-' + lang]
            assert '1111/1111' in block[0]['data-i18n-' + lang]
            assert '56/66' in block[0]['data-i18n-' + lang]
        assert soup.find(id='conditional-null-prior')
        assert soup.find(id='conditional-null-noise')
