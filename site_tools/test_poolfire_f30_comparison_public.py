import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_f30_comparison_20260906'


def test_aggregate_counts_and_cost_boundary():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    arms = {row['arm']: row for row in data['arms']}
    assert data['coverage']['cells'] == 505 and data['coverage']['geometries'] == 1
    assert data['coverage']['camera_counts'] == [9]
    assert (arms['ray_set_k1']['passing_cells'], arms['ray_set_k1']['complete_trajectories']) == (409, 2)
    assert (arms['linear_response_k1']['passing_cells'], arms['linear_response_k1']['complete_trajectories']) == (412, 1)
    assert arms['cgls4']['passing_cells'] == 505 and arms['cgls4']['is_reference']
    assert arms['field_k1']['logical_calls'] == {'A': 2, 'AT': 1}
    assert arms['ray_set_k1']['logical_calls'] == {'A': 2, 'AT': 2}
    for arm in arms.values():
        assert sum(r['passing_cells'] for r in arm['trajectory_results']) == arm['passing_cells']
        assert sum(r['passing_cells'] == 101 for r in arm['trajectory_results']) == arm['complete_trajectories']
        assert len(arm['trajectory_results']) == 5
    assert not data['neural_comparator']['hidden_weights_trained']
    assert not any(data['claims_fixed_false'].values())
    assert not data['cost_boundary']['fresh_wall_rss_comparison_completed']
    assert not data['criterion']['absolute_reference_adequacy_proven']


def test_weak_message_not_uniformly_harmless():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    diagnostic = data['diagnostics']['global_context']
    assert diagnostic['field_norm_contribution_max'] < .000245
    assert diagnostic['projected_field_norm_contribution_max'] < .000274
    assert diagnostic['unchanged_joint_classifications'] == 505
    assert not diagnostic['strict_uniform_removability_passed']
    assert diagnostic['ablation_all_four_better_cells'] == 41
    assert diagnostic['parent_all_four_better_cells'] == 12


def test_private_boundary_bilingual_current_and_archives():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    text = (ROOT / f'docs/{STEM}.md').read_text()
    for forbidden in ('/Users/', '/Volumes/', 'private_results', 'sha256', 'cameraData', 'checkpoint', '.pt'):
        assert forbidden not in json.dumps(data) and forbidden not in text
    assert '# PoolFire:' in text and '完整轨迹' in text and 'not 505 independent' in text
    current = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert current['scientific_status'] == data['scientific_status']
    assert current['current_decision']['v283_fixed_reference_closed']
    assert STEM in current['public_evidence']['summary']
    for file in ('index.html', 'operator-learning/index.html'):
        soup = BeautifulSoup((ROOT / file).read_text(), 'html.parser')
        section = soup.select_one('#poolfire-f30-comparison')
        assert section and soup.select_one('#v283-nodal-reference')
        for node in section.select('h2,p[data-i18n-zh],figcaption'):
            assert node.get('data-i18n-zh') and node.get('data-i18n-en')
        assert '409' in soup.select_one('header').get_text()
        image = section.select_one('img')
        assert image.get('data-i18n-alt-zh') and image.get('data-i18n-alt-en')
    daily = BeautifulSoup((ROOT / 'operator-learning/daily-progress.html').read_text(), 'html.parser')
    assert daily.select_one('#latest')['data-date'] == '2026-09-06'
    assert daily.select_one('#day-2026-09-05')
    assert (ROOT / f'assets/figures/{STEM}.png').stat().st_size > 10000
