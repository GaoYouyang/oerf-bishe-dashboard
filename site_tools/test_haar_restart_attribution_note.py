import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def test_attribution_preserves_closed_algorithm_and_mixed_scope():
    path = ROOT/'docs/poolfire_haar_restart_attribution_2026-10-06_public_summary.json'
    data = json.loads(path.read_text())
    assert data['decision'] == 'MIXED_LEARNED_CORRECTION_AND_RESTART_EFFECTS_HAAR_V1'
    assert data['original_algorithm_decision'] == 'FAIL_LEARNED_SOLVER_DUAL_HAAR8_WARM63_CLOSED'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 27
    assert data['parameters_refitted'] == 0 and data['control_standalone_calls'] == {'A': 96, 'AT': 96}
    assert data['control_basic_absolute_strata'] == 33 and data['control_strict_four_metric_matches'] == 0
    assert data['learned_all_four_worse_than_restart_cells'] == data['restart_control_all_four_worse_than_cold_cells'] == 99
    assert data['initial_loss_median_relative_drop']['fold_train'] > 0 and data['initial_loss_median_relative_drop']['held'] > 0
    assert all(.7 < s < .9 for s in data['signed_error_difference_restart_share_medians'].values())
    assert data['fold_train_assessments'] == 990 and data['distinct_held_assessments'] == 99
    for name in ('full_sequence_authorized', 'neural_operator_comparison_completed', 'algorithm_breakthrough',
                 'paper_success', 'resource_speedup', 'external_generalization', 'curved_ray_validated', 'real_bost'):
        assert data[name] is False
    for marker in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'theta'):
        assert marker not in path.read_text()


def test_attribution_is_bilingual_and_not_reinterpreted_as_acceleration():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        sections = soup.select('#poolfire-haar-restart-attribution-20261006')
        assert len(sections) == 1
        section = sections[0]
        pairs = section.select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) >= 5 and all(n['data-i18n-zh'] and n['data-i18n-en'] for n in pairs)
        assert '不是有效加速' in section.get_text() and '不是完整序列' in section.get_text()
        assert '混合效应' in section.get_text() and '不否定全部学习初值' in section.get_text()
        assert section.find('a')['href'].endswith('poolfire_haar_restart_attribution_2026-10-06_public_summary.json')
        if 'daily-progress' in name:
            assert section['data-categories'] == 'research decision'
