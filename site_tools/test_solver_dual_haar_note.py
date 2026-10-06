import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def test_haar_is_scoped_valid_negative_with_strong_controls():
    path = ROOT/'docs/poolfire_solver_dual_haar_2026-10-06_public_summary.json'
    record = json.loads(path.read_text())
    assert record['decision'] == 'FAIL_LEARNED_SOLVER_DUAL_HAAR8_WARM63_CLOSED'
    assert record['independent_checks_passed'] == record['independent_checks_total'] == 33
    assert record['basic_absolute_strata_passed'] == record['basic_absolute_strata_total'] == 33
    assert record['strict_four_metric_matched_cells'] == 0 and record['cells_total'] == 99
    assert record['shared_trainable_parameter_count'] == 8
    assert record['primary_standalone_calls'] == {'A': 96, 'AT': 96}
    assert record['bp_control_standalone_calls'] == {'A': 96, 'AT': 95}
    assert record['same_budget_plain_cgls_comparison']['all_four_harm_cells'] == 99
    assert record['cheaper_bp_comparison']['all_four_harm_cells'] == 99
    assert record['same_parameter_linear_control_comparison']['all_four_no_worse_cells'] == 63
    assert record['same_parameter_linear_control_comparison']['all_four_harm_cells'] == 35
    for name in ('held_trajectory_target_in_fit', 'CFD_truth_in_fit', 'neural_operator_comparison_completed',
                 'full_sequence_authorized', 'algorithm_breakthrough', 'paper_success', 'resource_speedup',
                 'external_generalization', 'curved_ray_validated', 'real_bost'):
        assert record[name] is False
    assert record['earlier_modest_learning_signal_preserved']
    for marker in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'theta'):
        assert marker not in path.read_text()


def test_haar_note_is_bilingual_scoped_and_linked():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        sections = soup.select('#poolfire-solver-dual-haar-20261006')
        assert len(sections) == 1
        section = sections[0]
        pairs = section.select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) >= 5 and all(n['data-i18n-zh'] and n['data-i18n-en'] for n in pairs)
        assert '不是有效加速' in section.get_text() and '不否定全部非线性先验' in section.get_text()
        assert '不是完整序列' in section.get_text() and '原因尚未唯一定位' in section.get_text()
        assert section.find('a')['href'].endswith('poolfire_solver_dual_haar_2026-10-06_public_summary.json')
        if 'daily-progress' in name:
            assert section['data-categories'] == 'research decision'
