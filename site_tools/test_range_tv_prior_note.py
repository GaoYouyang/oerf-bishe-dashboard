import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def test_range_tv_is_valid_negative_not_acceleration():
    path = ROOT/'docs/poolfire_range_tv_prior_2026-10-06_public_summary.json'
    record = json.loads(path.read_text())
    assert record['decision'] == 'FAIL_LEARNED_RANGE_TV1_WARM96_CLOSED'
    assert record['independent_checks_passed'] == record['independent_checks_total'] == 30
    assert record['basic_absolute_strata_passed'] == record['basic_absolute_strata_total'] == 33
    assert record['strict_four_metric_matched_cells'] == 0 and record['cells_total'] == 99
    assert record['same_budget_plain_cgls_comparison']['all_four_harm_cells'] == 99
    assert record['cheaper_bp_comparison']['all_four_harm_cells'] == 99
    assert record['shared_trainable_parameter_count'] == 1
    assert record['primary_standalone_calls'] == {'A': 99, 'AT': 99}
    assert record['bp_control_standalone_calls'] == {'A': 99, 'AT': 98}
    assert record['earlier_modest_learning_signal_preserved']
    for name in ('neural_operator_tested', 'held_trajectory_truth_in_fit', 'finite_teacher_in_fit',
                 'full_sequence_authorized', 'algorithm_breakthrough', 'paper_success',
                 'resource_speedup', 'external_generalization', 'real_bost'):
        assert record[name] is False
    for marker in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'alpha'):
        assert marker not in path.read_text()


def test_range_tv_note_is_bilingual_and_preserves_failure_scope():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        sections = soup.select('#poolfire-range-tv-prior-20261006')
        assert len(sections) == 1
        section = sections[0]
        pairs = section.select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) >= 5 and all(n['data-i18n-zh'] and n['data-i18n-en'] for n in pairs)
        assert '不是有效加速' in section.get_text() and '不否定全部物理先验' in section.get_text()
        assert '不重训、不改门' in section.get_text() and '不是完整序列' in section.get_text()
        assert section.find('a')['href'].endswith('poolfire_range_tv_prior_2026-10-06_public_summary.json')
        if 'daily-progress' in name:
            assert section['data-categories'] == 'research decision'
