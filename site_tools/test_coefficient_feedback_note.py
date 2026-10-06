import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-coefficient-feedback-20261006'
SUMMARY = 'poolfire_coefficient_feedback_2026-10-06_public_summary.json'


def test_feedback_scientific_boundary_and_aggregates():
    text = (ROOT/'docs'/SUMMARY).read_text()
    report = json.loads(text)
    assert report['independent_checks_passed'] == report['independent_checks_total'] == 26
    assert report['counts'] == {'transported_harm':33,'feedback_reversal':0,'mixed':66,'adaptive_all_four_harm':99}
    assert report['counterfactual_strict_matches'] == 0 and report['counterfactual_basic_strata'] == 32
    assert report['ratios']['frozen_vs_restart']['observation_rel_l2']['min'] > 1
    assert report['ratios']['adaptive_vs_frozen']['observation_rel_l2']['max'] < 1
    assert report['parameters_refitted'] == 0 and not report['formal_arrays_rerun']
    assert report['original_algorithm_closed'] and report['engineering_failure_preserved']
    assert report['audit_calls_each_mode'] == {'A':14652,'AT':12870}
    for field in ('full_sequence_authorized','training_authorized','neural_operator_comparison_completed','algorithm_breakthrough','paper_success','resource_speedup','external_generalization','curved_ray_validated','real_bost'):
        assert report[field] is False
    assert all(marker not in text for marker in ('/Users/', 'private_data/', 'private_results/', 'sha256', 'checkpoint'))


def test_feedback_note_has_complete_bilingual_scope():
    for name in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html','learning_log.html'):
        soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        notes = soup.select('#'+MARKER)
        assert len(notes) == 1
        pairs = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) >= 5
        zh = ' '.join(p['data-i18n-zh'] for p in pairs)
        en = ' '.join(p['data-i18n-en'] for p in pairs)
        assert all(x in zh for x in ('不是完整序列','不是有效加速','不否定全部学习初值'))
        assert all(x in en for x in ('not full sequences','not effective acceleration','does not refute all learned initializers'))
        assert notes[0].select_one('a')['href'].endswith(SUMMARY)
        if name.endswith('daily-progress.html'):
            assert notes[0]['data-date'] == '2026-10-06'
