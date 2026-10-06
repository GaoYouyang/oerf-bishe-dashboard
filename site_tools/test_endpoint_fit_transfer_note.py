import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-endpoint-fit-transfer-20261006'
SUMMARY = 'poolfire_endpoint_fit_transfer_2026-10-06_public_summary.json'


def test_final_fit_diagnosis_is_not_capacity_or_speed_success():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'NO_MATERIAL_FINAL_ENDPOINT_FIT_GAIN'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 17
    assert data['new_fold_train_endpoint_evaluations_each_implementation'] == 990
    assert data['unique_queries'] == data['held_endpoints_reused'] == 99
    assert data['fold_train_evaluations_repeat_queries'] and data['held_endpoints_rerun'] == 0
    assert data['final_iteration_evaluated_not_inferred_from_preupdate_history']
    assert data['fold_gain_summaries']['train']['positive_folds'] == 11
    assert data['fold_gain_summaries']['held']['positive_folds'] == 11
    assert .054 < data['fold_gain_summaries']['train']['median_percent'] < .055
    assert .061 < data['fold_gain_summaries']['held']['median_percent'] < .062
    assert data['source_recipe_stays_closed']
    assert data['original_strict_matches_unchanged'] == 0 and data['original_basic_strata_unchanged'] == 33
    assert data['new_fits'] == data['new_teacher_construction'] == data['new_CFD_reads'] == 0
    assert data['new_calls_each_implementation'] == {'A': 92103, 'AT': 90123}
    for key in ('large_training_gain_held_collapse_observed', 'capacity_impossibility_proven',
                'unique_cause_proven', 'more_training_authorized', 'full_sequence_authorized',
                'fresh_deployment_benchmark', 'algorithm_breakthrough', 'paper_success',
                'resource_speedup', 'external_generalization', 'curved_ray_validated', 'real_bost'):
        assert data[key] is False
    assert all(value not in text for value in ('/Users/', 'private_data/', 'private_results/', 'sha256', 'checkpoint'))


def test_final_fit_notes_keep_bilingual_scope_and_distinct_units():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        pairs = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) == 4
        zh = ' '.join(p['data-i18n-zh'] for p in pairs)
        en = ' '.join(p['data-i18n-en'] for p in pairs)
        assert all(term in zh for term in ('不是新增990个样本', '不是CFD场误差', '不是有效加速', '不否定全部学习初值', '不授权增加训练'))
        assert all(term in en for term in ('not 990 new samples', 'not CFD field errors', 'not effective acceleration', 'does not refute all learned initializers', 'does not authorize more training'))
        assert notes[0].select_one('a')['href'].endswith(SUMMARY)
