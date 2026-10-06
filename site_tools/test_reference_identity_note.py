import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = 'poolfire_reference_identity_2026-10-07_public_summary.json'
MARKER = 'poolfire-cost-identity-20261007'


def test_identity_audit_cannot_requalify_resource_experiment():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['benchmark_decision'] == 'INCONCLUSIVE_RESOURCE_VALIDITY'
    assert data['retrospective_audit_decision'] == 'REFERENCE_ENDPOINT_IDENTITY_MISMATCH_ONLY'
    assert data['audit_checks_passed'] == data['audit_checks_total'] == 49
    assert data['candidate_parent_identity_passes'] == data['candidate_parent_identity_total'] == 297
    assert data['reference_parent_identity_passes'] == 198
    assert data['reference_parent_identity_total'] == 297
    assert data['reference_failures_camera_count'] == 9
    assert all(value <= data['identity_threshold'] for value in data['candidate_maxima'].values())
    assert all(value > data['identity_threshold'] for value in data['reference_maxima'].values())
    assert data['within_method_repeats_bitwise_equal'] and data['original_classical_sentinel_result_preserved']
    for name in ('new_audit_A', 'new_audit_AT', 'new_audit_solvers', 'new_audit_fits', 'new_audit_CFD_truth_reads'):
        assert data[name] == 0
    for name in ('unique_ordering_causality_proven', 'raw_timing_performance_verdict_authorized',
                 'benchmark_retrospectively_requalified', 'trials_retried', 'learned_initializer',
                 'full_sequence', 'resource_speedup', 'algorithm_breakthrough', 'paper_success',
                 'external_generalization', 'real_bost', 'main_goal_complete'):
        assert data[name] is False
    assert all(term not in text for term in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_identity_notes_are_paired_and_keep_main_goal():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        nodes = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(nodes) == 4
        zh = ' '.join(node['data-i18n-zh'] for node in nodes)
        en = ' '.join(node['data-i18n-en'] for node in nodes)
        assert all(term in zh for term in ('九相机', '不能判速度或内存胜负', '49/49', '主目标仍未完成', '尚未证明排序是唯一原因'))
        assert all(term in en for term in ('nine-camera', 'no speed or memory winner', '49/49', 'main goal remains unmet', 'not been proved the unique cause'))
        link = notes[0].select_one('a')['href']
        assert (ROOT / name).parent.joinpath(link).resolve() == ROOT / 'docs' / SUMMARY
