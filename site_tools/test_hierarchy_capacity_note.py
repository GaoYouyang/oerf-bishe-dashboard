import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = 'poolfire_hierarchy_capacity_2026-10-07_public_summary.json'
MARKER = 'poolfire-hierarchy-capacity-20261007'


def test_negative_capacity_not_whole_direction_failure():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'FAIL_FIXED_HIERARCHY_INVERSE_FACTOR_SENTINEL_CLOSED'
    assert data['primary_strict_matches'] == data['primary_absolute_strata'] == 0
    assert set(data['primary_individual_matches'].values()) == {0}
    assert data['exact_factor_strict_matches'] == data['reference_strict_matches'] == 99
    assert data['exact_factor_absolute_strata'] == 33
    assert data['independent_checks_passed'] == data['independent_checks_total']
    assert data['standalone_calls'] == {'A': 34, 'AT': 34, 'hierarchy_scans': 68}
    assert data['full_normal_and_inverse_setup_nonfree'] and data['fixed_recipe_closed']
    assert data['full_factor_positive_witness_retained']
    for control in ('CGLS35', 'PCGLS35', 'KernelBank90-Warm33'):
        assert data['descriptive_controls'][control]['all_four_harm'] == 99
    for key in ('candidate_generation_reads_CFD_truth', 'all_hierarchical_methods_disproved',
                'learned_initializer', 'full_sequence_authorized', 'fresh_deployment_benchmark',
                'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization',
                'real_bost', 'neural_training_authorized', 'gpu_rental_authorized', 'main_goal_complete'):
        assert data[key] is False
    assert all(word not in text for word in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_bilingual_scope_and_failed_accuracy_storage():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        nodes = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(nodes) == 7
        zh = ' '.join(n['data-i18n-zh'] for n in nodes)
        en = ' '.join(n['data-i18n-en'] for n in nodes)
        assert all(t in zh for t in ('0/99', '0/33', '99/99', '三帧训练哨兵', '不是完整序列', '68次层次扫描', '精度失败', '主目标仍未完成'))
        assert all(t in en for t in ('0/99', '0/33', '99/99', 'three-frame train sentinels', 'not full sequences', '68 hierarchy scans', 'accuracy fails', 'main goal remains unmet'))
        image = notes[0].select_one('img')
        assert image['data-i18n-alt-zh'] and image['data-i18n-alt-en']
        assert (image['width'], image['height']) == ('1320', '540')
        assert (ROOT / name).parent.joinpath(image['src']).resolve().is_file()
        assert notes[0].select_one('a')['href'].endswith(SUMMARY)
