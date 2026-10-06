import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-normal-factor-classical-20261007'
SUMMARY = 'poolfire_normal_factor_classical_2026-10-07_public_summary.json'


def test_classical_match_does_not_complete_learned_goal():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'PASS_NORMAL_FACTOR_CLASSICAL_SENTINEL_ONLY'
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 24
    assert data['strict_matches']['NormalFactor-PCGLS34'] == data['strict_matches']['CGLS128'] == 99
    assert data['basic_strata']['NormalFactor-PCGLS34'] == 33
    assert set(data['primary_individual_matches'].values()) == {99}
    for name in ('CGLS35', 'PCGLS35', 'KernelBank90-Warm33'):
        assert data['strict_matches'][name] == 0
        assert data['descriptive_controls'][name]['all_four_no_worse'] == 99
    assert data['standalone_calls'] == {'A': 34, 'AT': 34, 'factor_solves': 34}
    assert data['reference_standalone_calls'] == {'A': 128, 'AT': 128}
    assert data['factor_solves_per_audit'] == data['preconditioner_actions_per_audit'] + 9 == 4506
    assert min(data['factor_storage_bytes_by_camera_count'].values()) > 400_000_000
    assert data['setup_and_factor_solves_nonfree'] and data['inherited_control_training_and_labels_nonfree']
    assert data['new_fits'] == data['new_teacher_construction'] == 0
    for key in ('candidate_generation_reads_CFD_truth', 'learned_initializer',
                'original_learning_recipes_reopened', 'full_sequence_authorized',
                'arbitrary_pose_validation', 'fresh_deployment_benchmark',
                'algorithm_breakthrough', 'paper_success', 'resource_speedup',
                'external_generalization', 'curved_ray_validated', 'real_bost',
                'neural_training_authorized', 'gpu_rental_authorized', 'main_goal_complete'):
        assert data[key] is False
    assert all(word not in text for word in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_bilingual_note_and_nonfree_factor_warning():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        nodes = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(nodes) == 7
        zh = ' '.join(node['data-i18n-zh'] for node in nodes)
        en = ' '.join(node['data-i18n-en'] for node in nodes)
        assert all(term in zh for term in ('99/99', '33/33', '三帧训练哨兵', '不是完整序列', '34次因子求解', '学习算法突破', '主目标仍未完成'))
        assert all(term in en for term in ('99/99', '33/33', 'three-frame train sentinels', 'not full sequences', '34 factor solves', 'main goal remains unmet'))
        image = notes[0].select_one('img')
        assert image['data-i18n-alt-zh'] and image['data-i18n-alt-en']
        assert (image['width'], image['height']) == ('1320', '540')
        assert (ROOT / name).parent.joinpath(image['src']).resolve().is_file()
        assert notes[0].select_one('a')['href'].endswith(SUMMARY)
