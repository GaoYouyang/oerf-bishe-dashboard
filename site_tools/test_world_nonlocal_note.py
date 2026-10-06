import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = 'poolfire_world_nonlocal_2026-10-07_public_summary.json'
MARKER = 'poolfire-world-nonlocal-20261007'


def test_world_nonlocal_negative_is_not_whole_route_refutation():
    text = (ROOT / 'docs' / SUMMARY).read_text()
    data = json.loads(text)
    assert data['decision'] == 'FAIL_WORLD_NONLOCAL_RANGE_KERNEL2_CLOSED'
    assert data['cells'] == 99 and data['absolute_strata_total'] == 33
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 110
    for name in ('WorldRangeKernel2-Warm33', 'WorldGeometryKernel2-Warm33'):
        assert data['strict_matches'][name] == 0 and data['basic_strata'][name] == 33
    assert data['strict_matches']['NormalFactor-PCGLS34'] == 99
    assert data['reference_adequate'] and data['exact_recipe_closed']
    assert data['parameters_per_arm_fold'] == 2 and data['fit_cells_per_fold'] == 90
    assert data['held_teacher_reads'] == data['candidate_generation_CFD_truth_reads'] == 0
    assert data['shared_canonical_CSR_input'] and data['shared_orchestration_disclosed']
    assert data['standalone_primary_calls'] == data['standalone_geometry_control_calls'] == {'A': 35, 'AT': 35}
    for control in ('CGLS35', 'WorldGeometryKernel2-Warm33'):
        evidence = data['descriptive_controls'][control]
        assert evidence['all_four_harm'] == 99 and evidence['all_four_no_worse'] == 0
        assert all(value > 1 for value in evidence['median_ratios'].values())
    for name in ('broader_C_route_disproved', 'full_sequence_authorized', 'fresh_deployment_benchmark',
                 'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization',
                 'curved_ray_validated', 'real_bost', 'gpu_rental_authorized', 'main_goal_complete'):
        assert data[name] is False
    assert all(term not in text for term in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint', 'theta'))


def test_world_nonlocal_notes_and_figures_are_bilingual():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT / name).read_text(), 'html.parser')
        notes = soup.select('#' + MARKER)
        assert len(notes) == 1
        nodes = notes[0].select('[data-i18n-zh][data-i18n-en]')
        assert len(nodes) == 7
        zh = ' '.join(node['data-i18n-zh'] for node in nodes)
        en = ' '.join(node['data-i18n-en'] for node in nodes)
        assert all(term in zh for term in ('0/99', '33/33', '110/110', '没有速度或内存优势', '主目标仍未完成', '不否定全部算子学习'))
        assert all(term in en for term in ('0/99', '33/33', '110/110', 'no speed or memory advantage', 'main goal remains unmet', 'does not refute all operator learning'))
        base = (ROOT / name).parent
        assert base.joinpath(notes[0].select_one('a')['href']).resolve() == ROOT / 'docs' / SUMMARY
        image = notes[0].select_one('img')
        for lang in ('zh', 'en'):
            assert base.joinpath(image['data-i18n-src-' + lang]).resolve().is_file()
            assert image['data-i18n-alt-' + lang]
