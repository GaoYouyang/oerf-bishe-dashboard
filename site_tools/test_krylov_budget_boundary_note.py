import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def test_theory_note_preserves_algorithm_and_scope_boundaries():
    path = ROOT/'docs/krylov_budget_boundary_2026-10-06_public_note.json'
    record = json.loads(path.read_text())
    assert record['category'] == 'research_note'
    assert record['finite_checks_are_not_universal_proof']
    assert record['exact_illustrative_instances'] == 36
    assert record['independent_exact_implementations'] == 2
    caveat = record['same_space_field_tradeoff_caveat']
    assert caveat['field_error_norm_ratio'] < 1 < caveat['observed_residual_norm_ratio'] < 1.01
    assert caveat['visible_algebraic_truth']
    assert not caveat['deployable_predictor'] and not caveat['fixed_refinement_path_success']
    assert record['former_recipe_verdicts_unchanged']
    assert record['new_CFD_reads'] == record['new_CFD_scoring'] == record['new_training'] == 0
    for key in ('algorithm_breakthrough', 'matched_accuracy_speedup', 'paper_success',
                'resource_speedup', 'external_generalization', 'real_bost'):
        assert record[key] is False
    for marker in ('/Users/', 'private_results/', 'private_data/', 'sha256'):
        assert marker not in path.read_text()


def test_bilingual_theory_note_is_research_not_algorithm_pass():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        soup = BeautifulSoup((ROOT/name).read_text(), 'html.parser')
        sections = soup.select('#krylov-budget-boundary-note-20261006')
        assert len(sections) == 1
        section = sections[0]
        pairs = section.select('[data-i18n-zh][data-i18n-en]')
        assert len(pairs) >= 4 and all(n['data-i18n-zh'] and n['data-i18n-en'] for n in pairs)
        if 'daily-progress' in name:
            assert section['data-categories'] == 'research decision'
        assert section.find('a')['href'].endswith('krylov_budget_boundary_2026-10-06_public_note.json')
        assert '不是预测器' in section.get_text() and '不是首创算法' in section.get_text()
