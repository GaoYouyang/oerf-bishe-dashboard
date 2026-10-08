import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
NAME = 'poolfire_primal_balanced_coarse_2026-10-09_public_summary.json'
MARKER = 'poolfire-primal-balanced-coarse-20261009'


def test_valid_classical_failure_and_descriptive_benefits():
    d = json.loads((SITE / 'docs' / NAME).read_text())
    assert d['cells'] == 99 and d['trajectories'] == 11 and d['three_frame_strata'] == 33
    assert d['reference_adequate'] and d['independent_checks_passed'] == d['independent_checks_total'] == 54
    assert all(d['matched_cells'][m] == 0 and d['absolute_sampled_strata'][m] == 33 for m in d['new_methods'])
    assert d['matched_cells']['CGLS128'] == d['matched_cells']['DirectRidge-Warm34'] == 99
    assert d['descriptive_comparator']['CGLS31']['lower_error_counts'] == [30, 30, 24, 79]
    assert d['descriptive_comparator']['Coarse-JacobiPCGLS28']['strict_all_four_lower_cells'] == 90
    assert d['descriptive_comparison_is_not_a_new_gate']
    assert max(d['independent_maxima'].values()) < 1e-10


def test_nonfree_cost_and_evidence_scope():
    d = json.loads((SITE / 'docs' / NAME).read_text())
    cost = d['call_ledger']
    assert cost['standalone_each_new_arm'] == {'A': 29, 'AT': 29}
    assert cost['online_all_three_arms_each_mode'] == {'A': 8613, 'AT': 8613}
    assert cost['callback_counters_have_overlapping_views_not_additive_work']
    assert cost['formal_prior_cost_reconstructed_from_frozen_counted_source_and_completed_models']
    assert d['fixed_recipe_closed'] and d['no_scientific_formula_or_gate_changed']
    assert d['metadata_failure_preserved_and_geometry_resumed_without_formal_refactorization']
    assert d['twelve_cameras_tested_only_on_manufactured_matrices']
    assert all(d[k] is False for k in ('all_balancing_methods_refuted', 'new_predictor_authorized',
        'algorithm_breakthrough', 'resource_speedup', 'paper_success', 'fresh_wall_RSS_tested',
        'complete_sequences_tested', 'external_generalization', 'real_bost', 'curved_ray_validated', 'main_goal_complete'))
    assert not any(s in (SITE / 'docs' / NAME).read_text() for s in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))


def test_bilingual_note_and_prior_results_preserved():
    for file in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / file).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-finite-loss-decomposition-20261009"')]
        assert note.count('data-i18n-zh=') == note.count('data-i18n-en=') == 7
        assert all(s in note for s in ('0/99', '33/33', '30/30/24/79', '90/99', '54/54', '29A+29AT', NAME))
        assert 'not learned success' in note and 'not complete sequences' in note
        assert 'not an accuracy-preserving call reduction' in note
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_primal_balanced_coarse']['matched_cells'] == 0
    assert current['latest_finite_loss_decomposition']['actual_harm'] == 98
