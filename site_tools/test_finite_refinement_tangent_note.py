import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
NAME = 'poolfire_finite_refinement_tangent_2026-10-09_public_summary.json'
MARKER = 'poolfire-finite-refinement-tangent-20261009'


def test_local_and_finite_evidence_are_not_conflated():
    d = json.loads((SITE / 'docs' / NAME).read_text())
    assert d['cells'] == 99 and d['three_frame_strata'] == 33 and d['trajectories'] == 11
    assert d['frames'] == [0, 50, 100] and d['camera_counts'] == [5, 7, 9]
    assert d['seed_locally_improving'] == 99
    assert d['final_locally_improving'] == 92 and d['final_locally_worsening'] == 7
    assert d['final_locally_neutral'] == 0
    assert [d['by_camera'][str(c)]['final_improving'] for c in (5, 7, 9)] == [31, 32, 29]
    assert [d['by_camera'][str(c)]['final_worsening'] for c in (5, 7, 9)] == [2, 1, 4]
    assert d['inherited_full_sequence_evidence']['strict_joint_matched'] == 0
    assert d['inherited_full_sequence_evidence']['final_teacher_field_worsened'] == 3304
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 22
    assert max(d['independent_maxima'].values()) < 1e-10
    assert d['tangent_remainder']['p50'] < d['tangent_remainder']['p90_higher'] < d['tangent_remainder']['max']


def test_nonfree_attribution_and_closed_model_boundaries():
    d = json.loads((SITE / 'docs' / NAME).read_text())
    assert d['call_ledger']['formal'] == {'A': 6120, 'AT': 5916}
    assert d['call_ledger']['independent'] == {'A': 6120, 'AT': 6018}
    assert d['call_ledger']['new_fits'] == d['call_ledger']['new_CFD_reads'] == 0
    assert d['inherited_cost_nonfree'] and d['old_recipe_still_closed']
    assert d['all_derivatives_sealed_before_teacher_scoring']
    assert all(not d[k] for k in ('amplitude_search_performed', 'safe_shared_amplitude_established',
        'causal_root_cause_established', 'full_sequence_derivatives_tested', 'new_predictor_authorized',
        'algorithm_breakthrough', 'resource_speedup', 'paper_success', 'fresh_wall_RSS_tested',
        'external_generalization', 'real_bost', 'curved_ray_validated', 'main_goal_complete'))
    assert not any(s in (SITE / 'docs' / NAME).read_text() for s in ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))


def test_paired_note_prior_evidence_and_figure():
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-seed-refinement-attribution-20261009"')]
        assert note.count('data-i18n-zh=') >= 8 and note.count('data-i18n-en=') >= 8
        assert all(s in note for s in ('99/99', '92/99', '7/99', '31/2', '32/1', '29/4', '22/22', '0/3333', '6120A+5916AT', '6120A+6018AT'))
        assert 'not a new algorithm' in note and 'cannot reopen' in note and 'without amplitude search' in note
        assert NAME in note and 'data-i18n-alt-en=' in note
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_finite_refinement_tangent']['final_locally_improving'] == 92
    assert current['latest_seed_refinement_attribution']['seed_improved_final_worsened'] == 3304
    assert current['latest_signed_prolongation_learning']['matched_cells'] == 0
    assert (SITE / 'assets/poolfire_finite_refinement_tangent_2026-10-09.png').is_file()
