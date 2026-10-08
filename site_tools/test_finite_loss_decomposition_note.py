import json
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
NAME = 'poolfire_finite_loss_decomposition_2026-10-09'
MARKER = 'poolfire-finite-loss-decomposition-20261009'


def test_counts_and_squared_distance_identity():
    d = json.loads((SITE / 'docs' / (NAME + '_public_summary.json')).read_text())
    assert d['cells'] == 99 and d['three_frame_strata'] == 33 and d['trajectories'] == 11
    assert [d[k] for k in ('local_harm', 'local_benefit_quadratic_flip',
        'linearized_benefit_remainder_flip', 'actual_benefit')] == [7, 59, 32, 1]
    assert d['actual_harm'] == 98 and sum(d['sign_pairs'].values()) == 99
    for count in (5, 7, 9):
        row = d['by_camera'][str(count)]
        assert row['cells'] == sum(row[k] for k in ('local_harm', 'quadratic_flip', 'remainder_flip', 'actual_benefit')) == 33
    assert [d['by_camera'][str(c)]['quadratic_flip'] for c in (5, 7, 9)] == [12, 21, 26]
    assert [d['by_camera'][str(c)]['remainder_flip'] for c in (5, 7, 9)] == [18, 11, 3]
    terms = d['all_mean_terms']
    assert abs(sum(terms[n] for n in ('L', 'Q', 'C', 'R')) - terms['actual_change']) < 1e-14
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 13
    assert max(d['independent_maxima'].values()) < 1e-14


def test_boundaries_and_zero_new_queries():
    d = json.loads((SITE / 'docs' / (NAME + '_public_summary.json')).read_text())
    assert all(v == 0 for v in d['call_ledger'].values())
    assert d['inherited_cost_nonfree'] and d['old_recipe_still_closed']
    assert d['unit_endpoint_is_old_sealed_endpoint']
    assert d['inherited_full_sequence_evidence']['strict_joint_matched'] == 0
    for key in ('amplitude_search_performed', 'causal_root_cause_established', 'full_sequence_decomposition_tested',
            'new_predictor_authorized', 'algorithm_breakthrough', 'resource_speedup', 'paper_success',
            'fresh_wall_RSS_tested', 'external_generalization', 'real_bost', 'curved_ray_validated', 'main_goal_complete'):
        assert d[key] is False
    assert not any(s in (SITE / 'docs' / (NAME + '_public_summary.json')).read_text() for s in
        ('/Users/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))


def test_bilingual_note_preserves_prior_evidence():
    for file in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (SITE / file).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        note = text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-finite-refinement-tangent-20261009"')]
        assert note.count('data-i18n-zh=') == note.count('data-i18n-en=') == 6
        assert all(s in note for s in ('59/99', '32/99', '7/99', '98/99', '12/21/26', '18/11/3', '13/13', '0/3333'))
        assert 'not a causal failure proof' in note and 'not CFD truth error' in note
        assert 'without amplitude search' in note and NAME in note and 'data-i18n-alt-en=' in note
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert current['latest_finite_loss_decomposition']['actual_harm'] == 98
    assert current['latest_finite_refinement_tangent']['final_locally_improving'] == 92
    assert (SITE / 'assets' / (NAME + '.png')).is_file()
