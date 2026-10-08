import json
from pathlib import Path

SITE=Path(__file__).resolve().parents[1]
NAME='poolfire_world_phase_flux_2026-10-08_public_summary.json'
MARKER='poolfire-world-phase-flux-20261008'


def test_world_flux_is_a_learned_negative_not_an_engineering_success():
    data=json.loads((SITE/'docs'/NAME).read_text())
    assert data['decision']=='FAIL_SIGNED_WORLD_PHASE_FLUX3_WARM30_CLOSED'
    assert data['learned_initializer_tested'] and data['primary_recipe_closed']
    assert data['parameters_per_outer_model']=={'primary':3,'linear_control':2}
    assert data['independent_checks_passed']==data['independent_checks_total']==23
    assert data['reference_adequate'] and data['matched_cells']['CGLS128']==99
    for name in('WorldFlux3-Warm30','WorldFluxLinear2-Warm30','CGLS32'):
        assert data['matched_cells'][name]==0 and data['basic_strata'][name]==33
    assert data['matched_cells']['NormalFactor-PCGLS34']==99
    assert data['camera_counts']==[5,7,9] and data['sentinels']==99 and data['nodes']==11849
    assert data['fit_cells_each_fold']==90 and data['held_cells_each_fold']==9
    assert data['complete_trajectory_exclusion_for_fit'] and data['shared_canonical_CSR_and_structural_incidence_disclosed']
    assert all(not data[k] for k in('held_teacher_used_in_fit_or_prediction','CFD_truth_used_in_fit_or_prediction',
        'algorithm_breakthrough','paper_success','resource_speedup','external_generalization','real_bost',
        'curved_ray_validated','gpu_rental_authorized','full_sequence_tested','main_goal_complete',
        'fresh_deployment_benchmark','native_12_camera_tested','larger_network_rescue_authorized'))
    assert max(data['independent_maxima'].values())<1e-7
    assert data['primary_online_calls']=={'A':33,'AT':32}
    assert data['linear_online_calls']=={'A':32,'AT':32}
    assert data['BP_online_calls']=={'A':32,'AT':31}
    assert data['actual_main_audit_calls_each_mode']=={'A':23265,'AT':19800}
    assert data['additional_permutation_calls_each_mode']=={'A':99,'AT':132}
    assert data['additional_unwrapped_geometry_probe_A_each_mode']==6
    assert data['inherited_geometry_forward_equivalents']==35547 and data['geometry_setup_and_training_nonfree']
    assert data['descriptive_ratios_are_not_new_gates']
    assert abs(data['descriptive_error_ratios_to_finite_reference']['WorldFlux3-Warm30']['observation_rel_l2']['p90_higher']-6.237106954533023)<1e-8


def test_world_flux_bilingual_scope_and_privacy():
    for name in('index.html','operator-learning/index.html','operator-learning/daily-progress.html','learning_log.html'):
        text=(SITE/name).read_text();assert text.count(f'id="{MARKER}"')==1
        note=text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-weak-direction-attribution-20261008"')]
        assert note.count('data-i18n-zh=')>=6 and note.count('data-i18n-en=')>=6
        assert all(s in note for s in('23/23','99','0/99','33/33','33A+32AT','23265A+19800AT'))
        assert 'not a full sequence' in note
        assert 'cannot substitute for success of a learned initializer' in note
        assert NAME in note and 'data-i18n-alt-en=' in note
    payload=(SITE/'docs'/NAME).read_text()
    assert not any(s in payload for s in('/Users/','private_results/','private_data/','sha256','checkpoint'))
    current=json.loads((SITE/'operator-learning/current-evidence.json').read_text())
    assert current['latest_world_phase_flux']['matched_cells']==0
    assert not current['latest_world_phase_flux']['main_goal_complete']
