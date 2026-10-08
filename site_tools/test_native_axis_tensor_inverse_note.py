import json
from pathlib import Path

SITE=Path(__file__).resolve().parents[1]
NAME='poolfire_native_axis_tensor_inverse_2026-10-08_public_summary.json'
MARKER='poolfire-native-axis-tensor-inverse-20261008'


def test_native_axis_tensor_result_is_a_necessary_negative_only():
    data=json.loads((SITE/'docs'/NAME).read_text())
    assert data['decision']=='FAIL_NATIVE_AXIS_TENSOR_INVERSE16_CLOSED'
    assert data['independent_checks_passed']==data['independent_checks_total']==18
    assert data['camera_counts']==[5,7,9] and data['nodes']==11849 and data['shape_zyx']==[41,17,17]
    assert data['bond_ranks']==[16,16] and data['matrix_rank_is_not_TT_rank']
    assert data['strict_probe_passes']==data['passed_camera_counts']==0 and data['probe_count']==36
    assert data['primary_recipe_closed'] and data['primary_error_is_not_a_reconstruction_error']
    assert data['field_action_and_image_action_error_gate']==.01
    for c in('5','7','9'):
        assert not data['camera_pass'][c]
        assert data['geometry'][c]['packed_bytes']==844160
        assert data['geometry'][c]['storage_fraction']<.001
        assert data['geometry'][c]['stationarity']<1e-8
        for metric in('field_error','image_error'):
            assert data['strata'][c]['NativeAxisMPO16'][metric]>100
            assert data['strata'][c]['NativeAxisMPO16'][metric]<data['strata'][c]['NativePhysicalDiagonal'][metric]
    assert max(data['independent_maxima'].values())<1e-7
    assert data['actual_main_audit_calls_each_mode']=={'A':252,'AT':72}
    assert data['additional_permutation_calls_each_mode']=={'A':9,'AT':9}
    assert data['full_inverse_rhs_each_mode']==data['inherited_geometry_forward_equivalents']==35547
    assert data['actual_observation_CFD_teacher_reads']==data['new_fits']==0
    assert data['geometry_setup_and_target_construction_nonfree'] and data['packed_payload_is_not_RSS']
    assert data['represented_actions_compared'] and data['shared_canonical_CSR_and_RNG_disclosed']
    assert all(not data[k] for k in('core_gauges_compared','native_12_camera_tested','rank_or_axis_order_rescue_authorized',
        'learned_initializer_tested','algorithm_breakthrough','paper_success','resource_speedup','external_generalization',
        'real_bost','curved_ray_validated','gpu_rental_authorized','full_sequence_tested','main_goal_complete',
        'fresh_deployment_benchmark','fresh_wall_RSS_tested','uniform_operator_accuracy_tested'))


def test_native_axis_note_retains_bilingual_scope_and_privacy():
    for name in('index.html','operator-learning/index.html','operator-learning/daily-progress.html','learning_log.html'):
        text=(SITE/name).read_text();assert text.count(f'id="{MARKER}"')==1
        note=text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-world-phase-flux-20261008"')]
        assert note.count('data-i18n-zh=')>=6 and note.count('data-i18n-en=')>=6
        assert all(s in note for s in('18/18','36','0/36','844160','252A+72AT','35547'))
        assert 'not percentages or CFD reconstruction errors' in note
        assert 'The learned C-route goal remains incomplete' in note
        assert NAME in note and 'data-i18n-alt-en=' in note
    payload=(SITE/'docs'/NAME).read_text()
    assert not any(s in payload for s in('/Users/','private_results/','private_data/','sha256','checkpoint'))
    current=json.loads((SITE/'operator-learning/current-evidence.json').read_text())
    assert current['latest_native_axis_tensor_inverse']['strict_probe_passes']==0
    assert not current['latest_native_axis_tensor_inverse']['main_goal_complete']
    assert current['latest_world_phase_flux']['matched_cells']==0
