import json
from pathlib import Path

SITE=Path(__file__).resolve().parents[1]
NAME='poolfire_causal_innovation_history_2026-10-08_public_summary.json'
MARKER='poolfire-causal-innovation-history-20261008'


def test_causal_history_full_sequence_does_not_count_bootstrap_as_learning_success():
    d=json.loads((SITE/'docs'/NAME).read_text())
    assert d['decision']=='FAIL_CAUSAL_INNOVATION3_WARM30_CLOSED'
    assert d['independent_checks_passed']==d['independent_checks_total']==28
    assert d['cells']==3333 and d['post_bootstrap_cells']==3300 and d['complete_strata']==33
    assert d['trajectories']==11 and d['frames_per_trajectory']==101 and d['camera_counts']==[5,7,9]
    assert d['reference_adequate'] and d['matched_cells']['DirectRidge-Warm34']==3333
    for m in('CausalCarry-Warm30','CausalLinear2-Warm30','CausalInnovation3-Warm30'):
        assert d['matched_cells'][m]==33 and d['absolute_strata'][m]==33
        assert d['post_evaluation_descriptive_diagnostics'][m]['bootstrap_joint_matches']==33
        assert d['post_evaluation_descriptive_diagnostics'][m]['post_bootstrap_joint_matches']==0
        assert d['post_evaluation_descriptive_diagnostics'][m]['post_bootstrap_individual_metric_matches']['observation_rel_l2']==0
    assert d['teacher_forcing_used_in_training'] and not d['teacher_forcing_used_in_held_rollout']
    assert d['future_observation_reads_for_prediction']==d['held_teacher_reads_for_prediction']==0
    assert d['shared_primary_parameters_per_fold']==3 and d['shared_linear_parameters_per_fold']==2
    assert d['new_fits_each_mode']==22 and d['train_cells_per_fold']==3000
    assert max(d['independent_maxima'].values())<1e-7


def test_causal_history_costs_and_failure_are_not_erased():
    d=json.loads((SITE/'docs'/NAME).read_text())
    assert d['full101_query_sequence_exact_calls_each_arm']=={'A':3228,'AT':3228}
    assert d['cold31_full101_query_sequence_exact_calls']=={'A':3131,'AT':3131}
    assert d['subsequent_query_exact_calls_each_arm']=={'A':31,'AT':31}
    assert d['bootstrap_pairs_per_stream_per_arm']==128
    assert d['discarded_first_cell_scoring_A']==19 and d['original_scoring_failure_preserved']
    assert d['complete_audit_main_calls_formal_including_preserved_failure']=={'A':892438,'AT':733425}
    assert d['complete_audit_main_calls_independent']=={'A':892419,'AT':733425}
    assert d['geometry_and_teacher_setup_nonfree'] and d['audit_shares_bootstrap_between_causal_arms']
    assert d['scoring_layout_repaired_only_after_prediction_seal'] and not d['formal_predictions_repeated']
    assert d['learned_initializer_tested'] and d['whole_sequence_evaluated'] and d['primary_recipe_closed']
    assert all(not d[k] for k in('algorithm_breakthrough','paper_success','resource_speedup','external_generalization',
        'real_bost','curved_ray_validated','native_12_camera_tested','history_loss_depth_parameter_rescue_authorized',
        'matched_exact_call_reduction_established','fresh_deployment_benchmark','fresh_wall_RSS_tested','main_goal_complete'))


def test_causal_history_bilingual_note_and_privacy():
    for name in('index.html','operator-learning/index.html','operator-learning/daily-progress.html','learning_log.html'):
        text=(SITE/name).read_text();assert text.count(f'id="{MARKER}"')==1
        note=text[text.index(f'id="{MARKER}"'):text.index('id="poolfire-native-axis-tensor-inverse-20261008"')]
        assert note.count('data-i18n-zh=')>=6 and note.count('data-i18n-en=')>=6
        assert all(s in note for s in('28/28','3333','33/3333','0/3300','957','3228A+3228AT','3131A+3131AT'))
        assert 'These post-evaluation descriptive error ratios are not percentages' in note
        assert 'The learned C-route goal remains incomplete' in note
        assert NAME in note and 'data-i18n-alt-en=' in note
    payload=(SITE/'docs'/NAME).read_text()
    assert not any(s in payload for s in('/Users/','private_results/','private_data/','sha256','checkpoint'))
    current=json.loads((SITE/'operator-learning/current-evidence.json').read_text())
    assert current['latest_causal_innovation_history']['post_bootstrap_matched_cells']==0
    assert not current['latest_causal_innovation_history']['main_goal_complete']
    assert current['latest_native_axis_tensor_inverse']['strict_probe_passes']==0
