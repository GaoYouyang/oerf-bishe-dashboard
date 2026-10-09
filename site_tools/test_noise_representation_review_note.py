import json
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_noise_representation_review_2026-10-09'
MARKER = 'poolfire-noise-representation-review-20261009'
PAGES = ('index.html', 'operator-learning/index.html',
         'operator-learning/daily-progress.html', 'learning_log.html')


def summary():
    return json.loads((ROOT / f'docs/{NAME}_public_summary.json').read_text())


def test_noise_is_not_experimental_or_complete_sequence():
    data = summary()
    assert data['scope']['clean_anchors'] == 99
    assert data['scope']['camera_counts'] == [5, 7, 9]
    assert not data['scope']['new_validation_test_opened']
    assert not data['scope']['paired_experiment']
    noise = data['controlled_noise']
    assert noise['relative_observation_l2_levels'] == [0.001, 0.01]
    assert noise['cells'] == 198 and noise['sampled_strata'] == 66
    assert noise['direct_initializer_matched_cells'] == [99, 2]
    assert noise['direct_initializer_matched_sampled_strata'] == [33, 0]
    assert noise['all_classical_arms_absolute_sampled_strata_passed'] == 66
    assert noise['noise_is_controlled_not_measured']
    assert noise['gaussian_direction_globally_normalized']
    assert noise['same_direction_at_both_levels']
    assert not noise['full_sequence_tested']


def test_learned_failure_and_teacher_attribution_are_distinct():
    data = summary()
    learner = data['learned_sentinel']
    assert learner['complete_trajectory_excluded_from_each_fit']
    assert learner['all_fits_sealed_before_held_prediction']
    assert learner['all_predictions_sealed_before_truth_scoring']
    assert learner['matched_cells'] == learner['linear_control_matched_cells'] == 0
    assert learner['cells'] == 99 and learner['absolute_sampled_strata_passed'] == 33
    assert learner['logical_calls_per_query'] == {'A': 34, 'AT': 33}
    assert learner['fixed_recipe_closed'] and not learner['complete_sequence_tested']
    assert 'original binding omission disclosed' in learner['inherited_helper_closure']
    audit = data['seed_fit_attribution']
    assert 0.8858 < audit['outside_feature_span_fraction_sample_median'] < 0.8859
    assert 0.8477 < audit['outside_feature_span_fraction_sample_mean'] < 0.8478
    assert 'teacher-visible' in audit['role']
    assert 'teacher-field' in audit['objective']
    assert audit['new_refinement_iterations'] == audit['new_trainable_parameter_fits'] == 0
    for key in ('final_CGLS_endpoint_lower_bound', 'actual_CFD_error_lower_bound',
                'irrecoverability_proof', 'original_closed_recipe_reopened'):
        assert not audit[key]
    assert not data['interpretation']['all_attention_methods_ruled_out']
    assert not data['interpretation']['C_route_ruled_out']
    assert all(value is False for value in data['claims'].values())


def test_batched_cost_does_not_overwrite_old_workload():
    data = summary()['classical_full_sequence_context']
    assert data['matched_cells_each_arm'] == data['cells'] == 3333
    assert data['complete_strata'] == 33
    assert not data['stable_wall_win'] and not data['joint_wall_RSS_win']
    assert data['older_scalar_workload_evidence_preserved']
    evidence = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    latest = evidence['latest_research_synthesis']
    assert latest['classical_matched_cells'] == 3333
    assert latest['learned_matched_cells'] == 0 and latest['learned_anchor_cells'] == 99
    assert 'classical clean complete sequences' in latest['display_counts_scope']
    assert evidence['latest_full_temporal_direct_cost']['faster_all_pairs']
    assert not evidence['latest_batched_classical_cost_review']['stable_wall_win']
    for page in PAGES[:2]:
        text = (ROOT / page).read_text()
        assert text.index('const synthesis = evidence.latest_research_synthesis;') < text.index(
            'const fullTemporal = evidence.latest_full_temporal_direct_cost;')
        assert 'Latest: faster full sequences, but expensive memory' not in text
        assert 'poolfire-full-temporal-direct-cost-20261007' in text


class SectionParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.nodes = []
        self.opening = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id') == MARKER:
            assert self.opening is None
            self.opening = (tag, attrs)
            self.active = True
        if self.active:
            self.nodes.append((tag, attrs))

    def handle_endtag(self, tag):
        if self.active and tag == self.opening[0]:
            self.active = False


def test_single_combined_bilingual_note_and_links():
    for page in PAGES:
        text = (ROOT / page).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        assert text.index(MARKER) < text.index('id="poolfire-batched-resource-review-20261009"')
        parser = SectionParser()
        parser.feed(text)
        pairs = [attrs for _, attrs in parser.nodes if 'data-i18n-zh' in attrs]
        assert len(pairs) == 6
        assert all(attrs['data-i18n-zh'] and attrs['data-i18n-en'] for attrs in pairs)
        for tag, attrs in parser.nodes:
            if tag == 'a':
                target = attrs['href'].split('?')[0]
                assert (ROOT / page).parent.joinpath(target).resolve().is_file()
        if 'daily-progress' in page:
            assert parser.opening[0] == 'article'
            assert parser.opening[1]['data-date'] == '2026-10-09'
    for suffix in ('public_summary.json', 'plain_language.md'):
        text = (ROOT / f'docs/{NAME}_{suffix}').read_text()
        assert not any(token in text for token in (
            '/Users/', '/Volumes/', 'private_results/', 'private_data/', 'sha256', 'checkpoint'))
