import json
from html.parser import HTMLParser
from pathlib import Path


SITE = Path(__file__).resolve().parents[1]
NAME = 'poolfire_batched_resource_review_2026-10-09'
MARKER = 'poolfire-batched-resource-review-20261009'
PAGES = ('index.html', 'operator-learning/index.html',
         'operator-learning/daily-progress.html', 'learning_log.html')


def summary():
    return json.loads((SITE / f'docs/{NAME}_public_summary.json').read_text())


def test_complete_accuracy_and_execution_boundary():
    data = summary()
    assert data['scope']['cells'] == 3333
    assert data['scope']['complete_strata'] == 33
    assert data['scope']['camera_counts'] == [5, 7, 9]
    assert not data['scope']['new_validation_test_opened']
    assert not data['scope']['paired_experiment']
    for arm in ('CGLS128', 'DirectRidge-Warm34'):
        assert data['accuracy']['matched_cells'][arm] == 3333
        assert data['accuracy']['absolute_strata'][arm] == 33
        assert data['accuracy']['complete_matched_strata'][arm] == 33
        assert data['accuracy']['repeats_bitwise'][arm]
    execution = data['execution']
    assert execution['both_arms_same_policy']
    assert execution['factor_setup_and_output_included']
    assert execution['historical_geometry_and_input_generation_excluded_nonfree']
    assert not execution['block_Krylov']
    assert not execution['streaming_latency']
    assert not execution['cross_run_speedup_causally_isolated']


def test_resource_judgment_and_calls_are_separate():
    data = summary()
    resource = data['resource']
    assert resource['decision'] == 'NO_JOINT_FULL_TEMPORAL_CACHED_DIRECT_RESOURCE_ADVANTAGE'
    assert not resource['faster_all_pairs'] and not resource['lower_rss_all_pairs']
    ratios = resource['paired_ratios']
    assert len(ratios) == 3
    assert sum(pair['wall_ratio'] < 1 for pair in ratios) == 1
    assert all(pair['rss_ratio'] > 5 for pair in ratios)
    assert data['logical_calls_per_query']['CGLS128'] == {'A': 128, 'AT': 128}
    assert data['logical_calls_per_query']['DirectRidge-Warm34'] == {'A': 35, 'AT': 35, 'factor_RHS': 1}
    assert data['interpretation']['old_workload_results_preserved']
    assert data['interpretation']['fewer_calls_do_not_imply_wall_speedup']
    assert all(value is False for value in data['claims'].values())


def test_actual_error_audit_is_narrow_and_privacy_safe():
    data = summary()
    audit = data['previous_CFD_objective_review']
    assert audit['cells'] == audit['each_metric_teacher_improves'] == 99
    assert audit['each_metric_task_harms'] == 0
    assert 'not full-sequence or LOTO' in audit['role']
    assert all(0 < gain < 1 for gain in audit['field_gain_median_percent'])
    assert not audit['weak_certificate_is_noise_or_irreducibility_proof']
    for suffix in ('public_summary.json', 'plain_language.md'):
        raw = (SITE / f'docs/{NAME}_{suffix}').read_text()
        assert not any(token in raw for token in ('/Users/', '/Volumes/', 'private_results/',
                                                 'private_data/', 'sha256', 'checkpoint'))


class ReviewParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.nodes = []
        self.start = None

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if attributes.get('id') == MARKER:
            assert not self.active and self.start is None
            self.active = True
            self.start = (tag, attributes)
        if self.active:
            self.nodes.append((tag, attributes))

    def handle_endtag(self, tag):
        if self.active and tag == self.start[0]:
            self.active = False


def test_paired_bilingual_sections_links_and_preserved_history():
    for name in PAGES:
        text = (SITE / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        assert text.count('id="poolfire-global-review-20261009"') == 1
        assert text.index(MARKER) < text.index('poolfire-global-review-20261009')
        parser = ReviewParser()
        parser.feed(text)
        pairs = [attrs for _, attrs in parser.nodes if 'data-i18n-zh' in attrs]
        assert len(pairs) == 5
        assert all(attrs['data-i18n-zh'] and attrs['data-i18n-en'] for attrs in pairs)
        for tag, attrs in parser.nodes:
            if tag in ('a', 'img'):
                target = (SITE / name).parent / attrs['href' if tag == 'a' else 'src']
                assert target.resolve().is_file()
            if tag == 'img':
                assert attrs['data-i18n-alt-zh'] and attrs['data-i18n-alt-en']
        if 'daily-progress' in name:
            assert parser.start[0] == 'article'
            assert parser.start[1]['data-date'] == '2026-10-09'
    current = json.loads((SITE / 'operator-learning/current-evidence.json').read_text())
    assert not current['latest_batched_classical_cost_review']['stable_wall_win']
    assert 'latest_global_review' in current
