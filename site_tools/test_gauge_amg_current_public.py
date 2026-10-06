"""Publication-only checks; no private scientific input is read."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')
BLOCK = 'poolfire-gauge-amg-section-20261006'


class Section(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active, self.tag, self.nodes = False, None, []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get('id') == BLOCK:
            self.active, self.tag = True, tag
        if self.active and 'data-i18n-zh' in attrs:
            self.nodes.append(attrs)

    def handle_endtag(self, tag):
        if tag == self.tag:
            self.active = False


@pytest.mark.parametrize('page', PAGES)
def test_tradeoff_is_bilingual_scoped_and_not_acceleration(page):
    text = (ROOT/page).read_text()
    parser = Section(); parser.feed(text)
    assert text.count(f'id="{BLOCK}"') == 1
    assert len(parser.nodes) == 4
    assert all(node.get('data-i18n-en') for node in parser.nodes)
    english = ' '.join(node['data-i18n-en'] for node in parser.nodes)
    for phrase in ('no training', 'matched accuracy fails', 'most 3D field errors are higher',
                   'shared frozen matrix', 'independently rebuilt physics', 'failures remain preserved',
                   'unchanged science contract', 'unique cause is not established', 'not free',
                   'no learned acceleration', 'not an impossibility proof'):
        assert phrase in english
    prose = ' '.join(node[key] for node in parser.nodes for key in ('data-i18n-zh', 'data-i18n-en'))
    assert not re.search(r'\b[0-9a-f]{40,64}\b|\b\d+\.\d{3,}', prose)
    for token in ('/Users/', 'private_results', 'sha256', 'FROZEN.json', 'PCGLS35', 'max_coarse', 'lambda'):
        assert token not in prose


def test_latest_aggregate_preserves_scientific_boundaries_and_history():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    result = evidence['latest_native_classical_control']
    assert result['updated'] == '2026-10-06'
    assert result['cells'] == 99 and result['strata'] == 33
    assert result['basic_strata_passed'] == 18 and result['matched_cells_passed'] == 0
    assert result['observation_only_matched_cells'] == 52
    assert result['independent_checks_passed'] == 16
    for key in ('whole_direction_refuted', 'algorithm_breakthrough', 'paper_success',
                'resource_speedup', 'external_generalization', 'real_bost'):
        assert result[key] is False
    assert evidence['latest_hybrid_gcv_reference']['reference_stable_strata'] == 0
    assert 'latest_case2_sarc_v285' in evidence
    for page in PAGES[:2]:
        assert 'const current = evidence.latest_direct_ridge_cost || evidence.latest_direct_ridge_initializer || evidence.latest_native_classical_control;' in (ROOT/page).read_text()


def test_note_keeps_cost_recovery_and_scope():
    note = (ROOT/'docs/native_observation_support_2026-10-05.md').read_text()
    assert '### 10月6日 观测拟合与三维场精度分离' in note
    latest = note.split('### October 6 Observation fit versus 3D field accuracy')[1]
    for phrase in ('most 3D field errors worsen', 'not the whole C route', 'time-limit',
                   'No accuracy gate is relaxed', 'are not free', 'not deployment latency',
                   'three-frame sentinel', 'paper success', 'remain unpublished'):
        assert phrase in latest
    assert not re.search(r'\b[0-9a-f]{40,64}\b', latest)


def test_fixed_direction_capacity_is_not_global_refutation_or_learning():
    evidence = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    result = evidence['latest_fixed_direction_capacity']
    assert result['cells'] == result['conservatively_infeasible_cells'] == 99
    assert result['field_individually_unreachable_cells'] == 99
    assert result['original_feasible_cells'] == result['observation_control_matched_cells'] == 0
    assert result['independent_checks_passed'] == 13 and result['post_open_attribution_only']
    assert result['new_training_parameters'] == 0
    assert not any(result[key] for key in ('whole_direction_refuted', 'algorithm_breakthrough',
        'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    for page in PAGES:
        parser = Section(); parser.feed((ROOT/page).read_text())
        nodes = [node for node in parser.nodes if node.get('id') == 'poolfire-fixed-direction-capacity-20261006']
        assert len(nodes) == 1
        english = nodes[0]['data-i18n-en']
        assert 'truth-aware' in english and 'not a refutation of all learned initialization' in english
        assert 'post-open capacity attribution' in english and 'remains unproven' in english
