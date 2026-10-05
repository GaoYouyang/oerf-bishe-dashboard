"""Check aggregate publication boundaries without opening private scientific files."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html',
         'operator-learning/daily-progress.html', 'learning_log.html')
BLOCK = 'poolfire-whole-observation-section-20261006'


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
def test_local_gain_does_not_override_failure_or_claim_whole_sequence(page):
    text = (ROOT / page).read_text()
    parser = Section(); parser.feed(text)
    assert text.count(f'id="{BLOCK}"') == 1 and len(parser.nodes) == 4
    assert all(node.get('data-i18n-en') for node in parser.nodes)
    english = ' '.join(node['data-i18n-en'] for node in parser.nodes)
    for phrase in ('held trajectory is excluded', '33/33', '0/99', '97/99', '3.2%',
                   '65/99', '10/99', 'are not free', 'not full-sequence validation',
                   'not fewer calls at matched accuracy', 'remain unpublished'):
        assert phrase in english
    for token in ('/Users/', 'private_results', 'FROZEN.json', 'sha256', 'bank90', 'theta', 'ridge'):
        assert token not in english
    assert not re.search(r'\b[0-9a-f]{40,64}\b', english)


def test_aggregate_history_and_false_claims_remain_intact():
    evidence = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    result = evidence['latest_whole_observation_learning']
    assert result['cells'] == 99 and result['strata'] == result['basic_strata_passed'] == 33
    assert result['matched_cells_passed'] == 0 and result['independent_checks_passed'] == 14
    assert result['plain_cgls_all_four_nonworse_cells'] == 97
    assert result['plain_cgls_field_or_interior_harm_cells'] == 2
    assert result['pcgls_all_four_nonworse_cells'] == 65
    assert result['pcgls_all_four_nonbetter_cells'] == 10
    assert result['evaluation_is_three_frame_sentinel_only'] and result['fixed_recipe_closed']
    assert not any(result[k] for k in ('whole_direction_refuted', 'full_sequence_authorized',
        'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'))
    assert evidence['latest_native_classical_control']['matched_cells_passed'] == 0
    assert evidence['latest_fixed_direction_capacity']['conservatively_infeasible_cells'] == 99


def test_note_preserves_scope_cost_and_existing_verdicts():
    note = (ROOT / 'docs/native_observation_support_2026-10-05.md').read_text()
    latest = note.split('### October 6 Whole-observation learning helps locally but is insufficient')[1]
    for phrase in ('0/99', '97/99', '65/99', '10/99', 'not a stationarity certificate',
                   'not proof of information sufficiency', 'are not free',
                   'No fresh wall/RSS', 'weights remain unpublished'):
        assert phrase in latest
