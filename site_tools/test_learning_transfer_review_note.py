import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
NAME = 'poolfire_learning_transfer_review_2026-10-10'
MARKER = 'poolfire-learning-transfer-review-20261010'
PAGES = ('index.html', 'operator-learning/index.html',
         'operator-learning/daily-progress.html', 'learning_log.html')


def test_learning_signal_is_not_strong_control_or_loto_success():
    data = json.loads((ROOT / f'docs/{NAME}_public_summary.json').read_text())
    scope, learner, audit = data['scope'], data['learned_capacity'], data['fixed_state_attribution']
    assert scope['cells'] == learner['actual_CFD_pareto_improves_own_unfit'] == 99
    assert scope['camera_counts'] == [5, 7, 9]
    assert not scope['complete_trajectory_LOTO'] and not scope['complete_sequence']
    assert not scope['paired_experiment'] and not scope['new_validation_test_opened']
    assert learner['matched_cells'] == 0
    assert learner['equal_call_cold_at_least_one_metric_better'] == 99
    assert learner['fixed_recipe_closed']
    assert learner['logical_calls_per_query'] == {'A': 35, 'AT': 35}
    assert all(value > 0 for value in learner['median_actual_CFD_relative_improvement_own_unfit'])
    assert audit['actual_mean_teacher_endpoint_loss_better_than_frozen_history'] == 99
    assert audit['exact_finite_transfer_faithful_cells'] == 0
    assert not audit['approximate_differential_gradient_usefulness_tested']
    assert not audit['causal_percentage_claim']
    assert all(value is False for value in data['claims'].values())


class Note(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tag = None
        self.active = False
        self.pairs = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id') == MARKER:
            self.active, self.tag = True, tag
        if self.active and 'data-i18n-zh' in attrs:
            self.pairs.append((attrs['data-i18n-zh'], attrs.get('data-i18n-en')))
        if self.active and tag == 'a':
            self.links.append(attrs['href'])

    def handle_endtag(self, tag):
        if self.active and tag == self.tag:
            self.active = False


def test_bilingual_single_note_links_and_headline_scope():
    for name in PAGES:
        text = (ROOT / name).read_text()
        assert text.count(f'id="{MARKER}"') == 1
        assert text.index(MARKER) < text.index('id="poolfire-noise-representation-review-20261009"')
        note = Note()
        note.feed(text)
        assert len(note.pairs) == 6 and all(a and b for a, b in note.pairs)
        for link in note.links:
            parts = urlsplit(link)
            assert (ROOT / name).parent.joinpath(parts.path).resolve().is_file()
            if 'doc' in parse_qs(parts.query):
                assert (ROOT / parse_qs(parts.query)['doc'][0]).is_file()
    latest = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())['latest_research_synthesis']
    assert 'in-sample' in latest['learned_counts_scope']
    assert latest['classical_matched_cells'] == 3333
    assert latest['learned_matched_cells'] == 0
    assert MARKER in latest['note'] and NAME in latest['summary']
    focus = (ROOT / 'operator-learning/index.html').read_text()
    assert 'fixed route closed' not in focus
    assert 'learned acceleration unproven' in focus


def test_public_note_has_no_private_execution_identity():
    for suffix in ('public_summary.json', 'plain_language.md'):
        text = (ROOT / f'docs/{NAME}_{suffix}').read_text()
        for token in ('/Users/', '/Volumes/', 'private_results/', 'private_data/',
                      'sha256', 'checkpoint', 'FROZEN.json', 'OPENED.json'):
            assert token not in text
    assert '## 中文' in (ROOT / f'docs/{NAME}_plain_language.md').read_text()
    assert '## English' in (ROOT / f'docs/{NAME}_plain_language.md').read_text()
