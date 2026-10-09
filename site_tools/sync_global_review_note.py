"""Copy one substantive bilingual note without rewriting existing page trees."""

import json
from html.parser import HTMLParser
from pathlib import Path
from xml.sax.saxutils import quoteattr

SITE = Path(__file__).resolve().parents[1]
MARKER = 'poolfire-global-review-20261009'
PREVIOUS = 'poolfire-detector-precision-20261009'

class Markers(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.markers = {}
        self.active = None
        self.depth = 0
        self.feed(text)

    def character_offset(self):
        line, column = self.getpos()
        return sum(map(len, self.lines[:line - 1])) + column

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        identity = attributes.get('id')
        if identity in (MARKER, PREVIOUS):
            assert identity not in self.markers and self.active is None
            self.markers[identity] = dict(start=self.character_offset(), tag=tag, attrs=attributes,
                                          opening=self.get_starttag_text())
            self.active, self.depth = identity, 1
        elif self.active and tag == self.markers[self.active]['tag']:
            self.depth += 1

    def handle_endtag(self, tag):
        if self.active and tag == self.markers[self.active]['tag']:
            self.depth -= 1
            if self.depth == 0:
                end = self.text.index('>', self.character_offset()) + 1
                self.markers[self.active]['end'] = end
                self.active = None


root_text = (SITE / 'index.html').read_text()
root = Markers(root_text).markers[MARKER]
content = root_text[root['start'] + len(root['opening']):root['end'] - len('</' + root['tag'] + '>')]
for name in ('operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
    path = SITE / name
    text = path.read_text()
    parsed = Markers(text).markers
    assert MARKER not in parsed
    target = parsed[PREVIOUS]
    attributes = dict(id=MARKER, **{'class':target['attrs']['class']})
    for attribute in ('data-date', 'data-categories'):
        if attribute in target['attrs']:
            attributes[attribute] = target['attrs'][attribute]
    prefix = '../' if name.startswith('operator-learning/') else ''
    body = content.replace('src="assets/poolfire_global_review_2026-10-09.png"',
                           f'src="{prefix}assets/poolfire_global_review_2026-10-09.png"')
    body = body.replace('href="docs/poolfire_global_review_2026-10-09_public_summary.json"',
                        f'href="{prefix}docs/poolfire_global_review_2026-10-09_public_summary.json"')
    opening = '<' + target['tag'] + ' ' + ' '.join(key + '=' + quoteattr(value) for key, value in attributes.items()) + '>'
    fragment = opening + body + '</' + target['tag'] + '>\n'
    offset = target['start']
    path.write_text(text[:offset] + fragment + text[offset:])

path = SITE / 'operator-learning/current-evidence.json'
data = json.loads(path.read_text())
assert 'latest_global_review' not in data
note = dict(updated='2026-10-09', cells=99, matched_cells=0, basic_absolute_sampled_strata=33,
            reference_matched_cells=99, independent_readonly_diagnostic=True, parent_native_resource_valid=False,
            original_parent_promoted=False, zero_primary_metric_wins_against_global_and_cold=True,
            camera_counts=[5, 7, 9], complete_sequence=False, complete_trajectory_LOTO=False,
            fixed_recipe_closed=True, new_predictor_authorized=False, algorithm_breakthrough=False,
            resource_speedup=False, external_generalization=False, real_bost=False,
            summary='docs/poolfire_global_review_2026-10-09_public_summary.json')
path.write_text(json.dumps(dict(latest_global_review=note, **data), indent=2, ensure_ascii=False) + '\n')
