from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')
BLOCK = 'poolfire-input-roundoff-section-20261006'


class Block(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active, self.tag, self.nodes = False, None, []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get('id') == BLOCK:
            self.active, self.tag = True, tag
        if self.active:
            self.nodes.append((tag, attrs))

    def handle_endtag(self, tag):
        if tag == self.tag:
            self.active = False


@pytest.mark.parametrize('page', PAGES)
def test_roundoff_scope_bilingual_and_privacy(page):
    source = (ROOT/page).read_text()
    parser = Block()
    parser.feed(source)
    assert source.count(f'id="{BLOCK}"') == 1 and parser.nodes[1][0] == 'h2'
    nodes = [attrs for _, attrs in parser.nodes if 'data-i18n-zh' in attrs]
    assert len(nodes) == 3 and all(attrs.get('data-i18n-en') for attrs in nodes)
    english = ' '.join(attrs['data-i18n-en'] for attrs in nodes)
    for required in ('remains inconclusive', 'same exact input', 'nine-camera', 'original teacher-consistency gate',
                     'without tolerance relaxation', 'not proof', 'not a new learned algorithm',
                     'before independent aggregation', 'shared frozen discrete matrix', 'No new truth',
                     'remain unchanged', 'no rescue', 'no call, time or memory saving'):
        assert required in english
    prose = ' '.join(attrs[k] for attrs in nodes for k in ('data-i18n-zh', 'data-i18n-en'))
    assert not re.search(r'\b\d{4,}\b|\b\d+\.\d{3,}', prose)
    for private in ('/Users/', 'private_results', 'sha256', 'FROZEN.json', 'state_dict', 'trainable_parameters'):
        assert private not in prose


def test_note_dates_and_numerical_limits():
    source = (ROOT/'docs/native_observation_support_2026-10-05.md').read_text()
    assert '### 10月6日 信息审计与输入舍入' in source and '### October 6 Information audit and input rounding' in source
    assert 'All diagnostic work is offline cost' in source
    assert 'their paper alone does not establish the cause' in source
