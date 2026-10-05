from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')
BLOCK = 'poolfire-boundary-filter-section-20261005'


class Section(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active, self.attributes, self.tag = False, [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id') == BLOCK:
            self.active, self.tag = True, tag
        if self.active:
            self.attributes.append((tag, attrs))

    def handle_endtag(self, tag):
        if tag == self.tag:
            self.active = False


@pytest.mark.parametrize('name', PAGES)
def test_boundary_control_scope_and_privacy(name):
    source = (ROOT / name).read_text()
    parser = Section()
    parser.feed(source)
    assert source.count(f'id="{BLOCK}"') == 1 and parser.attributes[1][0] == 'h2'
    translated = [a for _, a in parser.attributes if 'data-i18n-zh' in a]
    assert len(translated) == 3 and all(a.get('data-i18n-en') for a in translated)
    prose = ' '.join(a[k] for a in translated for k in ('data-i18n-zh', 'data-i18n-en'))
    for required in ('four-metric matched accuracy fails', 'Cheaper BP initialization',
                     'every tested cell', 'not a learned model', 'exact curl-free constraint',
                     'clipped interpolation branch', 'preserving the failed record', 'shared frozen matrix',
                     'independently reconstructed physics', 'seal all states before truth scoring',
                     'not free', 'not an impossibility proof'):
        assert required in prose
    assert not re.search(r'\b\d{4,}\b|\b\d+\.\d{3,}', prose)
    for private in ('/Users/', 'private_results', 'FROZEN.json', 'sha256', 'state_dict',
                    'k0=', 'learning_rate', 'trainable_parameters', 'parameter count'):
        assert private not in prose


def test_boundary_note_qualitative_bilingual():
    source = (ROOT / 'docs/native_observation_support_2026-10-05.md').read_text()
    assert '### 有限积分盒导数补偿初值' in source
    assert '### Finite box derivative compensation initializer' in source
    assert 'leaving the candidate and scientific gates unchanged' in source
    assert 'Additional derivative actions and setup are not free' in source
