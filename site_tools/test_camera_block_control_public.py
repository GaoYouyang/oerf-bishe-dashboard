from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')
BLOCK = 'poolfire-camera-block-section-20261005'


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
def test_camera_block_control_scope_and_privacy(name):
    source = (ROOT / name).read_text()
    parser = Section()
    parser.feed(source)
    assert source.count(f'id="{BLOCK}"') == 1 and parser.attributes[1][0] == 'h2'
    translated = [a for _, a in parser.attributes if 'data-i18n-zh' in a]
    assert len(translated) == 3 and all(a.get('data-i18n-en') for a in translated)
    prose = ' '.join(a[k] for a in translated for k in ('data-i18n-zh', 'data-i18n-en'))
    for required in ('all four metrics', 'every tested training sentinel', 'matched accuracy',
                     'still fails', 'not a learned model or a new invention',
                     'shared frozen matrix', 'independently reconstructed physics',
                     'seal before truth scoring', 'remain failed', 'pass unchanged',
                     'not free', 'not an impossibility proof'):
        assert required in prose
    assert not re.search(r'\b\d{4,}\b|\b\d+\.\d{3,}', prose)
    for private in ('/Users/', 'private_results', 'FROZEN.json', 'sha256', 'state_dict',
                    'cutoff=', 'rank=', 'trainable_parameters', 'parameter count'):
        assert private not in prose


def test_camera_block_note_bilingual_and_classical():
    source = (ROOT / 'docs/native_observation_support_2026-10-05.md').read_text()
    assert '### 完整相机块初值' in source and '### Full camera block initializer' in source
    assert 'existing classical block projection idea' in source
    assert 'Factorization, cache storage and block actions are not free' in source
