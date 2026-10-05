from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')
BLOCK = 'poolfire-neumann-half-section-20261006'


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
def test_classical_control_boundary_and_bilingual_privacy(page):
    text = (ROOT/page).read_text()
    parser = Block(); parser.feed(text)
    assert text.count(f'id="{BLOCK}"') == 1
    nodes = [attrs for _, attrs in parser.nodes if 'data-i18n-zh' in attrs]
    assert len(nodes) == 3 and all(attrs.get('data-i18n-en') for attrs in nodes)
    english = ' '.join(attrs['data-i18n-en'] for attrs in nodes)
    for phrase in ('no training', 'least-squares objective', 'independent direct-cosine',
                   'basic accuracy passes', 'matched accuracy fails', 'half-order recipe closes',
                   'same operator-call budget', 'not uniformly', 'without being the only bottleneck',
                   'not free', 'stronger classical competition', 'not learned acceleration',
                   'resource speedup', 'real BOST'):
        assert phrase in english
    prose = ' '.join(attrs[key] for attrs in nodes for key in ('data-i18n-zh', 'data-i18n-en'))
    assert not re.search(r'\b\d{4,}\b|\b\d+\.\d{3,}', prose)
    for secret in ('/Users/', 'private_results', 'sha256', 'FROZEN.json', 'lambda_min', 'D^-1/2'):
        assert secret not in prose


def test_note_does_not_replace_learned_goal():
    text = (ROOT/'docs/native_observation_support_2026-10-05.md').read_text()
    assert '### 10月6日 固定频谱经典对照' in text
    assert '### October 6 Fixed spectral classical control' in text
    for phrase in ('zero new fitted parameters', 'some regress', 'not a replacement for four per-cell gates',
                   'not deployment latency', 'Private formulas, protocols, arrays, source and figures remain unpublished'):
        assert phrase in text
