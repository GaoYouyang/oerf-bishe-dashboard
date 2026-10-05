from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html','operator-learning/index.html','operator-learning/daily-progress.html','learning_log.html')
BLOCK = 'poolfire-prenormal-section-20261006'


class Block(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active, self.tag, self.nodes = False, None, []

    def handle_starttag(self,tag,attributes):
        attrs = dict(attributes)
        if attrs.get('id') == BLOCK:
            self.active,self.tag = True,tag
        if self.active:
            self.nodes.append((tag,attrs))

    def handle_endtag(self,tag):
        if tag == self.tag:
            self.active = False


@pytest.mark.parametrize('page',PAGES)
def test_prenormal_scope_and_bilingual_privacy(page):
    source = (ROOT/page).read_text()
    block = Block()
    block.feed(source)
    assert source.count(f'id="{BLOCK}"') == 1
    nodes = [attrs for _,attrs in block.nodes if 'data-i18n-zh' in attrs]
    assert len(nodes) == 3 and all(attrs.get('data-i18n-en') for attrs in nodes)
    english = ' '.join(attrs['data-i18n-en'] for attrs in nodes)
    for phrase in ('before exact cross-camera propagation','Complete-trajectory held-out training',
                   'basic accuracy passes','matched accuracy fails','equal-call-budget',
                   'without width, target/loss or refinement-depth rescue','Post-open description',
                   'near-budget plain CGLS remains better','parameter-matched causal proof',
                   'share a frozen matrix','before truth scoring','not free','opened train sentinels only',
                   'not a full sequence','resource speedup','does not prove'):
        assert phrase in english
    prose = ' '.join(attrs[k] for attrs in nodes for k in ('data-i18n-zh','data-i18n-en'))
    assert not re.search(r'\b\d{4,}\b|\b\d+\.\d{3,}',prose)
    for secret in ('/Users/','private_results','sha256','FROZEN.json','state_dict','trainable_parameters'):
        assert secret not in prose


def test_note_separates_control_and_main_result():
    text = (ROOT/'docs/native_observation_support_2026-10-05.md').read_text()
    assert '### 10月6日 观测前置选择学习实验' in text
    assert '### October 6 Learning observation selection before physics propagation' in text
    for phrase in ('not a parameter-matched causal proof','Both fits use a shared frozen matrix',
                   'Private data, protocols, arrays, source, weights and figures remain unpublished',
                   'Neither supplies evidence of BOST acceleration'):
        assert phrase in text
