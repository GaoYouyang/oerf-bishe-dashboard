from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')
BLOCK = 'poolfire-tensor-instance-section-20261005'


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
def test_tensor_control_scope_cost_and_shared_input(name):
    source = (ROOT / name).read_text()
    parser = Section()
    parser.feed(source)
    assert source.count(f'id="{BLOCK}"') == 1
    translated = [a for _, a in parser.attributes if 'data-i18n-zh' in a]
    assert len(translated) == 3 and all(a.get('data-i18n-en') for a in translated)
    assert parser.attributes[1][0] == 'h2'
    content = str(translated)
    prose = ' '.join(a[k] for a in translated for k in ('data-i18n-zh', 'data-i18n-en'))
    assert not re.search(r'\b\d{4,}\b|\b\d+\.\d{3,}', prose)
    for claim in ('four-metric matched accuracy fails', 'total optimization plus refinement calls',
                  'Plain CGLS matches', 'at lower budget', 'shared frozen discrete matrix input',
                  'operator coefficients are not independently generated', 'sealed before truth scoring',
                  'not full TDBOST reproduction', 'no call, time or memory advantage',
                  'not an impossibility proof'):
        assert claim in content
    for private in ('/Users/', 'private_results', 'FROZEN.json', 'sha256', 'state_dict',
                    'learning_rate', 'trainable_parameters', 'sensitive numerical value', 'parameter count'):
        assert private not in content


def test_tensor_control_note_is_bilingual_and_qualitative():
    source = (ROOT / 'docs/native_observation_support_2026-10-05.md').read_text()
    assert '### 同算子张量初值与更简单的有限步基线' in source
    assert '### Same operator tensor initializer and simpler finite baseline' in source
    assert 'does not disprove the original algorithm' in source
    assert 'without fitting, refinement reruns or gate changes' in source
    assert 'old optimization-reference failures remain unchanged' in source
    for private in ('/Users/', 'private_results', 'FROZEN.json', 'sha256', 'sensitive numerical value'):
        assert private not in source
