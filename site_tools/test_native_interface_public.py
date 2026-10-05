from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')


class Extract(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.tag = None
        self.attributes = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id') == 'native-interface-check-20261005':
            assert not self.active
            self.active, self.tag = True, tag
        if self.active:
            self.attributes.append(attrs)

    def handle_endtag(self, tag):
        if self.active and tag == self.tag:
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.text.append(data)


@pytest.mark.parametrize('page', PAGES)
def test_native_interface_bilingual_and_private(page):
    parser = Extract()
    parser.feed((ROOT / page).read_text())
    translated = [a for a in parser.attributes if 'data-i18n-zh' in a]
    assert len(translated) == 5
    assert all(a.get('data-i18n-en') for a in translated)
    content = '\n'.join(parser.text) + '\n' + str(parser.attributes)
    for forbidden in ('jetflame', 'spray', 'Vq', '.mat', '/Users/', 'private_results', '101×174', '1774974', '1,774,974', '10.43655'):
        assert forbidden not in content
    assert 'necessary observation-support check' in content
    assert 'equally dense old-view control' in content
    assert 'not four-metric reconstruction success' in content
    assert 'truth-visible fixed-pair audit' in content
    assert 'not TV reconstruction success' in content
    assert 'unregularized reference remains inconclusive' in content
    assert 'nearly identical 2D observations' in content
    assert 'fail convergence certification' in content
    assert 'no training or speed claim is authorized' in content
    assert any('native_observation_support_2026-10-05.md' in a.get('href', '') for a in parser.attributes)


def test_qualitative_note_has_no_private_arrays_or_counts():
    note = (ROOT / 'docs/native_observation_support_2026-10-05.md').read_text()
    assert '## 中文' in note and '## English' in note
    assert 'not an open external benchmark' in note
    for forbidden in ('jetflame', 'spray', 'Vq', '.mat', '/Users/', 'private_results', '101×174', '1774974', '1,774,974', '10.43655', '60/60', '43/60'):
        assert forbidden not in note
