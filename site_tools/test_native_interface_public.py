from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')


class Extract(HTMLParser):
    def __init__(self, block_id='native-interface-check-20261005'):
        super().__init__()
        self.block_id = block_id
        self.active = False
        self.tag = None
        self.attributes = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id') == self.block_id:
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
    assert len(translated) == 6
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
    assert 'post-open unchanged-CGLS refinement' in content
    assert 'without harming 3D accuracy' in content
    assert 'only the refinement increment, not the full TV initializer' in content
    assert 'preregistered finite dual norm' in content
    assert 'not unbounded-dual, general-PCGLS or PoolFire impossibility' in content
    assert any('native_observation_support_2026-10-05.md' in a.get('href', '') for a in parser.attributes)


def test_qualitative_note_has_no_private_arrays_or_counts():
    note = (ROOT / 'docs/native_observation_support_2026-10-05.md').read_text()
    assert '## 中文' in note and '## English' in note
    assert 'not an open external benchmark' in note
    assert 'equal-total-cost algorithm advantage' in note
    assert 'complete nonlinear TV initial state' in note
    assert 'Historical costs' in note
    assert 'bounded representation family' in note
    assert 'General PCGLS' in note
    assert 'identical states, observations and selected parameters' in note
    assert 'old failed reference cannot be relabeled as success' in note
    for forbidden in ('jetflame', 'spray', 'Vq', '.mat', '/Users/', 'private_results', '101×174', '1774974', '1,774,974', '10.43655', '60/60', '43/60'):
        assert forbidden not in note


@pytest.mark.parametrize('page', PAGES)
def test_main_poolfire_attribution_does_not_repair_reference(page):
    parser = Extract('hybrid-gcv-reference-20261005')
    parser.feed((ROOT / page).read_text())
    content = '\n'.join(parser.text) + '\n' + str(parser.attributes)
    assert 'identical-input normal-action replay' in content
    assert '12/12 independent checks on all 99 existing sentinels' in content
    assert 'full-space stationarity gate in 0/99 cells' in content
    assert 'parent failed verdict is unchanged' in content
    assert 'not an accuracy veto on all finite-budget reconstructions' in content


@pytest.mark.parametrize('page', PAGES)
def test_fixed_target_lift_is_not_learning_or_reference_rescue(page):
    parser = Extract('poolfire-fixed-target-lift-20261005')
    parser.feed((ROOT / page).read_text())
    translations = [a for a in parser.attributes if 'data-i18n-zh' in a]
    assert len(translations) == 1 and translations[0].get('data-i18n-en')
    content = str(parser.attributes)
    for claim in ('representation gate passes', 'sentinel targets only', 'coefficients read the targets',
                  'preparation is expensive', 'No training or call saving', 'Old reference failures remain unchanged',
                  'complete-trajectory prediction and fair total cost are still untested'):
        assert claim in content
    for private in ('/Users/', 'private_results', 'psi=U', '1e6', 'alpha0', '7.276'):
        assert private not in content


@pytest.mark.parametrize('page', PAGES)
def test_normal_gate_learning_failure_is_not_global_impossibility(page):
    parser = Extract('poolfire-normal-gate-loto-20261005')
    source = (ROOT / page).read_text()
    parser.feed(source)
    assert source.count('id="poolfire-normal-gate-loto-20261005"') == 1
    translations = [a for a in parser.attributes if 'data-i18n-zh' in a]
    assert len(translations) == 1 and translations[0].get('data-i18n-en')
    content = str(parser.attributes)
    for claim in ('strict complete-trajectory-held-out', 'meets basic absolute accuracy',
                  'not four-metric matched accuracy', 'not a veto caused by an already passing cheap control',
                  'separately train, lift, physically replay', 'without full-sequence or larger-network escalation',
                  'old optimization-reference failures are unchanged', 'no impossibility proof'):
        assert claim in content
    for private in ('/Users/', 'private_results', 'NormalGate66', '66-parameter', '0/99',
                    '1390.50', '0.434806', '0.886948', '0861a253'):
        assert private not in content


def test_learning_note_retains_scope_and_prior_verdicts():
    note = (ROOT / 'docs/native_observation_support_2026-10-05.md').read_text()
    assert '### 主线小模型学习哨兵判决' in note
    assert '### Main-route small-model learning sentinel' in note
    assert 'basic reconstruction accuracy is not matched-accuracy acceleration' in note
    assert 'not unopened generalization, noise robustness or real BOST' in note
    assert 'old optimization-reference failures remain unchanged' in note
    for private in ('NormalGate66', '0/99', '35,547', '0.434806', '1390.50'):
        assert private not in note
