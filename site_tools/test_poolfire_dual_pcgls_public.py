import json
from pathlib import Path

from bs4 import BeautifulSoup
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_dual_pcgls_20260907'


def test_separate_audit_and_necessary_scope():
    d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    assert d['status'] == 'FAIL_SEALED_SQUARED_DUAL_PCGLS_NECESSARY_COST_GATE'
    assert d['independent_status'] == 'PASS_INDEPENDENT_SEALED_PCGLS_ENDPOINT_AUDIT'
    assert d['parent_exit_code'] == 1 and d['parent_not_rerun']
    assert (d['unique_opened_frames'], d['camera_count'], d['paths']) == (5, 9, 2)
    assert d['certified_per_path'] == d['accurate_per_path'] == [0, 0]
    scores = np.array(d['scores'])
    assert scores.shape == (2, 5, 4) and (scores[:, :, :3] > .01).all()
    assert (scores[:, :, 3] < .0033).all()
    assert d['online'] == dict(A=8634, AT=8634) and d['transform_applications'] == 8634
    assert d['ablation_not_run'] and d['inherited_geometry_certificate_not_free']
    assert not any(d[k] for k in ('full505_authorized', 'training_authorized', 'new_training',
                                  'full_sequence', 'algorithm_breakthrough', 'paper_success',
                                  'external_generalization', 'resource_speedup', 'real_bost'))


def test_bilingual_notes_and_no_private_payload():
    d = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert STEM in d['latest_dual_pcgls']['summary']
    assert d['next_scientific_gate'] == d['next_scientific_gate_en']
    assert d['latest_dual_pcgls']['parent_exit_code'] == 1
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        notes = soup.select('#dual-pcgls-result')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('0/5', '1%', '0.33%', '5.6%-8.9%'))
        if 'daily' in rel:
            assert len(soup.select('#latest')) == 1
    for ext in ('md', 'json'):
        value = (ROOT/'docs'/f'{STEM}.{ext}').read_text()
        assert not any(v in value for v in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    assert (ROOT/'assets/figures'/f'{STEM}.png').stat().st_size > 10000
