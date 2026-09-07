import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def test_independent_stage_not_new_success():
    doc = json.loads((ROOT/'docs/poolfire_camera_mix_ablation_20260907.json').read_text())
    d = doc['stage_attribution']
    assert d['status'] == 'P33_INTERIOR_HARM_PREEXISTS_K1'
    assert d['independent_status'] == 'PASS_INDEPENDENT_CAMERA_MIX_STAGE'
    assert d['trajectories'][2]['stage_counts'][2] == [101, 0, 0, 0]
    assert d['stage_counts'][2] == [227, 25, 56, 197]
    assert d['post_passing'] == [0, 0] and d['new_trainable_parameters'] == 0
    assert d['independent_endpoints'] == 1010 and d['independent_maxima']['terms'] < 1e-12
    assert d['source_reconstructed_offline_ledger'] and d['online'] == dict(A=2, AT=2)
    assert not any(d[k] for k in ('algorithm_breakthrough', 'paper_success', 'resource_speedup', 'real_bost'))
    assert doc['status'] == 'MIXED_CURRENT_CROSS_CAMERA_EFFECT'


def test_bilingual_stage_and_public_only():
    for rel in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html'):
        soup = BeautifulSoup((ROOT/rel).read_text(), 'html.parser')
        notes = soup.select('#camera-mix-stage-result')
        assert len(notes) == 1
        for lang in ('zh', 'en'):
            assert all(v in notes[0][f'data-i18n-{lang}'] for v in ('101', '0.725', '0.633', '25', '56', '1%'))
    for rel in ('docs/poolfire_camera_mix_ablation_20260907.json', 'docs/poolfire_camera_mix_ablation_20260907.md'):
        text = (ROOT/rel).read_text()
        assert not any(v in text for v in ('/Users/', '/Volumes/', 'private_results', 'sha256', '.pt', 'parameters.json'))
    assert (ROOT/'assets/figures/poolfire_camera_mix_stage_20260907.png').stat().st_size > 10000
