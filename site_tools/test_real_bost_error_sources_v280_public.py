import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = 'real_bost_error_sources_v280'


def test_v280_independent_scope_and_false_claims():
    data = json.loads((ROOT / f'docs/{STEM}_public_summary.json').read_text())
    assert data['scientific_status'] == 'PASS_INDEPENDENT_LINEAR_SOURCE_BUDGET_V280'
    assert data['coverage']['cells'] == 1404
    assert data['coverage']['cell_space_rows'] == 5616
    assert data['coverage']['cameras'] == 9
    assert all(value is False for value in data['claims_fixed_false'].values())
    assert data['independent_validation']['checks_passed'] == data['independent_validation']['checks_total'] == 15
    assert max(data['independent_validation']['maxima'].values()) < 1e-8
    assert data['cost']['online_savings_claim'] is False


def test_v280_complete_aggregate_roster_and_no_stacked_quantiles():
    data = json.loads((ROOT / f'docs/{STEM}_public_summary.json').read_text())
    rows = data['summaries']
    identities = {(r['condition'], r['time'], r['space']) for r in rows}
    assert len(rows) == len(identities) == 48
    assert {r['count'] for r in rows} == {117}
    clean = [r['signed_shares']['p50'][1] for r in rows if r['space']=='field' and r['condition']=='clean']
    assert .8227 < min(clean) < .8228 and .8410 < max(clean) < .8411
    assert any(min(r['signed_shares']['minimum']) < -.01 for r in rows)
    code = (ROOT / f'site_tools/build_{STEM}_figure.py').read_text()
    assert 'stackplot' not in code and '.bar(' not in code


def test_v280_publication_is_bilingual_and_consistent():
    note = (ROOT / f'docs/{STEM}_result_2026-09-05.md').read_text()
    assert '# v280：' in note and '# v280:' in note
    assert '不能相加' in note or '不保证相加' in note
    assert 'v279 remains inconclusive' in note
    assert '不是配对实测像素位移' in note
    assert 'paired measured pixel displacement' in note
    current = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert current['scientific_status'] == 'FAIL_FIXED_NODAL_TSVD_REFERENCE_V283'
    assert current['metrics']['v280_cells'] == 1404
    assert current['current_decision']['v280_attribution_only'] is True
    for name in ('index.html','operator-learning/index.html','operator-learning/daily-progress.html'):
        text = (ROOT/name).read_text()
        assert STEM in text and '82.27' in text
    assert (ROOT/'operator-learning/daily-progress.html').read_text().count('data-date="2026-09-05"') == 1


def test_v280_redacted_content_and_figure():
    for name in (f'docs/{STEM}_public_summary.json', f'docs/{STEM}_result_2026-09-05.md'):
        text = (ROOT/name).read_text()
        assert all(t not in text for t in ('/Users/','/Volumes/','private_results','hexplane','cameraData'))
    figure = ROOT/f'assets/figures/{STEM}.png'
    assert figure.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    assert figure.stat().st_size > 50000
