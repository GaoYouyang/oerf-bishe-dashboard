import json
from pathlib import Path

from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
STEM='real_bost_nodal_reference_v283'


def test_frozen_negative_reference_not_algorithm():
    data=json.loads((ROOT/f'docs/{STEM}_public_summary.json').read_text())
    assert data['scientific_decision']=='FAIL_FIXED_NODAL_TSVD_REFERENCE_V283'
    assert data['coverage']['cells']==1404 and data['coverage']['interior_unknowns']==5880
    assert data['coverage']['rays_per_camera']==576
    assert data['coverage']['calibration_files']==13 and data['coverage']['distinct_geometries']==11
    assert data['passing_strata']=={'TSVD':8,'Zero':0,'BP':0,'CGLS16':0,'PCGLS16':0}
    assert not any(data['claims_fixed_false'].values())
    assert data['cost']['dense_setup_not_free'] and not data['cost']['resource_comparison']
    for private in ('/Users/','/Volumes/','.cpu.pt','cameraData','sha256','checkpoint'):
        assert private not in json.dumps(data)


def test_all_five_camera_strata_fail_despite_image_fit():
    data=json.loads((ROOT/f'docs/{STEM}_public_summary.json').read_text())
    selected=[s for s in data['summaries'] if s['arm']=='TSVD']
    assert len(selected)==12
    for row in selected:
        assert row['count']==117
        if row['name'].startswith('5:'):
            assert not row['pass']
            assert row['metrics']['gradient']['p90']>1.25
            assert row['metrics']['observation']['worst']<5.2e-7
        else:
            assert row['pass']


def test_latest_and_historical_bilingual_sections():
    current=json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    assert current['current_decision']['v283_fixed_reference_closed']
    for path in ('index.html','operator-learning/index.html'):
        soup=BeautifulSoup((ROOT/path).read_text(),'html.parser')
        section=soup.select_one('#v283-nodal-reference')
        assert section and '五相机' in section.get_text()
        for node in section.select('h2,p[data-i18n-zh],figcaption'):
            assert node.get('data-i18n-zh') and node.get('data-i18n-en')
        assert soup.select_one('#v282-finite-background')
    home=BeautifulSoup((ROOT/'index.html').read_text(),'html.parser')
    assert STEM in home.select_one('#v283-nodal-reference img')['src']
    report=(ROOT/f'docs/{STEM}_result_2026-09-05.md').read_text()
    assert '# v283: fitting projections' in report and '不是实测 BOST' in report
    assert (ROOT/f'assets/figures/{STEM}.png').stat().st_size>10000
