import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PAGES=('index.html','operator-learning/index.html','operator-learning/daily-progress.html','learning_log.html')


def test_aggregate_gate_and_scope():
    d=json.loads((ROOT/'operator-learning/current-evidence.json').read_text())['latest_signed_query_feature_witness']
    assert (d['cells'],d['basic_strata_passed'],d['matched_cells_passed'],d['independent_checks_passed'])==(99,33,0,12)
    assert d['individual_metric_match_counts']==[1,6,17,0]
    assert d['classical_control_matched_cells']==[0,0,0]
    assert d['target_visible_projection_witness_only'] and d['best_initial_projection_is_not_all_refined_paths_optimum']
    assert d['directions_sealed_before_target_reads'] and d['fit_only_geometry_normalization']
    assert all(not d[k] for k in ('predictor_trained','predictor_authorized','whole_representation_refuted',
        'whole_direction_refuted','full_sequence_authorized','algorithm_breakthrough','paper_success',
        'resource_speedup','external_generalization','real_bost'))
    assert not re.search(r'/Users/|/Volumes/|[a-f0-9]{64}',json.dumps(d))


def test_four_pages_bilingual_and_figure():
    for path in PAGES:
        file=ROOT/path;section=BeautifulSoup(file.read_text(),'html.parser').select_one('#poolfire-signed-query-section-20261006')
        assert section is not None
        nodes=section.select('[data-i18n-zh]')
        assert len(nodes)==6 and all(n.get('data-i18n-en') for n in nodes)
        english=' '.join(n['data-i18n-en'] for n in nodes)
        assert 'No predictor is trained' in english and 'not the only one' in english
        assert 'not optimum over all refined paths' in english and 'not the oracle' in english
        assert '0/99' in english and '1/6/17/0' in english
        image=section.find('img');assert image['data-i18n-alt-zh'] and image['data-i18n-alt-en']
        assert (file.parent/image['src']).resolve().is_file()
        assert not re.search(r'/Users/|/Volumes/|[a-f0-9]{64}',str(section))


def test_figure_pixels_and_dimensions():
    image=Image.open(ROOT/'assets/figures/poolfire_signed_query_capacity_20261006.png').convert('RGB')
    assert image.size==(1320,600)
    assert len(image.getcolors(maxcolors=1000000))>100
