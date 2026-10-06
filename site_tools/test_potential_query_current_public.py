import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html')


def test_gate_scope_and_recovered_equivalence():
    data = json.loads((ROOT/'operator-learning/current-evidence.json').read_text())
    d = data['latest_spatial_potential_projection_witness']
    assert (d['cells'], d['basic_strata_passed'], d['matched_cells_passed']) == (99, 33, 0)
    assert (d['cheap_control_basic_strata_passed'], d['cheap_control_matched_cells_passed']) == (33, 0)
    assert d['recovered_independent_checks_passed'] == 14
    assert d['exact_empty_band_equivalence_certificate'] and d['original_failure_and_inconclusive_records_preserved']
    assert d['active_coefficients_and_physical_outputs_bitwise_unchanged']
    assert d['no_tolerances_changed_or_refinement_repeated']
    assert d['target_visible_initial_projection_witness_only'] and d['directions_and_control_sealed_before_targets']
    assert d['earlier_limited_learning_evidence_preserved']
    assert data['latest_whole_observation_learning']['plain_cgls_all_four_nonworse_cells'] == 97
    assert all(not d[k] for k in ('predictor_trained', 'predictor_authorized', 'whole_direction_refuted',
        'full_sequence_authorized', 'algorithm_breakthrough', 'paper_success', 'resource_speedup',
        'external_generalization', 'real_bost'))
    assert not re.search(r'/Users/|/Volumes/|[a-f0-9]{64}', json.dumps(d))


def test_four_pages_bilingual_privacy_and_chart():
    for name in PAGES:
        file = ROOT/name
        section = BeautifulSoup(file.read_text(), 'html.parser').select_one('#poolfire-potential-query-section-20261006')
        nodes = section.select('[data-i18n-zh]')
        assert len(nodes) == 7 and all(n.get('data-i18n-en') for n in nodes)
        en = ' '.join(n['data-i18n-en'] for n in nodes)
        assert 'No predictor is trained' in en and '0/99' in en and '33/33' in en
        assert 'bitwise unchanged' in en and 'No tolerance is relaxed' in en
        assert 'not optimum over every refined path' in en and 'not the actual cost' in en
        assert 'not full sequences' in en and 'Earlier limited learned gains remain valid' in en
        image = section.find('img')
        assert image['data-i18n-alt-zh'] and image['data-i18n-alt-en']
        assert (file.parent/image['src']).resolve().is_file()
        assert not re.search(r'/Users/|/Volumes/|[a-f0-9]{64}', str(section))


def test_chart_is_nonblank():
    image = Image.open(ROOT/'assets/figures/poolfire_potential_query_capacity_20261006.png').convert('RGB')
    assert image.size == (1320, 600) and len(image.getcolors(maxcolors=1000000)) > 100
