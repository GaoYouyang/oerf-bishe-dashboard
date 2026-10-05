"""Publication consistency checks, not scientific recomputation."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_projected_gcv_sentinel_2026-10-05_public_summary.json'


def summary():
    return json.loads((ROOT / 'docs' / STEM).read_text())


def test_fixed_negative_and_scope():
    s = summary()
    assert s['status'] == 'FAIL_PROJECTED_GCV_SENTINEL_REFERENCE_ESCALATION_CLOSED'
    assert s['scope']['cells'] == 11 * 3 * 3 == 99
    assert s['scope']['scored_rows_per_implementation'] == 396
    assert s['scope']['three_frame_sentinel_not_complete_trajectory_validation']
    assert s['independent_checks_passed'] == 14
    for arm in ('primary', 'control'):
        assert s['arms'][arm] == {'absolute_strata': 36, 'stable_strata': 0, 'passing_strata': 0}
    for key in ('full_3333_escalation_authorized', 'new_predictor_training_authorized',
                'algorithm_breakthrough', 'paper_success', 'resource_speedup', 'external_generalization', 'real_bost'):
        assert s[key] is False


def test_numerical_cost_and_readback():
    s = summary()
    assert max(s['maxima'].values()) < 1e-8
    assert all(v == 0 for v in s['sealed_control_reproduction'].values())
    assert s['cost']['shared_solver_per_cell'] == {'A': 128, 'AT': 128}
    assert s['cost']['both_rules_share_directions_and_projected_svd']
    assert not s['cost']['fresh_wall_rss_measured']
    for k in ('64', '128'):
        d = s['primary_minus_control_metric_diagnostics'][k]
        assert d['different_lambda_cells'] == 99
        for metric in ('field_rel_l2_gauge_centered', 'interior_gradient_rel_l2', 'observation_rel_l2'):
            assert d['primary_minus_control_metrics'][metric]['worse_cells'] == 99


def test_bilingual_linked_surfaces_and_retained_parent():
    evidence = json.loads((ROOT / 'operator-learning/current-evidence.json').read_text())
    assert evidence['latest_projected_gcv_sentinel'] == summary()
    assert evidence['latest_hybrid_gcv_reference']['scope']['cells'] == 3333
    assert not evidence['current_decision']['ordinary_projected_gcv_full_escalation_authorized']
    for name in ('index.html', 'operator-learning/index.html', 'operator-learning/daily-progress.html', 'learning_log.html'):
        text = (ROOT / name).read_text()
        assert STEM in text
        assert 'ordinary projected GCV and the old rule' in text
        assert '14/14' in text and '99' in text
    doc = (ROOT / 'docs/poolfire_hybrid_gcv_reference_2026-10-05.md').read_text()
    assert '独立机制对照' in doc and 'Independent Mechanism Control' in doc and STEM in doc


def test_redaction():
    data = (ROOT / 'docs' / STEM).read_text()
    for value in ('/Users/', 'private_results', 'sha256', 'spray', '.pt', 'PHY2309489'):
        assert value not in data
    assert not re.search(r'\b[0-9a-f]{40,64}\b', data)
