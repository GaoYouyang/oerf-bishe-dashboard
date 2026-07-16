from __future__ import annotations

import numpy as np

from demo_t16_operator.run_v5r_reserved_view_reliability_diagnosis import (
    residual_gain,
    rule_masks,
)


def test_residual_gain_prefers_lower_error() -> None:
    operator = np.eye(2)
    label = np.asarray([1.0, 2.0])
    candidate = np.asarray([0.9, 1.8])
    baseline = np.asarray([0.5, 1.0])
    assert residual_gain(operator, label, candidate, baseline) > 0.0
    assert residual_gain(operator, label, baseline, candidate) < 0.0


def test_sign_rules_are_fixed_boolean_combinations() -> None:
    rows = [
        {"field_uid": "a", "source_gain_vs_pbb9": 0.1, "reserved_gain_vs_pbb9": -0.1},
        {"field_uid": "b", "source_gain_vs_pbb9": 0.2, "reserved_gain_vs_pbb9": 0.3},
    ]
    masks = rule_masks(rows)
    assert masks["source_positive"] == {"a": True, "b": True}
    assert masks["reserved_positive"] == {"a": False, "b": True}
    assert masks["source_and_reserved_positive"] == {"a": False, "b": True}
    assert masks["source_or_reserved_positive"] == {"a": True, "b": True}
