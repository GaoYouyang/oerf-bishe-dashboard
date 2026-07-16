from __future__ import annotations

import json

import numpy as np

from demo_t16_operator.psu_b0_residual_risk import RISK_FEATURE_NAMES
from site_tools.analyze_psu_b0_residual_risk_postopen import (
    build_public_summary,
    feature_contrasts,
    summarize_view_strata,
)


def _row(*, views: int, gain: float, trusted: bool, shift: float) -> dict:
    return {
        "active_view_count": views,
        "actual_gain_percent": gain,
        "trusted": trusted,
        "standardized_features": [
            shift + 0.05 * index for index in range(len(RISK_FEATURE_NAMES))
        ],
    }


def test_view_strata_reports_accepted_tail_risk() -> None:
    rows = [
        _row(views=6, gain=2.0, trusted=True, shift=0.0),
        _row(views=6, gain=-2.0, trusted=True, shift=1.0),
        _row(views=7, gain=1.0, trusted=True, shift=0.2),
        _row(views=7, gain=-8.0, trusted=False, shift=2.0),
    ]
    summary = summarize_view_strata(rows)
    assert summary[0]["active_view_count"] == 6
    assert summary[0]["accepted_row_count"] == 2
    assert summary[0]["harm_over_one_percent_count"] == 1
    assert summary[0]["harm_over_one_percent_rate"] == 0.5
    assert summary[1]["minimum_gain_percent"] == 1.0


def test_feature_contrasts_rank_the_shifted_observable() -> None:
    width = len(RISK_FEATURE_NAMES)
    safe = []
    for offset in (-0.2, 0.0, 0.2):
        values = np.zeros(width)
        values[5] = offset
        safe.append({"standardized_features": values.tolist()})
    harmful_values = np.zeros(width)
    harmful_values[5] = 3.0
    contrasts = feature_contrasts(
        [{"standardized_features": harmful_values.tolist()}],
        safe,
    )
    assert contrasts[0]["feature"] == RISK_FEATURE_NAMES[5]
    assert contrasts[0]["harmful_vs_safe_effect"] > 10.0


def test_public_summary_strips_private_feature_rows_and_paths() -> None:
    private = {
        "status": "postopen",
        "evidence_scope": "synthetic",
        "source_audit_public": {},
        "accepted_harm_public": [],
        "view_strata": [],
        "calibration_view_support": [],
        "feature_contrasts": [],
        "failure_modes": [],
        "support_order_mismatch": {},
        "exact_view_conformal_probe": {},
        "next_candidate_hypothesis": {},
        "literature_boundary": [],
        "claim_boundary": {},
        "configuration_private": {"root": "/secret/root"},
        "feature_rows_private": [{"features": [1.0, 2.0]}],
    }
    public = build_public_summary(private)
    payload = json.dumps(public, sort_keys=True)
    assert "/secret/root" not in payload
    assert '"features"' not in payload
    assert public["public_export_policy"][
        "contains_all_per_sample_feature_rows"
    ] is False
