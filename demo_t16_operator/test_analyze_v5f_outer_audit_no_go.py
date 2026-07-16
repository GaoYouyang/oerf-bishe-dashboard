from __future__ import annotations

import pytest

from demo_t16_operator.analyze_v5f_outer_audit_no_go import (
    diagnose_method,
    safe_correlation,
)


def test_safe_correlation_detects_negative_relation() -> None:
    assert safe_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_diagnosis_counts_accepted_audit_harm() -> None:
    rows = [
        {
            "reconstruction_method": "gcv",
            "radius_changed_from_metadata": "True",
            "route_ge_0p0pct": "True",
            "minimum_outer_error_reduction_percent": "2.0",
            "raw_audit_error_reduction_percent": "-3.0",
        },
        {
            "reconstruction_method": "gcv",
            "radius_changed_from_metadata": "True",
            "route_ge_0p0pct": "False",
            "minimum_outer_error_reduction_percent": "-1.0",
            "raw_audit_error_reduction_percent": "4.0",
        },
    ]
    result = diagnose_method(rows, "gcv")
    assert result["outer_nonworse_accepted_count"] == 1
    assert result["accepted_audit_harm_count"] == 1
    assert result["outer_vs_audit_correlation_changed_radius"] == pytest.approx(-1.0)
