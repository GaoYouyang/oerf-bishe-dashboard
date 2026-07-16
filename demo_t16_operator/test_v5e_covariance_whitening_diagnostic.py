from __future__ import annotations

import pytest

from demo_t16_operator.run_v5e_covariance_whitening_diagnostic import (
    paired_summary,
    validated_variants,
)


def test_validated_variants_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown whitening"):
        validated_variants(
            {
                "variants": [
                    {"id": "bad", "whitening": "mystery", "method": "gcv"}
                ]
            }
        )


def test_validated_variants_requires_paired_modes_per_method() -> None:
    with pytest.raises(ValueError, match="requires one variant"):
        validated_variants(
            {
                "variants": [
                    {
                        "id": "diagonal_gcv",
                        "whitening": "diagonal",
                        "method": "gcv",
                    },
                    {
                        "id": "exact_upre",
                        "whitening": "exact_covariance_oracle",
                        "method": "upre",
                    },
                ]
            }
        )


def test_paired_summary_reports_exact_minus_diagonal() -> None:
    variants = [
        {"id": "diagonal_gcv", "whitening": "diagonal", "method": "gcv"},
        {
            "id": "exact_gcv",
            "whitening": "exact_covariance_oracle",
            "method": "gcv",
        },
    ]
    base = {
        "mean_selected_whitened_discrepancy": 0.5,
        "mean_selected_df_corrected_discrepancy": 1.0,
    }
    summaries = [
        {"variant_id": "diagonal_gcv", "nearest_bank_match_count": 2, **base},
        {"variant_id": "exact_gcv", "nearest_bank_match_count": 4, **base},
    ]
    result = paired_summary(summaries, variants)
    assert result[0]["match_delta_exact_minus_diagonal"] == 2
