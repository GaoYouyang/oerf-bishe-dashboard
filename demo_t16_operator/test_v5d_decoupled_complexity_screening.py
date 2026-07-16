import pytest

from demo_t16_operator.run_v5d_decoupled_complexity_screening import (
    summarize_screening,
    validated_variants,
)


def test_variant_contract_rejects_duplicates_and_nested_cv() -> None:
    with pytest.raises(ValueError, match="unique"):
        validated_variants(
            {"variants": [{"id": "same", "method": "gcv"}, {"id": "same", "method": "upre"}]}
        )
    with pytest.raises(ValueError, match="phase-A"):
        validated_variants({"variants": [{"id": "nested", "method": "nested_cv"}]})


def test_screening_rank_prioritizes_truth_match_then_nonboundary() -> None:
    rows = [
        {
            "variant_id": "boundary",
            "method": "gcv",
            "nearest_bank_match": True,
            "selected_radius_boundary": False,
            "fold_kappa_boundary_count": 4,
            "fold_score_deletion_radius_stability_fraction": 1.0,
            "mean_outer_validation_mse": 1.0,
            "mean_selected_effective_df_fraction": 0.5,
            "mean_selected_whitened_discrepancy": 1.0,
            "mean_selected_df_corrected_discrepancy": 1.1,
        },
        {
            "variant_id": "interior",
            "method": "upre",
            "nearest_bank_match": True,
            "selected_radius_boundary": False,
            "fold_kappa_boundary_count": 0,
            "fold_score_deletion_radius_stability_fraction": 0.75,
            "mean_outer_validation_mse": 2.0,
            "mean_selected_effective_df_fraction": 0.6,
            "mean_selected_whitened_discrepancy": 0.8,
            "mean_selected_df_corrected_discrepancy": 0.9,
        },
    ]

    summary = summarize_screening(rows, ["boundary", "interior"])

    assert summary[0]["variant_id"] == "interior"
    assert summary[0]["development_rank"] == 1
