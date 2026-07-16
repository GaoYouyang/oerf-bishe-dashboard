"""Tests for the PSU spectral-preconditioner pilot helpers."""

from __future__ import annotations

import pytest
import torch

from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _aggregate,
    _unique_masks,
)


def test_unique_masks_respect_forbidden_and_ranges() -> None:
    first, keys = _unique_masks(
        count=5,
        view_count=5,
        minimum_active=3,
        maximum_active=4,
        seed=8,
        forbidden=set(),
    )
    second, next_keys = _unique_masks(
        count=4,
        view_count=5,
        minimum_active=2,
        maximum_active=3,
        seed=9,
        forbidden=keys,
    )
    assert first.shape == (5, 5)
    assert second.shape == (4, 5)
    assert torch.all((first.sum(dim=1) >= 3) & (first.sum(dim=1) <= 4))
    assert torch.all((second.sum(dim=1) >= 2) & (second.sum(dim=1) <= 3))
    assert len(next_keys) == 9


def test_aggregate_computes_paired_gain_against_sobolev() -> None:
    rows = []
    for sample, baseline, learned in (
        ("a", 1.0, 0.8),
        ("b", 2.0, 2.2),
    ):
        for method, field in (
            ("sobolev_selected", baseline),
            ("learned_seed_1", learned),
        ):
            rows.append(
                {
                    "sample_id": sample,
                    "split": "test_iid",
                    "method": method,
                    "field_relative_l2": field,
                    "gradient_relative_l2": field,
                    "front_top10_f1": 0.5,
                    "combined_loss": field,
                    "measurement_relative_l2": 0.2,
                }
            )
    output = _aggregate(rows, baseline_method="sobolev_selected")
    learned = next(row for row in output if row["method"] == "learned_seed_1")
    assert learned["field_gain_vs_sobolev_mean_percent"] == pytest.approx(5.0)
    assert learned["field_harm_over_one_percent_rate"] == 0.5
