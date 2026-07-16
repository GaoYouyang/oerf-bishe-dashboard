from __future__ import annotations

import pytest

from site_tools.analyze_psu_b0_pcgls_no_go import (
    _binomial_greater_tail,
    holm_adjust,
    paired_split_summary,
)


def test_binomial_tail_and_holm_adjustment() -> None:
    assert _binomial_greater_tail(4, 4) == pytest.approx(1.0 / 16.0)
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_paired_summary_averages_learned_seeds_by_field() -> None:
    rows = []
    for sample_id, pcgls, learned in (
        ("a", 0.4, (0.5, 0.6)),
        ("b", 0.8, (0.7, 0.9)),
    ):
        rows.append(
            {
                "split": "validation",
                "sample_id": sample_id,
                "method": "pcgls",
                "field_relative_l2": pcgls,
            }
        )
        for seed, value in enumerate(learned):
            rows.append(
                {
                    "split": "validation",
                    "sample_id": sample_id,
                    "method": f"raw_seed_{seed}",
                    "field_relative_l2": value,
                }
            )
    summary = paired_split_summary(
        rows,
        split="validation",
        pcgls_method="pcgls",
        learned_prefix="raw_seed_",
        bootstrap_seed=1,
    )
    assert summary["sample_count"] == 2
    assert summary["pcgls_win_count"] == 1
    assert summary["learned_field_relative_l2_mean"] == pytest.approx(0.675)
