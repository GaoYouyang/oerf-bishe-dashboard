"""Tests for the conditioned-PCGLS development figure."""

from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_conditioned_pcgls_development import (
    build_figure,
)


def test_build_figure_writes_all_formats(tmp_path: Path) -> None:
    seeds = (101, 102, 103)
    summaries = []
    for seed_index, seed in enumerate(seeds):
        for split_index, split in enumerate(
            ("risk_validation", "risk_calibration")
        ):
            mean = 0.1 - 0.1 * seed_index - 0.05 * split_index
            summaries.append(
                {
                    "split": split,
                    "candidate_method": f"conditioned_pcgls_seed_{seed}",
                    "mean_field_gain_percent": mean,
                    "p10_field_gain_percent": mean - 0.4,
                    "bootstrap_mean_95_interval_percent": [
                        mean - 0.2,
                        mean + 0.2,
                    ],
                    "secondary_metric_gain": {
                        "measurement_relative_l2": {
                            "mean_gain_percent": 0.8 + seed_index
                        }
                    },
                }
            )
    report = {
        "paired_gain_summary": summaries,
        "training": [
            {
                "seed": seed,
                "learning_curve": [
                    {
                        "epoch": 5,
                        "validation_combined_loss": 0.56,
                    },
                    {
                        "epoch": 10,
                        "validation_combined_loss": 0.55,
                    },
                ],
            }
            for seed in seeds
        ],
    }
    output = tmp_path / "figure.png"
    build_figure(report, output)
    assert output.stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000
    assert output.with_suffix(".svg").stat().st_size > 1000
