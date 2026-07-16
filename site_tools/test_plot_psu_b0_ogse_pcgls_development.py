"""Tests for the OGSE-PCGLS development figure."""

from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_ogse_pcgls_development import build_figure


def test_build_figure_writes_all_formats(tmp_path: Path) -> None:
    summaries = []
    for method_index, method in enumerate(
        ("ogse_strict", "ogse_diagnostic")
    ):
        for split_index, split in enumerate(
            ("risk_validation", "risk_calibration")
        ):
            mean = 2.5 + 0.4 * method_index - 0.7 * split_index
            summaries.append(
                {
                    "candidate_method": method,
                    "split": split,
                    "mean_field_gain_percent": mean,
                    "minimum_field_gain_percent": (
                        -0.001 if method_index == 0 else -6.0
                    ),
                    "harm_over_one_percent_rate": (
                        0.0 if method_index == 0 else 0.08
                    ),
                    "bootstrap_mean_95_interval_percent": [
                        mean - 0.8,
                        mean + 0.8,
                    ],
                }
            )
    report = {
        "expert_banks": {
            "4": {
                "trajectory": [
                    {
                        "bank_size": index,
                        "mean_oracle_field_gain_percent": float(index),
                    }
                    for index in range(1, 5)
                ]
            },
            "6": {
                "trajectory": [
                    {
                        "bank_size": index,
                        "mean_oracle_field_gain_percent": float(index),
                    }
                    for index in range(1, 7)
                ]
            },
        },
        "paired_gain_summary": summaries,
    }
    output = tmp_path / "figure.png"
    build_figure(report, output)
    assert output.stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000
    assert output.with_suffix(".svg").stat().st_size > 1000
