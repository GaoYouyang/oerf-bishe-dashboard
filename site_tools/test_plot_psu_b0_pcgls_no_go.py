from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_pcgls_no_go import build_figure


def test_build_figure_writes_three_formats(tmp_path: Path) -> None:
    summaries = []
    for index, split in enumerate(
        (
            "risk_validation",
            "risk_calibration",
            "fresh_iid_support",
            "fresh_family_ood",
            "fresh_correlated_noise_ood",
            "fresh_family_noise_ood",
            "fresh_geometry_ood",
            "fresh_joint_ood",
            "fresh_exact_operator_control",
        )
    ):
        summaries.append(
            {
                "split": split,
                "sample_count": 24,
                "pcgls_relative_error_reduction_mean_percent": 4.0 + index,
                "bootstrap_mean_95_interval_percent": [3.0 + index, 5.0 + index],
                "pcgls_win_rate": 0.9,
                "pcgls_win_count": 22,
            }
        )
    frontier = []
    for method, calls, error in (
        ("sobolev_selected", 8, 0.7),
        ("raw_learned_seed_mean", 8, 0.68),
        ("gated_learned_seed_mean", 8, 0.69),
        ("pcgls_3_selected", 6, 0.66),
        ("pcgls_4_selected", 8, 0.62),
    ):
        frontier.append(
            {
                "method": method,
                "total_operator_calls": calls,
                "field_relative_l2_mean": error,
            }
        )
    output = tmp_path / "figure.png"
    build_figure(
        {
            "paired_split_summary": summaries,
            "pooled_fresh_frontier": frontier,
        },
        output,
    )
    assert output.exists()
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".svg").exists()
