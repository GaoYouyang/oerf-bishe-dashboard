from __future__ import annotations

from pathlib import Path

from site_tools.plot_psu_b0_edge_superiorization_no_go import build_figure


def test_plot_builds_from_frozen_smoke_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    scale = (
        root
        / "demo_t16_operator/results/psu_b0_edge_superiorization_scale_smoke"
    )
    tail = (
        root
        / "demo_t16_operator/results/psu_b0_edge_superiorization_tail_smoke"
    )
    figure = build_figure(
        scale_report_path=scale / "report.json",
        scale_rows_path=scale / "metric_rows.csv",
        tail_report_path=tail / "report.json",
        tail_rows_path=tail / "metric_rows.csv",
    )
    assert len(figure.axes) == 4
