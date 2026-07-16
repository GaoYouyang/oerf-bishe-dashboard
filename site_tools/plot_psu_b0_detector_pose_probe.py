#!/usr/bin/env python3
"""Plot the detector-graph and camera-pose routing mechanism probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "pooled": "#147d70",
    "detector": "#d1902f",
    "combined": "#3578a8",
    "negative": "#b14f61",
    "grid": "#dfe7e4",
    "ink": "#52636b",
}


def _paired_lookup(report: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row["candidate_method"]), str(row["split"])): row
        for row in report["paired_gain_summary"]
    }


def build_figure(report: dict, output: Path) -> None:
    lookup = _paired_lookup(report)
    methods = (
        (
            "vd0b_pooled_initial_normal_strict",
            "Pooled g0",
            COLORS["pooled"],
            "pooled_initial_normal",
        ),
        (
            "vd0b_detector_graph_front_strict",
            "Detector graph",
            COLORS["detector"],
            "detector_graph_front",
        ),
        (
            "vd0b_pooled_plus_detector_graph_front_strict",
            "Pooled + detector",
            COLORS["combined"],
            "pooled_plus_detector_graph_front",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))
    fig.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)

    x = np.arange(len(methods), dtype=np.float64)
    width = 0.34
    for split_index, (split, label) in enumerate(
        (
            ("risk_validation", "Validation"),
            ("risk_calibration", "Calibration"),
        )
    ):
        values = [
            float(lookup[(method, split)]["mean_field_gain_percent"])
            for method, _, _, _ in methods
        ]
        axes[0, 0].bar(
            x + (split_index - 0.5) * width,
            values,
            width=width,
            color=[
                color if split_index == 0 else "#aebbb8"
                for _, _, color, _ in methods
            ],
            edgecolor=[color for _, _, color, _ in methods],
            linewidth=1.1,
            label=label,
        )
    axes[0, 0].axhline(
        2.0,
        color=COLORS["ink"],
        linestyle="--",
        linewidth=1.0,
    )
    axes[0, 0].set_xticks(x, [label for _, label, _, _ in methods])
    axes[0, 0].set_ylabel("Mean field error reduction (%)")
    axes[0, 0].set_title(
        "A  Detector features transfer, but do not beat pooled safely",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    axes[0, 0].legend(frameon=False, fontsize=8.5)

    calibration_front = [
        float(
            lookup[(method, "risk_calibration")][
                "secondary_metric_gain"
            ]["front_top10_f1"]["mean_gain_percent"]
        )
        for method, _, _, _ in methods
    ]
    calibration_harm = [
        100.0
        * float(
            lookup[(method, "risk_calibration")][
                "harm_over_one_percent_rate"
            ]
        )
        for method, _, _, _ in methods
    ]
    axes[0, 1].bar(
        x - width / 2,
        calibration_front,
        width=width,
        color=[color for _, _, color, _ in methods],
        label="Front-F1 mean gain",
    )
    harm_axis = axes[0, 1].twinx()
    harm_axis.bar(
        x + width / 2,
        calibration_harm,
        width=width,
        color="#d9a6af",
        edgecolor=COLORS["negative"],
        label="Field harm rate",
    )
    axes[0, 1].axhline(0.0, color=COLORS["ink"], linewidth=1.0)
    harm_axis.axhline(
        5.0,
        color=COLORS["negative"],
        linestyle="--",
        linewidth=0.9,
    )
    axes[0, 1].set_xticks(x, [label for _, label, _, _ in methods])
    axes[0, 1].set_ylabel("Calibration front-F1 mean gain (%)")
    harm_axis.set_ylabel("Calibration field harm rate (%)")
    axes[0, 1].set_title(
        "B  Front safety remains negative and harm exceeds 5%",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    handles_a, labels_a = axes[0, 1].get_legend_handles_labels()
    handles_b, labels_b = harm_axis.get_legend_handles_labels()
    axes[0, 1].set_zorder(harm_axis.get_zorder() + 1)
    axes[0, 1].patch.set_visible(False)
    axes[0, 1].legend(
        handles_a + handles_b,
        labels_a + labels_b,
        frameon=False,
        fontsize=8.2,
        loc="lower left",
    )
    harm_axis.spines["top"].set_visible(False)
    harm_axis.grid(False)

    stress_names = (
        "leave_one_morphology_family_out",
        "leave_one_noise_profile_out",
    )
    for _, label, color, feature_set in methods:
        values = [
            float(
                report["stress_audits"][feature_set]["strict"][stress_name][
                    "mean_field_gain_percent"
                ]
            )
            for stress_name in stress_names
        ]
        axes[1, 0].plot(
            [0, 1],
            values,
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
    axes[1, 0].axhline(0.0, color=COLORS["ink"], linewidth=1.0)
    axes[1, 0].set_xticks(
        [0, 1],
        ["Leave one family out", "Leave one noise profile out"],
    )
    axes[1, 0].set_ylabel("Train stress mean field gain (%)")
    axes[1, 0].set_title(
        "C  Real detector coordinates recover a stress signal",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    axes[1, 0].legend(frameon=False, fontsize=8.2)

    graph = report["detector_graph_diagnostics"]
    runtime = report["runtime"]
    combined = lookup[
        (
            "vd0b_pooled_plus_detector_graph_front_strict",
            "risk_calibration",
        )
    ]
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.94,
        "D  Decision: stop set encoder; request real covariance",
        fontsize=11.2,
        fontweight="bold",
        transform=axes[1, 1].transAxes,
    )
    lines = (
        (
            "Detector graph",
            (
                f"{graph['view_count']} views x "
                f"{graph['rays_per_view']} rays, k={graph['neighbor_count']}"
            ),
            "REAL COORDS",
        ),
        (
            "Median nearest / 8th-neighbor distance",
            (
                f"{graph['nearest_neighbor_distance_median']:.4f} / "
                f"{graph['furthest_selected_neighbor_distance_median']:.4f}"
            ),
            "AUDITED",
        ),
        (
            "Combined calibration field / front",
            (
                f"{combined['mean_field_gain_percent']:.3f}% / "
                f"{combined['secondary_metric_gain']['front_top10_f1']['mean_gain_percent']:.3f}%"
            ),
            "NO-GO",
        ),
        (
            "Wall time / peak RSS",
            (
                f"{runtime['wall_seconds']:.2f} s / "
                f"{runtime['maximum_rss_bytes'] / 1e6:.0f} MB"
            ),
            "LOCAL",
        ),
    )
    y = 0.78
    for title, value, status in lines:
        axes[1, 1].text(
            0.0,
            y,
            title,
            fontsize=9.2,
            color=COLORS["ink"],
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.0,
            y - 0.07,
            value,
            fontsize=12.0,
            fontweight="bold",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.94,
            y - 0.035,
            status,
            fontsize=8.0,
            color=(
                COLORS["negative"]
                if status == "NO-GO"
                else COLORS["pooled"]
            ),
            fontweight="bold",
            ha="right",
            transform=axes[1, 1].transAxes,
        )
        y -= 0.19

    fig.suptitle(
        "VD0-B detector-graph + camera-pose probe",
        x=0.05,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#243b42",
    )
    fig.text(
        0.05,
        0.015,
        (
            "Post-open mechanism probe on real PSU support/detector geometry "
            "with analytic morphologies and synthetic noise; no real "
            "deflection values and no superiority claim."
        ),
        fontsize=8.2,
        color=COLORS["ink"],
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    build_figure(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
