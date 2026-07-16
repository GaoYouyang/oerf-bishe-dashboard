#!/usr/bin/env python3
"""Plot the adjoint-only view-decomposed routing mechanism probe."""

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
    "view": "#d1902f",
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
            "vd0_pooled_initial_normal_strict",
            "Pooled g0",
            COLORS["pooled"],
        ),
        (
            "vd0_view_conflict_strict",
            "View conflict",
            COLORS["view"],
        ),
        (
            "vd0_pooled_plus_view_conflict_strict",
            "Pooled + view",
            COLORS["combined"],
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
            for method, _, _ in methods
        ]
        axes[0, 0].bar(
            x + (split_index - 0.5) * width,
            values,
            width=width,
            color=[
                color if split_index == 0 else "#aebbb8"
                for _, _, color in methods
            ],
            edgecolor=[color for _, _, color in methods],
            linewidth=1.1,
            label=label,
        )
    axes[0, 0].axhline(
        2.0,
        color=COLORS["ink"],
        linestyle="--",
        linewidth=1.0,
    )
    axes[0, 0].set_xticks(x, [label for _, label, _ in methods])
    axes[0, 0].set_ylabel("Mean field error reduction (%)")
    axes[0, 0].set_title(
        "A  Adjoint-only view features do not beat pooled transfer",
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
        for method, _, _ in methods
    ]
    calibration_harm = [
        100.0
        * float(
            lookup[(method, "risk_calibration")][
                "harm_over_one_percent_rate"
            ]
        )
        for method, _, _ in methods
    ]
    axes[0, 1].bar(
        x - width / 2,
        calibration_front,
        width=width,
        color=[color for _, _, color in methods],
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
    axes[0, 1].set_xticks(x, [label for _, label, _ in methods])
    axes[0, 1].set_ylabel("Calibration front-F1 mean gain (%)")
    harm_axis.set_ylabel("Calibration field harm rate (%)")
    axes[0, 1].set_title(
        "B  Combined features lose utility and still harm fronts",
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

    stress = report["stress_audits"]
    stress_names = (
        "leave_one_morphology_family_out",
        "leave_one_noise_profile_out",
    )
    for method_index, (_, label, color) in enumerate(methods):
        feature_set = (
            "pooled_initial_normal",
            "view_conflict",
            "pooled_plus_view_conflict",
        )[method_index]
        values = [
            float(
                stress[feature_set]["strict"][stress_name][
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
        "C  View conflict softens family shift but fails noise shift",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    axes[1, 0].legend(frameon=False, fontsize=8.2)

    checks = report["regeneration_checks"]
    runtime = report["runtime"]
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.94,
        "D  Decision: interface PASS, representation NO-GO",
        fontsize=11.2,
        fontweight="bold",
        transform=axes[1, 1].transAxes,
    )
    combined_validation = float(
        lookup[(methods[2][0], "risk_validation")][
            "mean_field_gain_percent"
        ]
    )
    combined_calibration = float(
        lookup[(methods[2][0], "risk_calibration")][
            "mean_field_gain_percent"
        ]
    )
    lines = (
        (
            "Grouped sum relative error",
            f"{checks['maximum_group_sum_relative_error']:.2e}",
            "PASS",
        ),
        (
            "Wall time / peak RSS",
            (
                f"{runtime['wall_seconds']:.2f} s / "
                f"{runtime['maximum_rss_bytes'] / 1e6:.0f} MB"
            ),
            "LOCAL",
        ),
        (
            "Combined validation / calibration",
            f"{combined_validation:.3f}% / {combined_calibration:.3f}%",
            "FAIL",
        ),
        (
            "Next permitted feature work",
            "image-plane front proxy + camera pose",
            "ONE STEP",
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
            color=(
                COLORS["negative"]
                if status == "FAIL"
                else COLORS["pooled"]
            ),
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.96,
            y - 0.04,
            status,
            fontsize=8.3,
            ha="right",
            color=COLORS["ink"],
            transform=axes[1, 1].transAxes,
        )
        y -= 0.2

    fig.suptitle(
        "PSU B0 VD0-A: camera-wise adjoint conflict mechanism probe",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
        color="#24343a",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
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
