#!/usr/bin/env python3
"""Plot the real-versus-synthetic detector-feature domain audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "validation": "#147d70",
    "calibration": "#3578a8",
    "real": "#b14f61",
    "gold": "#d1902f",
    "grid": "#dfe7e4",
    "ink": "#52636b",
}


def build_figure(report: dict, output: Path) -> None:
    configuration = report.get("configuration", report.get("configuration_public"))
    if configuration is None:
        raise KeyError("report must contain configuration or configuration_public")
    comparisons = (
        (
            "Validation",
            report["synthetic_internal_comparisons"][
                "risk_validation_vs_risk_train"
            ],
            COLORS["validation"],
        ),
        (
            "Calibration",
            report["synthetic_internal_comparisons"][
                "risk_calibration_vs_risk_train"
            ],
            COLORS["calibration"],
        ),
        (
            "Real PSU subsets",
            report["real_measurement_comparison"],
            COLORS["real"],
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))
    fig.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=COLORS["grid"], linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)

    x = np.arange(len(comparisons), dtype=np.float64)
    width = 0.34
    center_median = [
        row["robust_center_distance"]["median"]
        for _, row, _ in comparisons
    ]
    nearest_median = [
        row["nearest_reference_distance"]["median"]
        for _, row, _ in comparisons
    ]
    axes[0, 0].bar(
        x - width / 2,
        center_median,
        width=width,
        color=[color for _, _, color in comparisons],
        label="Distance to train center",
    )
    axes[0, 0].bar(
        x + width / 2,
        nearest_median,
        width=width,
        color="#c4cfcc",
        edgecolor=[color for _, _, color in comparisons],
        label="Nearest train row",
    )
    axes[0, 0].set_xticks(x, [label for label, _, _ in comparisons])
    axes[0, 0].set_ylabel("Robust normalized distance (median)")
    axes[0, 0].set_title(
        "A  Real data are farther from synthetic train",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    axes[0, 0].legend(frameon=False, fontsize=8.5)

    outside_any = [
        100.0
        * row["candidate_rows_outside_any_train_95pct_feature_envelope"]
        for _, row, _ in comparisons
    ]
    mean_feature_outside = [
        100.0 * row["mean_informative_feature_outside_fraction"]
        for _, row, _ in comparisons
    ]
    axes[0, 1].bar(
        x - width / 2,
        outside_any,
        width=width,
        color=[color for _, _, color in comparisons],
        label="Rows outside any feature",
    )
    axes[0, 1].bar(
        x + width / 2,
        mean_feature_outside,
        width=width,
        color="#c4cfcc",
        edgecolor=[color for _, _, color in comparisons],
        label="Mean feature outside rate",
    )
    axes[0, 1].set_xticks(x, [label for label, _, _ in comparisons])
    axes[0, 1].set_ylabel("Outside train 95% envelope (%)")
    axes[0, 1].set_title(
        "B  Every real camera subset leaves the envelope",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    axes[0, 1].legend(frameon=False, fontsize=8.5)

    top = report["real_measurement_comparison"][
        "highest_outside_fraction_features"
    ][:6]
    labels = [
        row["feature"]
        .replace("neighbor_contrast_to_signal_ratio_", "contrast ")
        .replace("log_local_jacobian_rms_", "log Jacobian ")
        for row in top
    ][::-1]
    lower = np.asarray([row["train_q025"] for row in top][::-1])
    upper = np.asarray([row["train_q975"] for row in top][::-1])
    real = np.asarray([row["candidate_median"] for row in top][::-1])
    y = np.arange(len(top), dtype=np.float64)
    axes[1, 0].hlines(
        y,
        lower,
        upper,
        color=COLORS["validation"],
        linewidth=7,
        alpha=0.62,
        label="Synthetic train 2.5-97.5%",
    )
    axes[1, 0].scatter(
        real,
        y,
        color=COLORS["real"],
        s=48,
        zorder=3,
        label="Real subset median",
    )
    axes[1, 0].set_yticks(y, labels)
    axes[1, 0].set_xlabel("Self-normalized detector feature value")
    axes[1, 0].set_title(
        "C  Real local contrast and Jacobian exceed the synthetic range",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    axes[1, 0].legend(frameon=False, fontsize=8.2)

    real_summary = report["real_measurement_comparison"]
    runtime = report["runtime"]
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.94,
        "D  Decision: repair the data model before the neural model",
        fontsize=11.2,
        fontweight="bold",
        transform=axes[1, 1].transAxes,
    )
    lines = (
        (
            "Independent real physical fields",
            str(configuration["real_physical_field_count"]),
            "LIMIT",
        ),
        (
            "Deterministic camera subsets",
            str(configuration["real_camera_subset_count"]),
            "NOT IID",
        ),
        (
            "Real center / nearest median distance",
            (
                f"{real_summary['robust_center_distance']['median']:.3f} / "
                f"{real_summary['nearest_reference_distance']['median']:.3f}"
            ),
            "OOD",
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
    y_text = 0.78
    for title, value, status in lines:
        axes[1, 1].text(
            0.0,
            y_text,
            title,
            fontsize=9.2,
            color=COLORS["ink"],
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.0,
            y_text - 0.07,
            value,
            fontsize=12.0,
            fontweight="bold",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.94,
            y_text - 0.035,
            status,
            fontsize=8.0,
            color=(
                COLORS["real"]
                if status in {"LIMIT", "OOD"}
                else COLORS["validation"]
            ),
            fontweight="bold",
            ha="right",
            transform=axes[1, 1].transAxes,
        )
        y_text -= 0.19

    fig.suptitle(
        "Real PSU vs synthetic detector-feature domain audit",
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
            "One real physical flow field under 130 deterministic camera "
            "subsets; per-view RMS normalization is not measured noise "
            "whitening. No reconstruction or route training."
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
