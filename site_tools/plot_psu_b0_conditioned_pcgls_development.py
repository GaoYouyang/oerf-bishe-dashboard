#!/usr/bin/env python3
"""Plot the fixed-SPD conditioned-PCGLS development no-go."""

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
    "calibration": "#b14f61",
    "neutral": "#52636b",
    "light": "#dce6e3",
}


def build_figure(report: dict, output: Path) -> None:
    summaries = report["paired_gain_summary"]
    training = report["training"]
    seeds = [int(row["seed"]) for row in training]
    short = {seed: str(seed)[-2:] for seed in seeds}
    lookup = {
        (
            int(str(row["candidate_method"]).rsplit("_", 1)[-1]),
            str(row["split"]),
        ): row
        for row in summaries
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    fig.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dfe7e4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)

    x = np.arange(len(seeds), dtype=np.float64)
    for offset, split, color, label in (
        (-0.12, "risk_validation", COLORS["validation"], "Validation"),
        (0.12, "risk_calibration", COLORS["calibration"], "Calibration"),
    ):
        values = [lookup[(seed, split)] for seed in seeds]
        mean = np.asarray([row["mean_field_gain_percent"] for row in values])
        low = np.asarray(
            [row["bootstrap_mean_95_interval_percent"][0] for row in values]
        )
        high = np.asarray(
            [row["bootstrap_mean_95_interval_percent"][1] for row in values]
        )
        axes[0, 0].errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            linewidth=1.8,
            markersize=7,
            label=label,
        )
    axes[0, 0].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[0, 0].axhline(
        2.0,
        color="#88969b",
        linewidth=1.0,
        linestyle="--",
    )
    axes[0, 0].set_xticks(x, [f"Seed {short[seed]}" for seed in seeds])
    axes[0, 0].set_ylabel("Field error reduction vs static PCGLS-4 (%)")
    axes[0, 0].set_title(
        "A  No seed reaches the 2% development gate",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].legend(frameon=False)

    width = 0.34
    validation_p10 = np.asarray(
        [lookup[(seed, "risk_validation")]["p10_field_gain_percent"] for seed in seeds]
    )
    calibration_p10 = np.asarray(
        [lookup[(seed, "risk_calibration")]["p10_field_gain_percent"] for seed in seeds]
    )
    axes[0, 1].bar(
        x - width / 2,
        validation_p10,
        width,
        color=COLORS["validation"],
        label="Validation p10",
    )
    axes[0, 1].bar(
        x + width / 2,
        calibration_p10,
        width,
        color=COLORS["calibration"],
        label="Calibration p10",
    )
    axes[0, 1].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[0, 1].set_xticks(x, [f"Seed {short[seed]}" for seed in seeds])
    axes[0, 1].set_ylabel("Paired field gain p10 (%)")
    axes[0, 1].set_title(
        "B  Lower tails remain negative",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].legend(frameon=False)

    for seed in seeds:
        for split, marker, color in (
            ("risk_validation", "o", COLORS["validation"]),
            ("risk_calibration", "s", COLORS["calibration"]),
        ):
            row = lookup[(seed, split)]
            measurement = row["secondary_metric_gain"][
                "measurement_relative_l2"
            ]["mean_gain_percent"]
            field = row["mean_field_gain_percent"]
            axes[1, 0].scatter(
                measurement,
                field,
                marker=marker,
                color=color,
                s=62,
            )
            axes[1, 0].annotate(
                f"{short[seed]} {'V' if split.endswith('validation') else 'C'}",
                (measurement, field),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
            )
    axes[1, 0].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[1, 0].axvline(0.0, color="#88969b", linewidth=1.0)
    axes[1, 0].set_xlabel("Measurement residual reduction (%)")
    axes[1, 0].set_ylabel("Field error reduction (%)")
    axes[1, 0].set_title(
        "C  Better projection fit does not transfer to the 3-D field",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    for row in training:
        curve = row["learning_curve"]
        axes[1, 1].plot(
            [value["epoch"] for value in curve],
            [value["validation_combined_loss"] for value in curve],
            marker="o",
            linewidth=1.8,
            label=f"Seed {short[int(row['seed'])]}",
        )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Validation combined loss")
    axes[1, 1].set_title(
        "D  Optimization moves, but the PCGLS frontier barely moves",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        "Development no-go: low-dimensional conditioned SPD-PCGLS V1",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#1f3438",
    )
    fig.text(
        0.055,
        0.952,
        (
            "Three seeds, 2,527 parameters, fixed 4F+4AT; training uses risk_train, "
            "selection uses validation, calibration is a one-time development check."
        ),
        ha="left",
        fontsize=10,
        color="#53666b",
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.93), h_pad=2.2, w_pad=1.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    build_figure(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
