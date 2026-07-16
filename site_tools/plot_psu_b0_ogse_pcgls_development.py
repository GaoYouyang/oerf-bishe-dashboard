#!/usr/bin/env python3
"""Plot the observable greedy spectral-expert PCGLS audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "strict": "#147d70",
    "diagnostic": "#b14f61",
    "oracle": "#d1902f",
    "neutral": "#52636b",
    "gate": "#dfe7e4",
}


def _summary_lookup(report: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row["candidate_method"]), str(row["split"])): row
        for row in report["paired_gain_summary"]
    }


def build_figure(report: dict, output: Path) -> None:
    lookup = _summary_lookup(report)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    fig.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dfe7e4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)

    bank_items = sorted(
        report["expert_banks"].items(),
        key=lambda item: int(item[0]),
    )
    largest_trajectory = bank_items[-1][1]["trajectory"]
    axes[0, 0].plot(
        [row["bank_size"] for row in largest_trajectory],
        [
            row["mean_oracle_field_gain_percent"]
            for row in largest_trajectory
        ],
        marker="o",
        color=COLORS["neutral"],
        linewidth=1.8,
        label="Greedy additions",
    )
    final_sizes = [int(size) for size, _ in bank_items]
    final_gains = [
        float(record["trajectory"][-1]["mean_oracle_field_gain_percent"])
        for _, record in bank_items
    ]
    axes[0, 0].scatter(
        final_sizes,
        final_gains,
        color=COLORS["oracle"],
        edgecolor="#ffffff",
        linewidth=1.0,
        s=82,
        zorder=3,
        label="Screened bank sizes",
    )
    axes[0, 0].axhline(
        2.0,
        color=COLORS["neutral"],
        linewidth=1.0,
        linestyle="--",
    )
    axes[0, 0].set_xlabel("Number of fixed SPD experts")
    axes[0, 0].set_ylabel("Train truth-oracle field gain (%)")
    axes[0, 0].set_title(
        "A  A small spectral bank contains useful conditional headroom",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0, 0].legend(frameon=False, fontsize=8.5)

    methods = ("ogse_strict", "ogse_diagnostic")
    method_labels = ("Risk-gated", "Diagnostic")
    split_specs = (
        ("risk_validation", "Validation", -0.12, "o"),
        ("risk_calibration", "Calibration", 0.12, "s"),
    )
    x = np.arange(len(methods), dtype=np.float64)
    for split, label, offset, marker in split_specs:
        rows = [lookup[(method, split)] for method in methods]
        means = np.asarray([row["mean_field_gain_percent"] for row in rows])
        lows = np.asarray(
            [row["bootstrap_mean_95_interval_percent"][0] for row in rows]
        )
        highs = np.asarray(
            [row["bootstrap_mean_95_interval_percent"][1] for row in rows]
        )
        axes[0, 1].errorbar(
            x + offset,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt=marker,
            color=(
                COLORS["strict"]
                if split == "risk_validation"
                else COLORS["diagnostic"]
            ),
            capsize=4,
            linewidth=1.8,
            markersize=7,
            label=label,
        )
    axes[0, 1].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[0, 1].axhline(
        2.0,
        color=COLORS["neutral"],
        linewidth=1.0,
        linestyle="--",
    )
    axes[0, 1].set_xticks(x, method_labels)
    axes[0, 1].set_ylabel("Field error reduction vs static PCGLS-4 (%)")
    axes[0, 1].set_title(
        "B  The safe route misses the 2% calibration gate",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0, 1].legend(frameon=False)

    labels = []
    minimum = []
    harm_rate = []
    colors = []
    for method, method_label in zip(methods, method_labels, strict=True):
        for split, split_label, _, _ in split_specs:
            row = lookup[(method, split)]
            labels.append(f"{method_label}\n{split_label}")
            minimum.append(float(row["minimum_field_gain_percent"]))
            harm_rate.append(100.0 * float(row["harm_over_one_percent_rate"]))
            colors.append(
                COLORS["strict"]
                if method == "ogse_strict"
                else COLORS["diagnostic"]
            )
    tail_x = np.arange(len(labels), dtype=np.float64)
    axes[1, 0].bar(tail_x, minimum, color=colors, width=0.64)
    axes[1, 0].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[1, 0].set_xticks(tail_x, labels, fontsize=8.5)
    axes[1, 0].set_ylabel("Worst paired field gain (%)")
    axes[1, 0].set_title(
        "C  Removing false interventions eliminates catastrophic tails",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )
    for index, rate in enumerate(harm_rate):
        label_y = (
            -0.55
            if minimum[index] > -0.4
            else minimum[index] + 0.35
        )
        axes[1, 0].text(
            index,
            label_y,
            f"harm >1%: {rate:.1f}%",
            ha="center",
            va="top" if minimum[index] > -0.4 else "bottom",
            fontsize=8,
            color="#263b40",
        )

    axes[1, 1].axvspan(
        0.0,
        5.0,
        color=COLORS["gate"],
        alpha=0.75,
        label="Risk gate",
    )
    axes[1, 1].axhspan(
        2.0,
        6.0,
        color="#edf4ef",
        alpha=0.65,
    )
    for method, method_label in zip(methods, method_labels, strict=True):
        for split, split_label, _, marker in split_specs:
            row = lookup[(method, split)]
            risk = 100.0 * float(row["harm_over_one_percent_rate"])
            gain = float(row["mean_field_gain_percent"])
            color = (
                COLORS["strict"]
                if method == "ogse_strict"
                else COLORS["diagnostic"]
            )
            axes[1, 1].scatter(
                risk,
                gain,
                marker=marker,
                color=color,
                s=72,
            )
            axes[1, 1].annotate(
                f"{method_label} {split_label}",
                (risk, gain),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.2,
            )
    axes[1, 1].axvline(
        5.0,
        color=COLORS["neutral"],
        linewidth=1.0,
        linestyle="--",
    )
    axes[1, 1].axhline(
        2.0,
        color=COLORS["neutral"],
        linewidth=1.0,
        linestyle="--",
    )
    axes[1, 1].set_xlim(-0.25, max(8.0, max(harm_rate) + 1.0))
    axes[1, 1].set_ylim(0.0, 4.2)
    axes[1, 1].set_xlabel("Samples harmed by more than 1% (%)")
    axes[1, 1].set_ylabel("Mean field error reduction (%)")
    axes[1, 1].set_title(
        "D  No route satisfies both utility and safety gates",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
    )

    fig.suptitle(
        "OGSE-PCGLS v2: observable morphology helps, but does not yet pass",
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
            "Shared first adjoint, fixed 4F+4AT, train-only expert-bank and "
            "OOF selector screening; validation/calibration are post-open "
            "diagnostics and fresh remains excluded."
        ),
        ha="left",
        fontsize=9.8,
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
