#!/usr/bin/env python3
"""Plot the v5s development-only low-rank operator screen."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v5s_dco_low_rank_screening"
OUTPUT = RESULTS / "v5s_dco_low_rank_no_go.png"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    sweep = _rows("sweep.csv")
    rig_rows = _rows("development_rig_metrics.csv")
    summary = report["development_summary"]
    selected = report["selected_development_hyperparameters"]
    methods = [
        "zero_nominal",
        "mean_discrepancy",
        "nearest_geometry",
        "full_matrix_geometry_ridge",
        "gc_biloc_ridge",
        "oracle_shared_subspace",
    ]
    labels = ["nominal", "mean", "nearest", "full ridge", "GC-BiLOC", "oracle subspace"]
    colors = ["#8a989b", "#6b8e7b", "#6d83a5", "#146f66", "#b45e52", "#8a651b"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#b7c5c2",
            "axes.labelcolor": "#26383e",
            "xtick.color": "#53666c",
            "ytick.color": "#53666c",
            "figure.facecolor": "#f3f6f4",
            "axes.facecolor": "#ffffff",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
        "V5S development screen: global low-rank operator correction is a no-go",
        fontsize=17,
        fontweight="bold",
        color="#17252b",
    )

    axis = axes[0, 0]
    rank_pairs = sorted(
        {
            (int(row["rank_measurement"]), int(row["rank_voxel"]))
            for row in sweep
        }
    )
    best_by_pair = []
    for rank_measurement, rank_voxel in rank_pairs:
        values = [
            float(row["development_mean_relative_discrepancy_error"])
            for row in sweep
            if int(row["rank_measurement"]) == rank_measurement
            and int(row["rank_voxel"]) == rank_voxel
        ]
        best_by_pair.append(min(values))
    pair_labels = [f"{left}x{right}" for left, right in rank_pairs]
    bars = axis.bar(np.arange(len(rank_pairs)), best_by_pair, color="#315f93")
    axis.axhline(
        summary["full_matrix_geometry_ridge"]["mean_relative_discrepancy_error"],
        color="#146f66",
        linestyle="--",
        linewidth=1.5,
        label="full-matrix ridge",
    )
    axis.set_xticks(np.arange(len(rank_pairs)), pair_labels, rotation=50, ha="right")
    axis.set_ylim(0.45, 1.02)
    axis.set_ylabel("Best development discrepancy error")
    axis.set_xlabel("Measurement rank x voxel rank")
    axis.set_title("More rank helps, but every shared subspace trails the cheap baseline")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.16)
    best_index = int(np.argmin(best_by_pair))
    bars[best_index].set_color("#b45e52")

    axis = axes[0, 1]
    discrepancy = [summary[name]["mean_relative_discrepancy_error"] for name in methods]
    bars = axis.bar(np.arange(len(methods)), discrepancy, color=colors)
    axis.axhline(0.5, color="#8a651b", linewidth=1, linestyle="--", label="oracle gate")
    axis.set_xticks(np.arange(len(methods)), labels, rotation=27, ha="right")
    axis.set_ylabel("Mean relative discrepancy error")
    axis.set_ylim(0.0, 1.08)
    axis.set_title("GC-BiLOC: 0.828; full-matrix ridge: 0.522")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)
    for bar, value in zip(bars, discrepancy, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )

    axis = axes[1, 0]
    per_method = {name: [] for name in methods}
    for row in rig_rows:
        per_method[row["method"]].append(float(row["relative_discrepancy_error"]))
    full = np.asarray(per_method["full_matrix_geometry_ridge"])
    low_rank = np.asarray(per_method["gc_biloc_ridge"])
    axis.scatter(full, low_rank, color="#b45e52", s=48, alpha=0.82)
    limit = max(float(full.max()), float(low_rank.max())) + 0.03
    axis.plot([0.0, limit], [0.0, limit], color="#53666c", linewidth=1, linestyle="--")
    for index, (x_value, y_value) in enumerate(zip(full, low_rank, strict=True), start=30):
        axis.annotate(
            str(index),
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            color="#53666c",
        )
    axis.set_xlim(0.35, limit)
    axis.set_ylim(0.35, limit)
    axis.set_xlabel("Full-matrix ridge error")
    axis.set_ylabel("GC-BiLOC error")
    axis.set_title("All 12 development rigs lie above equality")
    axis.grid(alpha=0.16)

    axis = axes[1, 1]
    forward = [100.0 * summary[name]["mean_probe_forward_relative_error"] for name in methods]
    gradient_loss = [
        100.0 * (1.0 - summary[name]["mean_probe_gradient_cosine"]) for name in methods
    ]
    x = np.arange(len(methods))
    axis.bar(x - 0.2, forward, 0.4, color="#315f93", label="forward error")
    axis.bar(x + 0.2, gradient_loss, 0.4, color="#b45e52", label="100 x (1 - grad cosine)")
    axis.set_xticks(x, labels, rotation=27, ha="right")
    axis.set_ylabel("Percent")
    axis.set_title("Good gradient angles do not rescue inaccurate forward amplitudes")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)

    figure.text(
        0.5,
        0.005,
        (
            f"Selected {selected['rank_measurement']}x{selected['rank_voxel']} ranks; "
            "development-only evidence, no reconstruction and no fresh validation claim."
        ),
        ha="center",
        fontsize=9,
        color="#53666c",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
