#!/usr/bin/env python3
"""Plot the v5u oracle-calibrated renderer-residual no-go."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v5u_calibrated_renderer_residual_screening"
OUTPUT = RESULTS / "v5u_calibrated_renderer_low_rank_no_go.png"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    metrics = _rows("development_rig_metrics.csv")
    summary = report["development_summary"]
    methods = [
        "zero_calibrated_low_renderer",
        "mean_renderer_residual",
        "nearest_calibrated_geometry",
        "full_matrix_geometry_ridge",
        "cal_hosvd_ridge",
        "oracle_shared_subspace",
    ]
    labels = ["calibrated low", "mean", "nearest", "full ridge", "CAL-HOSVD", "oracle subspace"]
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
        "V5U post-open screen: calibration does not make the renderer gap globally low-rank",
        fontsize=16,
        fontweight="bold",
        color="#17252b",
    )

    axis = axes[0, 0]
    current = report["singular_energy"]
    raw = report["raw_v5s_singular_energy"]
    energy_labels = ["measurement-4", "measurement-16", "voxel-4", "voxel-16"]
    keys = ["measurement_first_4", "measurement_first_16", "voxel_first_4", "voxel_first_16"]
    x = np.arange(len(keys))
    axis.bar(x - 0.2, [raw[key] for key in keys], 0.4, color="#6d83a5", label="raw v5s")
    axis.bar(x + 0.2, [current[key] for key in keys], 0.4, color="#b45e52", label="after calibration")
    axis.set_xticks(x, energy_labels, rotation=22, ha="right")
    axis.set_ylim(0.0, 0.45)
    axis.set_ylabel("Cumulative squared singular energy")
    axis.set_title("Calibration barely changes the global HOSVD spectrum")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    errors = [summary[name]["mean_relative_renderer_residual_error"] for name in methods]
    bars = axis.bar(np.arange(len(methods)), errors, color=colors)
    axis.axhline(0.5, color="#8a651b", linestyle="--", linewidth=1, label="oracle gate")
    axis.set_xticks(np.arange(len(methods)), labels, rotation=27, ha="right")
    axis.set_ylabel("Mean relative renderer-residual error")
    axis.set_ylim(0.0, 1.08)
    axis.set_title("CAL-HOSVD 0.809; full-matrix ridge 0.476")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)
    for bar, value in zip(bars, errors, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )

    axis = axes[1, 0]
    per_method: dict[str, dict[str, float]] = {name: {} for name in methods}
    for row in metrics:
        per_method[row["method"]][row["rig_id"]] = float(
            row["relative_renderer_residual_error"]
        )
    rigs = sorted(per_method["cal_hosvd_ridge"])
    full = np.asarray([per_method["full_matrix_geometry_ridge"][rig] for rig in rigs])
    low_rank = np.asarray([per_method["cal_hosvd_ridge"][rig] for rig in rigs])
    axis.scatter(full, low_rank, color="#b45e52", s=48, alpha=0.84)
    limit = max(float(full.max()), float(low_rank.max())) + 0.04
    axis.plot([0.0, limit], [0.0, limit], color="#53666c", linewidth=1, linestyle="--")
    for rig, x_value, y_value in zip(rigs, full, low_rank, strict=True):
        axis.annotate(
            rig.split("-")[-1],
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            color="#53666c",
        )
    axis.set_xlim(0.3, limit)
    axis.set_ylim(0.3, limit)
    axis.set_xlabel("Full-matrix ridge residual error")
    axis.set_ylabel("CAL-HOSVD residual error")
    axis.set_title("Every development rig favors the uncompressed baseline")
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
    axis.set_title("Only 8.39% of raw discrepancy norm is removed by geometry alignment")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)

    figure.text(
        0.5,
        0.005,
        "Truth geometry is supplied and complete residual matrices are used in training; this is not a deployable correction result.",
        ha="center",
        fontsize=9,
        color="#53666c",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
