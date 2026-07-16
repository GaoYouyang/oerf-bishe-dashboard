#!/usr/bin/env python3
"""Plot the post-open v5t camera-local tangent diagnosis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v5t_camera_local_tangent_diagnosis"
V5S_RESULTS = ROOT / "results" / "v5s_dco_low_rank_screening"
OUTPUT = RESULTS / "v5t_camera_local_tangent_diagnosis.png"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    metrics = _rows(RESULTS / "rig_metrics.csv")
    decomposition = _rows(RESULTS / "gap_decomposition.csv")
    v5s_metrics = _rows(V5S_RESULTS / "development_rig_metrics.csv")
    summary = report["development_summary"]
    methods = [
        "low_fidelity_nominal",
        "high_fidelity_nominal",
        "first_order_tangent_oracle",
        "diagonal_second_order_oracle",
        "additive_secant_oracle",
    ]
    labels = ["low nominal", "high nominal", "1st tangent", "diag. 2nd", "secant oracle"]
    colors = ["#8a989b", "#6d83a5", "#315f93", "#b45e52", "#146f66"]

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
        "V5T post-open diagnosis: calibrate first, then test structured residuals",
        fontsize=17,
        fontweight="bold",
        color="#17252b",
    )

    axis = axes[0, 0]
    rigs = [row["rig_id"].split("-")[-1] for row in decomposition]
    renderer = [
        float(row["renderer_gap_fraction_of_total_norm"]) for row in decomposition
    ]
    parameter = [
        float(row["parameter_gap_fraction_of_total_norm"]) for row in decomposition
    ]
    x = np.arange(len(rigs))
    axis.bar(x - 0.2, renderer, 0.4, color="#315f93", label="renderer gap")
    axis.bar(x + 0.2, parameter, 0.4, color="#b45e52", label="parameter gap")
    axis.set_xticks(x, rigs, rotation=45, ha="right")
    axis.set_ylabel("Gap norm / total discrepancy norm")
    axis.set_title("Renderer and parameter gaps overlap; fractions need not sum to one")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    total_error = [
        summary[name]["mean_total_discrepancy_relative_error"] for name in methods
    ]
    bars = axis.bar(np.arange(len(methods)), total_error, color=colors)
    v5s_full = float(report["v5s_full_matrix_ridge_mean_relative_discrepancy_error"])
    axis.axhline(
        v5s_full,
        color="#8a651b",
        linestyle="--",
        linewidth=1.5,
        label="v5s full-matrix ridge",
    )
    axis.set_xticks(np.arange(len(methods)), labels, rotation=27, ha="right")
    axis.set_ylabel("Mean total discrepancy relative error")
    axis.set_ylim(0.0, 5.75)
    axis.set_title("Diagonal second order is numerically unstable; secant oracle is best")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)
    for bar, value in zip(bars, total_error, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.09,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )

    axis = axes[1, 0]
    secant = {
        row["rig_id"]: float(row["total_discrepancy_relative_error"])
        for row in metrics
        if row["method"] == "additive_secant_oracle"
    }
    full_ridge = {
        row["rig_id"]: float(row["relative_discrepancy_error"])
        for row in v5s_metrics
        if row["method"] == "full_matrix_geometry_ridge"
    }
    common = sorted(set(secant) & set(full_ridge))
    x_values = np.asarray([full_ridge[rig] for rig in common])
    y_values = np.asarray([secant[rig] for rig in common])
    axis.scatter(x_values, y_values, color="#146f66", s=50, alpha=0.86)
    limit = max(float(x_values.max()), float(y_values.max())) + 0.04
    axis.plot([0.0, limit], [0.0, limit], color="#53666c", linewidth=1, linestyle="--")
    for rig, x_value, y_value in zip(common, x_values, y_values, strict=True):
        axis.annotate(
            rig.split("-")[-1],
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            color="#53666c",
        )
    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_xlabel("V5S full-matrix ridge discrepancy error")
    axis.set_ylabel("Truth-offset secant discrepancy error")
    axis.set_title("Oracle wins 12/12 rigs, but truth offsets make it non-deployable")
    axis.grid(alpha=0.16)

    axis = axes[1, 1]
    parameter_error = [
        summary[name]["mean_parameter_gap_relative_error"] for name in methods
    ]
    forward_error = [
        100.0 * summary[name]["mean_probe_forward_relative_error"] for name in methods
    ]
    x = np.arange(len(methods))
    axis.bar(x - 0.2, parameter_error, 0.4, color="#b45e52", label="parameter-gap error")
    axis.bar(x + 0.2, forward_error, 0.4, color="#315f93", label="forward error (%)")
    axis.axhline(0.35, color="#8a651b", linestyle="--", linewidth=1, label="parameter gate")
    axis.set_xticks(x, labels, rotation=27, ha="right")
    axis.set_ylabel("Mixed diagnostic scale")
    axis.set_title("Secant improves task error but still misses 56.1% of parameter gap")
    axis.grid(axis="y", alpha=0.16)
    axis.legend(fontsize=8)

    figure.text(
        0.5,
        0.005,
        "Truth-side parameter offsets are supplied to all structured oracles; this is representation evidence only.",
        ha="center",
        fontsize=9,
        color="#53666c",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
