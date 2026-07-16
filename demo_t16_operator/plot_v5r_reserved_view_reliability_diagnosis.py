#!/usr/bin/env python3
"""Plot the v5r reserved-view reliability no-go."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v5r_reserved_view_reliability_diagnosis"
OUTPUT = RESULTS / "v5r_reserved_view_no_go.png"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    fields = _rows("field_rows.csv")
    cells = _rows("cell_rows.csv")
    rules = report["rules"]
    correlations = report["correlations"]
    colors = {
        "tilted_flame_brush": "#146f66",
        "triple_jet_merger": "#b45e52",
        "pulsed_toroidal_plume": "#315f93",
    }
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
        "V5R reserved-view audit: better average gain, insufficient safety",
        fontsize=17,
        fontweight="bold",
        color="#17252b",
    )

    axis = axes[0, 0]
    for family, color in colors.items():
        selected = [row for row in fields if row["family"] == family]
        axis.scatter(
            [100.0 * float(row["reserved_gain_vs_pbb9"]) for row in selected],
            [100.0 * float(row["target_gain_vs_pbb9"]) for row in selected],
            s=20,
            alpha=0.62,
            color=color,
            label=family.replace("_", " "),
        )
    axis.axhline(0.0, color="#53666c", linewidth=1)
    axis.axvline(0.0, color="#53666c", linewidth=1)
    axis.set_xlabel("Reserved-view gain vs PBB-9 (%)")
    axis.set_ylabel("Two-target gain vs PBB-9 (%)")
    axis.set_title("Field Spearman 0.548; audit view is not used in reconstruction")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.16)

    axis = axes[0, 1]
    labels = [
        f"{row['rig_id'][-1]}-{row['family'].split('_')[0]}" for row in cells
    ]
    reserved_gain = [
        100.0 * float(row["mean_reserved_gain_vs_pbb9"]) for row in cells
    ]
    target_gain = [100.0 * float(row["mean_target_gain_vs_pbb9"]) for row in cells]
    x = np.arange(len(cells))
    axis.bar(x - 0.2, reserved_gain, 0.4, color="#315f93", label="reserved")
    axis.bar(x + 0.2, target_gain, 0.4, color="#146f66", label="targets")
    axis.axhline(0.0, color="#53666c", linewidth=1)
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.set_ylabel("Gain versus PBB-9 (%)")
    axis.set_title("Cell Spearman 0.674; sign transfer remains incomplete")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.16)

    axis = axes[1, 0]
    per_rig = correlations["per_rig"]
    rig_labels = [row["rig_id"][-1].upper() for row in per_rig]
    rig_values = [float(row["field_spearman"]) for row in per_rig]
    bars = axis.bar(
        np.arange(len(per_rig)),
        rig_values,
        color=["#146f66" if value >= 0.3 else "#b45e52" for value in rig_values],
    )
    axis.axhline(0.0, color="#53666c", linewidth=1)
    axis.axhline(0.3, color="#8a651b", linewidth=1, linestyle="--")
    axis.set_xticks(np.arange(len(per_rig)), rig_labels)
    axis.set_ylim(-0.05, 0.92)
    axis.set_xlabel("Fresh synthetic rig")
    axis.set_ylabel("Reserved-to-target field Spearman")
    axis.set_title("Rig D is nearly uninformative; transfer is not uniform")
    axis.grid(axis="y", alpha=0.16)
    for bar, value in zip(bars, rig_values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )

    axis = axes[1, 1]
    rule_names = [
        "source_positive",
        "reserved_positive",
        "source_and_reserved_positive",
        "source_or_reserved_positive",
    ]
    rule_labels = ["source", "reserved", "both", "either"]
    columns = [
        "gated_ratio_of_cluster_means_gain_vs_pbb9",
        "gated_positive_cell_fraction",
        "field_coverage",
        "selected_field_target_harm_fraction",
        "gated_worst_cell_degradation",
    ]
    column_labels = ["gain", "positive cells", "coverage", "selected harm", "worst cell"]
    matrix = 100.0 * np.asarray(
        [[float(rules[name][column]) for column in columns] for name in rule_names]
    )
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=70.0, aspect="auto")
    axis.set_xticks(np.arange(len(columns)), column_labels, rotation=18, ha="right")
    axis.set_yticks(np.arange(len(rule_labels)), rule_labels)
    axis.set_title("No untuned sign rule reaches a safe, stable cell profile")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="#17252b",
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03, label="Percent")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
