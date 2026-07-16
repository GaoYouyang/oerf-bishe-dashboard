#!/usr/bin/env python3
"""Visualize the post-open v5q topology diagnosis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v5q_postopen_topology_diagnosis"
OUTPUT = RESULTS / "v5q_topology_diagnosis.png"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    fields = _read_csv("field_features.csv")
    cells = _read_csv("cell_features.csv")
    correlations = _read_csv("correlations.csv")
    sanity = report["postopen_zero_threshold_sanity_check"]

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
        "V5Q post-open diagnosis: a source-consistency signal, not a gate",
        fontsize=17,
        fontweight="bold",
        color="#17252b",
    )

    family_colors = {
        "tilted_flame_brush": "#146f66",
        "triple_jet_merger": "#b45e52",
        "pulsed_toroidal_plume": "#315f93",
    }
    axis = axes[0, 0]
    for family, color in family_colors.items():
        selected = [row for row in fields if row["family"] == family]
        axis.scatter(
            [100.0 * float(row["candidate_source_gain_vs_pbb9"]) for row in selected],
            [100.0 * float(row["candidate_target_gain_vs_pbb9"]) for row in selected],
            s=20,
            alpha=0.62,
            color=color,
            label=family.replace("_", " "),
        )
    axis.axhline(0.0, color="#53666c", linewidth=1)
    axis.axvline(0.0, color="#53666c", linewidth=1)
    axis.set_xlabel("Candidate source-residual gain vs PBB-9 (%)")
    axis.set_ylabel("Held-out target gain vs PBB-9 (%)")
    axis.set_title("Field Spearman 0.554; direction agrees in 6/6 rigs")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.16)

    axis = axes[0, 1]
    labels = [
        f"{row['rig_id'][-1]}-{row['family'].split('_')[0]}" for row in cells
    ]
    source_gain = [
        100.0 * float(row["candidate_source_gain_vs_pbb9"]) for row in cells
    ]
    target_gain = [
        100.0 * float(row["candidate_target_gain_vs_pbb9"]) for row in cells
    ]
    x = np.arange(len(cells))
    axis.bar(x - 0.2, source_gain, 0.4, color="#315f93", label="source gain")
    axis.bar(x + 0.2, target_gain, 0.4, color="#146f66", label="target gain")
    axis.axhline(0.0, color="#53666c", linewidth=1)
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.set_ylabel("Gain versus PBB-9 (%)")
    axis.set_title("Cell Spearman 0.802, but labels are already opened")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.16)

    axis = axes[1, 0]
    top_names = [
        "candidate_source_gain_vs_pbb9",
        "candidate_minus_pbb9_source_visibility",
        "candidate_floor_fraction",
        "prior_member_disagreement_over_base",
        "pbb9_total_variation_over_rms",
    ]
    rows_by_name = {row["feature"]: row for row in correlations}
    labels = [
        "source residual gain",
        "difference visibility",
        "candidate floor fraction",
        "prior disagreement",
        "PBB total variation",
    ]
    columns = [
        "field_spearman",
        "cell_spearman",
        "within_rig_family_centered_spearman",
        "leave_one_rig_oriented_median",
    ]
    matrix = np.asarray(
        [
            [float(rows_by_name[name][column] or "nan") for column in columns]
            for name in top_names
        ]
    )
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_xticks(
        np.arange(4), ["field", "18 cells", "within cell", "LOO-rig oriented"]
    )
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("Most morphology proxies vanish within cells")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#17252b",
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03, label="Spearman")

    axis = axes[1, 1]
    metric_labels = [
        "overall gated\ngain",
        "positive\ncells",
        "field\ncoverage",
        "selected-field\nharm",
        "worst cell\ndegradation",
    ]
    metric_values = [
        100.0 * float(sanity["gated_ratio_of_cluster_means_gain_vs_pbb9"]),
        100.0 * float(sanity["gated_positive_cell_fraction"]),
        100.0 * float(sanity["field_coverage"]),
        100.0 * float(sanity["selected_field_target_harm_fraction"]),
        100.0 * float(sanity["gated_worst_cell_degradation"]),
    ]
    colors = ["#146f66", "#315f93", "#557da8", "#b45e52", "#b45e52"]
    bars = axis.bar(np.arange(len(metric_labels)), metric_values, color=colors)
    axis.set_xticks(np.arange(len(metric_labels)), metric_labels)
    axis.set_ylabel("Percent")
    axis.set_title("Natural source-gain>0 fallback remains unsafe")
    axis.grid(axis="y", alpha=0.16)
    for bar, value in zip(bars, metric_values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.text(
        4.42,
        50.5,
        "Post-open only: 22.8% of selected fields are harmed;\n"
        "no router or design lock is authorized.",
        ha="right",
        va="top",
        fontsize=9,
        color="#7f3f36",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
