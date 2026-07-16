#!/usr/bin/env python3
"""Plot the preregistered v5p fresh-development gate without retuning it."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results" / "v5p_fresh_budget_gate" / "report.json"
OUTPUT = ROOT / "results" / "v5p_fresh_budget_gate" / "v5p_gate.png"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    target = report["target_summary"]
    metrics = target["cluster_mean_target_standardized_rmse"]
    cells = target["cells"]
    timing = report["wall_clock_seconds"]

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
        "V5P fresh-development gate: all three preregistered criteria fail",
        fontsize=17,
        fontweight="bold",
        color="#17252b",
    )

    axis = axes[0, 0]
    method_keys = [
        "zero_correction",
        "shared_field_prior",
        "source_only_pbb_9",
        "source_only_pbb_11",
        "source_only_pbb_32",
        "sfio_papbb_budget_vector_v1",
    ]
    method_labels = ["zero", "prior", "PBB-9", "PBB-11", "PBB-32", "SFIO-PAPBB"]
    colors = ["#78909c", "#b45e52", "#315f93", "#557da8", "#244e78", "#146f66"]
    axis.bar(
        np.arange(len(method_keys)),
        [metrics[key] for key in method_keys],
        color=colors,
    )
    axis.set_xticks(np.arange(len(method_labels)), method_labels, rotation=18, ha="right")
    axis.set_ylabel("Cluster-mean target standardized RMSE")
    axis.set_title("360 new fields; lower is better")
    axis.grid(axis="y", alpha=0.18)

    axis = axes[0, 1]
    cell_gains = [100.0 * float(row["candidate_gain_vs_pbb9"]) for row in cells]
    cell_labels = [
        f"{row['rig_id'][-1]}-{row['family'].split('_')[0]}" for row in cells
    ]
    cell_colors = ["#39724f" if value > 0.0 else "#b45e52" for value in cell_gains]
    axis.bar(np.arange(len(cells)), cell_gains, color=cell_colors)
    axis.axhline(0.0, color="#26383e", linewidth=1)
    axis.axhline(-5.0, color="#8a651b", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(cells)), cell_labels, rotation=55, ha="right")
    axis.set_ylabel("Candidate gain versus PBB-9 (%)")
    axis.set_title("Only 8/18 cells improve; worst is -6.64%")
    axis.grid(axis="y", alpha=0.18)

    axis = axes[1, 0]
    families = sorted({str(row["family"]) for row in cells})
    family_gain = []
    family_positive = []
    for family in families:
        values = np.asarray(
            [
                float(row["candidate_gain_vs_pbb9"])
                for row in cells
                if row["family"] == family
            ]
        )
        family_gain.append(100.0 * float(np.mean(values)))
        family_positive.append(100.0 * float(np.mean(values > 0.0)))
    x = np.arange(len(families))
    axis.bar(x - 0.18, family_gain, 0.36, color="#146f66", label="mean gain (%)")
    axis.bar(
        x + 0.18,
        family_positive,
        0.36,
        color="#315f93",
        label="positive rigs (%)",
    )
    axis.axhline(3.0, color="#8a651b", linestyle="--", linewidth=1, label="gain gate")
    axis.axhline(75.0, color="#a34f43", linestyle=":", linewidth=1, label="sign gate")
    axis.set_xticks(
        x,
        [family.replace("_", "\n") for family in families],
    )
    axis.set_ylabel("Percent")
    axis.set_title("The apparent benefit is topology-specific")
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(axis="y", alpha=0.18)

    axis = axes[1, 1]
    time_labels = ["PBB-9", "PBB-11", "PBB-32", "SFIO-PAPBB"]
    time_values = [
        timing["pbb_9_seconds"],
        timing["pbb_11_seconds"],
        timing["pbb_32_seconds"],
        timing["candidate_total_before_target_decode_seconds"],
    ]
    axis.bar(
        np.arange(len(time_labels)),
        time_values,
        color=["#315f93", "#557da8", "#244e78", "#146f66"],
    )
    axis.set_xticks(np.arange(len(time_labels)), time_labels, rotation=18, ha="right")
    axis.set_ylabel("Local CPU wall-clock (seconds)")
    axis.set_title("Fewer source calls do not yield lower total time")
    axis.grid(axis="y", alpha=0.18)
    axis.text(
        0.02,
        0.96,
        "candidate 8F/9A; PBB-9 9F/9A\nCNN cost is included only in wall-clock",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        color="#53666c",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
