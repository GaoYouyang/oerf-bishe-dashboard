#!/usr/bin/env python3
"""Plot the v5n-v5o strong-baseline and hybrid-budget evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "v5o_prior_anchored_frontier" / "v5o_frontier.png"


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    hybrid = _read("results/v5o_prior_anchored_frontier/report.json")
    target = hybrid["extension_target"]
    metrics = target["cluster_mean_whitened_rmse"]
    frontier = hybrid["frontier"]

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
        "SFIO-PAPBB post-open diagnostic: a small low-budget signal, not a gate pass",
        fontsize=17,
        fontweight="bold",
        color="#17252b",
    )

    axis = axes[0, 0]
    method_keys = [
        "zero_correction",
        "shared_field_ensemble",
        "pure_pbb_32",
        "pure_pbb_budget_matched",
        "sfio_papbb_selected",
    ]
    labels = ["zero", "shared field", "PBB-32", "PBB-35", "SFIO-PAPBB"]
    values = [metrics[key] for key in method_keys]
    colors = ["#78909c", "#b45e52", "#315f93", "#244e78", "#146f66"]
    axis.bar(np.arange(len(values)), values, color=colors)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axis.set_ylabel("Cluster-mean whitened RMSE (lower is better)")
    axis.set_title("Frozen selection on 4 new rigs x 2 topologies")
    axis.grid(axis="y", alpha=0.18)

    axis = axes[0, 1]
    pairs = [row["estimated_forward_adjoint_pairs"] for row in frontier]
    learned = [row["sfio_papbb_cluster_mean_whitened_rmse"] for row in frontier]
    pure = [row["pure_pbb_cluster_mean_whitened_rmse"] for row in frontier]
    axis.plot(pairs, pure, "o-", color="#315f93", label="pure PBB")
    axis.plot(pairs, learned, "o-", color="#146f66", label="SFIO-PAPBB")
    axis.set_xscale("log", base=2)
    axis.set_xticks(pairs, [str(value) for value in pairs])
    axis.set_xlabel("Historical source-pair proxy")
    axis.set_ylabel("Cluster-mean whitened RMSE")
    axis.set_title("Accuracy versus operator-call proxy")
    axis.legend()
    axis.grid(alpha=0.18)

    axis = axes[1, 0]
    gains = [100.0 * row["sfio_papbb_gain_vs_budget_matched_pbb"] for row in frontier]
    bar_colors = ["#39724f" if value > 0.0 else "#b45e52" for value in gains]
    axis.bar(np.arange(len(gains)), gains, color=bar_colors)
    axis.axhline(0.0, color="#26383e", linewidth=1)
    axis.set_xticks(np.arange(len(pairs)), [str(value) for value in pairs])
    axis.set_xlabel("Historical source-pair proxy")
    axis.set_ylabel("Gain versus budget-matched PBB (%)")
    axis.set_title("Best observed low-budget gain is post-open only")
    axis.grid(axis="y", alpha=0.18)

    axis = axes[1, 1]
    cells = target["cells"]
    cell_gains = [100.0 * row["candidate_gain"] for row in cells]
    cell_labels = [
        f"{row['rig_id'][-1]}-{row['family'].replace('_', ' ')[:8]}" for row in cells
    ]
    cell_colors = ["#39724f" if value > 0.0 else "#b45e52" for value in cell_gains]
    axis.bar(np.arange(len(cells)), cell_gains, color=cell_colors)
    axis.axhline(0.0, color="#26383e", linewidth=1)
    axis.axhline(-5.0, color="#8a651b", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(cells)), cell_labels, rotation=35, ha="right")
    axis.set_ylabel("Selected hybrid gain versus PBB-35 (%)")
    axis.set_title("Only 4/8 cells improve; worst degradation is 9.44%")
    axis.grid(axis="y", alpha=0.18)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
