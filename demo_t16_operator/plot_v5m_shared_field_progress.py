#!/usr/bin/env python3
"""Plot the v5h-v5m evidence progression without reopening any data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "v5m_shared_field_extension" / "v5m_shared_field_progress.png"


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    fisher = _read("results/v5h_gc_rio_development/report.json")
    signed = _read("results/v5i_signed_probe_development/report.json")
    ensemble = _read("results/v5l_shared_field_ensemble_diagnostic/report.json")
    extension = _read("results/v5m_shared_field_extension/report.json")

    fisher_gains = [
        100.0
        * (1.0 - cell["gc_rio_correct_target_geometry"] / cell["zero_correction"])
        for cell in fisher["decision"]["cells"]
    ]
    signed_gains = [
        100.0
        * (1.0 - cell["sp_gc_rio_correct_target_geometry"] / cell["zero_correction"])
        for cell in signed["decision"]["cells"]
    ]
    ensemble_gains = [
        100.0 * cell["relative_gain_vs_zero"]
        for cell in ensemble["validation_target_summary"]["cells"]
    ]
    extension_gains = [
        100.0 * cell["relative_gain_vs_zero"]
        for cell in extension["target_summary"]["cells"]
    ]

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
        "GC-RIO development evidence: target-specific queries to a shared 3D field",
        fontsize=17,
        fontweight="bold",
        color="#17252b",
    )

    axis = axes[0, 0]
    x = np.arange(4)
    width = 0.24
    axis.bar(x - width, fisher_gains, width, label="v0 Fisher target query", color="#a34f43")
    axis.bar(x, signed_gains, width, label="v1 signed target probes", color="#8a651b")
    axis.bar(x + width, ensemble_gains, width, label="v2 shared-field ensemble", color="#146f66")
    axis.axhline(0.0, color="#26383e", linewidth=1)
    axis.axhline(10.0, color="#315f93", linewidth=1.2, linestyle="--", label="10% gate")
    axis.set_xticks(x, ["rig-a\njet", "rig-a\nshock", "rig-b\njet", "rig-b\nshock"])
    axis.set_ylabel("Relative gain vs zero correction (%)")
    axis.set_title("Opened development validation")
    axis.legend(fontsize=8, loc="upper left")
    axis.grid(axis="y", alpha=0.18)

    axis = axes[0, 1]
    labels = [
        f"{cell['rig_id'][-1]}-{cell['family'].replace('_', ' ')[:8]}"
        for cell in extension["target_summary"]["cells"]
    ]
    colors = ["#39724f" if value > 0 else "#b45e52" for value in extension_gains]
    axis.bar(np.arange(len(extension_gains)), extension_gains, color=colors)
    axis.axhline(0.0, color="#26383e", linewidth=1)
    axis.axhline(10.0, color="#315f93", linewidth=1.2, linestyle="--")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_ylabel("Relative gain vs zero correction (%)")
    axis.set_title("No-retraining extension: 4 new rigs x 2 new topologies")
    axis.grid(axis="y", alpha=0.18)

    axis = axes[1, 0]
    old_field = ensemble["validation_field_truth_diagnostic"]
    new_field = extension["field_truth_diagnostic"]
    base = [old_field["mean_base_relative_l2"], new_field["mean_base_relative_l2"]]
    learned = [
        old_field["mean_ensemble_relative_l2"],
        new_field["mean_ensemble_relative_l2"],
    ]
    x = np.arange(2)
    axis.bar(x - 0.18, base, 0.36, label="source-only base", color="#78909c")
    axis.bar(x + 0.18, learned, 0.36, label="shared-field ensemble", color="#146f66")
    axis.set_xticks(x, ["original validation\n40 fields", "new extension\n128 fields"])
    axis.set_ylabel("Mean 3D relative L2 (truth diagnostic)")
    axis.set_title("Projection gain corresponds to a smaller 3D field error")
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.18)

    axis = axes[1, 1]
    old_gain = 100.0 * ensemble["validation_target_summary"]["cell_mean_relative_gain_vs_zero"]
    new_gain = 100.0 * extension["target_summary"]["cell_mean_relative_gain_vs_zero"]
    old_positive = 100.0 * ensemble["validation_target_summary"]["positive_cells"] / ensemble["validation_target_summary"]["cell_count"]
    new_positive = 100.0 * extension["target_summary"]["positive_cell_fraction"]
    x = np.arange(2)
    axis.bar(x - 0.18, [old_gain, new_gain], 0.36, color="#146f66", label="cell-mean gain (%)")
    axis.bar(x + 0.18, [old_positive, new_positive], 0.36, color="#315f93", label="positive cells (%)")
    axis.axhline(10.0, color="#8a651b", linestyle="--", linewidth=1, label="gain gate 10%")
    axis.axhline(75.0, color="#a34f43", linestyle=":", linewidth=1, label="sign gate 75%")
    axis.set_xticks(x, ["post-open ensemble", "new-data extension"])
    axis.set_ylabel("Percent")
    axis.set_ylim(-10, 105)
    axis.set_title("Neither development screen authorizes the design-lock open")
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(axis="y", alpha=0.18)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
