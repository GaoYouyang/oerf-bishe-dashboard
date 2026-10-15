# ruff: noqa: E402
"""Plot only the sanitized native-cardinality milestone."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "docs/potential_native_cardinality_2026-10-15_public_summary.json").read_text())
labels = ["Learned0", "Linear0", "Native\nmean0", "Range0", "Native\nGMM0"]
colors = ["#137f72", "#3868a5", "#8d728c", "#7f8a91", "#c7584d"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
for ax, count in zip(axes, ("7", "9")):
    values = data["counts"][count]["joint"][:5]
    bars = ax.bar(labels, values, color=colors, width=.64)
    ax.bar_label(bars, labels=[f"{v:.5f}" for v in values], padding=4, fontsize=10)
    ax.set_title(f"Native {count} cameras | 909/909 matches, 9/9 tails", fontsize=12, pad=14)
    ax.set_ylim(0, max(values) * 1.24)
    ax.set_ylabel("Joint relative field / full-gradient MSE", fontsize=10)
    ax.grid(axis="y", color="#e6e9eb", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
fig.suptitle("Five-camera weights transfer without neural retraining or CGLS", fontsize=15, y=.97)
fig.text(.5, .09, "All five actions meet the same minimum gate. Range0 is cheaper; native GMM0 remains more accurate.\n"
         "Known nested clean cameras; development-visible complete-trajectory LOTO. No new timing or external claim.",
         ha="center", va="center", fontsize=10, color="#4d5963", linespacing=1.6)
fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.25, wspace=.28)
fig.savefig(ROOT / "assets/potential_native_cardinality_2026-10-15.png", dpi=160)
plt.close(fig)
