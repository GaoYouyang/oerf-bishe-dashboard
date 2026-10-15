# ruff: noqa: E402
"""Plot only the published, scoped three-way resource evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "docs/potential_lift_resource_2026-10-15_public_summary.json").read_text())
names = ["Range0", "Potential0", "GMM0"]
labels = ["Range0", "Learned0", "GMM0"]
colors = ["#7f8a91", "#137f72", "#c7584d"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.9))
fig.patch.set_facecolor("white")
metrics = [("Median whole-process wall (s)", "median_whole_process_wall_seconds", 1, ".2f"),
           ("Joint field / gradient MSE", "joint_field_full_gradient_relative_MSE", 1, ".5f"),
           ("Maximum lifetime RSS (GiB)", "maximum_lifetime_RSS_bytes", 1024**3, ".3f")]
for ax, (title, key, scale, fmt) in zip(axes, metrics):
    values = [data[key][name] / scale for name in names]
    bars = ax.bar(labels, values, color=colors, width=.62)
    ax.set_title(title, fontsize=11, pad=15)
    ax.set_ylim(0, max(values) * 1.24)
    ax.bar_label(bars, labels=[format(v, fmt) for v in values], padding=5, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e9eb", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)
fig.suptitle("Learned value without CGLS: a measured three-way tradeoff", fontsize=15, y=.98)
fig.text(.5, .055, "All three: 909/909 reference matches, 9/9 trajectory tails | 3 fixed fresh-process repetitions\n"
         "Range0 is cheaper; GMM0 is more accurate. Clean known-five-camera exploratory LOTO only.",
         ha="center", va="center", fontsize=9, color="#4d5963", linespacing=1.6)
fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.23, wspace=.35)
fig.savefig(ROOT / "assets/potential_lift_resource_2026-10-15.png", dpi=160)
plt.close(fig)
