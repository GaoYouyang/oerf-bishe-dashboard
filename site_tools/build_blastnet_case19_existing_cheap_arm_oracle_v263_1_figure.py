from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_existing_cheap_arm_oracle_v263_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/blastnet_case19_existing_cheap_arm_oracle_v263_1.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    arms = data["arms"]
    oracle = data["oracle"]
    labels = [arm["label"] for arm in arms]
    absolute = np.asarray([arm["absolute_pass_rigs"] for arm in arms])
    matched = np.asarray([arm["matched_pass_rigs"] for arm in arms])

    fig = plt.figure(figsize=(18, 7), facecolor="#f7f8fa")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1), wspace=0.22)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])

    y = np.arange(len(labels))
    left.barh(y - 0.18, absolute, height=0.34, color="#168f73", label="Absolute gate")
    left.barh(y + 0.18, matched, height=0.34, color="#d64b4b", label="K16-matched gate")
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlim(0, 13.6)
    left.set_xlabel("Passing rigs out of 13")
    left.set_title("Every arm has zero matched passes", loc="left", fontweight="bold")
    left.grid(axis="x", alpha=0.2)
    left.legend(frameon=False, loc="lower right")
    for index, value in enumerate(absolute):
        left.text(value + 0.15, index - 0.18, f"{value}/13", va="center", fontsize=9)
    for index, value in enumerate(matched):
        left.text(value + 0.15, index + 0.18, f"{value}/13", va="center", fontsize=9, color="#a52b2b")

    summaries = [
        oracle["best_joint_burden_p50_higher"],
        oracle["best_joint_burden_p90_higher"],
        oracle["best_joint_burden_worst"],
    ]
    names = ["p50", "p90-higher", "worst"]
    colors = ["#2878b5", "#f2a134", "#d64b4b"]
    bars = right.bar(names, summaries, color=colors, width=0.58)
    right.axhline(1.0, color="#222222", linewidth=2, linestyle="--", label="Passing line")
    right.set_ylim(0.98, 1.08)
    right.set_ylabel("Best truth-aware joint burden")
    right.set_title("Oracle still misses the frozen gate", loc="left", fontweight="bold")
    right.grid(axis="y", alpha=0.2)
    right.legend(frameon=False, loc="upper left")
    for bar, value in zip(bars, summaries, strict=True):
        right.text(bar.get_x() + bar.get_width() / 2, value + 0.0015, f"{value:.5f}", ha="center", fontsize=10)

    fig.suptitle(
        "v263.1 | Truth-aware selection over nine existing cheap arms: 0/13 joint passes",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.015,
        "Independent validation 19/19 | already-opened Case 19 frame zero | 0A+0AT new calls | necessary-capacity audit, not a deployment algorithm",
        fontsize=10,
        color="#444444",
    )
    for axis in (left, right):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
