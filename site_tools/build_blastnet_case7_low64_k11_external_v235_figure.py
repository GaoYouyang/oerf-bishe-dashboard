#!/usr/bin/env python3
"""Build the redacted v235 prospective Case 7 result figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/blastnet_case7_low64_k11_external_v235.png"


def main() -> None:
    labels = ["K16\nreference", "Direct Low64\nK11", "Zero\nPCGLS K11", "BP +\nPCGLS K10"]
    absolute = np.asarray([546, 546, 257, 259])
    matched = np.asarray([546, 330, 0, 0])
    colors = ["#315f93", "#146f66", "#a34f43", "#d18a32"]

    metrics = ["Field", "Full gradient", "Interior gradient", "Observation"]
    p50 = np.asarray([0.8727293903, 0.9205157055, 0.8269863365, 0.9442914182])
    p90 = np.asarray([1.4467969568, 1.2639313603, 1.2097507931, 1.8781056242])
    worst = np.asarray([1.6261819669, 1.3834536372, 1.3349295908, 2.0873833305])

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), constrained_layout=True)

    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, absolute, width, label="absolute-safe", color=colors, alpha=0.9)
    axes[0].bar(x + width / 2, matched, width, label="matched to K16", color=colors, hatch="//", alpha=0.55)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 590)
    axes[0].set_ylabel("cells (of 546)")
    axes[0].set_title("Absolute safety does not imply K16 equivalence")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, loc="upper right")
    for index, (absolute_count, matched_count) in enumerate(zip(absolute, matched, strict=True)):
        axes[0].text(index - width / 2, absolute_count + 10, str(absolute_count), ha="center", fontweight="bold")
        axes[0].text(index + width / 2, matched_count + 10, str(matched_count), ha="center", fontweight="bold")

    y = np.arange(len(metrics))
    axes[1].barh(y + 0.22, worst, 0.22, label="worst", color="#a34f43")
    axes[1].barh(y, p90, 0.22, label="p90-higher", color="#d18a32")
    axes[1].barh(y - 0.22, p50, 0.22, label="p50", color="#146f66")
    axes[1].axvline(1.02, color="#315f93", linestyle="--", linewidth=1.2, label="rig p90 ratio gate 1.02")
    axes[1].axvline(1.05, color="#17252b", linestyle=":", linewidth=1.2, label="cell/worst ratio gate 1.05")
    axes[1].set_yticks(y, metrics)
    axes[1].set_xlim(0, 2.2)
    axes[1].set_xlabel("Direct K11 / K16 error ratio")
    axes[1].set_title("The mismatch is broad, not one isolated metric")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
    )

    fig.suptitle(
        "v235 Case 7: fixed Direct Low64 K11 fails prospective matched accuracy",
        fontsize=14,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
