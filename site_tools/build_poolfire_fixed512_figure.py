"""Plot redacted worst-over-both-implementations complete-sequence errors."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
stem = "poolfire_fixed512_reference_20260906"
data = json.loads((root / f"docs/{stem}.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(12.8, 8), layout="constrained")
for ax, metric, title in zip(axes.ravel(), ("field", "full_gradient", "interior_gradient", "observation"),
                            ("Density", "Full gradient", "Interior gradient", "Observation")):
    for arm, label, color, offset in (("cgls", "CGLS K512", "#187565", -.18),
                                      ("jacobi_pcgls", "Jacobi PCGLS K512", "#a14670", .18)):
        values = [max(data["summaries"][path][arm]["trajectories"][fold]["tails"][metric]["worst"]
                      for path in ("formal", "independent")) * 100 for fold in range(5)]
        ax.bar(np.arange(5) + offset, values, width=.32, label=label, color=color)
    ax.axhline(1., color="#606060", linestyle="--", linewidth=1, label="1% absolute gate")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(np.arange(5), [f"T{i+1}" for i in range(5)])
    ax.set_ylabel("Worst relative error (%)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=.12)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle("505 opened clean frames: fixed-budget classical reference\nBoth methods pass every frame; not learned speedup or a minimum-cost proof", fontsize=14)
fig.savefig(root / f"assets/figures/{stem}.png", dpi=125,
            metadata={"Title": "Opened clean sequence fixed-budget reference"})
plt.close(fig)
