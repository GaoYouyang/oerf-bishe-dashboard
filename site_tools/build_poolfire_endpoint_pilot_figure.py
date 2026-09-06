"""Render only the public five-point diagnostic aggregate."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
stem = "poolfire_endpoint_pilot_20260906"
data = json.loads((root / f"docs/{stem}.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for ax, key, title in zip(axes.ravel(),
        ("field", "full_gradient", "interior_gradient", "observation"),
        ("Density", "Full gradient", "Interior gradient", "Observation")):
    for arm, color, label in (("cgls4", "#ad5446", "K4: four iterations"),
                             ("lsmr", "#197e73", "LSMR: 1945-2122 iterations"),
                             ("lsqr", "#426fa8", "Independent LSQR: 1878-2059 iterations")):
        values = [row[arm][key] for row in data["points"]]
        ax.plot(np.arange(5), values, "o-", label=label, color=color, markersize=4)
    ax.axhline(.01, color="#777777", linestyle="--", linewidth=1, label="1% pilot criterion")
    ax.set_yscale("log")
    ax.set_ylim(3e-9, 1)
    ax.set_xticks(np.arange(5), [row["source"] for row in data["points"]])
    ax.set_title(title, loc="left")
    ax.set_ylabel("Relative error (log scale)")
    ax.grid(axis="y", alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=10)
fig.suptitle("Five fixed midpoint samples: clean endpoint recovery\nNot a learned model, equal-cost comparison, or full-trajectory result", fontsize=15)
fig.savefig(root / f"assets/figures/{stem}.png", dpi=150,
            metadata={"Title": "Five-point clean endpoint recovery diagnostic"})
plt.close(fig)
