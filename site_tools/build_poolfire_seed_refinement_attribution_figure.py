"""Visualize certified field-only post-evaluation distances."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = "poolfire_seed_refinement_attribution_2026-10-09"
data = json.loads((ROOT / "docs" / f"{STEM}_public_summary.json").read_text())
linear = data["pooled"]["LinearProlong6-Warm28"]["teacher_field_squared_distance"]
learned = data["pooled"]["SignedProlong29-Warm28"]["teacher_field_squared_distance"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.83, wspace=.3)
for ax, phase, upper, decimals, title in [
    (axes[0], "seed", .88, 5, "Closer at the seed"),
    (axes[1], "final", .04, 6, "Farther after unchanged warm28"),
]:
    values = [learned[phase + "_classical"]["p90"], linear[phase + "_arm"]["p90"], learned[phase + "_arm"]["p90"]]
    ax.bar(["Classical", "Linear6", "Learned29"], values,
           color=["#777777", "#217b85", "#bb4e61"], width=.55)
    ax.set(ylim=(0, upper), title=title)
    ax.set_ylabel("Higher-p90 normalized squared\nteacher-field distance", fontsize=10)
    for i, value in enumerate(values):
        ax.text(i, value + upper * .025, f"{value:.{decimals}f}", ha="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Held seed-field gains do not imply finite-refinement gains", fontsize=14, y=.96)
fig.text(.5, .12, "Learned29: all 3333 seeds closer; 3304 final fields farther; 29 closer.\n"
         "Both arms: 33/33 positive mean seed strata; 0/33 positive mean final strata.",
         ha="center", fontsize=10)
fig.text(.5, .035, "Field-only distance to a finite teacher, not CFD error or complete seed loss. Independent checks: 20/20.",
         ha="center", fontsize=9, color="#555555")
fig.savefig(ROOT / "assets" / f"{STEM}.png", dpi=180)
plt.close(fig)
