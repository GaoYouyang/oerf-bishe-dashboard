"""Plot certified aggregate losses and error ratios, not private inputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = "poolfire_signed_prolongation_learning_2026-10-08"
data = json.loads((ROOT / "docs" / f"{STEM}_public_summary.json").read_text())
diagnostics = data["post_evaluation_descriptive_diagnostics"]
loss = diagnostics["mean_train_seed_loss"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.84, wspace=.32)
axes[0].bar(["Initial", "Linear6", "Learned29"],
            [loss["initial"], loss["linear_final"], loss["primary_final"]],
            color=["#777777", "#217b85", "#bb4e61"], width=.58)
axes[0].set(ylim=(0, .76), ylabel="Mean train seed loss",
            title="Training objective decreased")
for i, value in enumerate([loss["initial"], loss["linear_final"], loss["primary_final"]]):
    axes[0].text(i, value + .012, f"{value:.5f}", ha="center", fontsize=10)
ratios = diagnostics["p90_higher_error_ratio_to_CGLS128"]
x = np.arange(4)
for offset, name, label, color in [
    (-.23, "CGLS31", "Cold CGLS31", "#777777"),
    (0, "LinearProlong6-Warm28", "Linear6 + warm28", "#217b85"),
    (.23, "SignedProlong29-Warm28", "Learned29 + warm28", "#bb4e61"),
]:
    axes[1].bar(x + offset, ratios[name], width=.23, label=label, color=color)
axes[1].axhline(1, color="#333333", linewidth=1, linestyle="--")
axes[1].set(yscale="log", ylim=(.9, 9),
            xticks=x, xticklabels=["Field", "Full grad.", "Interior", "Observation"],
            ylabel="Higher-p90 error ratio to CGLS128",
            title="Final error did not improve")
axes[1].legend(fontsize=9, frameon=False, loc="upper left")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Full trajectory-disjoint learned prolongation: fixed recipe failed",
             fontsize=13, y=.96)
fig.text(.5, .12,
         "Learned29: basic absolute strata 33/33; joint four-error matches 0/3333\n"
         "Cold31 also passes basic strata. Strong nonfree direct35 matches 3333/3333.",
         ha="center", fontsize=10)
fig.text(.5, .035,
         "Post-evaluation ratios are not percentages or speedups. Independent checks: 28/28.",
         ha="center", fontsize=9, color="#555555")
fig.savefig(ROOT / "assets" / f"{STEM}.png", dpi=180)
plt.close(fig)
