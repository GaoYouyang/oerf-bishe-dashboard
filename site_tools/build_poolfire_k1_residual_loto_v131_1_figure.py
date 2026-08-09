from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_residual_loto_v131_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_residual_loto_v131_1.png"


def build_figure() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    trajectories = list(data["trajectory_p90_higher"])
    labels = ["14kW-05", "22kW-03", "33kW-01", "45kW-05", "58kW-03"]
    metrics = ["field", "gradient", "interior_gradient", "observation"]
    metric_labels = ["Field", "Full gradient", "Interior gradient", "Observation"]
    matrix = np.array(
        [[data["trajectory_p90_higher"][trajectory][metric] for metric in metrics] for trajectory in trajectories]
    )

    colors = ["#e7f2ea", "#f8e2ae", "#ef8b70", "#b83b3b"]
    cmap = LinearSegmentedColormap.from_list("gate_harm", colors)
    fig = plt.figure(figsize=(16, 9), facecolor="#f5f7f8")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.55, 1), height_ratios=(1, 0.55), hspace=0.34, wspace=0.24)

    ax = fig.add_subplot(grid[:, 0])
    image = ax.imshow(matrix, cmap=cmap, vmin=1.0, vmax=1.50, aspect="auto")
    ax.set_xticks(range(len(metric_labels)), metric_labels, fontsize=11)
    ax.set_yticks(range(len(labels)), labels, fontsize=11)
    ax.set_title("A. Held-out p90 error ratio: candidate / CGLS K4", loc="left", fontsize=15, weight="bold", pad=14)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            mark = "PASS" if value <= 1.02 else "FAIL"
            ax.text(column, row, f"{value:.3f}\n{mark}", ha="center", va="center", fontsize=10, weight="bold", color="#172126")
    ax.text(0.0, -0.11, "Frozen p90 gate <= 1.02; all 5 trajectories fail the complete eight-gate contract.", transform=ax.transAxes, fontsize=10, color="#4d5960")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    colorbar.set_label("Error ratio", fontsize=10)

    camera_ax = fig.add_subplot(grid[0, 1])
    camera_data = data["postopen_diagnostic"]["observation_p90_by_camera_count"]
    camera_counts = [5, 7, 9, 12]
    camera_values = [camera_data[str(count)] for count in camera_counts]
    bars = camera_ax.bar([str(count) for count in camera_counts], camera_values, color=["#c84c4c", "#d96e50", "#e69b52", "#5f8795"], width=0.62)
    camera_ax.axhline(1.02, color="#263238", linestyle="--", linewidth=1.5, label="p90 gate 1.02")
    camera_ax.set_ylim(1.0, 1.56)
    camera_ax.set_ylabel("Observation error ratio p90")
    camera_ax.set_xlabel("Active camera count")
    camera_ax.set_title("B. Sparse views amplify the miss", loc="left", fontsize=15, weight="bold", pad=14)
    camera_ax.legend(frameon=False, loc="upper right")
    camera_ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, camera_values):
        camera_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=10, weight="bold")

    diagnosis_ax = fig.add_subplot(grid[1, 1])
    diagnosis_ax.axis("off")
    diagnosis_ax.set_title("C. Why a high cosine still fails", loc="left", fontsize=15, weight="bold", pad=8)
    boxes = [
        ("Dual cosine (median)", "0.9915"),
        ("Dual rel-L2 (p90)", "0.4484"),
        ("Effective correction rel-L2 (p90)", "0.4384"),
        ("Exact-call budget (K4: 4 + 4)", "2A + 2A^T"),
    ]
    for index, (title, value) in enumerate(boxes):
        x = 0.01 + (index % 2) * 0.50
        y = 0.55 - (index // 2) * 0.45
        diagnosis_ax.text(x, y + 0.18, title, transform=diagnosis_ax.transAxes, fontsize=9.2, color="#5d686e")
        diagnosis_ax.text(x, y - 0.02, value, transform=diagnosis_ax.transAxes, fontsize=16, weight="bold", color="#1f333d")

    fig.suptitle("v131.1: minimal learned correction dual does not match K4", x=0.06, y=0.985, ha="left", fontsize=21, weight="bold", color="#18262d")
    fig.text(0.06, 0.945, "3,700 held-out variable-camera PoolFire cells | independent recomputation max difference = 0", fontsize=11.5, color="#53636b")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
