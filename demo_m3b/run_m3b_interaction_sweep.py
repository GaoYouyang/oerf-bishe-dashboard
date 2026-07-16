#!/usr/bin/env python3
"""Crossed multi-seed interaction study for the M3B 4D BOST toy.

The six-axis sweep is intentionally one-factor-at-a-time. This experiment asks
the next question: does the preferred temporal rank remain stable when noise,
view count, and dynamics change together? It uses a balanced crossed design,
shares one framewise reconstruction across all ranks in each observation cell,
and reports paired uncertainty, rank regret, and trade-offs.

This remains a compact straight-ray synthetic study. It is not a reproduction
of TDBOST and its random seeds represent observation noise only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_RESULTS = ROOT / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_m3b.run_m3b_4d_lowrank_bost import low_rank_temporal, make_4d_sequence
from demo_m3b.run_m3b_six_axis_sweep import (
    METRIC_NAMES,
    deflection_sequence,
    metric_suite,
    reconstruct_case,
)


DEFAULT_RANKS = [1, 2, 3, 5, 8]
DEFAULT_NOISES = [0.0, 0.07, 0.14, 0.28, 0.42]
DEFAULT_VIEWS = [3, 5, 7, 9]
DEFAULT_DYNAMICS = ["smooth", "fast", "chirp", "transient"]

LOWER_IS_BETTER = [
    "rel_l2_global",
    "paper_l2_squared",
    "temporal_gradient_rel_l2",
    "temporal_curvature_rel_l2",
    "centroid_rmse",
    "mass_trace_rmse",
    "heldout_reprojection_rel_l2",
]

IMPROVEMENT_COLUMNS = [f"{metric}_improvement_pct" for metric in LOWER_IS_BETTER]
SUMMARY_COLUMNS = IMPROVEMENT_COLUMNS + [
    "dynamics_energy_closeness_gain",
    "lowrank_rel_l2_global",
    "lowrank_temporal_gradient_rel_l2",
    "lowrank_mass_trace_rmse",
    "lowrank_heldout_reprojection_rel_l2",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def t_critical_95(sample_count: int) -> float:
    """Two-sided 95% Student-t critical value without a SciPy dependency."""
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        12: 2.201,
        16: 2.131,
        20: 2.093,
        30: 2.045,
    }
    if sample_count in values:
        return values[sample_count]
    lower = max((count for count in values if count <= sample_count), default=2)
    return values[lower] if sample_count < 30 else 1.96


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    ci95 = float(t_critical_95(len(array)) * std / math.sqrt(len(array))) if len(array) > 1 else 0.0
    win_rate = float(np.mean(array > 0.0))
    return mean, std, ci95, win_rate


def raw_row(
    design_id: str,
    seed: int,
    method: str,
    rank: int,
    noise_level: float,
    view_count: int,
    dynamics: str,
    reconstruction_seconds: float,
    postprocess_seconds: float,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "design_id": design_id,
        "seed": seed,
        "method": method,
        "rank": rank,
        "noise_level": noise_level,
        "view_count": view_count,
        "dynamics": dynamics,
        "reconstruction_seconds": reconstruction_seconds,
        "postprocess_seconds": postprocess_seconds,
        **metrics,
    }


def paired_row(
    design_id: str,
    seed: int,
    rank: int,
    noise_level: float,
    view_count: int,
    dynamics: str,
    baseline: dict[str, float],
    lowrank: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "design_id": design_id,
        "seed": seed,
        "rank": rank,
        "noise_level": noise_level,
        "view_count": view_count,
        "dynamics": dynamics,
    }
    for metric in LOWER_IS_BETTER:
        baseline_value = float(baseline[metric])
        lowrank_value = float(lowrank[metric])
        row[f"framewise_{metric}"] = baseline_value
        row[f"lowrank_{metric}"] = lowrank_value
        row[f"{metric}_improvement_pct"] = 100.0 * (baseline_value - lowrank_value) / (abs(baseline_value) + 1e-12)
    baseline_energy = float(baseline["dynamics_energy_ratio"])
    lowrank_energy = float(lowrank["dynamics_energy_ratio"])
    row["framewise_dynamics_energy_ratio"] = baseline_energy
    row["lowrank_dynamics_energy_ratio"] = lowrank_energy
    row["dynamics_energy_closeness_gain"] = abs(baseline_energy - 1.0) - abs(lowrank_energy - 1.0)
    return row


def cell_summary(paired_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, float, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in paired_rows:
        key = (int(row["rank"]), float(row["noise_level"]), int(row["view_count"]), str(row["dynamics"]))
        groups[key].append(row)
    output = []
    for key, rows in groups.items():
        rank, noise, views, dynamics = key
        item: dict[str, object] = {
            "rank": rank,
            "noise_level": noise,
            "view_count": views,
            "dynamics": dynamics,
            "seed_count": len(rows),
        }
        for column in SUMMARY_COLUMNS:
            values = [float(row[column]) for row in rows]
            mean, std, ci95, win_rate = summarize(values)
            item[f"{column}_mean"] = mean
            item[f"{column}_std"] = std
            item[f"{column}_ci95"] = ci95
            if column in IMPROVEMENT_COLUMNS or column == "dynamics_energy_closeness_gain":
                item[f"{column}_win_rate"] = win_rate
        output.append(item)
    return sorted(output, key=lambda row: (float(row["noise_level"]), int(row["view_count"]), str(row["dynamics"]), int(row["rank"])))


def marginal_matrix(
    paired_rows: list[dict[str, object]],
    row_factor: str,
    row_levels: list[object],
    column_factor: str,
    column_levels: list[object],
    metric: str,
) -> np.ndarray:
    matrix = np.zeros((len(row_levels), len(column_levels)), dtype=float)
    for row_index, row_level in enumerate(row_levels):
        for column_index, column_level in enumerate(column_levels):
            values = [
                float(row[metric])
                for row in paired_rows
                if row[row_factor] == row_level and row[column_factor] == column_level
            ]
            matrix[row_index, column_index] = float(np.mean(values))
    return matrix


def rank_selection(
    summary_rows: list[dict[str, object]],
    ranks: list[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    environments: dict[tuple[float, int, str], dict[int, dict[str, object]]] = defaultdict(dict)
    for row in summary_rows:
        key = (float(row["noise_level"]), int(row["view_count"]), str(row["dynamics"]))
        environments[key][int(row["rank"])] = row

    selections: list[dict[str, object]] = []
    regrets: dict[int, list[float]] = {rank: [] for rank in ranks}
    best_counts: Counter[int] = Counter()
    for environment, by_rank in sorted(environments.items()):
        noise, views, dynamics = environment
        best_rank = min(ranks, key=lambda rank: float(by_rank[rank]["lowrank_rel_l2_global_mean"]))
        best_error = float(by_rank[best_rank]["lowrank_rel_l2_global_mean"])
        best_counts[best_rank] += 1
        selection: dict[str, object] = {
            "noise_level": noise,
            "view_count": views,
            "dynamics": dynamics,
            "best_rank": best_rank,
            "best_rel_l2_global": best_error,
        }
        for rank in ranks:
            value = float(by_rank[rank]["lowrank_rel_l2_global_mean"])
            regret = 100.0 * (value - best_error) / (abs(best_error) + 1e-12)
            regrets[rank].append(regret)
            selection[f"rank_{rank}_regret_pct"] = regret
        selections.append(selection)

    regret_report = {
        str(rank): {
            "mean_pct": float(np.mean(values)),
            "median_pct": float(np.median(values)),
            "p95_pct": float(np.percentile(values, 95)),
            "max_pct": float(np.max(values)),
        }
        for rank, values in regrets.items()
    }
    robust_mean = min(ranks, key=lambda rank: regret_report[str(rank)]["mean_pct"])
    robust_p95 = min(ranks, key=lambda rank: regret_report[str(rank)]["p95_pct"])
    report = {
        "environment_cell_count": len(selections),
        "best_rank_counts": {str(rank): int(best_counts.get(rank, 0)) for rank in ranks},
        "regret_by_rank": regret_report,
        "robust_rank_by_mean_regret": robust_mean,
        "robust_rank_by_p95_regret": robust_p95,
    }
    return selections, report


def matrix_to_nested(row_levels: list[object], column_levels: list[object], matrix: np.ndarray) -> dict[str, object]:
    return {
        str(row): {str(column): float(matrix[i, j]) for j, column in enumerate(column_levels)}
        for i, row in enumerate(row_levels)
    }


def plot_heatmaps(
    paired_rows: list[dict[str, object]],
    ranks: list[int],
    noises: list[float],
    views: list[int],
    dynamics_modes: list[str],
    path: Path,
) -> dict[str, np.ndarray]:
    matrices = {
        "rank_noise_field": marginal_matrix(
            paired_rows, "rank", ranks, "noise_level", noises, "rel_l2_global_improvement_pct"
        ),
        "rank_views_heldout": marginal_matrix(
            paired_rows, "rank", ranks, "view_count", views, "heldout_reprojection_rel_l2_improvement_pct"
        ),
        "rank_dynamics_temporal": marginal_matrix(
            paired_rows, "rank", ranks, "dynamics", dynamics_modes, "temporal_gradient_rel_l2_improvement_pct"
        ),
    }
    specs = [
        (matrices["rank_noise_field"], [f"{value:.2f}" for value in noises], "noise multiplier", "Field L2 improvement (%)"),
        (matrices["rank_views_heldout"], [str(value) for value in views], "view count", "Held-out deflection improvement (%)"),
        (matrices["rank_dynamics_temporal"], dynamics_modes, "dynamics", "Temporal-gradient improvement (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.3), constrained_layout=True)
    for ax, (matrix, labels, xlabel, title) in zip(axes, specs):
        scale = max(float(np.max(np.abs(matrix))), 1.0)
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=-scale, vmax=scale, aspect="auto")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(ranks)), [str(rank) for rank in ranks])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("temporal rank")
        ax.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "white" if abs(matrix[row, column]) > 0.55 * scale else "#1f292d"
                ax.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", fontsize=8.5, color=color)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    seed_count = len({int(row["seed"]) for row in paired_rows})
    fig.suptitle(
        f"M3B crossed interactions: marginal paired gains over other factors and {seed_count} seeds",
        fontsize=13.5,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return matrices


def plot_rank_selection(
    selections: list[dict[str, object]],
    rank_report: dict[str, object],
    ranks: list[int],
    noises: list[float],
    views: list[int],
    dynamics_modes: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)

    counts = [int(rank_report["best_rank_counts"][str(rank)]) for rank in ranks]
    axes[0].bar([str(rank) for rank in ranks], counts, color="#2c786c")
    axes[0].set_title(f"Best field-L2 rank across {len(selections)} environments")
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel("environment cells")
    for index, value in enumerate(counts):
        axes[0].text(index, value, str(value), ha="center", va="bottom", fontsize=9)

    regret = rank_report["regret_by_rank"]
    means = [float(regret[str(rank)]["mean_pct"]) for rank in ranks]
    p95 = [float(regret[str(rank)]["p95_pct"]) for rank in ranks]
    x = np.arange(len(ranks))
    width = 0.38
    axes[1].bar(x - width / 2, means, width, color="#4f77a2", label="mean regret")
    axes[1].bar(x + width / 2, p95, width, color="#b97832", label="95th percentile")
    axes[1].set_xticks(x, [str(rank) for rank in ranks])
    axes[1].set_title("Cost of using one fixed rank")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("field-L2 regret (%)")
    axes[1].legend()

    rank_grid = np.zeros((len(dynamics_modes) * len(views), len(noises)), dtype=float)
    y_labels = []
    for dynamics_index, dynamics in enumerate(dynamics_modes):
        for view_index, view_count in enumerate(views):
            row_index = dynamics_index * len(views) + view_index
            y_labels.append(f"{dynamics[:4]} / {view_count}v")
            for noise_index, noise in enumerate(noises):
                match = next(
                    row
                    for row in selections
                    if float(row["noise_level"]) == noise
                    and int(row["view_count"]) == view_count
                    and row["dynamics"] == dynamics
                )
                rank_grid[row_index, noise_index] = int(match["best_rank"])
    image = axes[2].imshow(rank_grid, cmap="viridis", vmin=min(ranks), vmax=max(ranks), aspect="auto")
    axes[2].set_xticks(np.arange(len(noises)), [f"{noise:.2f}" for noise in noises])
    axes[2].set_yticks(np.arange(len(y_labels)), y_labels, fontsize=7.5)
    axes[2].set_title("Cell-wise best rank")
    axes[2].set_xlabel("noise multiplier")
    for row in range(rank_grid.shape[0]):
        for column in range(rank_grid.shape[1]):
            axes[2].text(column, row, str(int(rank_grid[row, column])), ha="center", va="center", color="white", fontsize=7)
    colorbar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label("best rank")
    for ax in axes[:2]:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("M3B rank-selection stability and regret", fontsize=13.5)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoffs(summary_rows: list[dict[str, object]], ranks: list[int], path: Path) -> None:
    colors = {1: "#855c8d", 2: "#4f77a2", 3: "#2c786c", 5: "#b97832", 8: "#a74e43"}
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    for rank in ranks:
        rows = [row for row in summary_rows if int(row["rank"]) == rank]
        field = np.asarray([float(row["rel_l2_global_improvement_pct_mean"]) for row in rows])
        mass = np.asarray([float(row["mass_trace_rmse_improvement_pct_mean"]) for row in rows])
        heldout = np.asarray([float(row["heldout_reprojection_rel_l2_improvement_pct_mean"]) for row in rows])
        axes[0].scatter(field, mass, s=22, alpha=0.52, color=colors[rank], label=f"rank {rank}")
        axes[1].scatter(field, heldout, s=22, alpha=0.52, color=colors[rank], label=f"rank {rank}")
    axes[0].set_title("Field gain vs mass-trace gain")
    axes[0].set_ylabel("mass-trace improvement (%)")
    axes[1].set_title("Field gain vs held-out gain")
    axes[1].set_ylabel("held-out improvement (%)")
    for ax in axes:
        ax.axhline(0.0, color="#3c454a", linewidth=1)
        ax.axvline(0.0, color="#3c454a", linewidth=1)
        ax.set_xlabel("field-L2 improvement (%)")
        ax.grid(True, alpha=0.22)
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("M3B crossed-design trade-offs: a smoother field is not automatically more physical", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rank3_operating_map(
    paired_rows: list[dict[str, object]],
    noises: list[float],
    views: list[int],
    path: Path,
) -> dict[str, np.ndarray]:
    rank3_rows = [row for row in paired_rows if int(row["rank"]) == 3]
    matrices = {
        "field": marginal_matrix(
            rank3_rows, "view_count", views, "noise_level", noises, "rel_l2_global_improvement_pct"
        ),
        "heldout": marginal_matrix(
            rank3_rows,
            "view_count",
            views,
            "noise_level",
            noises,
            "heldout_reprojection_rel_l2_improvement_pct",
        ),
        "mass": marginal_matrix(
            rank3_rows,
            "view_count",
            views,
            "noise_level",
            noises,
            "mass_trace_rmse_improvement_pct",
        ),
    }
    specs = [
        (matrices["field"], "Field L2 improvement (%)"),
        (matrices["heldout"], "Held-out deflection improvement (%)"),
        (matrices["mass"], "Mass-trace improvement (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.7), constrained_layout=True)
    for ax, (matrix, title) in zip(axes, specs):
        scale = max(float(np.max(np.abs(matrix))), 1.0)
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=-scale, vmax=scale, aspect="auto")
        ax.set_xticks(np.arange(len(noises)), [f"{noise:.2f}" for noise in noises], rotation=20, ha="right")
        ax.set_yticks(np.arange(len(views)), [str(view) for view in views])
        ax.set_xlabel("noise multiplier")
        ax.set_ylabel("view count")
        ax.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "white" if abs(matrix[row, column]) > 0.55 * scale else "#1f292d"
                ax.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", fontsize=8.5, color=color)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    seed_count = len({int(row["seed"]) for row in rank3_rows})
    fig.suptitle(
        f"M3B rank-3 operating map: noise x views, marginal over dynamics and {seed_count} paired seeds",
        fontsize=13.2,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return matrices


def build_report(
    paired_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    selections: list[dict[str, object]],
    rank_report: dict[str, object],
    matrices: dict[str, np.ndarray],
    rank3_matrices: dict[str, np.ndarray],
    ranks: list[int],
    noises: list[float],
    views: list[int],
    dynamics_modes: list[str],
    seed_count: int,
    raw_count: int,
    runtime_seconds: float,
) -> dict[str, object]:
    def rank_values(rank: int, column: str) -> list[float]:
        return [float(row[column]) for row in paired_rows if int(row["rank"]) == rank]

    rank_field_win_rate = {
        str(rank): float(np.mean(np.asarray(rank_values(rank, "rel_l2_global_improvement_pct")) > 0.0))
        for rank in ranks
    }
    rank3_summary = next(
        (
            row
            for row in summary_rows
            if int(row["rank"]) == 3
            and float(row["noise_level"]) == 0.14
            and int(row["view_count"]) == 5
            and row["dynamics"] == "smooth"
        ),
        None,
    )
    negative_rank3_cells = [
        {
            "noise_level": float(row["noise_level"]),
            "view_count": int(row["view_count"]),
            "dynamics": str(row["dynamics"]),
            "field_improvement_pct": float(row["rel_l2_global_improvement_pct_mean"]),
        }
        for row in summary_rows
        if int(row["rank"]) == 3 and float(row["rel_l2_global_improvement_pct_mean"]) < 0.0
    ]
    field_positive_mass_negative = [
        row
        for row in summary_rows
        if float(row["rel_l2_global_improvement_pct_mean"]) > 0.0
        and float(row["mass_trace_rmse_improvement_pct_mean"]) < 0.0
    ]
    global_rank_error = {
        str(rank): float(
            np.mean([float(row["lowrank_rel_l2_global_mean"]) for row in summary_rows if int(row["rank"]) == rank])
        )
        for rank in ranks
    }
    global_best_rank = min(ranks, key=lambda rank: global_rank_error[str(rank)])

    rank3_cells = [row for row in summary_rows if int(row["rank"]) == 3]
    rank3_ci_positive = sum(
        float(row["rel_l2_global_improvement_pct_mean"])
        - float(row["rel_l2_global_improvement_pct_ci95"])
        > 0.0
        for row in rank3_cells
    )
    rank3_ci_negative = sum(
        float(row["rel_l2_global_improvement_pct_mean"])
        + float(row["rel_l2_global_improvement_pct_ci95"])
        < 0.0
        for row in rank3_cells
    )

    def best_counts_by(factor: str, levels: list[object]) -> dict[str, dict[str, int]]:
        return {
            str(level): {
                str(rank): sum(
                    row[factor] == level and int(row["best_rank"]) == rank
                    for row in selections
                )
                for rank in ranks
            }
            for level in levels
        }

    tradeoff_by_rank = {}
    for rank in ranks:
        rows = [row for row in summary_rows if int(row["rank"]) == rank]
        positive = [row for row in rows if float(row["rel_l2_global_improvement_pct_mean"]) > 0.0]
        mass_loss = [row for row in positive if float(row["mass_trace_rmse_improvement_pct_mean"]) < 0.0]
        tradeoff_by_rank[str(rank)] = {
            "field_positive_cells": len(positive),
            "field_positive_mass_loss_cells": len(mass_loss),
            "fraction_among_field_positive": len(mass_loss) / max(len(positive), 1),
        }
    return {
        "experiment": "M3B crossed rank-noise-view-dynamics interaction sweep",
        "design": (
            f"{len(ranks)} ranks x {len(noises)} noise levels x "
            f"{len(views)} view counts x {len(dynamics_modes)} dynamics x paired seeds"
        ),
        "seed_count": seed_count,
        "factor_levels": {
            "rank": ranks,
            "noise_multiplier": noises,
            "view_count": views,
            "dynamics": dynamics_modes,
        },
        "observation_cells": len(noises) * len(views) * len(dynamics_modes) * seed_count,
        "environment_cells": len(noises) * len(views) * len(dynamics_modes),
        "framewise_rows": len(noises) * len(views) * len(dynamics_modes) * seed_count,
        "lowrank_rows": len(paired_rows),
        "method_rows": raw_count,
        "paired_rows": len(paired_rows),
        "runtime_seconds": runtime_seconds,
        "ci_method": f"paired mean +/- Student-t 95% CI; t={t_critical_95(seed_count):.3f} for n={seed_count}",
        "rank_selection": {
            **rank_report,
            "global_best_rank_by_mean_field_l2": global_best_rank,
            "global_mean_field_l2_by_rank": global_rank_error,
            "field_positive_win_rate_by_rank": rank_field_win_rate,
            "best_rank_counts_by_noise": best_counts_by("noise_level", noises),
            "best_rank_counts_by_views": best_counts_by("view_count", views),
            "best_rank_counts_by_dynamics": best_counts_by("dynamics", dynamics_modes),
        },
        "default_rank3_cell": rank3_summary,
        "rank3_negative_field_cells": {
            "count": len(negative_rank3_cells),
            "out_of": len(noises) * len(views) * len(dynamics_modes),
            "cells": negative_rank3_cells,
        },
        "rank3_field_sign_by_student_t_ci": {
            "positive": rank3_ci_positive,
            "negative": rank3_ci_negative,
            "crosses_zero": len(rank3_cells) - rank3_ci_positive - rank3_ci_negative,
            "total": len(rank3_cells),
        },
        "field_gain_but_mass_loss_cells": {
            "count": len(field_positive_mass_negative),
            "out_of": len(summary_rows),
            "fraction": len(field_positive_mass_negative) / max(len(summary_rows), 1),
            "by_rank": tradeoff_by_rank,
        },
        "marginal_interactions": {
            "rank_x_noise_field_improvement_pct": matrix_to_nested(ranks, noises, matrices["rank_noise_field"]),
            "rank_x_views_heldout_improvement_pct": matrix_to_nested(ranks, views, matrices["rank_views_heldout"]),
            "rank_x_dynamics_temporal_gradient_improvement_pct": matrix_to_nested(ranks, dynamics_modes, matrices["rank_dynamics_temporal"]),
            "rank3_noise_x_views_field_improvement_pct": matrix_to_nested(views, noises, rank3_matrices["field"]),
            "rank3_noise_x_views_heldout_improvement_pct": matrix_to_nested(views, noises, rank3_matrices["heldout"]),
            "rank3_noise_x_views_mass_trace_improvement_pct": matrix_to_nested(views, noises, rank3_matrices["mass"]),
        },
        "interpretation_boundary": [
            "The crossed factors are rank, observation-noise multiplier, view count, and synthetic dynamics; frame count and systematic bias remain separate validation axes.",
            "The eight seeds are paired observation-noise realizations, not eight independent phantom families or experimental operating points.",
            "Cell-wise best rank uses mean globally aligned field L2 and is not a deployable selector without ground truth.",
            "The straight-ray stack baseline and temporal SVD are a clean-room undergraduate toy, not the TDBOST architecture or OERF camera model.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=8, help="paired observation-noise seeds (default: 8)")
    parser.add_argument("--quick", action="store_true", help="run a reduced 3x3x2x2 grid with two seeds")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        seed_count = 2
        ranks = [2, 3, 5]
        noises = [0.0, 0.14, 0.42]
        views = [3, 5]
        dynamics_modes = ["smooth", "transient"]
    else:
        seed_count = args.seed_count
        ranks = DEFAULT_RANKS
        noises = DEFAULT_NOISES
        views = DEFAULT_VIEWS
        dynamics_modes = DEFAULT_DYNAMICS
    if seed_count < 2:
        raise SystemExit("seed-count must be at least 2")

    results = args.output_dir
    results.mkdir(parents=True, exist_ok=True)
    seeds = [20260710 + index for index in range(seed_count)]
    n, nz, frame_count = 24, 10, 18
    raw_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    total_observation_cells = len(noises) * len(views) * len(dynamics_modes) * seed_count
    completed = 0

    references = {
        dynamics: make_4d_sequence(n=n, nz=nz, nt=frame_count, dynamics=dynamics)
        for dynamics in dynamics_modes
    }
    for dynamics in dynamics_modes:
        reference = references[dynamics]
        for view_count in views:
            angles = np.linspace(0, 180, view_count, endpoint=False)
            heldout_angles = np.array([90.0 / view_count])
            reference_heldout = deflection_sequence(reference, heldout_angles)
            for noise_level in noises:
                for seed in seeds:
                    design_id = f"d-{dynamics}_v-{view_count}_n-{noise_level:.2f}_s-{seed}"
                    baseline, reconstruction_seconds = reconstruct_case(
                        reference, angles, seed, noise_level, bias="none"
                    )
                    baseline_metrics = metric_suite(
                        baseline, reference, heldout_angles, reference_heldout
                    )
                    raw_rows.append(
                        raw_row(
                            design_id,
                            seed,
                            "framewise",
                            0,
                            noise_level,
                            view_count,
                            dynamics,
                            reconstruction_seconds,
                            0.0,
                            baseline_metrics,
                        )
                    )
                    for rank in ranks:
                        post_started = time.perf_counter()
                        lowrank = low_rank_temporal(baseline, rank=rank)
                        postprocess_seconds = time.perf_counter() - post_started
                        lowrank_metrics = metric_suite(
                            lowrank, reference, heldout_angles, reference_heldout
                        )
                        raw_rows.append(
                            raw_row(
                                design_id,
                                seed,
                                "lowrank",
                                rank,
                                noise_level,
                                view_count,
                                dynamics,
                                reconstruction_seconds,
                                postprocess_seconds,
                                lowrank_metrics,
                            )
                        )
                        paired_rows.append(
                            paired_row(
                                design_id,
                                seed,
                                rank,
                                noise_level,
                                view_count,
                                dynamics,
                                baseline_metrics,
                                lowrank_metrics,
                            )
                        )
                    completed += 1
                    if completed % max(total_observation_cells // 16, 1) == 0 or completed == total_observation_cells:
                        print(
                            f"completed observation cells: {completed}/{total_observation_cells}",
                            flush=True,
                        )

    summary_rows = cell_summary(paired_rows)
    selections, rank_report = rank_selection(summary_rows, ranks)
    matrices = plot_heatmaps(
        paired_rows,
        ranks,
        noises,
        views,
        dynamics_modes,
        results / "m3b_interaction_heatmaps.png",
    )
    plot_rank_selection(
        selections,
        rank_report,
        ranks,
        noises,
        views,
        dynamics_modes,
        results / "m3b_rank_selection_stability.png",
    )
    plot_tradeoffs(summary_rows, ranks, results / "m3b_interaction_tradeoffs.png")
    rank3_matrices = plot_rank3_operating_map(
        paired_rows,
        noises,
        views,
        results / "m3b_rank3_operating_map.png",
    )
    runtime_seconds = time.perf_counter() - started
    report = build_report(
        paired_rows,
        summary_rows,
        selections,
        rank_report,
        matrices,
        rank3_matrices,
        ranks,
        noises,
        views,
        dynamics_modes,
        seed_count,
        len(raw_rows),
        runtime_seconds,
    )

    write_csv(results / "interaction_raw.csv", raw_rows)
    write_csv(results / "interaction_paired.csv", paired_rows)
    write_csv(results / "interaction_cell_summary.csv", summary_rows)
    write_csv(results / "interaction_rank_selection.csv", selections)
    with (results / "interaction_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    print(f"completed {len(raw_rows)} method rows and {len(paired_rows)} paired rows in {runtime_seconds:.1f} s")
    print(json.dumps(report["rank_selection"], indent=2))
    print(f"results: {results}")


if __name__ == "__main__":
    main()
