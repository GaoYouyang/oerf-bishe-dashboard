#!/usr/bin/env python3
"""Multi-seed six-axis stress test for the M3B 4D BOST toy.

This is an OFAT (one-factor-at-a-time) experiment, not a full factorial study.
It probes rank, deflection noise, frame count, view count, systematic bias, and
temporal dynamics while keeping the remaining settings fixed. The goal is to
produce honest uncertainty bars and failure signatures for an undergraduate
4D BOST discussion, not to reproduce the ACM TOG implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
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

from demo_m1.run_m1_3d_stack_bost import baseline_stack, synthesize_deflection_stack
from demo_m3b.run_m3b_4d_lowrank_bost import (
    add_deflection_noise,
    centroid_x,
    low_rank_temporal,
    make_4d_sequence,
    temporal_smoothness,
)


METRIC_NAMES = [
    "rel_l2_global",
    "paper_l2_squared",
    "mean_frame_rel_l2_global",
    "global_cc",
    "temporal_smoothness",
    "temporal_gradient_rel_l2",
    "temporal_curvature_rel_l2",
    "dynamics_energy_ratio",
    "centroid_rmse",
    "mass_trace_rmse",
    "heldout_reprojection_rel_l2",
]

AXIS_ORDER = ["rank", "noise", "frame_count", "view_count", "bias", "dynamics"]


def global_align(reconstruction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Use one affine calibration for the complete 3D+time sequence."""
    x = reconstruction.reshape(-1)
    y = reference.reshape(-1)
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    return scale * reconstruction + offset


def deflection_sequence(sequence: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return np.stack([synthesize_deflection_stack(frame, angles) for frame in sequence], axis=0)


def reconstruct_case(
    reference: np.ndarray,
    angles: np.ndarray,
    seed: int,
    noise_level: float,
    bias: str = "none",
) -> tuple[np.ndarray, float]:
    """Generate noisy observations and reconstruct each time frame.

    The bias cases are controlled signatures rather than calibrated camera
    models. They are intentionally simple so each failure can be interpreted.
    """
    if bias not in {"none", "geometry_2deg", "scale_drift", "flow_drift", "sync_lag"}:
        raise ValueError(f"unknown bias case: {bias}")
    rng = np.random.default_rng(seed)
    nt = reference.shape[0]
    n = reference.shape[-1]
    recon_angles = angles + 2.0 if bias == "geometry_2deg" else angles
    detector = np.linspace(-1.0, 1.0, n).reshape(1, n, 1)
    frames = []
    started = time.perf_counter()
    for frame_index in range(nt):
        source_index = max(0, frame_index - 1) if bias == "sync_lag" else frame_index
        deflection = synthesize_deflection_stack(reference[source_index], angles)
        if bias == "scale_drift":
            scale = 1.0 + 0.12 * np.sin(2 * np.pi * frame_index / max(nt, 1) + 0.35)
            deflection = deflection * scale
        elif bias == "flow_drift":
            drift_phase = 2.0 * frame_index / max(nt - 1, 1) - 1.0
            pattern = 0.72 * detector + 0.28 * detector**2
            deflection = deflection + 0.18 * float(np.std(deflection)) * drift_phase * pattern
        noisy = add_deflection_noise(deflection, rng=rng, noise_level=noise_level)
        frames.append(baseline_stack(noisy, recon_angles, n))
    return np.stack(frames, axis=0), time.perf_counter() - started


def metric_suite(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    heldout_angles: np.ndarray,
    reference_heldout: np.ndarray,
) -> dict[str, float]:
    aligned = global_align(reconstruction, reference)
    error = aligned - reference
    reference_norm = np.linalg.norm(reference) + 1e-12
    error_norm = np.linalg.norm(error)
    per_frame = [
        np.linalg.norm(error[index]) / (np.linalg.norm(reference[index]) + 1e-12)
        for index in range(reference.shape[0])
    ]

    gradient_ref = np.diff(reference, axis=0)
    gradient_rec = np.diff(aligned, axis=0)
    gradient_norm = np.linalg.norm(gradient_ref) + 1e-12
    if reference.shape[0] > 2:
        curvature_ref = np.diff(reference, n=2, axis=0)
        curvature_rec = np.diff(aligned, n=2, axis=0)
        curvature_error = np.linalg.norm(curvature_rec - curvature_ref) / (np.linalg.norm(curvature_ref) + 1e-12)
    else:
        curvature_error = float("nan")

    positive_ref = np.maximum(reference, 0.0)
    positive_rec = np.maximum(aligned, 0.0)
    mass_ref = positive_ref.sum(axis=(1, 2, 3))
    mass_rec = positive_rec.sum(axis=(1, 2, 3))
    mass_scale = float(np.mean(mass_ref)) + 1e-12

    reconstructed_heldout = deflection_sequence(aligned, heldout_angles)
    heldout_error = np.linalg.norm(reconstructed_heldout - reference_heldout) / (np.linalg.norm(reference_heldout) + 1e-12)
    correlation = np.corrcoef(aligned.reshape(-1), reference.reshape(-1))[0, 1]
    return {
        "rel_l2_global": float(error_norm / reference_norm),
        "paper_l2_squared": float((error_norm**2) / (reference_norm**2)),
        "mean_frame_rel_l2_global": float(np.mean(per_frame)),
        "global_cc": float(correlation),
        "temporal_smoothness": temporal_smoothness(aligned),
        "temporal_gradient_rel_l2": float(np.linalg.norm(gradient_rec - gradient_ref) / gradient_norm),
        "temporal_curvature_rel_l2": float(curvature_error),
        "dynamics_energy_ratio": float(np.linalg.norm(gradient_rec) / gradient_norm),
        "centroid_rmse": float(np.sqrt(np.mean((centroid_x(aligned) - centroid_x(reference)) ** 2))),
        "mass_trace_rmse": float(np.sqrt(np.mean((mass_rec - mass_ref) ** 2)) / mass_scale),
        "heldout_reprojection_rel_l2": float(heldout_error),
    }


def make_row(
    axis: str,
    condition: str,
    condition_value: str | float | int,
    seed: int,
    method: str,
    rank: int,
    view_count: int,
    frame_count: int,
    noise_level: float,
    bias: str,
    dynamics: str,
    reconstruction_seconds: float,
    postprocess_seconds: float,
    values: dict[str, float],
) -> dict[str, str | float | int]:
    return {
        "axis": axis,
        "condition": condition,
        "condition_value": condition_value,
        "seed": seed,
        "method": method,
        "rank": rank,
        "view_count": view_count,
        "frame_count": frame_count,
        "noise_level": noise_level,
        "bias": bias,
        "dynamics": dynamics,
        "reconstruction_seconds": reconstruction_seconds,
        "postprocess_seconds": postprocess_seconds,
        **values,
    }


def evaluate_pair(
    axis: str,
    condition: str,
    condition_value: str | float | int,
    reference: np.ndarray,
    angles: np.ndarray,
    seed: int,
    noise_level: float,
    rank: int,
    bias: str,
    dynamics: str,
) -> list[dict[str, str | float | int]]:
    heldout = np.array([90.0 / len(angles)])
    reference_heldout = deflection_sequence(reference, heldout)
    baseline, reconstruction_seconds = reconstruct_case(reference, angles, seed, noise_level, bias=bias)
    baseline_metrics = metric_suite(baseline, reference, heldout, reference_heldout)
    started = time.perf_counter()
    lowrank = low_rank_temporal(baseline, rank=rank)
    postprocess_seconds = time.perf_counter() - started
    lowrank_metrics = metric_suite(lowrank, reference, heldout, reference_heldout)
    context = {
        "axis": axis,
        "condition": condition,
        "condition_value": condition_value,
        "seed": seed,
        "rank": rank,
        "view_count": len(angles),
        "frame_count": reference.shape[0],
        "noise_level": noise_level,
        "bias": bias,
        "dynamics": dynamics,
        "reconstruction_seconds": reconstruction_seconds,
    }
    return [
        make_row(method="framewise", postprocess_seconds=0.0, values=baseline_metrics, **context),
        make_row(method="lowrank", postprocess_seconds=postprocess_seconds, values=lowrank_metrics, **context),
    ]


def evaluate_rank_axis(
    reference: np.ndarray,
    angles: np.ndarray,
    seed: int,
    noise_level: float,
    ranks: list[int],
) -> list[dict[str, str | float | int]]:
    heldout = np.array([90.0 / len(angles)])
    reference_heldout = deflection_sequence(reference, heldout)
    baseline, reconstruction_seconds = reconstruct_case(reference, angles, seed, noise_level)
    baseline_metrics = metric_suite(baseline, reference, heldout, reference_heldout)
    rows = []
    for rank in ranks:
        started = time.perf_counter()
        lowrank = low_rank_temporal(baseline, rank=rank)
        postprocess_seconds = time.perf_counter() - started
        lowrank_metrics = metric_suite(lowrank, reference, heldout, reference_heldout)
        context = {
            "axis": "rank",
            "condition": f"rank_{rank}",
            "condition_value": rank,
            "seed": seed,
            "rank": rank,
            "view_count": len(angles),
            "frame_count": reference.shape[0],
            "noise_level": noise_level,
            "bias": "none",
            "dynamics": "smooth",
            "reconstruction_seconds": reconstruction_seconds,
        }
        rows.append(make_row(method="framewise", postprocess_seconds=0.0, values=baseline_metrics, **context))
        rows.append(make_row(method="lowrank", postprocess_seconds=postprocess_seconds, values=lowrank_metrics, **context))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, std, ci95


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (row["axis"], row["condition"], row["condition_value"], row["method"], row["rank"])
        groups[key].append(row)
    summary = []
    for key, items in groups.items():
        axis, condition, condition_value, method, rank = key
        output: dict[str, object] = {
            "axis": axis,
            "condition": condition,
            "condition_value": condition_value,
            "method": method,
            "rank": rank,
            "seed_count": len(items),
        }
        for metric in METRIC_NAMES + ["reconstruction_seconds", "postprocess_seconds"]:
            values = [float(item[metric]) for item in items if np.isfinite(float(item[metric]))]
            mean, std, ci95 = mean_std_ci(values)
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            output[f"{metric}_ci95"] = ci95
        summary.append(output)
    axis_position = {axis: index for index, axis in enumerate(AXIS_ORDER)}
    return sorted(summary, key=lambda row: (axis_position[str(row["axis"])], str(row["condition"]), str(row["method"])))


def paired_improvements(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (row["axis"], row["condition"], row["condition_value"], row["seed"], row["rank"])
        groups[key][str(row["method"])] = row
    output = []
    lower_is_better = [
        "rel_l2_global",
        "paper_l2_squared",
        "temporal_gradient_rel_l2",
        "temporal_curvature_rel_l2",
        "centroid_rmse",
        "mass_trace_rmse",
        "heldout_reprojection_rel_l2",
    ]
    for key, methods in groups.items():
        if "framewise" not in methods or "lowrank" not in methods:
            continue
        baseline = methods["framewise"]
        lowrank = methods["lowrank"]
        axis, condition, condition_value, seed, rank = key
        row: dict[str, object] = {
            "axis": axis,
            "condition": condition,
            "condition_value": condition_value,
            "seed": seed,
            "rank": rank,
        }
        for metric in lower_is_better:
            base_value = float(baseline[metric])
            low_value = float(lowrank[metric])
            row[f"{metric}_improvement_pct"] = 100.0 * (base_value - low_value) / (abs(base_value) + 1e-12)
        base_energy = float(baseline["dynamics_energy_ratio"])
        low_energy = float(lowrank["dynamics_energy_ratio"])
        row["dynamics_energy_closeness_gain"] = abs(base_energy - 1.0) - abs(low_energy - 1.0)
        output.append(row)
    axis_position = {axis: index for index, axis in enumerate(AXIS_ORDER)}
    return sorted(output, key=lambda row: (axis_position[str(row["axis"])], str(row["condition"]), int(row["seed"])))


def rows_for(rows: list[dict[str, object]], axis: str, method: str, metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["axis"] == axis and row["method"] == method:
            grouped[str(row["condition"])].append(float(row[metric]))
    return grouped


def paired_for(rows: list[dict[str, object]], axis: str, metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["axis"] == axis:
            grouped[str(row["condition"])].append(float(row[metric]))
    return grouped


def errorbar_series(grouped: dict[str, list[float]], order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means, cis = [], []
    for condition in order:
        mean, _, ci = mean_std_ci(grouped[condition])
        means.append(mean)
        cis.append(ci)
    return np.asarray(means), np.asarray(cis)


def plot_six_axis_overview(
    raw_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    seed_count: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    rank_order = ["rank_1", "rank_2", "rank_3", "rank_5", "rank_8", "rank_12"]
    rank_x = np.array([1, 2, 3, 5, 8, 12])
    rank_low, rank_ci = errorbar_series(rows_for(raw_rows, "rank", "lowrank", "rel_l2_global"), rank_order)
    rank_base, _ = errorbar_series(rows_for(raw_rows, "rank", "framewise", "rel_l2_global"), rank_order)
    axes[0, 0].errorbar(rank_x, rank_low, yerr=rank_ci, marker="o", color="#246a73", capsize=3, label="low-rank")
    axes[0, 0].plot(rank_x, rank_base, linestyle="--", color="#9b4d3f", label="framewise")
    axes[0, 0].set_title("Rank: global relative L2")
    axes[0, 0].set_xlabel("rank")
    axes[0, 0].legend()

    panel_specs = [
        (axes[0, 1], "noise", ["noise_0.00", "noise_0.07", "noise_0.14", "noise_0.28", "noise_0.42"], ["0", "0.07", "0.14", "0.28", "0.42"], "rel_l2_global_improvement_pct", "Noise: field L2 improvement", "noise multiplier"),
        (axes[0, 2], "frame_count", ["frames_8", "frames_12", "frames_18", "frames_30"], ["8", "12", "18", "30"], "temporal_gradient_rel_l2_improvement_pct", "Frames: temporal-gradient improvement", "frame count"),
        (axes[1, 0], "view_count", ["views_3", "views_5", "views_7", "views_9"], ["3", "5", "7", "9"], "heldout_reprojection_rel_l2_improvement_pct", "Views: held-out improvement", "view count"),
        (axes[1, 1], "bias", ["none", "geometry_2deg", "scale_drift", "flow_drift", "sync_lag"], ["none", "geometry", "scale", "flow drift", "sync lag"], "rel_l2_global_improvement_pct", "Bias: field L2 improvement", "bias type"),
        (axes[1, 2], "dynamics", ["smooth", "fast", "chirp", "transient"], ["smooth", "fast", "chirp", "transient"], "temporal_gradient_rel_l2_improvement_pct", "Dynamics: temporal-gradient improvement", "dynamics"),
    ]
    for ax, axis, order, labels, metric, title, xlabel in panel_specs:
        means, cis = errorbar_series(paired_for(paired_rows, axis, metric), order)
        x = np.arange(len(order))
        ax.bar(x, means, yerr=cis, color=["#3d7f75", "#4f77a2", "#ba7d35", "#9b4d3f", "#6d6592"][: len(order)], capsize=3)
        ax.axhline(0.0, color="#3c454a", linewidth=1)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("paired improvement (%)")
    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"M3B six-axis OFAT sweep: {seed_count} paired noise seeds, mean and 95% CI", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rank_stability(raw_rows: list[dict[str, object]], path: Path) -> None:
    rank_order = ["rank_1", "rank_2", "rank_3", "rank_5", "rank_8", "rank_12"]
    ranks = np.array([1, 2, 3, 5, 8, 12])
    panels = [
        ("rel_l2_global", "Global relative L2", "lower is better"),
        ("temporal_gradient_rel_l2", "Temporal-gradient error", "lower is better"),
        ("heldout_reprojection_rel_l2", "Held-out deflection error", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes, panels):
        low_mean, low_ci = errorbar_series(rows_for(raw_rows, "rank", "lowrank", metric), rank_order)
        base_mean, base_ci = errorbar_series(rows_for(raw_rows, "rank", "framewise", metric), rank_order)
        ax.errorbar(ranks, low_mean, yerr=low_ci, marker="o", capsize=3, color="#246a73", label="low-rank")
        ax.errorbar(ranks, base_mean, yerr=base_ci, marker="s", capsize=3, linestyle="--", color="#9b4d3f", label="framewise")
        ax.set_title(title)
        ax.set_xlabel("rank")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("M3B rank stability across paired random seeds", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bias_dynamics(raw_rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), constrained_layout=True)
    specs = [
        (axes[0], "bias", ["none", "geometry_2deg", "scale_drift", "flow_drift", "sync_lag"], ["none", "geometry", "scale", "flow drift", "sync lag"], "rel_l2_global", "Systematic-bias field error"),
        (axes[1], "dynamics", ["smooth", "fast", "chirp", "transient"], ["smooth", "fast", "chirp", "transient"], "dynamics_energy_ratio", "Dynamics-energy ratio (ideal = 1)"),
    ]
    for ax, axis, order, labels, metric, title in specs:
        base_mean, base_ci = errorbar_series(rows_for(raw_rows, axis, "framewise", metric), order)
        low_mean, low_ci = errorbar_series(rows_for(raw_rows, axis, "lowrank", metric), order)
        x = np.arange(len(order))
        width = 0.38
        ax.bar(x - width / 2, base_mean, width, yerr=base_ci, color="#4f77a2", capsize=3, label="framewise")
        ax.bar(x + width / 2, low_mean, width, yerr=low_ci, color="#ba7d35", capsize=3, label="rank 3")
        if metric == "dynamics_energy_ratio":
            ax.axhline(1.0, color="#3c454a", linewidth=1, linestyle="--")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("M3B failure signatures: denoising does not guarantee bias correction or event retention", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(
    raw_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    seed_count: int,
    runtime_seconds: float,
) -> dict[str, object]:
    rank_group = rows_for(raw_rows, "rank", "lowrank", "rel_l2_global")
    best_condition = min(rank_group, key=lambda condition: np.mean(rank_group[condition]))
    rank_value = int(best_condition.split("_")[-1])

    def paired_mean(axis: str, condition: str, metric: str) -> float:
        values = [float(row[metric]) for row in paired_rows if row["axis"] == axis and row["condition"] == condition]
        return float(np.mean(values))

    def raw_mean(axis: str, condition: str, method: str, metric: str) -> float:
        values = [
            float(row[metric])
            for row in raw_rows
            if row["axis"] == axis and row["condition"] == condition and row["method"] == method
        ]
        return float(np.mean(values))

    bias_summary = {
        condition: {
            "field_l2_improvement_pct": paired_mean("bias", condition, "rel_l2_global_improvement_pct"),
            "temporal_gradient_improvement_pct": paired_mean("bias", condition, "temporal_gradient_rel_l2_improvement_pct"),
        }
        for condition in ["none", "geometry_2deg", "scale_drift", "flow_drift", "sync_lag"]
    }
    dynamics_summary = {
        condition: {
            "field_l2_improvement_pct": paired_mean("dynamics", condition, "rel_l2_global_improvement_pct"),
            "temporal_gradient_improvement_pct": paired_mean("dynamics", condition, "temporal_gradient_rel_l2_improvement_pct"),
            "energy_closeness_gain": paired_mean("dynamics", condition, "dynamics_energy_closeness_gain"),
        }
        for condition in ["smooth", "fast", "chirp", "transient"]
    }
    return {
        "experiment": "M3B six-axis OFAT sweep",
        "seed_count": seed_count,
        "runtime_seconds": runtime_seconds,
        "default_case": {
            "n": 24,
            "nz": 10,
            "frames": 18,
            "views": 5,
            "noise_multiplier": 0.14,
            "rank": 3,
            "bias": "none",
            "dynamics": "smooth",
        },
        "best_rank_by_mean_global_rel_l2": rank_value,
        "default_rank3_absolute_metrics": {
            metric: {
                "framewise": raw_mean("rank", "rank_3", "framewise", metric),
                "lowrank": raw_mean("rank", "rank_3", "lowrank", metric),
            }
            for metric in [
                "rel_l2_global",
                "paper_l2_squared",
                "temporal_gradient_rel_l2",
                "temporal_curvature_rel_l2",
                "dynamics_energy_ratio",
                "centroid_rmse",
                "mass_trace_rmse",
                "heldout_reprojection_rel_l2",
            ]
        },
        "default_rank3_paired_improvement_pct": {
            "field_rel_l2": paired_mean("rank", "rank_3", "rel_l2_global_improvement_pct"),
            "paper_l2_squared": paired_mean("rank", "rank_3", "paper_l2_squared_improvement_pct"),
            "temporal_gradient": paired_mean("rank", "rank_3", "temporal_gradient_rel_l2_improvement_pct"),
            "temporal_curvature": paired_mean("rank", "rank_3", "temporal_curvature_rel_l2_improvement_pct"),
            "centroid": paired_mean("rank", "rank_3", "centroid_rmse_improvement_pct"),
            "mass_trace": paired_mean("rank", "rank_3", "mass_trace_rmse_improvement_pct"),
            "heldout_reprojection": paired_mean("rank", "rank_3", "heldout_reprojection_rel_l2_improvement_pct"),
        },
        "noise_summary": {
            condition: paired_mean("noise", condition, "rel_l2_global_improvement_pct")
            for condition in ["noise_0.00", "noise_0.07", "noise_0.14", "noise_0.28", "noise_0.42"]
        },
        "frame_summary": {
            condition: paired_mean("frame_count", condition, "temporal_gradient_rel_l2_improvement_pct")
            for condition in ["frames_8", "frames_12", "frames_18", "frames_30"]
        },
        "view_summary": {
            condition: paired_mean("view_count", condition, "heldout_reprojection_rel_l2_improvement_pct")
            for condition in ["views_3", "views_5", "views_7", "views_9"]
        },
        "bias_summary": bias_summary,
        "dynamics_summary": dynamics_summary,
        "interpretation_boundary": [
            "This is a lightweight OFAT toy, not a full factorial design or TDBOST reproduction.",
            "Positive smoothing metrics do not prove that camera geometry, synchronization, or optical-flow bias was corrected.",
            "The paper-style squared L2 and norm-ratio L2 are reported separately and must not be mixed.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=8, help="number of paired noise seeds (default: 8)")
    parser.add_argument("--quick", action="store_true", help="run two seeds for a fast smoke test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_count = 2 if args.quick else args.seed_count
    if seed_count < 2:
        raise SystemExit("seed-count must be at least 2")
    results = args.output_dir
    results.mkdir(parents=True, exist_ok=True)
    seeds = [20260710 + index for index in range(seed_count)]
    n, nz = 24, 10
    default_nt, default_views, default_noise, default_rank = 18, 5, 0.14, 3
    ranks = [1, 2, 3, 5, 8, 12]
    started = time.perf_counter()
    raw_rows: list[dict[str, object]] = []

    default_reference = make_4d_sequence(n=n, nz=nz, nt=default_nt, dynamics="smooth")
    default_angles = np.linspace(0, 180, default_views, endpoint=False)
    print(f"rank axis: {seed_count} paired seeds x {len(ranks)} ranks", flush=True)
    for seed in seeds:
        raw_rows.extend(evaluate_rank_axis(default_reference, default_angles, seed, default_noise, ranks))

    noise_levels = [0.0, 0.07, 0.14, 0.28, 0.42]
    print(f"noise axis: {seed_count} paired seeds x {len(noise_levels)} levels", flush=True)
    for noise in noise_levels:
        for seed in seeds:
            raw_rows.extend(evaluate_pair("noise", f"noise_{noise:.2f}", noise, default_reference, default_angles, seed, noise, default_rank, "none", "smooth"))

    frame_counts = [8, 12, 18, 30]
    print(f"frame axis: {seed_count} paired seeds x {len(frame_counts)} levels", flush=True)
    for frame_count in frame_counts:
        reference = make_4d_sequence(n=n, nz=nz, nt=frame_count, dynamics="smooth")
        for seed in seeds:
            raw_rows.extend(evaluate_pair("frame_count", f"frames_{frame_count}", frame_count, reference, default_angles, seed, default_noise, default_rank, "none", "smooth"))

    view_counts = [3, 5, 7, 9]
    print(f"view axis: {seed_count} paired seeds x {len(view_counts)} levels", flush=True)
    for view_count in view_counts:
        angles = np.linspace(0, 180, view_count, endpoint=False)
        for seed in seeds:
            raw_rows.extend(evaluate_pair("view_count", f"views_{view_count}", view_count, default_reference, angles, seed, default_noise, default_rank, "none", "smooth"))

    biases = ["none", "geometry_2deg", "scale_drift", "flow_drift", "sync_lag"]
    print(f"bias axis: {seed_count} paired seeds x {len(biases)} signatures", flush=True)
    for bias in biases:
        for seed in seeds:
            raw_rows.extend(evaluate_pair("bias", bias, bias, default_reference, default_angles, seed, default_noise, default_rank, bias, "smooth"))

    dynamics_modes = ["smooth", "fast", "chirp", "transient"]
    print(f"dynamics axis: {seed_count} paired seeds x {len(dynamics_modes)} modes", flush=True)
    for dynamics in dynamics_modes:
        reference = make_4d_sequence(n=n, nz=nz, nt=default_nt, dynamics=dynamics)
        for seed in seeds:
            raw_rows.extend(evaluate_pair("dynamics", dynamics, dynamics, reference, default_angles, seed, default_noise, default_rank, "none", dynamics))

    summary_rows = aggregate_rows(raw_rows)
    paired_rows = paired_improvements(raw_rows)
    runtime_seconds = time.perf_counter() - started
    report = build_report(raw_rows, paired_rows, seed_count, runtime_seconds)

    write_csv(results / "six_axis_raw.csv", raw_rows)
    write_csv(results / "six_axis_summary.csv", summary_rows)
    write_csv(results / "six_axis_paired_improvements.csv", paired_rows)
    with (results / "six_axis_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    plot_six_axis_overview(raw_rows, paired_rows, seed_count, results / "m3b_six_axis_overview.png")
    plot_rank_stability(raw_rows, results / "m3b_rank_seed_stability.png")
    plot_bias_dynamics(raw_rows, results / "m3b_bias_dynamics_diagnostic.png")

    print(f"completed {len(raw_rows)} method rows in {runtime_seconds:.1f} s", flush=True)
    print(json.dumps(report["default_rank3_paired_improvement_pct"], indent=2), flush=True)
    print(f"best rank by multi-seed global relative L2: {report['best_rank_by_mean_global_rel_l2']}", flush=True)
    print(f"results: {results}", flush=True)


if __name__ == "__main__":
    main()
