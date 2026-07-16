#!/usr/bin/env python3
"""Family-generalized observable rank selection for the M3B 4D BOST toy.

The earlier M3B crossed sweep establishes that one fixed temporal rank is not
optimal across noise, view count, and dynamics. This experiment removes two
remaining shortcuts:

1. it crosses four synthetic morphology families with dynamics, noise, views,
   and paired observation-noise seeds;
2. it evaluates rank selectors without field truth at test time using strict
   leave-one-phantom-family-out (LOFO) validation.

The forward model remains a compact straight-ray BOS-like stack. The experiment
tests synthetic transfer and selector logic; it is not a TDBOST reproduction or
evidence of real OERF generalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
DEFAULT_CONFIG = ROOT / "configs" / "family_selector.json"
DEFAULT_RESULTS = ROOT / "results" / "family_selector"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_m1.run_m1_3d_stack_bost import baseline_stack, synthesize_deflection_stack
from demo_m3b.run_m3b_4d_lowrank_bost import centroid_x, gaussian3d, make_grid_3d
from demo_m3b.run_m3b_interaction_sweep import t_critical_95
from demo_m3b.run_m3b_six_axis_sweep import deflection_sequence, metric_suite


NO_HOLDOUT_FEATURES = [
    "rank_fraction",
    "rank_fraction_squared",
    "log_rank",
    "spectral_energy",
    "spectral_tail",
    "spectral_gap",
    "log_support_residual",
    "temporal_gradient_ratio",
    "temporal_curvature_ratio",
    "mass_curvature",
    "negative_fraction",
    "observation_temporal_noise_proxy",
    "view_fraction",
    "rank_x_noise_proxy",
    "rank_x_spectral_tail",
]

HOLDOUT_FEATURES = NO_HOLDOUT_FEATURES + [
    "log_heldout_residual",
    "log_heldout_support_ratio",
]

CAPACITY_SPECTRUM_FEATURES = [
    "rank_fraction",
    "rank_fraction_squared",
    "log_rank",
    "spectral_energy",
    "spectral_tail",
    "spectral_gap",
    "observation_temporal_noise_proxy",
    "view_fraction",
    "rank_x_noise_proxy",
    "rank_x_spectral_tail",
]

SUPPORT_FEATURES = [
    "rank_fraction",
    "rank_fraction_squared",
    "log_rank",
    "log_support_residual",
    "observation_temporal_noise_proxy",
    "view_fraction",
    "rank_x_noise_proxy",
]

TEMPORAL_FEATURES = [
    "rank_fraction",
    "rank_fraction_squared",
    "log_rank",
    "temporal_gradient_ratio",
    "temporal_curvature_ratio",
    "mass_curvature",
    "negative_fraction",
    "observation_temporal_noise_proxy",
    "view_fraction",
    "rank_x_noise_proxy",
]

MODEL_FEATURE_SETS = {
    "capacity_spectrum": CAPACITY_SPECTRUM_FEATURES,
    "support": SUPPORT_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "no_holdout": NO_HOLDOUT_FEATURES,
    "with_holdout": HOLDOUT_FEATURES,
}

MODEL_LABELS = {
    "capacity_spectrum": "capacity + spectrum",
    "support": "support consistency",
    "temporal": "temporal statistics",
    "no_holdout": "all observable, no held-out",
    "with_holdout": "all observable + held-out",
}

SELECTOR_ORDER = [
    "fixed_rank3",
    "train_fixed_rank",
    "energy95",
    "support_min",
    "heldout_min",
    "lofo_ridge_no_holdout",
    "lofo_ridge_with_holdout",
]

SELECTOR_LABELS = {
    "fixed_rank3": "fixed rank 3",
    "train_fixed_rank": "train-family fixed rank",
    "energy95": "95% singular energy",
    "support_min": "support residual minimum",
    "heldout_min": "held-out residual minimum",
    "lofo_ridge_no_holdout": "LOFO ridge, no held-out",
    "lofo_ridge_with_holdout": "LOFO ridge + held-out",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_checksum_manifest(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "family_selector_checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def phase_for_dynamics(fraction: float, dynamics: str) -> float:
    if dynamics == "smooth" or dynamics == "transient":
        return 2.0 * np.pi * fraction
    if dynamics == "chirp":
        return 2.0 * np.pi * (0.25 * fraction + 1.25 * fraction**2)
    raise ValueError(f"unknown dynamics: {dynamics}")


def make_family_sequence(
    family: str,
    dynamics: str,
    n: int = 24,
    nz: int = 10,
    nt: int = 18,
) -> np.ndarray:
    """Generate four deliberately different synthetic 3D+time morphologies."""
    allowed = {"blobs_sheet", "vortex_ring", "expanding_shell", "jet_filaments"}
    if family not in allowed:
        raise ValueError(f"unknown phantom family: {family}")
    if dynamics not in {"smooth", "chirp", "transient"}:
        raise ValueError(f"unknown dynamics: {dynamics}")

    xx, yy, zz = make_grid_3d(n, nz)
    frames = []
    for frame_index in range(nt):
        fraction = frame_index / max(nt - 1, 1)
        phase = phase_for_dynamics(fraction, dynamics)
        event_weight = np.exp(-0.5 * ((fraction - 0.58) / 0.055) ** 2)

        if family == "blobs_sheet":
            c1 = (-0.30 + 0.16 * np.sin(phase), 0.06 * np.cos(phase), -0.18 + 0.10 * np.sin(phase + 0.5))
            c2 = (0.27 + 0.07 * np.sin(phase + 1.1), -0.22 + 0.13 * np.cos(phase), 0.20 * np.cos(phase + 0.2))
            width_scale = 1.0 + 0.10 * np.sin(2.0 * phase)
            volume = (
                gaussian3d(xx, yy, zz, c1, (0.20 * width_scale, 0.30, 0.42), 1.0)
                + gaussian3d(xx, yy, zz, c2, (0.17, 0.22 * width_scale, 0.34), 0.72)
            )
            sheet_center = 0.12 * np.sin(3.5 * yy + 2.0 * zz + phase)
            volume += 0.22 * np.exp(-((xx - sheet_center) ** 2) / (2 * 0.052**2)) * np.exp(-(yy**2) / 0.82) * np.exp(-(zz**2) / 1.25)

        elif family == "vortex_ring":
            center_x = 0.16 * np.sin(phase)
            center_y = 0.12 * np.cos(phase)
            radial = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
            ring_radius = 0.43 + 0.055 * np.sin(1.5 * phase)
            ring = np.exp(-((radial - ring_radius) ** 2) / (2 * 0.075**2) - ((zz - 0.10 * np.sin(phase)) ** 2) / (2 * 0.19**2))
            azimuth = np.arctan2(yy - center_y, xx - center_x)
            modulation = 0.76 + 0.24 * np.cos(2.0 * azimuth - 1.7 * phase)
            core = gaussian3d(
                xx,
                yy,
                zz,
                (center_x + ring_radius * np.cos(phase), center_y + ring_radius * np.sin(phase), 0.0),
                (0.12, 0.12, 0.22),
                0.48,
            )
            volume = ring * modulation + core

        elif family == "expanding_shell":
            center = np.array([0.10 * np.sin(phase), -0.08 * np.cos(0.7 * phase), 0.05 * np.sin(1.3 * phase)])
            radius = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
            front_radius = 0.24 + 0.38 * fraction + 0.025 * np.sin(phase)
            shell = np.exp(-((radius - front_radius) ** 2) / (2 * 0.042**2))
            wake = np.exp(-((radius - 0.72 * front_radius) ** 2) / (2 * 0.09**2))
            anisotropy = 0.68 + 0.32 * np.maximum(np.cos(np.arctan2(yy, xx) - phase), 0.0)
            volume = shell * anisotropy + 0.26 * wake

        else:
            center_x = 0.17 * np.sin(2.4 * zz + phase) + 0.035 * np.sin(7.0 * zz - 0.7 * phase)
            center_y = -0.11 * np.cos(2.0 * zz - 0.8 * phase)
            radial2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
            width = 0.085 + 0.055 * (zz + 1.0) / 2.0
            axial = np.exp(-((zz + 0.10 - 0.18 * np.sin(phase)) ** 2) / (2 * 0.62**2))
            jet = np.exp(-radial2 / (2 * width**2)) * axial
            eddy1 = gaussian3d(xx, yy, zz, (0.26 * np.sin(phase), 0.24 * np.cos(phase), 0.35), (0.16, 0.13, 0.19), 0.58)
            eddy2 = gaussian3d(xx, yy, zz, (-0.30 * np.cos(0.8 * phase), 0.20 * np.sin(phase), -0.42), (0.12, 0.16, 0.16), 0.42)
            volume = jet + eddy1 + eddy2

        if dynamics == "transient":
            if family == "expanding_shell":
                transient = gaussian3d(xx, yy, zz, (0.24, -0.18, 0.08), (0.10, 0.11, 0.14), 1.0)
            elif family == "vortex_ring":
                transient = gaussian3d(xx, yy, zz, (-0.30, 0.22, -0.05), (0.11, 0.13, 0.15), 0.78)
            else:
                transient = gaussian3d(xx, yy, zz, (0.04, 0.34, -0.08), (0.11, 0.13, 0.18), 0.78)
            volume = volume + event_weight * transient

        frames.append(np.maximum(volume, 0.0))

    sequence = np.stack(frames, axis=0)
    sequence /= float(np.max(sequence)) + 1e-12
    return sequence


def add_noise(array: np.ndarray, rng: np.random.Generator, noise_level: float) -> np.ndarray:
    output = np.empty_like(array)
    for frame_index, frame in enumerate(array):
        scale = noise_level * float(np.std(frame))
        output[frame_index] = frame + scale * rng.normal(size=frame.shape)
    return output


def simulate_observation(
    reference: np.ndarray,
    support_angles: np.ndarray,
    heldout_angles: np.ndarray,
    seed: int,
    noise_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_clean = deflection_sequence(reference, support_angles)
    heldout_clean = deflection_sequence(reference, heldout_angles)
    support_observed = add_noise(support_clean, np.random.default_rng(seed), noise_level)
    heldout_observed = add_noise(heldout_clean, np.random.default_rng(seed + 1_000_003), noise_level)
    frames = [
        baseline_stack(support_observed[index], support_angles, reference.shape[-1])
        for index in range(reference.shape[0])
    ]
    return np.stack(frames, axis=0), support_observed, heldout_observed, heldout_clean


def low_rank_candidates(sequence: np.ndarray, ranks: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    nt = sequence.shape[0]
    matrix = sequence.reshape(nt, -1)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    output = {}
    for rank in ranks:
        effective = min(rank, len(singular_values))
        reconstruction = (u[:, :effective] * singular_values[:effective]) @ vt[:effective] + mean
        output[rank] = reconstruction.reshape(sequence.shape)
    return output, singular_values


def affine_projection_residual(prediction: np.ndarray, observation: np.ndarray) -> float:
    x = prediction.reshape(-1)
    y = observation.reshape(-1)
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    aligned = scale * prediction + offset
    return float(np.linalg.norm(aligned - observation) / (np.linalg.norm(observation - np.mean(observation)) + 1e-12))


def normalized_temporal_norm(sequence: np.ndarray, order: int = 1) -> float:
    difference = np.diff(sequence, n=order, axis=0)
    return float(np.linalg.norm(difference) / (np.linalg.norm(sequence) + 1e-12))


def mass_curvature(sequence: np.ndarray) -> float:
    mass = np.maximum(sequence, 0.0).sum(axis=(1, 2, 3))
    if len(mass) < 3:
        return 0.0
    return float(np.linalg.norm(np.diff(mass, n=2)) / (np.linalg.norm(mass - np.mean(mass)) + 1e-12))


def observation_temporal_noise_proxy(observation: np.ndarray) -> float:
    first = np.diff(observation, axis=0)
    second = np.diff(observation, n=2, axis=0)
    return float(np.linalg.norm(second) / (np.linalg.norm(first) + 1e-12))


def spectral_features(singular_values: np.ndarray, rank: int) -> tuple[float, float, float]:
    energy = singular_values**2
    total = float(np.sum(energy)) + 1e-12
    effective = min(rank, len(singular_values))
    captured = float(np.sum(energy[:effective]) / total)
    tail = float(np.sqrt(np.sum(energy[effective:]) / total))
    if 0 < effective < len(singular_values):
        gap = float(np.log10((singular_values[effective - 1] + 1e-12) / (singular_values[effective] + 1e-12)))
    else:
        gap = 0.0
    return captured, tail, gap


def candidate_row(
    cell_id: str,
    family: str,
    dynamics: str,
    noise_level: float,
    view_count: int,
    seed: int,
    rank: int,
    max_rank: int,
    candidate: np.ndarray,
    framewise: np.ndarray,
    reference: np.ndarray,
    support_angles: np.ndarray,
    heldout_angles: np.ndarray,
    support_observed: np.ndarray,
    heldout_observed: np.ndarray,
    heldout_clean: np.ndarray,
    singular_values: np.ndarray,
) -> dict[str, object]:
    field_metrics = metric_suite(candidate, reference, heldout_angles, heldout_clean)
    support_prediction = deflection_sequence(candidate, support_angles)
    heldout_prediction = deflection_sequence(candidate, heldout_angles)
    support_residual = affine_projection_residual(support_prediction, support_observed)
    heldout_residual = affine_projection_residual(heldout_prediction, heldout_observed)
    spectral_energy, spectral_tail, spectral_gap = spectral_features(singular_values, rank)
    baseline_gradient = normalized_temporal_norm(framewise, order=1)
    baseline_curvature = normalized_temporal_norm(framewise, order=2)
    gradient_ratio = normalized_temporal_norm(candidate, order=1) / (baseline_gradient + 1e-12)
    curvature_ratio = normalized_temporal_norm(candidate, order=2) / (baseline_curvature + 1e-12)
    rank_fraction = min(rank, max_rank) / max(max_rank, 1)
    noise_proxy = observation_temporal_noise_proxy(support_observed)
    return {
        "cell_id": cell_id,
        "family": family,
        "dynamics": dynamics,
        "noise_level": noise_level,
        "view_count": view_count,
        "seed": seed,
        "rank": rank,
        "is_full_rank": int(rank >= max_rank),
        "field_rel_l2": float(field_metrics["rel_l2_global"]),
        "paper_l2_squared": float(field_metrics["paper_l2_squared"]),
        "temporal_gradient_rel_l2": float(field_metrics["temporal_gradient_rel_l2"]),
        "mass_trace_rmse": float(field_metrics["mass_trace_rmse"]),
        "clean_heldout_reprojection_rel_l2": float(field_metrics["heldout_reprojection_rel_l2"]),
        "support_residual": support_residual,
        "heldout_residual": heldout_residual,
        "rank_fraction": rank_fraction,
        "rank_fraction_squared": rank_fraction**2,
        "log_rank": float(np.log1p(rank)),
        "spectral_energy": spectral_energy,
        "spectral_tail": spectral_tail,
        "spectral_gap": spectral_gap,
        "log_support_residual": float(np.log(support_residual + 1e-12)),
        "temporal_gradient_ratio": gradient_ratio,
        "temporal_curvature_ratio": curvature_ratio,
        "mass_curvature": mass_curvature(candidate),
        "negative_fraction": float(np.mean(candidate < 0.0)),
        "observation_temporal_noise_proxy": noise_proxy,
        "view_fraction": view_count / 9.0,
        "rank_x_noise_proxy": rank_fraction * noise_proxy,
        "rank_x_spectral_tail": rank_fraction * spectral_tail,
        "log_heldout_residual": float(np.log(heldout_residual + 1e-12)),
        "log_heldout_support_ratio": float(np.log((heldout_residual + 1e-12) / (support_residual + 1e-12))),
    }


def annotate_oracle(rows: list[dict[str, object]]) -> None:
    by_cell: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row["cell_id"])].append(row)
    for candidates in by_cell.values():
        full = max(candidates, key=lambda row: int(row["rank"]))
        full_error = float(full["field_rel_l2"])
        oracle = min(candidates, key=lambda row: float(row["field_rel_l2"]))
        oracle_error = float(oracle["field_rel_l2"])
        for row in candidates:
            field_error = float(row["field_rel_l2"])
            row["full_rank_field_rel_l2"] = full_error
            row["field_error_ratio_to_full"] = field_error / (full_error + 1e-12)
            row["target_log_error_ratio"] = float(np.log((field_error + 1e-12) / (full_error + 1e-12)))
            row["oracle_rank"] = int(oracle["rank"])
            row["oracle_field_rel_l2"] = oracle_error
            row["oracle_regret_pct"] = 100.0 * (field_error - oracle_error) / (oracle_error + 1e-12)


def matrix(rows: list[dict[str, object]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=float)


def fit_ridge(
    rows: list[dict[str, object]],
    ridge_lambda: float,
    features: list[str],
) -> dict[str, object]:
    x = matrix(rows, features)
    y = np.asarray([float(row["target_log_error_ratio"]) for row in rows], dtype=float)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std[x_std < 1e-10] = 1.0
    x_scaled = (x - x_mean) / x_std
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    penalty = ridge_lambda * np.eye(x_scaled.shape[1])
    coefficients = np.linalg.solve(x_scaled.T @ x_scaled + penalty, x_scaled.T @ y_centered)
    return {
        "features": features,
        "lambda": ridge_lambda,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "coefficients": coefficients,
    }


def ridge_predict(model: dict[str, object], rows: list[dict[str, object]]) -> np.ndarray:
    features = list(model["features"])
    x = matrix(rows, features)
    x_scaled = (x - np.asarray(model["x_mean"])) / np.asarray(model["x_std"])
    return float(model["y_mean"]) + x_scaled @ np.asarray(model["coefficients"])


def group_cells(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        output[str(row["cell_id"])].append(row)
    return output


def predicted_selector_regret(
    model: dict[str, object],
    rows: list[dict[str, object]],
) -> float:
    predictions = ridge_predict(model, rows)
    indexed = list(zip(rows, predictions))
    by_cell: dict[str, list[tuple[dict[str, object], float]]] = defaultdict(list)
    for row, prediction in indexed:
        by_cell[str(row["cell_id"])].append((row, float(prediction)))
    regrets = []
    for candidates in by_cell.values():
        selected = min(candidates, key=lambda item: item[1])[0]
        regrets.append(float(selected["oracle_regret_pct"]))
    return float(np.mean(regrets))


def tune_lambda(
    training_rows: list[dict[str, object]],
    families: list[str],
    lambdas: list[float],
    features: list[str],
) -> tuple[float, dict[str, float]]:
    scores = {}
    for ridge_lambda in lambdas:
        fold_scores = []
        for validation_family in families:
            inner_train = [row for row in training_rows if row["family"] != validation_family]
            inner_validation = [row for row in training_rows if row["family"] == validation_family]
            model = fit_ridge(inner_train, ridge_lambda, features)
            fold_scores.append(predicted_selector_regret(model, inner_validation))
        scores[str(ridge_lambda)] = float(np.mean(fold_scores))
    best_lambda = min(lambdas, key=lambda value: (scores[str(value)], value))
    return best_lambda, scores


def train_lofo_models(
    rows: list[dict[str, object]],
    families: list[str],
    lambdas: list[float],
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, object]]:
    models: dict[str, dict[str, dict[str, object]]] = {
        mode: {} for mode in MODEL_FEATURE_SETS
    }
    audit: dict[str, object] = {}
    for heldout_family in families:
        training_rows = [row for row in rows if row["family"] != heldout_family]
        training_families = [family for family in families if family != heldout_family]
        fold_audit = {
            "heldout_family": heldout_family,
            "training_families": training_families,
            "training_candidate_rows": len(training_rows),
        }
        for mode, features in MODEL_FEATURE_SETS.items():
            best_lambda, inner_scores = tune_lambda(training_rows, training_families, lambdas, features)
            model = fit_ridge(training_rows, best_lambda, features)
            models[mode][heldout_family] = model
            fold_audit[mode] = {
                "selected_lambda": best_lambda,
                "inner_lofo_mean_regret_by_lambda": inner_scores,
                "feature_count": len(features),
            }
        audit[heldout_family] = fold_audit
    return models, audit


def selection_record(
    selector: str,
    selected: dict[str, object],
    oracle: dict[str, object],
    rank3: dict[str, object],
    full: dict[str, object],
) -> dict[str, object]:
    selected_error = float(selected["field_rel_l2"])
    oracle_error = float(oracle["field_rel_l2"])
    regret = 100.0 * (selected_error - oracle_error) / (oracle_error + 1e-12)
    return {
        "cell_id": selected["cell_id"],
        "family": selected["family"],
        "dynamics": selected["dynamics"],
        "noise_level": selected["noise_level"],
        "view_count": selected["view_count"],
        "seed": selected["seed"],
        "selector": selector,
        "selected_rank": int(selected["rank"]),
        "oracle_rank": int(oracle["rank"]),
        "selected_field_rel_l2": selected_error,
        "oracle_field_rel_l2": oracle_error,
        "rank3_field_rel_l2": float(rank3["field_rel_l2"]),
        "full_rank_field_rel_l2": float(full["field_rel_l2"]),
        "regret_pct": regret,
        "field_improvement_vs_rank3_pct": 100.0 * (float(rank3["field_rel_l2"]) - selected_error) / (float(rank3["field_rel_l2"]) + 1e-12),
        "field_improvement_vs_full_pct": 100.0 * (float(full["field_rel_l2"]) - selected_error) / (float(full["field_rel_l2"]) + 1e-12),
        "exact_oracle_rank": int(int(selected["rank"]) == int(oracle["rank"])),
        "within_one_pct_oracle": int(regret <= 1.0),
        "requires_heldout_view": int(selector in {"heldout_min", "lofo_ridge_with_holdout", "with_holdout"}),
    }


def select_candidates(
    rows: list[dict[str, object]],
    families: list[str],
    energy_threshold: float,
    lofo_models: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    by_cell = group_cells(rows)
    training_fixed_rank = {}
    for heldout_family in families:
        training = [row for row in rows if row["family"] != heldout_family]
        by_rank: dict[int, list[float]] = defaultdict(list)
        for row in training:
            by_rank[int(row["rank"])].append(float(row["field_error_ratio_to_full"]))
        training_fixed_rank[heldout_family] = min(by_rank, key=lambda rank: float(np.mean(by_rank[rank])))

    output = []
    for cell_id, candidates in sorted(by_cell.items()):
        candidates = sorted(candidates, key=lambda row: int(row["rank"]))
        family = str(candidates[0]["family"])
        oracle = min(candidates, key=lambda row: float(row["field_rel_l2"]))
        rank3 = next(row for row in candidates if int(row["rank"]) == 3)
        full = max(candidates, key=lambda row: int(row["rank"]))

        energy_candidates = [row for row in candidates if float(row["spectral_energy"]) >= energy_threshold]
        energy_choice = min(energy_candidates, key=lambda row: int(row["rank"])) if energy_candidates else full

        no_holdout_prediction = ridge_predict(lofo_models["no_holdout"][family], candidates)
        with_holdout_prediction = ridge_predict(lofo_models["with_holdout"][family], candidates)
        choices = {
            "fixed_rank3": rank3,
            "train_fixed_rank": next(row for row in candidates if int(row["rank"]) == training_fixed_rank[family]),
            "energy95": energy_choice,
            "support_min": min(candidates, key=lambda row: float(row["support_residual"])),
            "heldout_min": min(candidates, key=lambda row: float(row["heldout_residual"])),
            "lofo_ridge_no_holdout": candidates[int(np.argmin(no_holdout_prediction))],
            "lofo_ridge_with_holdout": candidates[int(np.argmin(with_holdout_prediction))],
        }
        for selector in SELECTOR_ORDER:
            selected = choices[selector]
            output.append(selection_record(selector, selected, oracle, rank3, full))
    return output


def select_model_ablation(
    rows: list[dict[str, object]],
    lofo_models: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    output = []
    for candidates in group_cells(rows).values():
        candidates = sorted(candidates, key=lambda row: int(row["rank"]))
        family = str(candidates[0]["family"])
        oracle = min(candidates, key=lambda row: float(row["field_rel_l2"]))
        rank3 = next(row for row in candidates if int(row["rank"]) == 3)
        full = max(candidates, key=lambda row: int(row["rank"]))
        for mode in MODEL_FEATURE_SETS:
            predictions = ridge_predict(lofo_models[mode][family], candidates)
            selected = candidates[int(np.argmin(predictions))]
            output.append(selection_record(mode, selected, oracle, rank3, full))
    return sorted(output, key=lambda row: (str(row["cell_id"]), str(row["selector"])))


def summarize_selector_rows(
    rows: list[dict[str, object]],
    selector_order: list[str] | None = None,
) -> list[dict[str, object]]:
    output = []
    order = selector_order or SELECTOR_ORDER
    for selector in order:
        subset = [row for row in rows if row["selector"] == selector]
        regrets = np.asarray([float(row["regret_pct"]) for row in subset])
        seed_means = []
        for seed in sorted({int(row["seed"]) for row in subset}):
            seed_means.append(float(np.mean([float(row["regret_pct"]) for row in subset if int(row["seed"]) == seed])))
        seed_std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
        seed_ci = t_critical_95(len(seed_means)) * seed_std / math.sqrt(len(seed_means)) if len(seed_means) > 1 else 0.0
        output.append(
            {
                "selector": selector,
                "cell_count": len(subset),
                "mean_regret_pct": float(np.mean(regrets)),
                "seed_clustered_ci95_pct": float(seed_ci),
                "median_regret_pct": float(np.median(regrets)),
                "p95_regret_pct": float(np.percentile(regrets, 95)),
                "max_regret_pct": float(np.max(regrets)),
                "exact_oracle_rank_rate": float(np.mean([int(row["exact_oracle_rank"]) for row in subset])),
                "within_one_pct_oracle_rate": float(np.mean([int(row["within_one_pct_oracle"]) for row in subset])),
                "mean_field_improvement_vs_rank3_pct": float(np.mean([float(row["field_improvement_vs_rank3_pct"]) for row in subset])),
                "field_better_than_rank3_rate": float(np.mean([float(row["field_improvement_vs_rank3_pct"]) > 0.0 for row in subset])),
                "mean_field_improvement_vs_full_pct": float(np.mean([float(row["field_improvement_vs_full_pct"]) for row in subset])),
            }
        )
    return output


def family_summary(rows: list[dict[str, object]], families: list[str]) -> list[dict[str, object]]:
    output = []
    for selector in SELECTOR_ORDER:
        for family in families:
            subset = [row for row in rows if row["selector"] == selector and row["family"] == family]
            output.append(
                {
                    "selector": selector,
                    "family": family,
                    "cell_count": len(subset),
                    "mean_regret_pct": float(np.mean([float(row["regret_pct"]) for row in subset])),
                    "p95_regret_pct": float(np.percentile([float(row["regret_pct"]) for row in subset], 95)),
                    "within_one_pct_oracle_rate": float(np.mean([int(row["within_one_pct_oracle"]) for row in subset])),
                    "mean_field_improvement_vs_rank3_pct": float(np.mean([float(row["field_improvement_vs_rank3_pct"]) for row in subset])),
                }
            )
    return output


def correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def within_cell_surrogate_audit(rows: list[dict[str, object]], observable: str) -> dict[str, float]:
    cell_correlations = []
    correct_pairs = 0
    comparable_pairs = 0
    for candidates in group_cells(rows).values():
        field = [float(row["field_rel_l2"]) for row in candidates]
        proxy = [float(row[observable]) for row in candidates]
        cell_correlations.append(correlation(proxy, field))
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                field_delta = field[left] - field[right]
                proxy_delta = proxy[left] - proxy[right]
                if abs(field_delta) < 1e-12 or abs(proxy_delta) < 1e-12:
                    continue
                comparable_pairs += 1
                correct_pairs += int(field_delta * proxy_delta > 0.0)
    return {
        "mean_within_cell_pearson": float(np.mean(cell_correlations)),
        "median_within_cell_pearson": float(np.median(cell_correlations)),
        "pairwise_rank_order_accuracy": correct_pairs / max(comparable_pairs, 1),
        "comparable_rank_pairs": comparable_pairs,
    }


def plot_oracle_map(rows: list[dict[str, object]], families: list[str], noises: list[float], views: list[int], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), constrained_layout=True)
    for ax, family in zip(axes.flat, families):
        matrix_values = np.zeros((len(views), len(noises)))
        for view_index, view_count in enumerate(views):
            for noise_index, noise_level in enumerate(noises):
                values = [
                    int(row["oracle_rank"])
                    for row in rows
                    if row["family"] == family
                    and int(row["view_count"]) == view_count
                    and float(row["noise_level"]) == noise_level
                ]
                matrix_values[view_index, noise_index] = float(np.mean(values))
        image = ax.imshow(matrix_values, cmap="viridis", vmin=1, vmax=max(int(row["rank"]) for row in rows), aspect="auto")
        ax.set_xticks(np.arange(len(noises)), [f"{value:.2f}" for value in noises])
        ax.set_yticks(np.arange(len(views)), [str(value) for value in views])
        ax.set_xlabel("noise multiplier")
        ax.set_ylabel("support views")
        ax.set_title(family.replace("_", " "))
        for row_index in range(matrix_values.shape[0]):
            for column_index in range(matrix_values.shape[1]):
                ax.text(column_index, row_index, f"{matrix_values[row_index, column_index]:.1f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="mean oracle rank")
    fig.suptitle("M3B oracle rank changes with morphology, noise, and view count", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selector_regret(summary_rows: list[dict[str, object]], path: Path) -> None:
    labels = [SELECTOR_LABELS[str(row["selector"])] for row in summary_rows]
    mean = np.asarray([float(row["mean_regret_pct"]) for row in summary_rows])
    ci = np.asarray([float(row["seed_clustered_ci95_pct"]) for row in summary_rows])
    p95 = np.asarray([float(row["p95_regret_pct"]) for row in summary_rows])
    within = np.asarray([100.0 * float(row["within_one_pct_oracle_rate"]) for row in summary_rows])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.3), constrained_layout=True)
    axes[0].bar(x, mean, yerr=ci, color="#2c786c", capsize=3, label="mean +/- seed-clustered 95% CI")
    axes[0].scatter(x, p95, marker="D", color="#a74e43", zorder=3, label="cell-wise p95")
    axes[0].set_ylabel("field-L2 regret to oracle (%)")
    axes[0].set_title("Selector regret under LOFO morphology transfer")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, within, color="#4f77a2")
    axes[1].set_ylabel("cells within 1% of oracle (%)")
    axes[1].set_title("Near-oracle operating-cell coverage")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.grid(True, axis="y", alpha=0.24)
    fig.suptitle("M3B observable temporal-rank selectors", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_family_generalization(
    family_rows: list[dict[str, object]],
    families: list[str],
    path: Path,
) -> None:
    matrix_values = np.zeros((len(SELECTOR_ORDER), len(families)))
    for selector_index, selector in enumerate(SELECTOR_ORDER):
        for family_index, family in enumerate(families):
            row = next(item for item in family_rows if item["selector"] == selector and item["family"] == family)
            matrix_values[selector_index, family_index] = float(row["mean_regret_pct"])
    fig, ax = plt.subplots(figsize=(10.8, 6.6), constrained_layout=True)
    scale = max(float(np.percentile(matrix_values, 95)), 1.0)
    image = ax.imshow(matrix_values, cmap="YlOrRd", vmin=0.0, vmax=scale, aspect="auto")
    ax.set_xticks(np.arange(len(families)), [family.replace("_", "\n") for family in families])
    ax.set_yticks(np.arange(len(SELECTOR_ORDER)), [SELECTOR_LABELS[name] for name in SELECTOR_ORDER])
    ax.set_title("Mean oracle regret on each held-out phantom family")
    for row_index in range(matrix_values.shape[0]):
        for column_index in range(matrix_values.shape[1]):
            color = "white" if matrix_values[row_index, column_index] > 0.55 * scale else "#263238"
            ax.text(column_index, row_index, f"{matrix_values[row_index, column_index]:.2f}", ha="center", va="center", color=color, fontsize=8.5)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="mean regret (%)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selection_distribution(selected_rows: list[dict[str, object]], ranks: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.0), constrained_layout=True)
    bottoms = np.zeros(len(SELECTOR_ORDER))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(ranks)))
    for rank, color in zip(ranks, colors):
        counts = np.asarray([
            sum(row["selector"] == selector and int(row["selected_rank"]) == rank for row in selected_rows)
            for selector in SELECTOR_ORDER
        ], dtype=float)
        fractions = 100.0 * counts / np.asarray([
            sum(row["selector"] == selector for row in selected_rows) for selector in SELECTOR_ORDER
        ])
        ax.bar(np.arange(len(SELECTOR_ORDER)), fractions, bottom=bottoms, label=f"rank {rank}", color=color)
        bottoms += fractions
    ax.set_xticks(np.arange(len(SELECTOR_ORDER)), [SELECTOR_LABELS[name] for name in SELECTOR_ORDER], rotation=24, ha="right")
    ax.set_ylabel("selected cells (%)")
    ax.set_title("Selector capacity choices reveal collapse and adaptation")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.22)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_observable_alignment(rows: list[dict[str, object]], path: Path) -> dict[str, object]:
    field_ratio = [float(row["field_error_ratio_to_full"]) for row in rows]
    heldout = [float(row["heldout_residual"]) for row in rows]
    support = [float(row["support_residual"]) for row in rows]
    ranks = sorted({int(row["rank"]) for row in rows})
    correlations = {
        "heldout_residual_vs_field_error_ratio_pearson": correlation(heldout, field_ratio),
        "support_residual_vs_field_error_ratio_pearson": correlation(support, field_ratio),
    }
    correlations.update(
        {
            "heldout_within_cell": within_cell_surrogate_audit(rows, "heldout_residual"),
            "support_within_cell": within_cell_surrogate_audit(rows, "support_residual"),
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    colors = {rank: color for rank, color in zip(ranks, plt.cm.viridis(np.linspace(0.08, 0.92, len(ranks))))}
    stride = max(len(rows) // 1800, 1)
    sampled = rows[::stride]
    for rank in ranks:
        subset = [row for row in sampled if int(row["rank"]) == rank]
        axes[0].scatter(
            [float(row["heldout_residual"]) for row in subset],
            [float(row["field_error_ratio_to_full"]) for row in subset],
            s=12,
            alpha=0.30,
            color=colors[rank],
            label=f"rank {rank}",
        )
        axes[1].scatter(
            [float(row["support_residual"]) for row in subset],
            [float(row["field_error_ratio_to_full"]) for row in subset],
            s=12,
            alpha=0.30,
            color=colors[rank],
        )
    axes[0].set_title(f"Held-out residual, r={correlations['heldout_residual_vs_field_error_ratio_pearson']:.3f}")
    axes[1].set_title(f"Support residual, r={correlations['support_residual_vs_field_error_ratio_pearson']:.3f}")
    for ax in axes:
        ax.set_xlabel("observable projection residual")
        ax.set_ylabel("field error / full-rank field error")
        ax.grid(True, alpha=0.22)
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Observable projection residuals are imperfect rank surrogates", fontsize=13.5)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return correlations


def plot_feature_ablation(summary_rows: list[dict[str, object]], path: Path) -> None:
    labels = [MODEL_LABELS[str(row["selector"])] for row in summary_rows]
    means = np.asarray([float(row["mean_regret_pct"]) for row in summary_rows])
    p95 = np.asarray([float(row["p95_regret_pct"]) for row in summary_rows])
    within = np.asarray([100.0 * float(row["within_one_pct_oracle_rate"]) for row in summary_rows])
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), constrained_layout=True)
    axes[0].bar(x - width / 2, means, width, color="#2c786c", label="mean regret")
    axes[0].bar(x + width / 2, p95, width, color="#a74e43", label="p95 regret")
    axes[0].set_ylabel("field-L2 regret to oracle (%)")
    axes[0].set_title("Which observable feature groups transfer?")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, within, color="#4f77a2")
    axes[1].set_ylabel("cells within 1% of oracle (%)")
    axes[1].set_title("Near-oracle coverage")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.grid(True, axis="y", alpha=0.24)
    fig.suptitle("M3B LOFO feature ablation: no test-family truth enters the selector", fontsize=13.5)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def serializable_model(model: dict[str, object]) -> dict[str, object]:
    return {
        "features": list(model["features"]),
        "lambda": float(model["lambda"]),
        "x_mean": np.asarray(model["x_mean"]).tolist(),
        "x_std": np.asarray(model["x_std"]).tolist(),
        "y_mean": float(model["y_mean"]),
        "coefficients": np.asarray(model["coefficients"]).tolist(),
    }


def build_report(
    config: dict[str, object],
    raw_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    ablation_rows: list[dict[str, object]],
    ablation_summary: list[dict[str, object]],
    models: dict[str, dict[str, dict[str, object]]],
    model_audit: dict[str, object],
    observable_correlations: dict[str, object],
) -> dict[str, object]:
    grid = config["grid"]
    best_observable = min(summary_rows, key=lambda row: float(row["mean_regret_pct"]))
    fixed = next(row for row in summary_rows if row["selector"] == "fixed_rank3")
    no_holdout = next(row for row in summary_rows if row["selector"] == "lofo_ridge_no_holdout")
    with_holdout = next(row for row in summary_rows if row["selector"] == "lofo_ridge_with_holdout")
    best_ablation = min(ablation_summary, key=lambda row: float(row["mean_regret_pct"]))
    cells = group_cells(raw_rows)
    oracle_counts = Counter(int(candidates[0]["oracle_rank"]) for candidates in cells.values())
    family_oracle_counts = {}
    for family in grid["families"]:
        cells = group_cells([row for row in raw_rows if row["family"] == family])
        family_oracle_counts[family] = dict(Counter(int(rows[0]["oracle_rank"]) for rows in cells.values()))
    return {
        "experiment": config["experiment_name"],
        "design": {
            "families": grid["families"],
            "dynamics": grid["dynamics"],
            "noise_levels": grid["noise_levels"],
            "view_counts": grid["view_counts"],
            "ranks": grid["ranks"],
            "seed_count": grid["seed_count"],
            "environment_cells_without_seed": len(grid["families"]) * len(grid["dynamics"]) * len(grid["noise_levels"]) * len(grid["view_counts"]),
            "observation_cells": len(group_cells(raw_rows)),
            "candidate_rows": len(raw_rows),
            "selector_rows": len(selected_rows),
            "evaluation_protocol": "nested lambda tuning inside leave-one-phantom-family-out outer folds",
        },
        "rebuild_modes": ["full_simulation", "resume_missing_cells", "reuse_raw_analysis"],
        "runtime_scope": "Wall time is printed by each command but excluded from the deterministic report because resume and reuse-raw cover different work.",
        "selector_summary": {str(row["selector"]): row for row in summary_rows},
        "family_summary": family_rows,
        "feature_ablation_summary": {str(row["selector"]): row for row in ablation_summary},
        "feature_ablation_rows": len(ablation_rows),
        "oracle_rank_counts": {str(rank): int(count) for rank, count in sorted(oracle_counts.items())},
        "oracle_rank_counts_by_family": {
            family: {str(rank): int(count) for rank, count in sorted(counts.items())}
            for family, counts in family_oracle_counts.items()
        },
        "observable_correlations": observable_correlations,
        "model_selection_audit": model_audit,
        "models": {
            mode: {family: serializable_model(model) for family, model in fold_models.items()}
            for mode, fold_models in models.items()
        },
        "key_findings": {
            "lowest_mean_regret_selector": str(best_observable["selector"]),
            "lowest_mean_regret_pct": float(best_observable["mean_regret_pct"]),
            "fixed_rank3_mean_regret_pct": float(fixed["mean_regret_pct"]),
            "fixed_rank3_p95_regret_pct": float(fixed["p95_regret_pct"]),
            "lofo_no_holdout_mean_regret_pct": float(no_holdout["mean_regret_pct"]),
            "lofo_with_holdout_mean_regret_pct": float(with_holdout["mean_regret_pct"]),
            "heldout_feature_delta_mean_regret_pct": float(no_holdout["mean_regret_pct"]) - float(with_holdout["mean_regret_pct"]),
            "lofo_with_holdout_mean_field_improvement_vs_rank3_pct": float(with_holdout["mean_field_improvement_vs_rank3_pct"]),
            "lofo_with_holdout_better_than_rank3_rate": float(with_holdout["field_better_than_rank3_rate"]),
            "best_feature_set": str(best_ablation["selector"]),
            "best_feature_set_mean_regret_pct": float(best_ablation["mean_regret_pct"]),
        },
        "claims_boundary": config["claims_boundary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--quick", action="store_true", help="run 3 families x 2 dynamics x 2 noises x 2 views x 2 seeds")
    parser.add_argument("--reuse-raw", action="store_true", help="reuse family_selector_raw.csv and rebuild selectors, reports, and figures")
    parser.add_argument("--resume", action="store_true", help="reuse completed cells in family_selector_raw.csv and simulate only missing design cells")
    return parser.parse_args()


def load_config(path: Path, quick: bool) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if quick:
        grid = config["grid"]
        grid["families"] = grid["families"][:3]
        grid["dynamics"] = ["smooth", "transient"]
        grid["noise_levels"] = [0.07, 0.28]
        grid["view_counts"] = [3, 7]
        grid["ranks"] = [2, 3, 5, 18]
        grid["seed_count"] = 2
    return config


def main() -> None:
    args = parse_args()
    if args.reuse_raw and args.resume:
        raise SystemExit("--reuse-raw and --resume are mutually exclusive")
    forbidden_features = {"family", "dynamics", "noise_level", "field_rel_l2", "oracle_rank", "target_log_error_ratio"}
    for mode, features in MODEL_FEATURE_SETS.items():
        leakage = forbidden_features.intersection(features)
        if leakage:
            raise SystemExit(f"selector feature leakage in {mode}: {sorted(leakage)}")
    config = load_config(args.config, args.quick)
    grid = config["grid"]
    families = list(grid["families"])
    dynamics_modes = list(grid["dynamics"])
    noises = [float(value) for value in grid["noise_levels"]]
    views = [int(value) for value in grid["view_counts"]]
    ranks = [int(value) for value in grid["ranks"]]
    seed_count = int(grid["seed_count"])
    seeds = [int(grid["seed_base"]) + index for index in range(seed_count)]
    if len(families) < 2:
        raise SystemExit("at least two phantom families are required for LOFO evaluation")
    if 3 not in ranks:
        raise SystemExit("rank 3 is required as the fixed baseline")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw_path = args.output_dir / "family_selector_raw.csv"
    if args.reuse_raw:
        if not raw_path.exists():
            raise SystemExit(f"cannot reuse missing raw table: {raw_path}")
        with raw_path.open(encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        print(f"reused {len(raw_rows)} candidate rows from {raw_path}")
    else:
        references = {
            (family, dynamics): make_family_sequence(
                family,
                dynamics,
                n=int(grid["n"]),
                nz=int(grid["nz"]),
                nt=int(grid["frame_count"]),
            )
            for family in families
            for dynamics in dynamics_modes
        }
        expected_cell_ids = {
            f"f-{family}_d-{dynamics}_v-{view_count}_n-{noise_level:.2f}_s-{seed}"
            for family in families
            for dynamics in dynamics_modes
            for view_count in views
            for noise_level in noises
            for seed in seeds
        }
        total_cells = len(expected_cell_ids)
        if args.resume and raw_path.exists():
            with raw_path.open(encoding="utf-8") as handle:
                raw_rows = list(csv.DictReader(handle))
            existing_cells = {str(row["cell_id"]) for row in raw_rows}
            print(f"resume found {len(existing_cells)} completed observation cells")
        else:
            raw_rows = []
            existing_cells = set()
        unexpected_cells = existing_cells - expected_cell_ids
        if unexpected_cells:
            preview = sorted(unexpected_cells)[:3]
            raise SystemExit(f"raw table contains cells outside the configured design: {preview}")
        missing_cells = len(expected_cell_ids - existing_cells)
        completed = 0
        for family in families:
            for dynamics in dynamics_modes:
                reference = references[(family, dynamics)]
                for view_count in views:
                    support_angles = np.linspace(0.0, 180.0, view_count, endpoint=False)
                    heldout_angles = np.asarray([90.0 / view_count])
                    for noise_level in noises:
                        for seed in seeds:
                            cell_id = f"f-{family}_d-{dynamics}_v-{view_count}_n-{noise_level:.2f}_s-{seed}"
                            if cell_id in existing_cells:
                                continue
                            framewise, support_observed, heldout_observed, heldout_clean = simulate_observation(
                                reference,
                                support_angles,
                                heldout_angles,
                                seed,
                                noise_level,
                            )
                            candidates, singular_values = low_rank_candidates(framewise, ranks)
                            for rank in ranks:
                                raw_rows.append(
                                    candidate_row(
                                        cell_id,
                                        family,
                                        dynamics,
                                        noise_level,
                                        view_count,
                                        seed,
                                        rank,
                                        max(ranks),
                                        candidates[rank],
                                        framewise,
                                        reference,
                                        support_angles,
                                        heldout_angles,
                                        support_observed,
                                        heldout_observed,
                                        heldout_clean,
                                        singular_values,
                                    )
                                )
                            completed += 1
                            if completed % max(missing_cells // 16, 1) == 0 or completed == missing_cells:
                                print(f"completed missing observation cells: {completed}/{missing_cells}", flush=True)

    annotate_oracle(raw_rows)
    lambdas = [float(value) for value in config["selectors"]["ridge_lambdas"]]
    models, model_audit = train_lofo_models(raw_rows, families, lambdas)
    selected_rows = select_candidates(
        raw_rows,
        families,
        float(config["selectors"]["energy_threshold"]),
        models,
    )
    summary_rows = summarize_selector_rows(selected_rows)
    family_rows = family_summary(selected_rows, families)
    ablation_rows = select_model_ablation(raw_rows, models)
    ablation_summary = summarize_selector_rows(ablation_rows, list(MODEL_FEATURE_SETS))
    observable_correlations = plot_observable_alignment(
        raw_rows,
        args.output_dir / "m3b_observable_alignment.png",
    )
    plot_oracle_map(raw_rows, families, noises, views, args.output_dir / "m3b_family_oracle_map.png")
    plot_selector_regret(summary_rows, args.output_dir / "m3b_selector_regret.png")
    plot_family_generalization(family_rows, families, args.output_dir / "m3b_selector_family_generalization.png")
    plot_selection_distribution(selected_rows, ranks, args.output_dir / "m3b_selector_rank_distribution.png")
    plot_feature_ablation(ablation_summary, args.output_dir / "m3b_selector_feature_ablation.png")

    runtime_seconds = time.perf_counter() - started
    report = build_report(
        config,
        raw_rows,
        selected_rows,
        summary_rows,
        family_rows,
        ablation_rows,
        ablation_summary,
        models,
        model_audit,
        observable_correlations,
    )
    write_csv(args.output_dir / "family_selector_raw.csv", raw_rows)
    write_csv(args.output_dir / "family_selector_selected.csv", selected_rows)
    write_csv(args.output_dir / "family_selector_summary.csv", summary_rows)
    write_csv(args.output_dir / "family_selector_family_summary.csv", family_rows)
    write_csv(args.output_dir / "family_selector_ablation.csv", ablation_rows)
    write_csv(args.output_dir / "family_selector_ablation_summary.csv", ablation_summary)
    with (args.output_dir / "family_selector_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    write_checksum_manifest(
        args.output_dir,
        [
            "family_selector_raw.csv",
            "family_selector_selected.csv",
            "family_selector_summary.csv",
            "family_selector_family_summary.csv",
            "family_selector_ablation.csv",
            "family_selector_ablation_summary.csv",
            "family_selector_report.json",
        ],
    )

    print(f"completed {len(raw_rows)} candidate rows and {len(selected_rows)} selector rows in {runtime_seconds:.1f} s")
    print(json.dumps(report["key_findings"], indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
