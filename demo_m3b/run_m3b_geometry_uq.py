#!/usr/bin/env python3
"""Leave-one-geometry-out rank selection and abstention for the M3B toy.

This experiment follows the cross-morphology selector with a harder question:
does an observable temporal-capacity selector transfer to an unseen camera
layout or a controlled calibration-offset signature? It also evaluates whether
ensemble disagreement can identify high-regret cells before field truth is
available and whether full-rank fallback actually reduces system risk.

The study remains a clean-room straight-ray synthetic benchmark. Geometry
families are controlled angular layouts, not calibrated OERF hardware.
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
DEFAULT_CONFIG = ROOT / "configs" / "geometry_uq.json"
DEFAULT_RESULTS = ROOT / "results" / "geometry_uq"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_m1.run_m1_3d_stack_bost import baseline_stack
from demo_m3b.run_m3b_family_selector import (
    HOLDOUT_FEATURES,
    NO_HOLDOUT_FEATURES,
    add_noise,
    annotate_oracle,
    candidate_row,
    fit_ridge,
    group_cells,
    low_rank_candidates,
    make_family_sequence,
    ridge_predict,
    summarize_selector_rows,
    write_csv,
)
from demo_m3b.run_m3b_six_axis_sweep import deflection_sequence


GEOMETRY_FEATURES = [
    "geometry_max_gap_fraction",
    "geometry_gap_std_fraction",
    "geometry_gap_entropy",
    "geometry_resultant",
    "geometry_coverage_fraction",
]

LOGO_NO_HOLDOUT_FEATURES = NO_HOLDOUT_FEATURES + GEOMETRY_FEATURES
LOGO_WITH_HOLDOUT_FEATURES = HOLDOUT_FEATURES + GEOMETRY_FEATURES

MODEL_FEATURE_SETS = {
    "no_holdout": LOGO_NO_HOLDOUT_FEATURES,
    "with_holdout": LOGO_WITH_HOLDOUT_FEATURES,
}

SELECTOR_ORDER = [
    "fixed_rank3",
    "train_fixed_rank",
    "energy95",
    "support_min",
    "heldout_min",
    "logo_ensemble_no_holdout",
    "logo_ensemble_with_holdout",
]

SELECTOR_LABELS = {
    "fixed_rank3": "fixed rank 3",
    "train_fixed_rank": "training-geometry fixed rank",
    "energy95": "95% singular energy",
    "support_min": "support residual minimum",
    "heldout_min": "held-out residual minimum",
    "logo_ensemble_no_holdout": "LOGO ensemble, no held-out",
    "logo_ensemble_with_holdout": "LOGO ensemble + held-out",
}

UQ_SCORE_FIELDS = [
    "prediction_std_percentile",
    "vote_entropy_percentile",
    "inverse_margin_percentile",
    "predicted_risk_percentile",
    "combined_uncertainty",
]

UQ_SCORE_LABELS = {
    "prediction_std_percentile": "prediction std",
    "vote_entropy_percentile": "rank-vote entropy",
    "inverse_margin_percentile": "inverse rank margin",
    "predicted_risk_percentile": "predicted residual risk",
    "combined_uncertainty": "combined uncertainty",
}


def write_checksum_manifest(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "geometry_uq_checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def geometry_angles(geometry: str, view_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return true observation angles and angles used by reconstruction."""
    if view_count < 3:
        raise ValueError("at least three views are required")
    step = 180.0 / view_count
    base = np.linspace(0.0, 180.0, view_count, endpoint=False)
    if geometry == "uniform":
        true = base
        reconstruction = base
    elif geometry == "rotated_uniform":
        true = np.mod(base + 0.37 * step, 180.0)
        reconstruction = true.copy()
    elif geometry == "limited_arc":
        true = np.linspace(20.0, 120.0, view_count)
        reconstruction = true.copy()
    elif geometry == "dual_cluster":
        left_count = (view_count + 1) // 2
        right_count = view_count - left_count
        left = np.linspace(12.0, 38.0, left_count)
        right = np.linspace(102.0, 128.0, right_count) if right_count else np.asarray([], dtype=float)
        true = np.concatenate([left, right])
        reconstruction = true.copy()
    elif geometry == "jittered":
        offsets = 0.22 * step * np.sin(1.7 * (np.arange(view_count) + 1.0))
        true = np.mod(base + offsets, 180.0)
        reconstruction = true.copy()
    elif geometry == "calibration_offset_2deg":
        true = base
        reconstruction = np.mod(base + 2.0, 180.0)
    else:
        raise ValueError(f"unknown geometry: {geometry}")
    order = np.argsort(true)
    true = np.asarray(true, dtype=float)[order]
    reconstruction = np.asarray(reconstruction, dtype=float)[order]
    return true, reconstruction


def circular_gaps(angles: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.mod(np.asarray(angles, dtype=float), 180.0))
    extended = np.concatenate([ordered, [ordered[0] + 180.0]])
    return np.diff(extended)


def largest_gap_midpoint(angles: np.ndarray) -> float:
    ordered = np.sort(np.mod(np.asarray(angles, dtype=float), 180.0))
    gaps = circular_gaps(ordered)
    index = int(np.argmax(gaps))
    return float(np.mod(ordered[index] + 0.5 * gaps[index], 180.0))


def geometry_descriptors(angles: np.ndarray) -> dict[str, float]:
    gaps = circular_gaps(angles)
    probabilities = gaps / (float(np.sum(gaps)) + 1e-12)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12))) / math.log(len(gaps))
    radians = np.deg2rad(2.0 * np.asarray(angles, dtype=float))
    resultant = abs(np.mean(np.exp(1j * radians)))
    max_gap = float(np.max(gaps))
    return {
        "geometry_max_gap_fraction": max_gap / 180.0,
        "geometry_gap_std_fraction": float(np.std(gaps)) / 180.0,
        "geometry_gap_entropy": entropy,
        "geometry_resultant": float(resultant),
        "geometry_coverage_fraction": 1.0 - max_gap / 180.0,
    }


def simulate_geometry_observation(
    reference: np.ndarray,
    true_angles: np.ndarray,
    reconstruction_angles: np.ndarray,
    heldout_angles: np.ndarray,
    seed: int,
    noise_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_clean = deflection_sequence(reference, true_angles)
    heldout_clean = deflection_sequence(reference, heldout_angles)
    support_observed = add_noise(support_clean, np.random.default_rng(seed), noise_level)
    heldout_observed = add_noise(heldout_clean, np.random.default_rng(seed + 1_000_033), noise_level)
    frames = [
        baseline_stack(support_observed[index], reconstruction_angles, reference.shape[-1])
        for index in range(reference.shape[0])
    ]
    return np.stack(frames, axis=0), support_observed, heldout_observed, heldout_clean


def grouped_tune_lambda(
    training_rows: list[dict[str, object]],
    training_geometries: list[str],
    lambdas: list[float],
    features: list[str],
) -> tuple[float, dict[str, float]]:
    scores = {}
    for ridge_lambda in lambdas:
        fold_regrets = []
        for validation_geometry in training_geometries:
            inner_train = [row for row in training_rows if row["geometry"] != validation_geometry]
            inner_validation = [row for row in training_rows if row["geometry"] == validation_geometry]
            model = fit_ridge(inner_train, ridge_lambda, features)
            fold_regrets.append(predicted_selector_regret(model, inner_validation))
        scores[str(ridge_lambda)] = float(np.mean(fold_regrets))
    best_lambda = min(lambdas, key=lambda value: (scores[str(value)], value))
    return best_lambda, scores


def predicted_selector_regret(model: dict[str, object], rows: list[dict[str, object]]) -> float:
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


def train_logo_models(
    rows: list[dict[str, object]],
    geometries: list[str],
    lambdas: list[float],
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, object]]:
    models: dict[str, dict[str, dict[str, object]]] = {
        mode: {} for mode in MODEL_FEATURE_SETS
    }
    audit = {}
    for heldout_geometry in geometries:
        training_geometries = [geometry for geometry in geometries if geometry != heldout_geometry]
        training_rows = [row for row in rows if row["geometry"] != heldout_geometry]
        fold_audit: dict[str, object] = {
            "heldout_geometry": heldout_geometry,
            "training_geometries": training_geometries,
            "training_candidate_rows": len(training_rows),
        }
        for mode, features in MODEL_FEATURE_SETS.items():
            best_lambda, inner_scores = grouped_tune_lambda(
                training_rows,
                training_geometries,
                lambdas,
                features,
            )
            central = fit_ridge(training_rows, best_lambda, features)
            ensemble = []
            ensemble_audit = []
            for omitted_training_geometry in training_geometries:
                ensemble_geometries = [
                    geometry for geometry in training_geometries if geometry != omitted_training_geometry
                ]
                ensemble_rows = [
                    row for row in training_rows if row["geometry"] != omitted_training_geometry
                ]
                ensemble.append(fit_ridge(ensemble_rows, best_lambda, features))
                ensemble_audit.append(
                    {
                        "omitted_training_geometry": omitted_training_geometry,
                        "fit_geometries": ensemble_geometries,
                        "candidate_rows": len(ensemble_rows),
                    }
                )
            models[mode][heldout_geometry] = {
                "central": central,
                "ensemble": ensemble,
            }
            fold_audit[mode] = {
                "features": features,
                "selected_lambda": best_lambda,
                "inner_logo_mean_regret_by_lambda": inner_scores,
                "ensemble_members": ensemble_audit,
            }
        audit[heldout_geometry] = fold_audit
    return models, audit


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    index = 0
    while index < len(array):
        end = index + 1
        while end < len(array) and array[order[end]] == array[order[index]]:
            end += 1
        average = 0.5 * (index + end - 1)
        ranks[order[index:end]] = average
        index = end
    return ranks


def percentile_ranks(values: list[float]) -> np.ndarray:
    ranks = rankdata(values)
    return ranks / max(len(values) - 1, 1)


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    xr = rankdata(x)
    yr = rankdata(y)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def binary_auc(scores: list[float], labels: list[int]) -> float:
    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    positives = int(np.sum(labels_array == 1))
    negatives = int(np.sum(labels_array == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = rankdata(scores_array.tolist()) + 1.0
    positive_rank_sum = float(np.sum(ranks[labels_array == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def vote_entropy(votes: list[int]) -> float:
    counts = np.asarray(list(Counter(votes).values()), dtype=float)
    probabilities = counts / float(np.sum(counts))
    denominator = math.log(max(len(votes), 2))
    return -float(np.sum(probabilities * np.log(probabilities + 1e-12))) / denominator


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
        "geometry": selected["geometry"],
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
        "requires_heldout_view": int("with_holdout" in selector or selector == "heldout_min"),
    }


def select_and_measure_uncertainty(
    rows: list[dict[str, object]],
    geometries: list[str],
    models: dict[str, dict[str, dict[str, object]]],
    energy_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_rows = []
    uncertainty_rows = []
    for heldout_geometry in geometries:
        training = [row for row in rows if row["geometry"] != heldout_geometry]
        by_rank: dict[int, list[float]] = defaultdict(list)
        for row in training:
            by_rank[int(row["rank"])].append(float(row["field_error_ratio_to_full"]))
        train_fixed_rank = min(by_rank, key=lambda rank: float(np.mean(by_rank[rank])))
        test_cells = group_cells([row for row in rows if row["geometry"] == heldout_geometry])
        fold_uq: dict[str, list[dict[str, object]]] = {mode: [] for mode in MODEL_FEATURE_SETS}

        for candidates in test_cells.values():
            candidates = sorted(candidates, key=lambda row: int(row["rank"]))
            oracle = min(candidates, key=lambda row: float(row["field_rel_l2"]))
            rank3 = next(row for row in candidates if int(row["rank"]) == 3)
            full = max(candidates, key=lambda row: int(row["rank"]))
            energy_candidates = [
                row for row in candidates if float(row["spectral_energy"]) >= energy_threshold
            ]
            energy_choice = min(energy_candidates, key=lambda row: int(row["rank"])) if energy_candidates else full
            baseline_choices = {
                "fixed_rank3": rank3,
                "train_fixed_rank": next(
                    row for row in candidates if int(row["rank"]) == train_fixed_rank
                ),
                "energy95": energy_choice,
                "support_min": min(candidates, key=lambda row: float(row["support_residual"])),
                "heldout_min": min(candidates, key=lambda row: float(row["heldout_residual"])),
            }
            for selector, selected in baseline_choices.items():
                selected_rows.append(selection_record(selector, selected, oracle, rank3, full))

            for mode in MODEL_FEATURE_SETS:
                ensemble = models[mode][heldout_geometry]["ensemble"]
                prediction_matrix = np.stack(
                    [ridge_predict(model, candidates) for model in ensemble],
                    axis=0,
                )
                mean_prediction = np.mean(prediction_matrix, axis=0)
                selected_index = int(np.argmin(mean_prediction))
                selected = candidates[selected_index]
                selector = f"logo_ensemble_{mode}"
                selected_rows.append(selection_record(selector, selected, oracle, rank3, full))

                votes = [int(candidates[int(np.argmin(member))]["rank"]) for member in prediction_matrix]
                sorted_prediction = np.sort(mean_prediction)
                margin = float(sorted_prediction[1] - sorted_prediction[0]) if len(sorted_prediction) > 1 else 0.0
                selected_regret = 100.0 * (
                    float(selected["field_rel_l2"]) - float(oracle["field_rel_l2"])
                ) / (float(oracle["field_rel_l2"]) + 1e-12)
                full_regret = 100.0 * (
                    float(full["field_rel_l2"]) - float(oracle["field_rel_l2"])
                ) / (float(oracle["field_rel_l2"]) + 1e-12)
                fold_uq[mode].append(
                    {
                        "cell_id": selected["cell_id"],
                        "geometry": heldout_geometry,
                        "family": selected["family"],
                        "dynamics": selected["dynamics"],
                        "noise_level": selected["noise_level"],
                        "view_count": selected["view_count"],
                        "seed": selected["seed"],
                        "mode": mode,
                        "selected_rank": int(selected["rank"]),
                        "oracle_rank": int(oracle["rank"]),
                        "selected_regret_pct": selected_regret,
                        "full_rank_regret_pct": full_regret,
                        "rank3_regret_pct": 100.0 * (
                            float(rank3["field_rel_l2"]) - float(oracle["field_rel_l2"])
                        ) / (float(oracle["field_rel_l2"]) + 1e-12),
                        "prediction_std": float(np.std(prediction_matrix[:, selected_index], ddof=1)) if len(ensemble) > 1 else 0.0,
                        "vote_entropy": vote_entropy(votes),
                        "rank_vote_count": len(set(votes)),
                        "rank_votes": json.dumps(votes, separators=(",", ":")),
                        "prediction_margin": margin,
                        "inverse_margin": -margin,
                        "predicted_risk": float(mean_prediction[selected_index]),
                    }
                )

        for mode, fold_rows in fold_uq.items():
            for source, target in [
                ("prediction_std", "prediction_std_percentile"),
                ("vote_entropy", "vote_entropy_percentile"),
                ("inverse_margin", "inverse_margin_percentile"),
                ("predicted_risk", "predicted_risk_percentile"),
            ]:
                percentiles = percentile_ranks([float(row[source]) for row in fold_rows])
                for row, percentile in zip(fold_rows, percentiles):
                    row[target] = float(percentile)
            for row in fold_rows:
                row["combined_uncertainty"] = float(
                    np.mean(
                        [
                            float(row["prediction_std_percentile"]),
                            float(row["vote_entropy_percentile"]),
                            float(row["inverse_margin_percentile"]),
                            float(row["predicted_risk_percentile"]),
                        ]
                    )
                )
                uncertainty_rows.append(row)
    return (
        sorted(selected_rows, key=lambda row: (str(row["cell_id"]), str(row["selector"]))),
        sorted(uncertainty_rows, key=lambda row: (str(row["cell_id"]), str(row["mode"]))),
    )


def geometry_summary(
    selected_rows: list[dict[str, object]],
    geometries: list[str],
) -> list[dict[str, object]]:
    output = []
    for selector in SELECTOR_ORDER:
        for geometry in geometries:
            subset = [
                row for row in selected_rows
                if row["selector"] == selector and row["geometry"] == geometry
            ]
            output.append(
                {
                    "selector": selector,
                    "geometry": geometry,
                    "cell_count": len(subset),
                    "mean_regret_pct": float(np.mean([float(row["regret_pct"]) for row in subset])),
                    "p95_regret_pct": float(np.percentile([float(row["regret_pct"]) for row in subset], 95)),
                    "within_one_pct_oracle_rate": float(np.mean([int(row["within_one_pct_oracle"]) for row in subset])),
                    "mean_field_improvement_vs_rank3_pct": float(np.mean([float(row["field_improvement_vs_rank3_pct"]) for row in subset])),
                }
            )
    return output


def uncertainty_audit(
    rows: list[dict[str, object]],
    high_risk_threshold: float,
) -> list[dict[str, object]]:
    output = []
    for mode in MODEL_FEATURE_SETS:
        subset = [row for row in rows if row["mode"] == mode]
        regrets = [float(row["selected_regret_pct"]) for row in subset]
        labels = [int(value > high_risk_threshold) for value in regrets]
        for score in UQ_SCORE_FIELDS:
            values = [float(row[score]) for row in subset]
            output.append(
                {
                    "mode": mode,
                    "score": score,
                    "cell_count": len(subset),
                    "high_risk_threshold_pct": high_risk_threshold,
                    "high_risk_prevalence": float(np.mean(labels)),
                    "spearman_score_vs_regret": spearman(values, regrets),
                    "high_risk_auc": binary_auc(values, labels),
                }
            )
    return output


def risk_coverage_rows(
    rows: list[dict[str, object]],
    coverages: list[float],
) -> list[dict[str, object]]:
    output = []
    for mode in MODEL_FEATURE_SETS:
        subset = [row for row in rows if row["mode"] == mode]
        for score in UQ_SCORE_FIELDS:
            ordered = sorted(subset, key=lambda row: (float(row[score]), str(row["cell_id"])))
            for target_coverage in coverages:
                accepted_count = max(1, int(math.ceil(target_coverage * len(ordered))))
                accepted = ordered[:accepted_count]
                rejected = ordered[accepted_count:]
                selective_regret = float(np.mean([float(row["selected_regret_pct"]) for row in accepted]))
                system_values = [float(row["selected_regret_pct"]) for row in accepted] + [
                    float(row["full_rank_regret_pct"]) for row in rejected
                ]
                output.append(
                    {
                        "mode": mode,
                        "score": score,
                        "target_coverage": target_coverage,
                        "actual_coverage": accepted_count / len(ordered),
                        "accepted_cells": accepted_count,
                        "rejected_cells": len(rejected),
                        "selective_mean_regret_pct": selective_regret,
                        "selective_p95_regret_pct": float(np.percentile([float(row["selected_regret_pct"]) for row in accepted], 95)),
                        "full_fallback_system_mean_regret_pct": float(np.mean(system_values)),
                        "rejected_mean_selector_regret_pct": float(np.mean([float(row["selected_regret_pct"]) for row in rejected])) if rejected else 0.0,
                        "rejected_mean_full_rank_regret_pct": float(np.mean([float(row["full_rank_regret_pct"]) for row in rejected])) if rejected else 0.0,
                    }
                )
    return output


def plot_geometry_layouts(geometries: list[str], view_count: int, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for ax, geometry in zip(axes.flat, geometries):
        true_angles, reconstruction_angles = geometry_angles(geometry, view_count)
        true_radians = np.deg2rad(2.0 * true_angles)
        reconstruction_radians = np.deg2rad(2.0 * reconstruction_angles)
        ax.scatter(true_radians, np.ones_like(true_radians), s=55, color="#196e63", label="true observation")
        if not np.allclose(true_angles, reconstruction_angles):
            ax.scatter(reconstruction_radians, np.full_like(reconstruction_radians, 0.82), s=52, marker="x", color="#a24d41", label="reconstruction geometry")
        heldout = largest_gap_midpoint(true_angles)
        ax.scatter([np.deg2rad(2.0 * heldout)], [1.16], marker="D", s=42, color="#315f8d", label="held-out")
        ax.set_ylim(0.0, 1.28)
        ax.set_yticklabels([])
        ax.set_title(geometry.replace("_", " "), fontsize=10.5)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    calibration_handles, calibration_labels = axes.flat[-1].get_legend_handles_labels()
    for handle, label in zip(calibration_handles, calibration_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"M3B camera-layout families at {view_count} support views", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selector_regret(summary_rows: list[dict[str, object]], path: Path) -> None:
    labels = [SELECTOR_LABELS[str(row["selector"])] for row in summary_rows]
    means = np.asarray([float(row["mean_regret_pct"]) for row in summary_rows])
    p95 = np.asarray([float(row["p95_regret_pct"]) for row in summary_rows])
    within = np.asarray([100.0 * float(row["within_one_pct_oracle_rate"]) for row in summary_rows])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), constrained_layout=True)
    axes[0].bar(x, means, color="#2c786c", label="mean regret")
    axes[0].scatter(x, p95, marker="D", color="#a24d41", label="p95 regret", zorder=3)
    axes[0].set_ylabel("field-L2 regret to oracle (%)")
    axes[0].set_title("Selector transfer to unseen geometry")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, within, color="#4f77a2")
    axes[1].set_ylabel("cells within 1% of oracle (%)")
    axes[1].set_title("Near-oracle geometry coverage")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.grid(True, axis="y", alpha=0.24)
    fig.suptitle("M3B leave-one-geometry-out capacity selection", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_geometry_heatmap(
    rows: list[dict[str, object]],
    geometries: list[str],
    path: Path,
) -> None:
    matrix = np.zeros((len(SELECTOR_ORDER), len(geometries)))
    for selector_index, selector in enumerate(SELECTOR_ORDER):
        for geometry_index, geometry in enumerate(geometries):
            match = next(
                row for row in rows
                if row["selector"] == selector and row["geometry"] == geometry
            )
            matrix[selector_index, geometry_index] = float(match["mean_regret_pct"])
    fig, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)
    scale = max(float(np.percentile(matrix, 95)), 1.0)
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=scale, aspect="auto")
    ax.set_xticks(np.arange(len(geometries)), [value.replace("_", "\n") for value in geometries])
    ax.set_yticks(np.arange(len(SELECTOR_ORDER)), [SELECTOR_LABELS[value] for value in SELECTOR_ORDER])
    ax.set_title("Mean oracle regret on each completely held-out geometry")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, column_index] > 0.55 * scale else "#263238"
            ax.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", color=color, fontsize=8.5)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="mean regret (%)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_failure_concentration(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if row["mode"] == "no_holdout"]
    dimensions = [
        ("family", "Morphology family"),
        ("dynamics", "Temporal dynamics"),
        ("noise_level", "Noise level"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for ax, (field, title) in zip(axes, dimensions):
        values = sorted({str(row[field]) for row in selected})
        means = []
        p95 = []
        for value in values:
            regrets = [
                float(row["selected_regret_pct"])
                for row in selected
                if str(row[field]) == value
            ]
            means.append(float(np.mean(regrets)))
            p95.append(float(np.percentile(regrets, 95)))
        x = np.arange(len(values))
        ax.bar(x, means, color="#196e63", label="mean")
        ax.scatter(x, p95, marker="D", color="#a24d41", label="p95", zorder=3)
        ax.set_xticks(x, [value.replace("_", "\n") for value in values])
        ax.set_ylabel("oracle regret (%)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.24)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Geometry transfer failures concentrate in morphology-dynamics-noise interactions",
        fontsize=13.5,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_uncertainty_alignment(
    rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    for ax, mode in zip(axes, MODEL_FEATURE_SETS):
        subset = [row for row in rows if row["mode"] == mode]
        audit = next(
            row for row in audit_rows
            if row["mode"] == mode and row["score"] == "combined_uncertainty"
        )
        ax.scatter(
            [float(row["combined_uncertainty"]) for row in subset],
            [float(row["selected_regret_pct"]) for row in subset],
            s=13,
            alpha=0.30,
            color="#315f8d" if mode == "no_holdout" else "#196e63",
        )
        ax.axhline(1.0, color="#a24d41", linestyle="--", linewidth=1)
        ax.set_xlabel("observable uncertainty percentile score")
        ax.set_ylabel("selector oracle regret (%)")
        ax.set_title(
            f"{mode.replace('_', ' ')}: Spearman {float(audit['spearman_score_vs_regret']):.3f}, AUC {float(audit['high_risk_auc']):.3f}"
        )
        ax.grid(True, alpha=0.22)
    fig.suptitle("Does ensemble uncertainty identify high-regret geometry cells?", fontsize=13.5)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_risk_coverage(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(UQ_SCORE_FIELDS)))
    for ax, mode in zip(axes, MODEL_FEATURE_SETS):
        for score, color in zip(UQ_SCORE_FIELDS, colors):
            subset = sorted(
                [row for row in rows if row["mode"] == mode and row["score"] == score],
                key=lambda row: float(row["actual_coverage"]),
            )
            ax.plot(
                [100.0 * float(row["actual_coverage"]) for row in subset],
                [float(row["selective_mean_regret_pct"]) for row in subset],
                marker="o",
                color=color,
                label=UQ_SCORE_LABELS[score],
            )
        ax.set_xlabel("accepted coverage (%)")
        ax.set_ylabel("accepted-cell mean regret (%)")
        ax.set_title(mode.replace("_", " "))
        ax.grid(True, alpha=0.24)
    axes[1].legend(fontsize=8)
    fig.suptitle("M3B risk-coverage: reject high-uncertainty geometry cells", fontsize=13.5)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_full_fallback(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), constrained_layout=True)
    for ax, mode in zip(axes, MODEL_FEATURE_SETS):
        subset = sorted(
            [
                row for row in rows
                if row["mode"] == mode and row["score"] == "combined_uncertainty"
            ],
            key=lambda row: float(row["actual_coverage"]),
        )
        coverage = [100.0 * float(row["actual_coverage"]) for row in subset]
        ax.plot(
            coverage,
            [float(row["selective_mean_regret_pct"]) for row in subset],
            marker="o",
            color="#196e63",
            label="accepted cells only",
        )
        ax.plot(
            coverage,
            [float(row["full_fallback_system_mean_regret_pct"]) for row in subset],
            marker="s",
            color="#a24d41",
            label="system risk with full-rank fallback",
        )
        ax.set_xlabel("selector coverage (%)")
        ax.set_ylabel("mean oracle regret (%)")
        ax.set_title(mode.replace("_", " "))
        ax.grid(True, alpha=0.24)
    axes[1].legend(fontsize=8)
    fig.suptitle("Abstention is useful only if the fallback is safer", fontsize=13.5)
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
    fold_rows: list[dict[str, object]],
    uq_rows: list[dict[str, object]],
    uq_audit: list[dict[str, object]],
    risk_rows: list[dict[str, object]],
    models: dict[str, dict[str, dict[str, object]]],
    model_audit: dict[str, object],
) -> dict[str, object]:
    grid = config["grid"]
    summary = {str(row["selector"]): row for row in summary_rows}
    no_holdout = summary["logo_ensemble_no_holdout"]
    with_holdout = summary["logo_ensemble_with_holdout"]
    fixed = summary["fixed_rank3"]
    combined_audit = {
        row["mode"]: row
        for row in uq_audit
        if row["score"] == "combined_uncertainty"
    }
    combined_risk = {
        mode: {
            str(row["target_coverage"]): row
            for row in risk_rows
            if row["mode"] == mode and row["score"] == "combined_uncertainty"
        }
        for mode in MODEL_FEATURE_SETS
    }
    audit_lookup = {
        (str(row["mode"]), str(row["score"])): row for row in uq_audit
    }
    risk_lookup = {
        (str(row["mode"]), str(row["score"]), str(row["target_coverage"])): row
        for row in risk_rows
    }
    no_holdout_prediction_std_rows = [
        row
        for row in risk_rows
        if row["mode"] == "no_holdout" and row["score"] == "prediction_std_percentile"
    ]
    best_prediction_std_risk = min(
        no_holdout_prediction_std_rows,
        key=lambda row: float(row["selective_mean_regret_pct"]),
    )
    high_risk_rows = [
        row
        for row in uq_rows
        if row["mode"] == "no_holdout" and float(row["selected_regret_pct"]) > 1.0
    ]
    concentration: dict[tuple[str, str, str], int] = Counter(
        (
            str(row["family"]),
            str(row["dynamics"]),
            str(row["noise_level"]),
        )
        for row in high_risk_rows
    )
    worst = sorted(
        [row for row in selected_rows if row["selector"] == "logo_ensemble_no_holdout"],
        key=lambda row: float(row["regret_pct"]),
        reverse=True,
    )[:12]
    return {
        "experiment": config["experiment_name"],
        "design": {
            "families": grid["families"],
            "dynamics": grid["dynamics"],
            "noise_levels": grid["noise_levels"],
            "view_counts": grid["view_counts"],
            "geometries": grid["geometries"],
            "ranks": grid["ranks"],
            "seed_count": grid["seed_count"],
            "environment_cells_without_seed": len(grid["families"]) * len(grid["dynamics"]) * len(grid["noise_levels"]) * len(grid["view_counts"]) * len(grid["geometries"]),
            "observation_cells": len(group_cells(raw_rows)),
            "candidate_rows": len(raw_rows),
            "selector_rows": len(selected_rows),
            "uncertainty_rows": len(uq_rows),
            "evaluation_protocol": "nested leave-one-geometry-out with ensemble members trained on proper subsets of training geometries",
        },
        "selector_summary": summary,
        "geometry_fold_summary": fold_rows,
        "uncertainty_audit": uq_audit,
        "risk_coverage": risk_rows,
        "model_selection_audit": model_audit,
        "models": {
            mode: {
                geometry: {
                    "central": serializable_model(bundle["central"]),
                    "ensemble": [serializable_model(model) for model in bundle["ensemble"]],
                }
                for geometry, bundle in geometry_models.items()
            }
            for mode, geometry_models in models.items()
        },
        "worst_no_holdout_cells": worst,
        "key_findings": {
            "fixed_rank3_mean_regret_pct": float(fixed["mean_regret_pct"]),
            "fixed_rank3_p95_regret_pct": float(fixed["p95_regret_pct"]),
            "logo_no_holdout_mean_regret_pct": float(no_holdout["mean_regret_pct"]),
            "logo_no_holdout_p95_regret_pct": float(no_holdout["p95_regret_pct"]),
            "logo_with_holdout_mean_regret_pct": float(with_holdout["mean_regret_pct"]),
            "logo_with_holdout_p95_regret_pct": float(with_holdout["p95_regret_pct"]),
            "combined_uq_no_holdout_spearman": float(combined_audit["no_holdout"]["spearman_score_vs_regret"]),
            "combined_uq_no_holdout_high_risk_auc": float(combined_audit["no_holdout"]["high_risk_auc"]),
            "combined_uq_with_holdout_spearman": float(combined_audit["with_holdout"]["spearman_score_vs_regret"]),
            "combined_uq_with_holdout_high_risk_auc": float(combined_audit["with_holdout"]["high_risk_auc"]),
            "prediction_std_no_holdout_spearman": float(
                audit_lookup[("no_holdout", "prediction_std_percentile")][
                    "spearman_score_vs_regret"
                ]
            ),
            "prediction_std_no_holdout_high_risk_auc": float(
                audit_lookup[("no_holdout", "prediction_std_percentile")]["high_risk_auc"]
            ),
            "prediction_std_no_holdout_selective_risk_at_50pct_coverage": float(
                risk_lookup[("no_holdout", "prediction_std_percentile", "0.5")][
                    "selective_mean_regret_pct"
                ]
            ),
            "prediction_std_no_holdout_full_fallback_risk_at_50pct_coverage": float(
                risk_lookup[("no_holdout", "prediction_std_percentile", "0.5")][
                    "full_fallback_system_mean_regret_pct"
                ]
            ),
            "prediction_std_best_retrospective_coverage": float(
                best_prediction_std_risk["actual_coverage"]
            ),
            "prediction_std_best_retrospective_selective_risk_pct": float(
                best_prediction_std_risk["selective_mean_regret_pct"]
            ),
            "high_risk_cell_count": len(high_risk_rows),
            "top_high_risk_family_dynamics_noise": [
                {
                    "family": key[0],
                    "dynamics": key[1],
                    "noise_level": key[2],
                    "cell_count": count,
                }
                for key, count in concentration.most_common(6)
            ],
            "no_holdout_selective_risk_at_50pct_coverage": float(combined_risk["no_holdout"]["0.5"]["selective_mean_regret_pct"]),
            "no_holdout_full_fallback_system_risk_at_50pct_coverage": float(combined_risk["no_holdout"]["0.5"]["full_fallback_system_mean_regret_pct"]),
            "with_holdout_selective_risk_at_50pct_coverage": float(combined_risk["with_holdout"]["0.5"]["selective_mean_regret_pct"]),
            "with_holdout_full_fallback_system_risk_at_50pct_coverage": float(combined_risk["with_holdout"]["0.5"]["full_fallback_system_mean_regret_pct"]),
        },
        "rebuild_modes": ["full_simulation", "resume_missing_cells", "reuse_raw_analysis"],
        "claims_boundary": config["claims_boundary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-raw", action="store_true")
    return parser.parse_args()


def load_config(path: Path, quick: bool) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if quick:
        grid = config["grid"]
        grid["families"] = grid["families"][:2]
        grid["dynamics"] = ["smooth", "transient"]
        grid["noise_levels"] = [0.0, 0.28]
        grid["view_counts"] = [5]
        grid["geometries"] = ["uniform", "limited_arc", "calibration_offset_2deg"]
        grid["ranks"] = [2, 3, 5, 18]
        grid["seed_count"] = 2
    return config


def main() -> None:
    args = parse_args()
    if args.resume and args.reuse_raw:
        raise SystemExit("--resume and --reuse-raw are mutually exclusive")
    forbidden = {
        "geometry",
        "family",
        "dynamics",
        "noise_level",
        "field_rel_l2",
        "oracle_rank",
        "target_log_error_ratio",
    }
    for mode, features in MODEL_FEATURE_SETS.items():
        leakage = forbidden.intersection(features)
        if leakage:
            raise SystemExit(f"selector feature leakage in {mode}: {sorted(leakage)}")

    config = load_config(args.config, args.quick)
    grid = config["grid"]
    families = list(grid["families"])
    dynamics_modes = list(grid["dynamics"])
    noises = [float(value) for value in grid["noise_levels"]]
    views = [int(value) for value in grid["view_counts"]]
    geometries = list(grid["geometries"])
    ranks = [int(value) for value in grid["ranks"]]
    seeds = [int(grid["seed_base"]) + index for index in range(int(grid["seed_count"]))]
    if len(geometries) < 3:
        raise SystemExit("at least three geometry families are required")
    if 3 not in ranks:
        raise SystemExit("rank 3 is required as a baseline")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "geometry_raw.csv"
    started = time.perf_counter()
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
            f"g-{geometry}_f-{family}_d-{dynamics}_v-{view_count}_n-{noise_level:.2f}_s-{seed}"
            for geometry in geometries
            for family in families
            for dynamics in dynamics_modes
            for view_count in views
            for noise_level in noises
            for seed in seeds
        }
        if args.resume and raw_path.exists():
            with raw_path.open(encoding="utf-8") as handle:
                raw_rows = list(csv.DictReader(handle))
            existing_cells = {str(row["cell_id"]) for row in raw_rows}
            unexpected = existing_cells - expected_cell_ids
            if unexpected:
                raise SystemExit(f"raw table contains unexpected cells: {sorted(unexpected)[:3]}")
            print(f"resume found {len(existing_cells)} completed observation cells")
        else:
            raw_rows = []
            existing_cells = set()
        missing_cells = len(expected_cell_ids - existing_cells)
        completed = 0

        for geometry in geometries:
            for view_count in views:
                true_angles, reconstruction_angles = geometry_angles(geometry, view_count)
                heldout_angles = np.asarray([largest_gap_midpoint(true_angles)])
                descriptors = geometry_descriptors(reconstruction_angles)
                for family in families:
                    for dynamics in dynamics_modes:
                        reference = references[(family, dynamics)]
                        for noise_level in noises:
                            for seed in seeds:
                                cell_id = f"g-{geometry}_f-{family}_d-{dynamics}_v-{view_count}_n-{noise_level:.2f}_s-{seed}"
                                if cell_id in existing_cells:
                                    continue
                                framewise, support_observed, heldout_observed, heldout_clean = simulate_geometry_observation(
                                    reference,
                                    true_angles,
                                    reconstruction_angles,
                                    heldout_angles,
                                    seed,
                                    noise_level,
                                )
                                candidates, singular_values = low_rank_candidates(framewise, ranks)
                                for rank in ranks:
                                    row = candidate_row(
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
                                        reconstruction_angles,
                                        heldout_angles,
                                        support_observed,
                                        heldout_observed,
                                        heldout_clean,
                                        singular_values,
                                    )
                                    row["geometry"] = geometry
                                    row["true_angles_deg"] = json.dumps(true_angles.tolist(), separators=(",", ":"))
                                    row["reconstruction_angles_deg"] = json.dumps(reconstruction_angles.tolist(), separators=(",", ":"))
                                    row["heldout_angle_deg"] = float(heldout_angles[0])
                                    row.update(descriptors)
                                    raw_rows.append(row)
                                completed += 1
                                if completed % max(missing_cells // 18, 1) == 0 or completed == missing_cells:
                                    write_csv(raw_path, raw_rows)
                                    print(f"completed missing observation cells: {completed}/{missing_cells}", flush=True)

    annotate_oracle(raw_rows)
    lambdas = [float(value) for value in config["selectors"]["ridge_lambdas"]]
    models, model_audit = train_logo_models(raw_rows, geometries, lambdas)
    selected_rows, uq_rows = select_and_measure_uncertainty(
        raw_rows,
        geometries,
        models,
        float(config["selectors"]["energy_threshold"]),
    )
    summary_rows = summarize_selector_rows(selected_rows, SELECTOR_ORDER)
    fold_rows = geometry_summary(selected_rows, geometries)
    uq_audit = uncertainty_audit(
        uq_rows,
        float(config["selectors"]["high_risk_regret_pct"]),
    )
    risk_rows = risk_coverage_rows(
        uq_rows,
        [float(value) for value in config["selectors"]["risk_coverages"]],
    )

    plot_geometry_layouts(geometries, max(views), args.output_dir / "m3b_geometry_layouts.png")
    plot_selector_regret(summary_rows, args.output_dir / "m3b_geometry_selector_regret.png")
    plot_geometry_heatmap(fold_rows, geometries, args.output_dir / "m3b_geometry_fold_heatmap.png")
    plot_failure_concentration(
        uq_rows,
        args.output_dir / "m3b_geometry_failure_concentration.png",
    )
    plot_uncertainty_alignment(uq_rows, uq_audit, args.output_dir / "m3b_geometry_uncertainty_alignment.png")
    plot_risk_coverage(risk_rows, args.output_dir / "m3b_geometry_risk_coverage.png")
    plot_full_fallback(risk_rows, args.output_dir / "m3b_geometry_full_fallback.png")

    report = build_report(
        config,
        raw_rows,
        selected_rows,
        summary_rows,
        fold_rows,
        uq_rows,
        uq_audit,
        risk_rows,
        models,
        model_audit,
    )
    write_csv(args.output_dir / "geometry_raw.csv", raw_rows)
    write_csv(args.output_dir / "geometry_selected.csv", selected_rows)
    write_csv(args.output_dir / "geometry_summary.csv", summary_rows)
    write_csv(args.output_dir / "geometry_fold_summary.csv", fold_rows)
    write_csv(args.output_dir / "geometry_uq_cells.csv", uq_rows)
    write_csv(args.output_dir / "geometry_uq_audit.csv", uq_audit)
    write_csv(args.output_dir / "geometry_risk_coverage.csv", risk_rows)
    with (args.output_dir / "geometry_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    write_checksum_manifest(
        args.output_dir,
        [
            "geometry_raw.csv",
            "geometry_selected.csv",
            "geometry_summary.csv",
            "geometry_fold_summary.csv",
            "geometry_uq_cells.csv",
            "geometry_uq_audit.csv",
            "geometry_risk_coverage.csv",
            "geometry_report.json",
        ],
    )

    runtime = time.perf_counter() - started
    print(f"completed {len(raw_rows)} candidate rows and {len(selected_rows)} selector rows in {runtime:.1f} s")
    print(json.dumps(report["key_findings"], indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
