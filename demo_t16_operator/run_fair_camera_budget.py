#!/usr/bin/env python3
"""Compare learned and numerical corrections under an equal camera budget.

The experiment fixes a total budget K. K-1 views build the initial field, one
Q_fit view calibrates or directly augments the reconstruction, and one prelocked
camera is reserved as Q_audit. Field truth is used only for retrospective metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .bost_physics import baseline_lift, forward_volume
    from .data import generate_dataset, load_npz, split_indices
    from .run_ablations import SPLIT_ORDER
    from .run_query_calibrated_nullspace import (
        clipped_line_search_alpha,
        load_null_corrector,
    )
    from .run_support_nullspace_corrector import (
        TorchSupportNullProjector,
        corrected_prediction,
        corrector_input,
        load_base_model,
        read_json,
        support_fit_state,
    )
    from .train_eval import _gradient_relative, _masked_projection_relative, _relative_norm
except ImportError:
    from bost_physics import baseline_lift, forward_volume
    from data import generate_dataset, load_npz, split_indices
    from run_ablations import SPLIT_ORDER
    from run_query_calibrated_nullspace import clipped_line_search_alpha, load_null_corrector
    from run_support_nullspace_corrector import (
        TorchSupportNullProjector,
        corrected_prediction,
        corrector_input,
        load_base_model,
        read_json,
        support_fit_state,
    )
    from train_eval import _gradient_relative, _masked_projection_relative, _relative_norm


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "fair_camera_budget.json"
DEFAULT_OUTPUT = ROOT / "results" / "fair_camera_budget"
METHODS = [
    "support_physics_lift",
    "union_physics_lift",
    "support_fit_base",
    "union_support_fit_direct",
    "learned_query_correction",
    "numeric_query_null_update",
]
LABELS = {
    "support_physics_lift": "S-only physics lift",
    "union_physics_lift": "S union Q physics lift",
    "support_fit_base": "S-only support fit",
    "union_support_fit_direct": "S union Q direct support fit",
    "learned_query_correction": "learned null + Q fit",
    "numeric_query_null_update": "numerical query-null update",
}
QUERY_STRATEGIES = ["fixed", "random", "max_gap", "adaptive_energy"]
METRIC_KEYS = [
    "field_rel_l2",
    "gradient_rel_l2",
    "support_reprojection_rel_l2",
    "audit_reprojection_rel_l2",
    "qfit_noisy_residual",
    "support_correction_leakage",
    "correction_norm_ratio",
    "field_improvement_vs_support_pct",
    "field_improvement_vs_union_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


def controlled_support_mask(
    max_views: int,
    support_count: int,
    fixed_query_index: int,
    audit_query_index: int,
) -> np.ndarray:
    """Choose deterministic support views while reserving Q_fit and Q_audit."""
    if not 0 <= fixed_query_index < max_views or not 0 <= audit_query_index < max_views:
        raise ValueError("reserved query index is outside the camera layout")
    if fixed_query_index == audit_query_index:
        raise ValueError("Q_fit and Q_audit must be different cameras")
    candidates = np.asarray(
        [
            index
            for index in range(max_views)
            if index not in {fixed_query_index, audit_query_index}
        ],
        dtype=int,
    )
    if support_count < 1 or support_count > len(candidates):
        raise ValueError("support_count exceeds cameras left after Q_fit/Q_audit reservation")
    positions = np.rint(np.linspace(0, len(candidates) - 1, support_count)).astype(int)
    selected = np.unique(candidates[positions])
    if len(selected) != support_count:
        raise RuntimeError("deterministic support selection produced duplicate views")
    mask = np.zeros(max_views, dtype=np.float64)
    mask[selected] = 1.0
    return mask


def _circular_distance(angle_a: float, angle_b: np.ndarray) -> np.ndarray:
    difference = np.abs(np.asarray(angle_b, dtype=np.float64) - float(angle_a))
    return np.minimum(difference, 180.0 - difference)


def select_query_index(
    strategy: str,
    support_mask: np.ndarray,
    angles: np.ndarray,
    fixed_query_index: int,
    audit_query_index: int,
    sample_seed: int,
    total_budget: int,
    direction_projection: np.ndarray,
    random_seed: int,
) -> int:
    available = np.asarray(
        [
            index
            for index in np.flatnonzero(np.asarray(support_mask) < 0.5)
            if int(index) != int(audit_query_index)
        ],
        dtype=int,
    )
    if len(available) < 1:
        raise ValueError("no Q_fit candidate remains after locking Q_audit")
    if strategy == "fixed":
        if fixed_query_index not in available:
            raise ValueError("the fixed query camera leaked into support")
        return int(fixed_query_index)
    if strategy == "random":
        rng = np.random.default_rng(
            int(random_seed) + int(sample_seed) * 1009 + int(total_budget) * 9176
        )
        return int(rng.choice(available))
    if strategy == "max_gap":
        support = np.flatnonzero(np.asarray(support_mask) > 0.5)
        scores = [
            float(np.min(_circular_distance(float(angles[index]), angles[support])))
            for index in available
        ]
        return int(available[int(np.argmax(scores))])
    if strategy == "adaptive_energy":
        energies = np.sum(
            np.asarray(direction_projection, dtype=np.float64)[:, available, :] ** 2,
            axis=(0, 2),
        )
        return int(available[int(np.argmax(energies))])
    raise ValueError(f"unknown query strategy: {strategy}")


def mask_for_index(max_views: int, index: int) -> np.ndarray:
    mask = np.zeros(max_views, dtype=np.float64)
    mask[int(index)] = 1.0
    return mask


def controlled_input(
    data: dict[str, np.ndarray],
    sample_index: int,
    mask: np.ndarray,
) -> np.ndarray:
    selected = np.flatnonzero(np.asarray(mask) > 0.5)
    n = int(data["field"].shape[-1])
    scale, offset = [float(value) for value in data["calibration"]]
    raw_lift = baseline_lift(
        data["observation"][sample_index][:, selected],
        data["angles"][selected],
        n,
    )
    inputs = np.asarray(data["inputs"][sample_index], dtype=np.float32).copy()
    inputs[0] = (scale * raw_lift + offset).astype(np.float32)
    inputs[2].fill(float(len(selected)) / len(mask))
    return inputs


def predict_support_fit(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    operator = torch.from_numpy(data["forward_matrix"]).to(device)
    bases: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    absolutes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_masks = masks[start : start + batch_size]
        x = torch.from_numpy(
            np.stack(
                [controlled_input(data, int(index), mask) for index, mask in zip(batch_indices, batch_masks)]
            )
        ).to(device)
        observed = torch.from_numpy(data["observation"][batch_indices]).to(device)
        mask_torch = torch.from_numpy(batch_masks.astype(np.float32)).to(device)
        with torch.no_grad():
            base, residual, absolute, weight = support_fit_state(
                model,
                x,
                observed,
                mask_torch,
                operator,
            )
        bases.append(base[:, 0].cpu().numpy())
        residuals.append(residual[:, 0].cpu().numpy())
        absolutes.append(absolute[:, 0].cpu().numpy())
        weights.append(weight[:, 0, 0, 0, 0].cpu().numpy())
    return (
        np.concatenate(bases),
        np.concatenate(residuals),
        np.concatenate(absolutes),
        np.concatenate(weights),
    )


def predict_learned_correction(
    corrector: torch.nn.Module,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    masks: np.ndarray,
    bases: np.ndarray,
    residuals: np.ndarray,
    absolutes: np.ndarray,
    projector: TorchSupportNullProjector,
    cap_ratio: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_masks = masks[start : start + batch_size]
        x = torch.from_numpy(
            np.stack(
                [controlled_input(data, int(index), mask) for index, mask in zip(batch_indices, batch_masks)]
            )
        ).to(device)
        base = torch.from_numpy(bases[start : start + batch_size, None].astype(np.float32)).to(device)
        residual = torch.from_numpy(
            residuals[start : start + batch_size, None].astype(np.float32)
        ).to(device)
        absolute = torch.from_numpy(
            absolutes[start : start + batch_size, None].astype(np.float32)
        ).to(device)
        mask_torch = torch.from_numpy(batch_masks.astype(np.float32)).to(device)
        inputs = corrector_input(x, base, residual, absolute)
        with torch.no_grad():
            _, correction, _ = corrected_prediction(
                "nullspace_correction",
                corrector,
                inputs,
                base,
                mask_torch,
                projector,
                cap_ratio,
            )
        outputs.append(correction[:, 0].cpu().numpy())
    return np.concatenate(outputs)


def numeric_query_null_update(
    base: np.ndarray,
    observed: np.ndarray,
    support_mask: np.ndarray,
    query_mask: np.ndarray,
    operator: np.ndarray,
    projector: TorchSupportNullProjector,
    ridge_relative: float,
    cap_ratio: float,
) -> np.ndarray:
    """Solve P_N A_Q^T (A_Q P_N A_Q^T + lambda I)^-1 r_Q."""
    basis = projector.numpy_basis(support_mask)
    query_indices = np.flatnonzero(np.asarray(query_mask) > 0.5)
    query_matrix = np.asarray(operator[query_indices], dtype=np.float64).reshape(-1, operator.shape[-1])
    compressed = query_matrix @ basis.T
    gram = compressed @ compressed.T
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    ridge = max(float(ridge_relative) * scale, 1e-10)
    inverse_system = gram + ridge * np.eye(gram.shape[0], dtype=np.float64)
    projected_base = forward_volume(base, operator)
    correction = np.zeros_like(base, dtype=np.float64)
    for depth_index in range(base.shape[0]):
        residual = (
            observed[depth_index, query_indices] - projected_base[depth_index, query_indices]
        ).reshape(-1)
        dual = np.linalg.solve(inverse_system, residual)
        coefficients = compressed.T @ dual
        correction[depth_index] = (basis.T @ coefficients).reshape(base.shape[1:])
    correction_norm = float(np.linalg.norm(correction))
    maximum = float(cap_ratio) * max(float(np.linalg.norm(base)), 1e-12)
    if correction_norm > maximum:
        correction *= maximum / correction_norm
    return correction


def noisy_residual(
    prediction: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
) -> float:
    active = np.asarray(mask, dtype=np.float64)[None, :, None]
    return _relative_norm((prediction - observed) * active, observed * active)


def method_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    target: np.ndarray,
    clean: np.ndarray,
    observed: np.ndarray,
    support_mask: np.ndarray,
    query_mask: np.ndarray,
    audit_mask: np.ndarray,
    operator: np.ndarray,
    support_error: float,
    union_error: float,
) -> dict[str, float]:
    projected = forward_volume(prediction, operator)
    correction_projection = forward_volume(prediction - reference, operator)
    field_error = _relative_norm(prediction - target, target)
    correction_norm = float(np.linalg.norm(prediction - reference))
    reference_norm = max(float(np.linalg.norm(reference)), 1e-12)
    return {
        "field_rel_l2": field_error,
        "gradient_rel_l2": _gradient_relative(prediction, target),
        "support_reprojection_rel_l2": _masked_projection_relative(
            projected, clean, support_mask
        ),
        "audit_reprojection_rel_l2": _masked_projection_relative(
            projected, clean, audit_mask
        ),
        "qfit_noisy_residual": noisy_residual(projected, observed, query_mask),
        "support_correction_leakage": _relative_norm(
            correction_projection * support_mask[None, :, None],
            clean * support_mask[None, :, None],
        ),
        "correction_norm_ratio": correction_norm / reference_norm,
        "field_improvement_vs_support_pct": 100.0
        * (support_error - field_error)
        / (support_error + 1e-12),
        "field_improvement_vs_union_pct": 100.0
        * (union_error - field_error)
        / (union_error + 1e-12),
    }


def test_indices_and_splits(data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[int, str]]:
    split_map = split_indices(data)
    indices = np.concatenate([split_map[name] for name in SPLIT_ORDER]).astype(int)
    names = {
        int(index): name for name in SPLIT_ORDER for index in split_map[name]
    }
    return indices, names


def evaluate_seed(
    seed: int,
    base_model: torch.nn.Module,
    corrector: torch.nn.Module,
    data: dict[str, np.ndarray],
    dataset_config: dict,
    corrector_config: dict,
    experiment_config: dict,
    device: torch.device,
) -> list[dict[str, object]]:
    indices, source_splits = test_indices_and_splits(data)
    operator = np.asarray(data["forward_matrix"], dtype=np.float64)
    angles = np.asarray(data["angles"], dtype=np.float64)
    max_views = len(angles)
    batch_size = int(dataset_config["training"]["batch_size"])
    projector = TorchSupportNullProjector(operator)
    rows: list[dict[str, object]] = []

    for total_budget in [int(value) for value in experiment_config["total_budgets"]]:
        support_count = total_budget - 1
        support_mask = controlled_support_mask(
            max_views,
            support_count,
            int(experiment_config["fixed_query_index"]),
            int(experiment_config["audit_query_index"]),
        )
        support_masks = np.broadcast_to(support_mask, (len(indices), max_views)).copy()
        bases, residuals, absolutes, support_weights = predict_support_fit(
            base_model,
            data,
            indices,
            support_masks,
            device,
            batch_size,
        )
        corrections = predict_learned_correction(
            corrector,
            data,
            indices,
            support_masks,
            bases,
            residuals,
            absolutes,
            projector,
            float(corrector_config["correction_cap_ratio"]),
            device,
            batch_size,
        )
        correction_projections = np.stack(
            [forward_volume(correction, operator) for correction in corrections]
        )
        support_lifts = np.stack(
            [controlled_input(data, int(index), support_mask)[0] for index in indices]
        )

        for strategy in experiment_config["query_strategies"]:
            query_indices = np.asarray(
                [
                    select_query_index(
                        str(strategy),
                        support_mask,
                        angles,
                        int(experiment_config["fixed_query_index"]),
                        int(experiment_config["audit_query_index"]),
                        int(data["sample_seed"][sample_index]),
                        total_budget,
                        correction_projections[local_index],
                        int(experiment_config["random_seed"]),
                    )
                    for local_index, sample_index in enumerate(indices)
                ],
                dtype=int,
            )
            query_masks = np.stack(
                [mask_for_index(max_views, index) for index in query_indices]
            )
            union_masks = support_masks + query_masks
            audit_mask = mask_for_index(
                max_views, int(experiment_config["audit_query_index"])
            )
            audit_masks = np.broadcast_to(audit_mask, (len(indices), max_views)).copy()
            if np.any(np.sum(union_masks * audit_masks, axis=1) > 0.0):
                raise RuntimeError("locked Q_audit leaked into reconstruction")
            unions, _, _, union_weights = predict_support_fit(
                base_model,
                data,
                indices,
                union_masks,
                device,
                batch_size,
            )
            union_lifts = np.stack(
                [
                    controlled_input(data, int(index), mask)[0]
                    for index, mask in zip(indices, union_masks)
                ]
            )

            for local_index, sample_index in enumerate(indices):
                base = np.asarray(bases[local_index], dtype=np.float64)
                correction = np.asarray(corrections[local_index], dtype=np.float64)
                union = np.asarray(unions[local_index], dtype=np.float64)
                target = np.asarray(data["field"][sample_index], dtype=np.float64)
                clean = np.asarray(data["clean_observation"][sample_index], dtype=np.float64)
                observed = np.asarray(data["observation"][sample_index], dtype=np.float64)
                query_mask = query_masks[local_index]
                audit_mask = audit_masks[local_index]
                base_projection = forward_volume(base, operator)
                correction_projection = correction_projections[local_index]
                alpha = clipped_line_search_alpha(
                    correction_projection,
                    observed - base_projection,
                    query_mask,
                    float(experiment_config["alpha_min"]),
                    float(experiment_config["alpha_max"]),
                )
                learned = base + alpha * correction
                numeric_correction = numeric_query_null_update(
                    base,
                    observed,
                    support_mask,
                    query_mask,
                    operator,
                    projector,
                    float(experiment_config["numeric_ridge_relative"]),
                    float(corrector_config["correction_cap_ratio"]),
                )
                numeric = base + numeric_correction
                support_error = _relative_norm(base - target, target)
                union_error = _relative_norm(union - target, target)
                predictions = {
                    "support_physics_lift": np.asarray(support_lifts[local_index]),
                    "union_physics_lift": np.asarray(union_lifts[local_index]),
                    "support_fit_base": base,
                    "union_support_fit_direct": union,
                    "learned_query_correction": learned,
                    "numeric_query_null_update": numeric,
                }
                references = {
                    "support_physics_lift": np.asarray(support_lifts[local_index]),
                    "union_physics_lift": np.asarray(union_lifts[local_index]),
                    "support_fit_base": base,
                    "union_support_fit_direct": union,
                    "learned_query_correction": base,
                    "numeric_query_null_update": base,
                }
                for method in METHODS:
                    row: dict[str, object] = {
                        "seed": seed,
                        "source_split": source_splits[int(sample_index)],
                        "sample_index": int(sample_index),
                        "sample_seed": int(data["sample_seed"][sample_index]),
                        "family_id": int(data["family_id"][sample_index]),
                        "noise_level": float(data["noise_level"][sample_index]),
                        "total_budget": total_budget,
                        "support_count": support_count,
                        "query_strategy": str(strategy),
                        "query_index": int(query_indices[local_index]),
                        "query_angle_deg": float(angles[query_indices[local_index]]),
                        "audit_index": int(experiment_config["audit_query_index"]),
                        "audit_angle_deg": float(
                            angles[int(experiment_config["audit_query_index"])]
                        ),
                        "audit_count": int(np.sum(audit_mask)),
                        "support_fit_residual_weight": float(support_weights[local_index]),
                        "union_fit_residual_weight": float(union_weights[local_index]),
                        "method": method,
                        "alpha": float(alpha) if method == "learned_query_correction" else 0.0,
                    }
                    row.update(
                        method_metrics(
                            predictions[method],
                            references[method],
                            target,
                            clean,
                            observed,
                            support_mask,
                            query_mask,
                            audit_mask,
                            operator,
                            support_error,
                            union_error,
                        )
                    )
                    rows.append(row)
    return rows


def collapse_model_seeds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["sample_index"]),
            int(row["total_budget"]),
            str(row["query_strategy"]),
            str(row["method"]),
        )
        groups[key].append(row)
    output: list[dict[str, object]] = []
    for key, subset in sorted(groups.items()):
        first = subset[0]
        output.append(
            {
                "sample_index": key[0],
                "sample_seed": int(first["sample_seed"]),
                "source_split": first["source_split"],
                "family_id": int(first["family_id"]),
                "noise_level": float(first["noise_level"]),
                "total_budget": key[1],
                "support_count": int(first["support_count"]),
                "query_strategy": key[2],
                "method": key[3],
                "model_seed_count": len(subset),
                **{
                    metric: float(np.mean([float(row[metric]) for row in subset]))
                    for metric in METRIC_KEYS
                },
            }
        )
    return output


def stratified_bootstrap_interval(
    rows: list[dict[str, object]],
    metric: str,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    domains = sorted({str(row["source_split"]) for row in rows})
    estimates = np.zeros(int(replicates), dtype=np.float64)
    for domain in domains:
        values = np.asarray(
            [float(row[metric]) for row in rows if row["source_split"] == domain]
        )
        indices = rng.integers(0, len(values), size=(int(replicates), len(values)))
        estimates += np.mean(values[indices], axis=1) / len(domains)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def domain_equal_weights(rows: list[dict[str, object]]) -> np.ndarray:
    domains = sorted({str(row["source_split"]) for row in rows})
    counts = {
        domain: sum(str(row["source_split"]) == domain for row in rows)
        for domain in domains
    }
    return np.asarray(
        [1.0 / len(domains) / counts[str(row["source_split"])] for row in rows],
        dtype=np.float64,
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = min(int(np.searchsorted(cumulative, float(quantile), side="left")), len(values) - 1)
    return float(sorted_values[index])


def weighted_lower_cvar(
    values: np.ndarray,
    weights: np.ndarray,
    tail_probability: float,
) -> float:
    order = np.argsort(values)
    remaining = float(tail_probability)
    weighted_sum = 0.0
    for index in order:
        take = min(float(weights[index]), remaining)
        weighted_sum += take * float(values[index])
        remaining -= take
        if remaining <= 1e-12:
            break
    return weighted_sum / float(tail_probability)


def summarize_clusters(
    cluster_rows: list[dict[str, object]],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in cluster_rows:
        groups[(int(row["total_budget"]), str(row["query_strategy"]), str(row["method"]))].append(row)
    rng = np.random.default_rng(int(bootstrap_seed))
    output: list[dict[str, object]] = []
    for (budget, strategy, method), subset in sorted(groups.items()):
        vs_support = np.asarray(
            [float(row["field_improvement_vs_support_pct"]) for row in subset]
        )
        vs_union = np.asarray(
            [float(row["field_improvement_vs_union_pct"]) for row in subset]
        )
        weights = domain_equal_weights(subset)
        ci_low, ci_high = stratified_bootstrap_interval(
            subset,
            "field_improvement_vs_union_pct",
            rng,
            bootstrap_replicates,
        )
        output.append(
            {
                "total_budget": budget,
                "query_strategy": strategy,
                "method": method,
                "independent_field_count": len(subset),
                "model_seed_count": int(subset[0]["model_seed_count"]),
                "source_domain_count": len({str(row["source_split"]) for row in subset}),
                "mean_field_rel_l2": float(
                    np.sum(weights * np.asarray([float(row["field_rel_l2"]) for row in subset]))
                ),
                "mean_audit_reprojection_rel_l2": float(
                    np.sum(
                        weights
                        * np.asarray(
                            [float(row["audit_reprojection_rel_l2"]) for row in subset]
                        )
                    )
                ),
                "mean_improvement_vs_support_pct": float(np.sum(weights * vs_support)),
                "mean_improvement_vs_union_pct": float(np.sum(weights * vs_union)),
                "median_improvement_vs_union_pct": weighted_quantile(vs_union, weights, 0.5),
                "p10_improvement_vs_union_pct": weighted_quantile(vs_union, weights, 0.1),
                "cvar10_improvement_vs_union_pct": weighted_lower_cvar(
                    vs_union, weights, 0.1
                ),
                "harm_rate_gt_1pct_vs_union": float(np.sum(weights * (vs_union < -1.0))),
                "improvement_vs_union_ci95_cluster_low": ci_low,
                "improvement_vs_union_ci95_cluster_high": ci_high,
                "maximum_support_correction_leakage": float(
                    np.max([float(row["support_correction_leakage"]) for row in subset])
                ),
            }
        )
    return output


def summary_row(
    summary: list[dict[str, object]],
    budget: int,
    strategy: str,
    method: str,
) -> dict[str, object]:
    return next(
        row
        for row in summary
        if int(row["total_budget"]) == int(budget)
        and row["query_strategy"] == strategy
        and row["method"] == method
    )


def build_verdicts(
    summary: list[dict[str, object]],
    budgets: list[int],
    minimum_same_budget_gain_pct: float,
    minimum_fixed_retention: float,
) -> list[dict[str, object]]:
    verdicts = []
    for budget in budgets:
        fixed = summary_row(summary, budget, "fixed", "learned_query_correction")
        adaptive = summary_row(summary, budget, "adaptive_energy", "learned_query_correction")
        fixed_gain = float(fixed["mean_improvement_vs_support_pct"])
        adaptive_gain = float(adaptive["mean_improvement_vs_support_pct"])
        retention = fixed_gain / adaptive_gain if adaptive_gain > 1e-12 else None
        fair_pass = (
            float(fixed["improvement_vs_union_ci95_cluster_low"])
            > float(minimum_same_budget_gain_pct)
            and float(fixed["p10_improvement_vs_union_pct"]) >= 0.0
            and float(fixed["harm_rate_gt_1pct_vs_union"]) <= 0.05
        )
        verdicts.append(
            {
                "total_budget": budget,
                "fixed_learned_gain_vs_support_pct": fixed_gain,
                "adaptive_learned_gain_vs_support_pct": adaptive_gain,
                "fixed_over_adaptive_retention": retention,
                "minimum_fixed_retention": float(minimum_fixed_retention),
                "retention_ratio_point_threshold_pass": bool(
                    retention is not None and retention >= minimum_fixed_retention
                ),
                "minimum_same_budget_gain_pct": float(minimum_same_budget_gain_pct),
                "same_budget_union_direct_pass": bool(fair_pass),
                "current_checkpoint_path_pass": bool(
                    fair_pass
                    and retention is not None
                    and retention >= minimum_fixed_retention
                ),
            }
        )
    return verdicts


def plot_same_budget(summary: list[dict[str, object]], path: Path) -> None:
    methods = [
        "union_support_fit_direct",
        "learned_query_correction",
        "numeric_query_null_update",
    ]
    budgets = sorted({int(row["total_budget"]) for row in summary})
    strategies = ["fixed", "max_gap", "adaptive_energy"]
    fig, axes = plt.subplots(1, len(budgets), figsize=(14.4, 4.7), sharey=True, constrained_layout=True)
    colors = ["#5d6b72", "#1c7669", "#ba6648"]
    for ax, budget in zip(np.atleast_1d(axes), budgets):
        x = np.arange(len(strategies))
        width = 0.24
        for method_index, (method, color) in enumerate(zip(methods, colors)):
            values = [
                float(summary_row(summary, budget, strategy, method)["mean_improvement_vs_union_pct"])
                for strategy in strategies
            ]
            ax.bar(
                x + (method_index - 1) * width,
                values,
                width=width,
                color=color,
                label=LABELS[method],
            )
        ax.axhline(0.0, color="#343b3d", linewidth=1)
        ax.set_xticks(x, [value.replace("_", "\n") for value in strategies])
        ax.set_title(f"reconstruction budget K={budget}")
        ax.grid(True, axis="y", alpha=0.22)
    axes[0].set_ylabel("field improvement over S union Q direct (%)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Equal-reconstruction-budget verdict: correction must beat direct use of Q")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_risk(summary: list[dict[str, object]], path: Path) -> None:
    budgets = sorted({int(row["total_budget"]) for row in summary})
    fig, ax = plt.subplots(figsize=(10.8, 5.1), constrained_layout=True)
    x = np.arange(len(budgets))
    width = 0.24
    for offset, strategy in enumerate(["fixed", "max_gap", "adaptive_energy"]):
        values = [
            100.0
            * float(
                summary_row(summary, budget, strategy, "learned_query_correction")[
                    "harm_rate_gt_1pct_vs_union"
                ]
            )
            for budget in budgets
        ]
        ax.bar(x + (offset - 1) * width, values, width=width, label=strategy)
    ax.set_xticks(x, [f"K={budget}" for budget in budgets])
    ax.set_ylabel("fields harmed by >1% vs union direct (%)")
    ax.set_title("A positive mean is insufficient when individual fields are harmed")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "fair_camera_budget_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def main() -> None:
    args = parse_args()
    experiment_config = read_json(args.config)
    base_path = args.config.parent / str(experiment_config["base_experiment_config"])
    corrector_path = args.config.parent / str(
        experiment_config["corrector_experiment_config"]
    )
    base_config = read_json(base_path)
    corrector_config = read_json(corrector_path)
    dataset_config = read_json(base_path.parent / str(base_config["dataset_config"]))
    device = torch.device(args.device or str(experiment_config["device"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT / "results" / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, dataset_path)
    data = load_npz(dataset_path)

    sample_rows: list[dict[str, object]] = []
    for seed in [int(value) for value in base_config["training_seeds"]]:
        base_model = load_base_model(base_config, dataset_config, data, seed, device)
        corrector = load_null_corrector(
            corrector_config,
            int(data["inputs"].shape[1]) + 2,
            seed,
            device,
        )
        seed_rows = evaluate_seed(
            seed,
            base_model,
            corrector,
            data,
            dataset_config,
            corrector_config,
            experiment_config,
            device,
        )
        sample_rows.extend(seed_rows)
        print(f"seed={seed}: evaluated {len(seed_rows)} method rows", flush=True)

    cluster_rows = collapse_model_seeds(sample_rows)
    summary = summarize_clusters(
        cluster_rows,
        int(experiment_config["bootstrap_seed"]),
        int(experiment_config["bootstrap_replicates"]),
    )
    budgets = [int(value) for value in experiment_config["total_budgets"]]
    verdicts = build_verdicts(
        summary,
        budgets,
        float(experiment_config["minimum_same_budget_gain_pct"]),
        float(experiment_config["minimum_fixed_retention"]),
    )
    scientific_status = (
        "PILOT_ONLY_CURRENT_CHECKPOINT_PATH_FAILS"
        if not any(bool(row["current_checkpoint_path_pass"]) for row in verdicts)
        else "PILOT_ONLY_TRAINING_MASK_MISMATCH"
    )
    write_csv(args.output_dir / "fair_camera_budget_samples.csv", sample_rows)
    write_csv(args.output_dir / "fair_camera_budget_clusters.csv", cluster_rows)
    write_csv(args.output_dir / "fair_camera_budget_summary.csv", summary)
    write_csv(args.output_dir / "fair_camera_budget_verdicts.csv", verdicts)
    plot_same_budget(summary, args.output_dir / "t16_fair_camera_budget_verdict.png")
    plot_risk(summary, args.output_dir / "t16_fair_camera_budget_harm.png")

    dashboard_payload = {
        "experiment": experiment_config["name"],
        "independent_test_fields": len(test_indices_and_splits(data)[0]),
        "model_seed_count": len(base_config["training_seeds"]),
        "scientific_status": scientific_status,
        "training_mask_mismatch": bool(experiment_config["training_mask_mismatch"]),
        "methods": METHODS,
        "labels": LABELS,
        "query_strategies": experiment_config["query_strategies"],
        "summary": summary,
        "verdicts": verdicts,
    }
    (args.output_dir / "fair_camera_budget_dashboard.json").write_text(
        json.dumps(dashboard_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "status": "completed_equal_camera_budget_red_team_audit",
        "experiment": experiment_config["name"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "protocol": {
            "budget_rule": "K-1 support views reconstruct; one Q_fit view calibrates or joins direct reconstruction",
            "audit_rule": "one predeclared camera outside support and Q_fit is locked as Q_audit and never fits alpha or selects a method",
            "fixed_query_index": int(experiment_config["fixed_query_index"]),
            "audit_query_index": int(experiment_config["audit_query_index"]),
            "query_strategies": experiment_config["query_strategies"],
            "field_truth_in_inference": False,
            "field_truth_for_retrospective_metrics_only": True,
            "source_split_view_labels_reused_for_fields_only": True,
            "training_mask_matches_controlled_evaluation": not bool(
                experiment_config["training_mask_mismatch"]
            ),
        },
        "design": {
            "independent_test_fields": len(test_indices_and_splits(data)[0]),
            "model_seed_count": len(base_config["training_seeds"]),
            "total_budgets": budgets,
            "query_strategy_count": len(experiment_config["query_strategies"]),
            "methods": METHODS,
            "sample_method_rows": len(sample_rows),
            "cluster_rows_after_seed_collapse": len(cluster_rows),
            "bootstrap_replicates": int(experiment_config["bootstrap_replicates"]),
        },
        "verdicts": verdicts,
        "scientific_status": scientific_status,
        "claims_boundary": [
            "The result is an equal-reconstruction-budget test inside the existing 8x16x16 linear synthetic forward model.",
            "K is the reconstruction-view budget; the locked Q_audit camera is an additional evaluation instrument, so installed-camera count is K+1.",
            "Q_audit is separate from Q_fit, but field truth remains available for retrospective synthetic metrics.",
            "The independent unit is the three-dimensional field; model seeds are collapsed before bootstrap intervals.",
            "The fixed camera is fixed for this canonical nine-view layout and is not yet an OERF hardware placement.",
            "The numerical query-null update is a linear baseline, not a nonlinear ray-geometry solver.",
            "Its ridge and correction cap are fixed diagnostics rather than validation-tuned numerical baselines.",
            "The reused checkpoints were trained with the earlier view-dropout distribution, not the controlled fixed-query masks; a training-matched rerun is required before a final algorithm verdict.",
            "The 88 fields have informed earlier v1-v2c development and therefore remain an exploratory development audit rather than a locked final test.",
            "Passing this audit would not remove the independent-forward and real-data requirements.",
        ],
    }
    (args.output_dir / "fair_camera_budget_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_checksums(
        args.output_dir,
        [
            "fair_camera_budget_samples.csv",
            "fair_camera_budget_clusters.csv",
            "fair_camera_budget_summary.csv",
            "fair_camera_budget_verdicts.csv",
            "fair_camera_budget_dashboard.json",
            "fair_camera_budget_report.json",
        ],
    )
    print(json.dumps({"verdicts": verdicts}, indent=2, allow_nan=False))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
