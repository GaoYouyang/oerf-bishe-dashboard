#!/usr/bin/env python3
"""Post-open SFIO-PAPBB accuracy/operator-budget diagnostic.

The neural shared field is used as a target-observation-free prior and warm
start. Projected BB iterations then enforce the measured source residual. All
selection happens on the existing training split; the v5m extension is scored
once with frozen hyperparameters and remains development evidence.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.data import build_dataset
    from demo_t16_operator.gc_rio.protocol import (
        make_development_config,
        sha256_json,
        sha256_state_dict,
    )
    from demo_t16_operator.run_v5m_shared_field_extension import (
        extension_dataset_config,
    )
    from demo_t16_operator.run_v5n_strong_classical_baselines import (
        BASE_CONFIG_PATH,
        EXTENSION_CONFIG_PATH,
        PBB_ITERATION_GRID,
        SOURCE_REPORT_PATH,
        _correction_map,
        _ensemble_corrections,
        _field_diagnostics,
        _score,
        _source_system,
        _target_predictions,
        projected_bb_correction,
    )
else:
    from .gc_rio.data import build_dataset
    from .gc_rio.protocol import (
        make_development_config,
        sha256_json,
        sha256_state_dict,
    )
    from .run_v5m_shared_field_extension import extension_dataset_config
    from .run_v5n_strong_classical_baselines import (
        BASE_CONFIG_PATH,
        EXTENSION_CONFIG_PATH,
        PBB_ITERATION_GRID,
        SOURCE_REPORT_PATH,
        _correction_map,
        _ensemble_corrections,
        _field_diagnostics,
        _score,
        _source_system,
        _target_predictions,
        projected_bb_correction,
    )


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "v5o_prior_anchored_frontier"
LEGACY_NETWORK_PROXY_PAIRS = 3
ANCHOR_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
REFINEMENT_GRID = PBB_ITERATION_GRID


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def prior_anchored_bb_correction(
    row: Mapping[str, Any],
    prior_correction: np.ndarray,
    *,
    iterations: int,
    relative_anchor: float,
    operator_call_counter: MutableMapping[str, int] | None = None,
) -> np.ndarray:
    """Refine a learned full-field prior with projected source-residual BB."""

    if iterations < 0 or relative_anchor < 0.0:
        raise ValueError("iterations and relative_anchor must be nonnegative")
    operator, residual, support = _source_system(row)
    base = np.asarray(row["base_field"], dtype=np.float64)[support]
    full_prior = np.asarray(prior_correction, dtype=np.float64)
    if full_prior.shape != support.shape or not np.all(np.isfinite(full_prior)):
        raise ValueError("prior_correction must be one finite full field")
    prior = np.maximum(base + full_prior[support], 0.0) - base
    correction = prior.copy()
    mean_diagonal = float(np.mean(np.sum(np.square(operator), axis=0)))
    anchor = float(relative_anchor) * mean_diagonal
    lipschitz = max(float(np.linalg.norm(operator, ord=2) ** 2) + anchor, 1e-10)
    previous_correction: np.ndarray | None = None
    previous_gradient: np.ndarray | None = None
    for _ in range(int(iterations)):
        predicted = operator @ correction
        if operator_call_counter is not None:
            operator_call_counter["source_forward"] = int(
                operator_call_counter.get("source_forward", 0)
            ) + 1
        gradient = operator.T @ (predicted - residual)
        if operator_call_counter is not None:
            operator_call_counter["source_adjoint"] = int(
                operator_call_counter.get("source_adjoint", 0)
            ) + 1
        gradient += anchor * (correction - prior)
        step = 1.0 / lipschitz
        if previous_correction is not None and previous_gradient is not None:
            displacement = correction - previous_correction
            gradient_change = gradient - previous_gradient
            denominator = float(displacement @ gradient_change)
            if denominator > 1e-12:
                step = float(displacement @ displacement) / denominator
                step = float(np.clip(step, 0.05 / lipschitz, 1.8 / lipschitz))
        previous_correction = correction.copy()
        previous_gradient = gradient.copy()
        correction = correction - step * gradient
        correction = np.maximum(base + correction, 0.0) - base
    output = np.zeros(len(support), dtype=np.float32)
    output[support] = correction.astype(np.float32)
    return output


def _score_corrections(
    bundle: Any,
    split: str,
    corrections: Mapping[str, np.ndarray],
) -> tuple[float, tuple[dict[str, Any], ...], np.ndarray, list[int], np.ndarray]:
    indices, prediction = _target_predictions(bundle, split, corrections)
    aggregate, cells, row_metric = _score(bundle, indices, prediction)
    return aggregate, cells, row_metric, indices, prediction


def _select_on_train(
    bundle: Any,
    priors: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for anchor in ANCHOR_GRID:
        for iterations in REFINEMENT_GRID:
            corrections = _correction_map(
                bundle,
                "train",
                lambda row, a=anchor, count=iterations: prior_anchored_bb_correction(
                    row,
                    priors[str(row["field_uid"])],
                    iterations=count,
                    relative_anchor=a,
                ),
            )
            aggregate, cells, _, _, _ = _score_corrections(
                bundle, "train", corrections
            )
            record = {
                "relative_anchor": anchor,
                "refinement_iterations": iterations,
                "estimated_forward_adjoint_pairs": LEGACY_NETWORK_PROXY_PAIRS
                + iterations,
                "train_cluster_mean_whitened_rmse": aggregate,
                "train_positive_cell_fraction_vs_zero": float(
                    np.mean(
                        [
                            cell["mean_whitened_rmse"]
                            < _zero_cell_metric(bundle, "train", cell)
                            for cell in cells
                        ]
                    )
                ),
            }
            rows.append(record)
            if best is None or aggregate < best["train_cluster_mean_whitened_rmse"]:
                best = record
    assert best is not None
    return dict(best), rows


def _zero_cell_metric(bundle: Any, split: str, cell: Mapping[str, Any]) -> float:
    indices = [
        int(row["row_index"])
        for row in bundle.rows
        if row["split"] == split
        and row["rig_id"] == cell["rig_id"]
        and row["family"] == cell["family"]
    ]
    zeros = np.zeros(
        (len(indices), bundle.rows[indices[0]]["target_operator"].shape[0]),
        dtype=np.float32,
    )
    _, cells, _ = _score(bundle, indices, zeros)
    return float(cells[0]["mean_whitened_rmse"])


def _method_summary(
    bundle: Any,
    methods: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[
    dict[str, float],
    dict[str, tuple[dict[str, Any], ...]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    aggregates: dict[str, float] = {}
    cells: dict[str, tuple[dict[str, Any], ...]] = {}
    predictions: dict[str, np.ndarray] = {}
    row_metrics: dict[str, np.ndarray] = {}
    for method, corrections in methods.items():
        aggregate, method_cells, metric, _, prediction = _score_corrections(
            bundle, "validation", corrections
        )
        aggregates[method] = aggregate
        cells[method] = method_cells
        predictions[method] = prediction
        row_metrics[method] = metric
    return aggregates, cells, predictions, row_metrics


def _cell_comparison(
    cells: Mapping[str, tuple[dict[str, Any], ...]],
    candidate: str,
    baseline: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, candidate_cell in enumerate(cells[candidate]):
        candidate_value = float(candidate_cell["mean_whitened_rmse"])
        baseline_value = float(cells[baseline][index]["mean_whitened_rmse"])
        rows.append(
            {
                "rig_id": candidate_cell["rig_id"],
                "family": candidate_cell["family"],
                "row_count": candidate_cell["row_count"],
                "candidate": candidate_value,
                "baseline": baseline_value,
                "candidate_gain": 1.0
                - candidate_value / max(baseline_value, 1e-12),
            }
        )
    return rows


def run() -> dict[str, Any]:
    base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION_CONFIG_PATH.read_text(encoding="utf-8"))
    source_report = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8"))
    train_bundle = build_dataset(make_development_config(base))
    train_priors, train_hashes = _ensemble_corrections(
        base, train_bundle, "train", source_report
    )
    selected, selection_rows = _select_on_train(train_bundle, train_priors)

    extension_bundle = build_dataset(extension_dataset_config(base, extension))
    extension_priors, extension_hashes = _ensemble_corrections(
        base, extension_bundle, "validation", source_report
    )
    selected_iterations = int(selected["refinement_iterations"])
    selected_anchor = float(selected["relative_anchor"])
    matched_iterations = LEGACY_NETWORK_PROXY_PAIRS + selected_iterations
    methods = {
        "zero_correction": _correction_map(
            extension_bundle,
            "validation",
            lambda row: np.zeros_like(row["base_field"]),
        ),
        "shared_field_ensemble": extension_priors,
        "pure_pbb_32": _correction_map(
            extension_bundle,
            "validation",
            lambda row: projected_bb_correction(row, 32),
        ),
        "pure_pbb_budget_matched": _correction_map(
            extension_bundle,
            "validation",
            lambda row: projected_bb_correction(row, matched_iterations),
        ),
        "sfio_papbb_selected": _correction_map(
            extension_bundle,
            "validation",
            lambda row: prior_anchored_bb_correction(
                row,
                extension_priors[str(row["field_uid"])],
                iterations=selected_iterations,
                relative_anchor=selected_anchor,
            ),
        ),
    }
    aggregates, cells, predictions, _ = _method_summary(extension_bundle, methods)
    cell_rows = _cell_comparison(
        cells, "sfio_papbb_selected", "pure_pbb_budget_matched"
    )
    gains = np.asarray([row["candidate_gain"] for row in cell_rows])

    frontier_rows: list[dict[str, Any]] = []
    for refinement in REFINEMENT_GRID:
        total_pairs = LEGACY_NETWORK_PROXY_PAIRS + int(refinement)
        warm = _correction_map(
            extension_bundle,
            "validation",
            lambda row, count=refinement: prior_anchored_bb_correction(
                row,
                extension_priors[str(row["field_uid"])],
                iterations=count,
                relative_anchor=selected_anchor,
            ),
        )
        pure = _correction_map(
            extension_bundle,
            "validation",
            lambda row, count=total_pairs: projected_bb_correction(row, count),
        )
        warm_metric, _, _, _, _ = _score_corrections(
            extension_bundle, "validation", warm
        )
        pure_metric, _, _, _, _ = _score_corrections(
            extension_bundle, "validation", pure
        )
        frontier_rows.append(
            {
                "refinement_iterations": refinement,
                "estimated_forward_adjoint_pairs": total_pairs,
                "relative_anchor": selected_anchor,
                "sfio_papbb_cluster_mean_whitened_rmse": warm_metric,
                "pure_pbb_cluster_mean_whitened_rmse": pure_metric,
                "sfio_papbb_gain_vs_budget_matched_pbb": 1.0
                - warm_metric / max(pure_metric, 1e-12),
            }
        )

    field_rows = _field_diagnostics(extension_bundle, methods)
    field_summary = {}
    for method in methods:
        values = [row for row in field_rows if row["method"] == method]
        field_summary[method] = {
            "mean_relative_l2": float(np.mean([row["relative_l2"] for row in values])),
            "mean_gain_vs_base": float(np.mean([row["gain_vs_base"] for row in values])),
            "better_fraction": float(
                np.mean([row["gain_vs_base"] > 0.0 for row in values])
            ),
        }

    report = {
        "schema": "v5o-sfio-papbb-frontier-1",
        "evidence_label": "adaptive_postopen_development_hybrid_frontier",
        "base_config_sha256": sha256_json(base),
        "extension_config_sha256": sha256_json(extension),
        "checkpoint_hashes": {
            "train": train_hashes,
            "extension": extension_hashes,
        },
        "selection": {
            **selected,
            "selection_split": "train",
            "target_labels_used_for_offline_selection": True,
            "extension_labels_not_used_for_selection": True,
        },
        "extension_target": {
            "cluster_mean_whitened_rmse": aggregates,
            "candidate_ratio_of_cluster_means_gain_vs_budget_matched_pbb": 1.0
            - aggregates["sfio_papbb_selected"]
            / aggregates["pure_pbb_budget_matched"],
            "candidate_cell_mean_gain_vs_budget_matched_pbb": float(np.mean(gains)),
            "candidate_positive_cell_fraction": float(np.mean(gains > 0.0)),
            "candidate_worst_cell_degradation": max(0.0, -float(np.min(gains))),
            "cells": cell_rows,
        },
        "extension_field_truth_diagnostic": field_summary,
        "frontier": frontier_rows,
        "prediction_hashes": {
            method: sha256_state_dict({"target_residual_prediction": prediction})
            for method, prediction in predictions.items()
        },
        "operator_budget_note": "V5O preserves the historical 3+k proxy used before call instrumentation. A later audit found that the adjoint-only checkpoints can reuse one source adjoint/Fisher statistic across seeds and need not execute discarded Krylov stacks. V5P therefore replaces this proxy with exact forward/adjoint counters; CNN cost and wall-clock remain separate.",
        "decision": "NO_DESIGN_LOCK_OPEN",
        "decision_reason": "This is an adaptive post-open diagnostic on previously inspected extension data; design-lock data are not constructed or opened.",
        "limitations": [
            "The hybrid mechanism and grid were specified after v5n exposed the PBB gap.",
            "Training labels select anchor strength and iterations; this is not unsupervised training.",
            "Three Krylov pairs are only an operator-call proxy for neural feature cost, not wall-clock equivalence.",
            "Scalar camera sigma does not whiten correlated or signal-dependent noise.",
            "Rows sharing one field are dependent; field and rig-family are the reporting units.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "train_selection.csv", selection_rows)
    _write_csv(OUTPUT_DIR / "extension_frontier.csv", frontier_rows)
    _write_csv(OUTPUT_DIR / "extension_cells.csv", cell_rows)
    _write_csv(OUTPUT_DIR / "extension_field_rows.csv", field_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "selection": report["selection"],
                "target": report["extension_target"],
                "field": report["extension_field_truth_diagnostic"],
                "frontier": report["frontier"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
