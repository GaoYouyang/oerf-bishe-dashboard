#!/usr/bin/env python3
"""Strong source-only numerical baselines for the v5m shared-field extension."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.data import build_dataset
    from demo_t16_operator.gc_rio.protocol import make_development_config, sha256_file, sha256_json, sha256_state_dict
    from demo_t16_operator.gc_rio.training import cluster_mean_metric, predictor_tensors, row_whitened_rmse
    from demo_t16_operator.run_v5h_gc_rio_development import _model
    from demo_t16_operator.run_v5m_shared_field_extension import extension_dataset_config
else:
    from .gc_rio.data import build_dataset
    from .gc_rio.protocol import make_development_config, sha256_file, sha256_json, sha256_state_dict
    from .gc_rio.training import cluster_mean_metric, predictor_tensors, row_whitened_rmse
    from .run_v5h_gc_rio_development import _model
    from .run_v5m_shared_field_extension import extension_dataset_config


ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "configs" / "v5h_gc_rio_development.json"
EXTENSION_CONFIG_PATH = ROOT / "configs" / "v5m_shared_field_extension.json"
SOURCE_REPORT_PATH = ROOT / "results" / "v5k_shared_field_development" / "report.json"
WORK_DIR = ROOT / "results" / "v5k_shared_field_work"
OUTPUT_DIR = ROOT / "results" / "v5n_strong_classical_baselines"
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
PBB_ITERATION_GRID = (0, 1, 2, 4, 8, 16, 32)


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


def _source_system(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    operator = np.asarray(row["source_operator"], dtype=np.float64)
    residual = np.asarray(row["source_residual"], dtype=np.float64)
    sigma = np.asarray(row["source_sigma"], dtype=np.float64)
    support = np.asarray(row["support"], dtype=bool)
    whitened_operator = (operator / sigma[:, None, None]).reshape(-1, operator.shape[-1])
    whitened_residual = (residual / sigma[:, None]).reshape(-1)
    return whitened_operator[:, support], whitened_residual, support


def support_ridge_correction(
    row: Mapping[str, Any], relative_ridge: float
) -> np.ndarray:
    """Solve one global relative-Tikhonov source correction."""

    operator, residual, support = _source_system(row)
    mean_diagonal = float(np.mean(np.sum(np.square(operator), axis=0)))
    ridge = max(float(relative_ridge) * mean_diagonal, 1e-10)
    normal = operator.T @ operator + ridge * np.eye(operator.shape[1])
    active = np.linalg.solve(normal, operator.T @ residual)
    correction = np.zeros(len(support), dtype=np.float32)
    correction[support] = active.astype(np.float32)
    return correction


def projected_bb_correction(
    row: Mapping[str, Any],
    iterations: int,
    *,
    initial_correction: np.ndarray | None = None,
    operator_call_counter: MutableMapping[str, int] | None = None,
) -> np.ndarray:
    """Projected Barzilai-Borwein iterations on the source residual objective."""

    operator, residual, support = _source_system(row)
    base = np.asarray(row["base_field"], dtype=np.float64)[support]
    if initial_correction is None:
        correction = np.zeros(operator.shape[1], dtype=np.float64)
    else:
        initial = np.asarray(initial_correction, dtype=np.float64)
        if initial.shape != support.shape or not np.all(np.isfinite(initial)):
            raise ValueError("initial_correction must be one finite full field")
        correction = np.maximum(base + initial[support], 0.0) - base
    if int(iterations) == 0:
        output = np.zeros(len(support), dtype=np.float32)
        output[support] = correction.astype(np.float32)
        return output
    lipschitz = max(float(np.linalg.norm(operator, ord=2) ** 2), 1e-10)
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


def _unique_fields(bundle: Any, split: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in bundle.rows:
        if row["split"] == split:
            output.setdefault(str(row["field_uid"]), row)
    return output


def _correction_map(
    bundle: Any,
    split: str,
    solver: Callable[[Mapping[str, Any]], np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        field_uid: solver(row)
        for field_uid, row in _unique_fields(bundle, split).items()
    }


def _target_predictions(
    bundle: Any, split: str, corrections: Mapping[str, np.ndarray]
) -> tuple[list[int], np.ndarray]:
    indices = [int(row["row_index"]) for row in bundle.rows if row["split"] == split]
    prediction = np.stack(
        [
            bundle.rows[index]["target_operator"]
            @ corrections[str(bundle.rows[index]["field_uid"])]
            for index in indices
        ]
    )
    return indices, prediction


def _score(
    bundle: Any, indices: Sequence[int], prediction: np.ndarray
) -> tuple[float, tuple[dict[str, Any], ...], np.ndarray]:
    labels = np.stack(
        [bundle.rows[int(index)]["target_residual_label"] for index in indices]
    )
    sigma = np.asarray(
        [bundle.rows[int(index)]["target_sigma"] for index in indices]
    )
    row_metric = row_whitened_rmse(prediction, labels, sigma)
    aggregate, cells = cluster_mean_metric(
        row_metric,
        [str(bundle.rows[int(index)]["rig_id"]) for index in indices],
        [str(bundle.rows[int(index)]["family"]) for index in indices],
    )
    return aggregate, cells, row_metric


def _select_classical_hyperparameters(bundle: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_ridge = None
    best_ridge_metric = float("inf")
    for value in RIDGE_GRID:
        corrections = _correction_map(
            bundle,
            "train",
            lambda row, relative=value: support_ridge_correction(row, relative),
        )
        indices, prediction = _target_predictions(bundle, "train", corrections)
        metric, _, _ = _score(bundle, indices, prediction)
        rows.append(
            {
                "method": "support_ridge",
                "hyperparameter": value,
                "train_cluster_mean_whitened_rmse": metric,
            }
        )
        if metric < best_ridge_metric:
            best_ridge_metric, best_ridge = metric, value
    best_iterations = None
    best_pbb_metric = float("inf")
    for value in PBB_ITERATION_GRID:
        corrections = _correction_map(
            bundle,
            "train",
            lambda row, count=value: projected_bb_correction(row, count),
        )
        indices, prediction = _target_predictions(bundle, "train", corrections)
        metric, _, _ = _score(bundle, indices, prediction)
        rows.append(
            {
                "method": "projected_bb",
                "hyperparameter": value,
                "train_cluster_mean_whitened_rmse": metric,
            }
        )
        if metric < best_pbb_metric:
            best_pbb_metric, best_iterations = metric, value
    return (
        {
            "ridge_relative_lambda": best_ridge,
            "ridge_train_metric": best_ridge_metric,
            "pbb_iterations": best_iterations,
            "pbb_train_metric": best_pbb_metric,
            "selection_split": "train",
            "target_labels_used_for_offline_selection": True,
        },
        rows,
    )


def _ensemble_corrections(
    base_config: Mapping[str, Any], bundle: Any, split: str, report: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    recorded = {
        (row["method"], int(row["model_seed"])): row["checkpoint_sha256"]
        for row in report["training_records"]
    }
    indices = [int(row["row_index"]) for row in bundle.rows if row["split"] == split]
    field_positions: dict[str, int] = {}
    for position, index in enumerate(indices):
        field_positions.setdefault(str(bundle.rows[index]["field_uid"]), position)
    members: list[np.ndarray] = []
    hashes = {}
    for seed in (3101, 3102, 3103):
        checkpoint = WORK_DIR / "sf_rio_adjoint_only" / str(seed) / "best.pt"
        digest = sha256_file(checkpoint)
        if digest != recorded[("sf_rio_adjoint_only", seed)]:
            raise RuntimeError(f"checkpoint drift for seed {seed}")
        model = _model(
            base_config,
            candidate="shared_field",
            use_target_geometry=False,
            seed=seed,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        values: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(indices), 16):
                selected = indices[start : start + 16]
                tensors = predictor_tensors(bundle.predictor_batch(selected), "cpu")
                values.append(model(**tensors).correction.numpy())
        members.append(np.concatenate(values, axis=0))
        hashes[str(seed)] = digest
    mean = np.mean(members, axis=0)
    return (
        {field_uid: mean[position] for field_uid, position in field_positions.items()},
        hashes,
    )


def _field_diagnostics(
    bundle: Any,
    corrections_by_method: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = _unique_fields(bundle, "validation")
    for field_uid, row in fields.items():
        truth = bundle.truth_fields[field_uid]
        base = row["base_field"]
        base_error = float(
            np.linalg.norm(base - truth) / max(np.linalg.norm(truth), 1e-12)
        )
        for method, corrections in corrections_by_method.items():
            prediction = base + corrections[field_uid]
            error = float(
                np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-12)
            )
            rows.append(
                {
                    "method": method,
                    "rig_id": row["rig_id"],
                    "family": row["family"],
                    "field_uid": field_uid,
                    "relative_l2": error,
                    "gain_vs_base": 1.0 - error / max(base_error, 1e-12),
                }
            )
    return rows


def run() -> dict[str, Any]:
    base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION_CONFIG_PATH.read_text(encoding="utf-8"))
    source_report = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8"))
    train_bundle = build_dataset(make_development_config(base))
    selected, selection_rows = _select_classical_hyperparameters(train_bundle)
    extension_bundle = build_dataset(extension_dataset_config(base, extension))

    corrections = {
        "zero_correction": _correction_map(
            extension_bundle,
            "validation",
            lambda row: np.zeros_like(row["base_field"]),
        ),
        "selected_support_ridge": _correction_map(
            extension_bundle,
            "validation",
            lambda row: support_ridge_correction(
                row, float(selected["ridge_relative_lambda"])
            ),
        ),
        "selected_projected_bb": _correction_map(
            extension_bundle,
            "validation",
            lambda row: projected_bb_correction(
                row, int(selected["pbb_iterations"])
            ),
        ),
    }
    ensemble, checkpoint_hashes = _ensemble_corrections(
        base, extension_bundle, "validation", source_report
    )
    corrections["shared_field_ensemble"] = ensemble

    predictions = {}
    indices = None
    prediction_hashes = {}
    for method, method_corrections in corrections.items():
        method_indices, prediction = _target_predictions(
            extension_bundle, "validation", method_corrections
        )
        if indices is None:
            indices = method_indices
        elif indices != method_indices:
            raise RuntimeError("method row order mismatch")
        predictions[method] = prediction
        prediction_hashes[method] = sha256_state_dict(
            {"target_residual_prediction": prediction}
        )
    assert indices is not None

    method_cells = {}
    method_aggregates = {}
    row_metrics = {}
    for method, prediction in predictions.items():
        aggregate, cells, metric = _score(extension_bundle, indices, prediction)
        method_cells[method] = cells
        method_aggregates[method] = aggregate
        row_metrics[method] = metric
    cell_rows = []
    for cell_index in range(len(method_cells["shared_field_ensemble"])):
        values = {
            method: float(cells[cell_index]["mean_whitened_rmse"])
            for method, cells in method_cells.items()
        }
        strongest_method = min(
            ("zero_correction", "selected_support_ridge", "selected_projected_bb"),
            key=lambda method: values[method],
        )
        baseline = values[strongest_method]
        ensemble_value = values["shared_field_ensemble"]
        reference = method_cells["shared_field_ensemble"][cell_index]
        cell_rows.append(
            {
                "rig_id": reference["rig_id"],
                "family": reference["family"],
                "row_count": reference["row_count"],
                **values,
                "strongest_classical_method": strongest_method,
                "ensemble_gain_vs_strongest_classical": 1.0
                - ensemble_value / max(baseline, 1e-12),
            }
        )
    gains = np.asarray(
        [row["ensemble_gain_vs_strongest_classical"] for row in cell_rows]
    )
    strongest_global_method = min(
        ("zero_correction", "selected_support_ridge", "selected_projected_bb"),
        key=lambda method: method_aggregates[method],
    )
    ratio_of_cluster_means = 1.0 - method_aggregates[
        "shared_field_ensemble"
    ] / method_aggregates[strongest_global_method]
    field_rows = _field_diagnostics(extension_bundle, corrections)
    field_summary = {}
    for method in corrections:
        values = [row for row in field_rows if row["method"] == method]
        field_summary[method] = {
            "mean_relative_l2": float(np.mean([row["relative_l2"] for row in values])),
            "mean_gain_vs_base": float(np.mean([row["gain_vs_base"] for row in values])),
            "better_fraction": float(np.mean([row["gain_vs_base"] > 0.0 for row in values])),
        }
    report = {
        "schema": "v5n-source-only-strong-classical-baselines-1",
        "evidence_label": "adaptive_postopen_development_strong_baseline",
        "base_config_sha256": sha256_json(base),
        "extension_config_sha256": sha256_json(extension),
        "checkpoint_hashes": checkpoint_hashes,
        "selected_on_train_only": selected,
        "prediction_hashes_before_extension_scoring": prediction_hashes,
        "operator_budget": {
            "projected_bb_forward_adjoint_pairs": int(selected["pbb_iterations"]),
            "support_ridge": "one direct normal-equation solve",
            "neural": "one source-statistics construction plus one CNN and target decode",
            "wall_clock_matched": False,
        },
        "extension_target": {
            "cluster_mean_whitened_rmse": method_aggregates,
            "strongest_global_classical_method": strongest_global_method,
            "ensemble_ratio_of_cluster_means_gain": ratio_of_cluster_means,
            "cell_mean_gain_vs_strongest_classical": float(np.mean(gains)),
            "positive_cell_fraction": float(np.mean(gains > 0.0)),
            "worst_cell_degradation": max(0.0, -float(np.min(gains))),
            "cells": cell_rows,
        },
        "extension_field_truth_diagnostic": field_summary,
        "decision": "NO_DESIGN_LOCK_OPEN",
        "decision_reason": "This adaptive comparison is not preregistered, wall-clock is unmatched, and the design-lock remains closed regardless of the observed ranking.",
        "limitations": [
            "Classical hyperparameters use train target labels, matching offline supervision but not a target-free training claim.",
            "The scalar camera sigma does not whiten correlated or signal-dependent noise.",
            "Rows sharing a field are not independent; field and rig are the reporting units.",
            "The extension rigs and topologies were designed after inspecting earlier development results.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "train_selection.csv", selection_rows)
    _write_csv(OUTPUT_DIR / "extension_cells.csv", cell_rows)
    _write_csv(OUTPUT_DIR / "extension_field_rows.csv", field_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "selection": report["selected_on_train_only"],
                "target": report["extension_target"],
                "field": report["extension_field_truth_diagnostic"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
