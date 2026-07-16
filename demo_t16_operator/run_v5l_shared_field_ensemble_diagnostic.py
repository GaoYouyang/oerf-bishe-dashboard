#!/usr/bin/env python3
"""Adaptive post-open ensemble and 3D-field diagnosis for v5k checkpoints."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.data import build_dataset
    from demo_t16_operator.gc_rio.protocol import make_development_config, sha256_file, sha256_state_dict
    from demo_t16_operator.gc_rio.training import cluster_mean_metric, predictor_tensors, row_whitened_rmse
    from demo_t16_operator.run_v5h_gc_rio_development import _model
else:
    from .gc_rio.data import build_dataset
    from .gc_rio.protocol import make_development_config, sha256_file, sha256_state_dict
    from .gc_rio.training import cluster_mean_metric, predictor_tensors, row_whitened_rmse
    from .run_v5h_gc_rio_development import _model


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5h_gc_rio_development.json"
SOURCE_REPORT = ROOT / "results" / "v5k_shared_field_development" / "report.json"
WORK_DIR = ROOT / "results" / "v5k_shared_field_work"
OUTPUT_DIR = ROOT / "results" / "v5l_shared_field_ensemble_diagnostic"
SEEDS = (3101, 3102, 3103)


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


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(prediction) - np.asarray(target))
        / max(np.linalg.norm(target), 1e-12)
    )


def _gradient_relative_l2(
    prediction: np.ndarray, target: np.ndarray, shape: tuple[int, int, int]
) -> float:
    predicted_gradients = np.concatenate(
        [item.reshape(-1) for item in np.gradient(prediction.reshape(shape))]
    )
    target_gradients = np.concatenate(
        [item.reshape(-1) for item in np.gradient(target.reshape(shape))]
    )
    return _relative_l2(predicted_gradients, target_gradients)


def _predict_corrections(
    model: torch.nn.Module,
    bundle: Any,
    indices: Sequence[int],
    *,
    batch_size: int,
) -> np.ndarray:
    corrections: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            selected = indices[start : start + int(batch_size)]
            tensors = predictor_tensors(bundle.predictor_batch(selected), "cpu")
            corrections.append(model(**tensors).correction.numpy())
    return np.concatenate(corrections, axis=0)


def run() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    bundle = build_dataset(make_development_config(config))
    if any(row["split"] == "design_lock" for row in bundle.rows):
        raise RuntimeError("design-lock rows must remain unconstructed")
    recorded = {
        (row["method"], int(row["model_seed"])): row["checkpoint_sha256"]
        for row in source_report["training_records"]
    }
    models = []
    checkpoint_hashes = {}
    for seed in SEEDS:
        checkpoint = WORK_DIR / "sf_rio_adjoint_only" / str(seed) / "best.pt"
        digest = sha256_file(checkpoint)
        if digest != recorded[("sf_rio_adjoint_only", seed)]:
            raise RuntimeError(f"checkpoint hash drift for seed {seed}")
        model = _model(
            config,
            candidate="shared_field",
            use_target_geometry=False,
            seed=seed,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        models.append(model)
        checkpoint_hashes[str(seed)] = digest

    target_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    prediction_hashes = {}
    batch_size = int(config["training"]["batch_size"])
    shape = (int(config["depth"]), int(config["grid_size"]), int(config["grid_size"]))
    for split in ("train", "validation"):
        indices = [
            int(row["row_index"]) for row in bundle.rows if row["split"] == split
        ]
        correction_members = [
            _predict_corrections(
                model, bundle, indices, batch_size=batch_size
            )
            for model in models
        ]
        ensemble_correction = np.mean(correction_members, axis=0)
        target_prediction = np.stack(
            [
                bundle.rows[index]["target_operator"] @ ensemble_correction[position]
                for position, index in enumerate(indices)
            ]
        )
        prediction_hashes[split] = sha256_state_dict(
            {
                "ensemble_correction": ensemble_correction,
                "target_residual_prediction": target_prediction,
            }
        )
        labels = np.stack(
            [bundle.rows[index]["target_residual_label"] for index in indices]
        )
        sigma = np.asarray([bundle.rows[index]["target_sigma"] for index in indices])
        metric = row_whitened_rmse(target_prediction, labels, sigma)
        zero_metric = row_whitened_rmse(np.zeros_like(target_prediction), labels, sigma)
        for position, index in enumerate(indices):
            row = bundle.rows[index]
            target_rows.append(
                {
                    "split": split,
                    "rig_id": row["rig_id"],
                    "family": row["family"],
                    "field_uid": row["field_uid"],
                    "target_view": int(row["target_view"]),
                    "ensemble_whitened_rmse": float(metric[position]),
                    "zero_whitened_rmse": float(zero_metric[position]),
                    "relative_gain_vs_zero": float(
                        1.0 - metric[position] / max(zero_metric[position], 1e-12)
                    ),
                }
            )
        first_positions: dict[str, int] = {}
        for position, index in enumerate(indices):
            first_positions.setdefault(str(bundle.rows[index]["field_uid"]), position)
        for field_uid, position in sorted(first_positions.items()):
            row = bundle.rows[indices[position]]
            truth = bundle.truth_fields[field_uid]
            base = row["base_field"]
            analytic = base + row["analytic_correction"]
            predicted = base + ensemble_correction[position]
            field_rows.append(
                {
                    "split": split,
                    "rig_id": row["rig_id"],
                    "family": row["family"],
                    "field_uid": field_uid,
                    "base_relative_l2": _relative_l2(base, truth),
                    "analytic_relative_l2": _relative_l2(analytic, truth),
                    "ensemble_relative_l2": _relative_l2(predicted, truth),
                    "ensemble_field_gain_vs_base": float(
                        1.0
                        - _relative_l2(predicted, truth)
                        / max(_relative_l2(base, truth), 1e-12)
                    ),
                    "base_gradient_relative_l2": _gradient_relative_l2(
                        base, truth, shape
                    ),
                    "ensemble_gradient_relative_l2": _gradient_relative_l2(
                        predicted, truth, shape
                    ),
                }
            )

    validation_target = [row for row in target_rows if row["split"] == "validation"]
    ensemble_values = np.asarray(
        [row["ensemble_whitened_rmse"] for row in validation_target]
    )
    zero_values = np.asarray([row["zero_whitened_rmse"] for row in validation_target])
    rigs = [str(row["rig_id"]) for row in validation_target]
    families = [str(row["family"]) for row in validation_target]
    ensemble_mean, ensemble_cells = cluster_mean_metric(
        ensemble_values, rigs, families
    )
    zero_mean, zero_cells = cluster_mean_metric(zero_values, rigs, families)
    cell_rows = []
    for ensemble_cell, zero_cell in zip(ensemble_cells, zero_cells, strict=True):
        gain = 1.0 - float(ensemble_cell["mean_whitened_rmse"]) / float(
            zero_cell["mean_whitened_rmse"]
        )
        cell_rows.append(
            {
                "rig_id": ensemble_cell["rig_id"],
                "family": ensemble_cell["family"],
                "row_count": ensemble_cell["row_count"],
                "ensemble_whitened_rmse": ensemble_cell["mean_whitened_rmse"],
                "zero_whitened_rmse": zero_cell["mean_whitened_rmse"],
                "relative_gain_vs_zero": gain,
            }
        )
    validation_fields = [row for row in field_rows if row["split"] == "validation"]
    report = {
        "schema": "v5l-shared-field-postopen-ensemble-1",
        "evidence_label": "adaptive_postopen_development_diagnostic",
        "source_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "checkpoint_hashes": checkpoint_hashes,
        "prediction_hashes_before_scoring": prediction_hashes,
        "ensemble_rule": "unweighted mean of the three pre-existing seed predictions",
        "selection_note": "The ensemble was evaluated after inspecting v5k and is not preregistered evidence.",
        "design_lock_rows_constructed": 0,
        "validation_target_summary": {
            "ensemble_cluster_mean_whitened_rmse": ensemble_mean,
            "zero_cluster_mean_whitened_rmse": zero_mean,
            "cell_mean_relative_gain_vs_zero": float(
                np.mean([row["relative_gain_vs_zero"] for row in cell_rows])
            ),
            "positive_cells": int(
                sum(row["relative_gain_vs_zero"] > 0.0 for row in cell_rows)
            ),
            "cell_count": len(cell_rows),
            "cells": cell_rows,
        },
        "validation_field_truth_diagnostic": {
            "field_count": len(validation_fields),
            "mean_base_relative_l2": float(
                np.mean([row["base_relative_l2"] for row in validation_fields])
            ),
            "mean_ensemble_relative_l2": float(
                np.mean([row["ensemble_relative_l2"] for row in validation_fields])
            ),
            "mean_field_gain_vs_base": float(
                np.mean([row["ensemble_field_gain_vs_base"] for row in validation_fields])
            ),
            "field_better_fraction": float(
                np.mean([row["ensemble_field_gain_vs_base"] > 0.0 for row in validation_fields])
            ),
            "mean_base_gradient_relative_l2": float(
                np.mean(
                    [row["base_gradient_relative_l2"] for row in validation_fields]
                )
            ),
            "mean_ensemble_gradient_relative_l2": float(
                np.mean(
                    [
                        row["ensemble_gradient_relative_l2"]
                        for row in validation_fields
                    ]
                )
            ),
        },
        "claim_boundary": [
            "The ensemble and truth-field metrics were chosen after v5k was opened.",
            "Validation truth is diagnostic only and cannot select a confirmatory method.",
            "No design-lock rig or label was constructed.",
            "All data remain synthetic and do not establish OERF or experimental performance.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "target_rows.csv", target_rows)
    _write_csv(OUTPUT_DIR / "field_rows.csv", field_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "target": report["validation_target_summary"],
                "field": report["validation_field_truth_diagnostic"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
