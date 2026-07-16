#!/usr/bin/env python3
"""No-retraining new-topology/rig extension for the v5k shared-field ensemble."""

from __future__ import annotations

import copy
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
    from demo_t16_operator.gc_rio.protocol import sha256_file, sha256_json, sha256_state_dict
    from demo_t16_operator.gc_rio.training import cluster_mean_metric, predictor_tensors, row_whitened_rmse
    from demo_t16_operator.run_v5h_gc_rio_development import _model
else:
    from .gc_rio.data import build_dataset
    from .gc_rio.protocol import sha256_file, sha256_json, sha256_state_dict
    from .gc_rio.training import cluster_mean_metric, predictor_tensors, row_whitened_rmse
    from .run_v5h_gc_rio_development import _model


ROOT = Path(__file__).resolve().parent
EXTENSION_PATH = ROOT / "configs" / "v5m_shared_field_extension.json"
WORK_DIR = ROOT / "results" / "v5k_shared_field_work"
OUTPUT_DIR = ROOT / "results" / "v5m_shared_field_extension"


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


def extension_dataset_config(
    base: Mapping[str, Any], extension: Mapping[str, Any]
) -> dict[str, Any]:
    """Construct only the new validation rigs and topologies."""

    config = copy.deepcopy(dict(base))
    config["seed"] = int(extension["extension_seed"])
    config["fields_per_family"] = int(extension["fields_per_family"])
    config["rigs"] = copy.deepcopy(extension["rigs"])
    config["splits"] = {
        "train": {"families": []},
        "validation": {"families": list(extension["families"])},
        "design_lock": {"families": []},
    }
    config["families"] = []
    return config


def _predict_corrections(
    model: torch.nn.Module, bundle: Any, indices: Sequence[int], batch_size: int
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            tensors = predictor_tensors(bundle.predictor_batch(selected), "cpu")
            values.append(model(**tensors).correction.numpy())
    return np.concatenate(values, axis=0)


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(prediction) - np.asarray(target))
        / max(np.linalg.norm(target), 1e-12)
    )


def run() -> dict[str, Any]:
    extension = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
    base_path = ROOT / extension["source_config"]
    report_path = ROOT / extension["source_report"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    config = extension_dataset_config(base, extension)
    bundle = build_dataset(config)
    if {row["split"] for row in bundle.rows} != {"validation"}:
        raise RuntimeError("extension may construct validation rows only")
    if set(bundle.manifest["split_rig_sets"]["validation"]) != {
        rig["id"] for rig in extension["rigs"]
    }:
        raise RuntimeError("extension rig manifest mismatch")
    if set(bundle.manifest["split_family_sets"]["validation"]) != set(
        extension["families"]
    ):
        raise RuntimeError("extension family manifest mismatch")

    recorded = {
        (row["method"], int(row["model_seed"])): row["checkpoint_sha256"]
        for row in source_report["training_records"]
    }
    seeds = [int(seed) for seed in extension["ensemble_seeds"]]
    models = []
    checkpoint_hashes = {}
    for seed in seeds:
        checkpoint = WORK_DIR / extension["source_method"] / str(seed) / "best.pt"
        digest = sha256_file(checkpoint)
        if digest != recorded[(extension["source_method"], seed)]:
            raise RuntimeError(f"checkpoint hash drift for seed {seed}")
        model = _model(
            base,
            candidate="shared_field",
            use_target_geometry=False,
            seed=seed,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        models.append(model)
        checkpoint_hashes[str(seed)] = digest

    indices = [int(row["row_index"]) for row in bundle.rows]
    corrections = [
        _predict_corrections(
            model, bundle, indices, int(base["training"]["batch_size"])
        )
        for model in models
    ]
    ensemble_correction = np.mean(corrections, axis=0)
    target_prediction = np.stack(
        [
            bundle.rows[index]["target_operator"] @ ensemble_correction[position]
            for position, index in enumerate(indices)
        ]
    )
    prediction_sha256 = sha256_state_dict(
        {
            "ensemble_correction": ensemble_correction,
            "target_residual_prediction": target_prediction,
        }
    )

    labels = np.stack(
        [bundle.rows[index]["target_residual_label"] for index in indices]
    )
    sigma = np.asarray([bundle.rows[index]["target_sigma"] for index in indices])
    ensemble_metric = row_whitened_rmse(target_prediction, labels, sigma)
    zero_metric = row_whitened_rmse(np.zeros_like(target_prediction), labels, sigma)
    rigs = [str(bundle.rows[index]["rig_id"]) for index in indices]
    families = [str(bundle.rows[index]["family"]) for index in indices]
    ensemble_mean, ensemble_cells = cluster_mean_metric(
        ensemble_metric, rigs, families
    )
    zero_mean, zero_cells = cluster_mean_metric(zero_metric, rigs, families)
    cell_rows = []
    for ensemble_cell, zero_cell in zip(ensemble_cells, zero_cells, strict=True):
        cell_rows.append(
            {
                "rig_id": ensemble_cell["rig_id"],
                "family": ensemble_cell["family"],
                "row_count": ensemble_cell["row_count"],
                "ensemble_whitened_rmse": ensemble_cell["mean_whitened_rmse"],
                "zero_whitened_rmse": zero_cell["mean_whitened_rmse"],
                "relative_gain_vs_zero": float(
                    1.0
                    - float(ensemble_cell["mean_whitened_rmse"])
                    / float(zero_cell["mean_whitened_rmse"])
                ),
            }
        )

    first_positions: dict[str, int] = {}
    for position, index in enumerate(indices):
        first_positions.setdefault(str(bundle.rows[index]["field_uid"]), position)
    field_rows = []
    for field_uid, position in sorted(first_positions.items()):
        row = bundle.rows[indices[position]]
        truth = bundle.truth_fields[field_uid]
        base_field = row["base_field"]
        predicted = base_field + ensemble_correction[position]
        field_rows.append(
            {
                "rig_id": row["rig_id"],
                "family": row["family"],
                "field_uid": field_uid,
                "base_relative_l2": _relative_l2(base_field, truth),
                "ensemble_relative_l2": _relative_l2(predicted, truth),
                "relative_gain_vs_base": float(
                    1.0
                    - _relative_l2(predicted, truth)
                    / max(_relative_l2(base_field, truth), 1e-12)
                ),
            }
        )
    gains = np.asarray([row["relative_gain_vs_zero"] for row in cell_rows])
    policy = extension["policy"]
    passed = (
        float(np.mean(gains)) >= float(policy["minimum_relative_gain_vs_zero"])
        and float(np.mean(gains > 0.0))
        >= float(policy["minimum_positive_rig_family_fraction"])
        and max(0.0, -float(np.min(gains)))
        <= float(policy["maximum_cell_degradation"])
    )
    report = {
        "schema": extension["schema"],
        "evidence_label": extension["evidence_label"],
        "source_config_sha256": sha256_json(base),
        "extension_config_sha256": sha256_json(extension),
        "checkpoint_hashes": checkpoint_hashes,
        "prediction_sha256_before_scoring": prediction_sha256,
        "checkpoint_retraining": False,
        "design_lock_rows_constructed": 0,
        "dataset_manifest": bundle.manifest,
        "target_summary": {
            "ensemble_cluster_mean_whitened_rmse": ensemble_mean,
            "zero_cluster_mean_whitened_rmse": zero_mean,
            "cell_mean_relative_gain_vs_zero": float(np.mean(gains)),
            "positive_cell_fraction": float(np.mean(gains > 0.0)),
            "worst_cell_degradation": max(0.0, -float(np.min(gains))),
            "cells": cell_rows,
        },
        "field_truth_diagnostic": {
            "field_count": len(field_rows),
            "mean_base_relative_l2": float(
                np.mean([row["base_relative_l2"] for row in field_rows])
            ),
            "mean_ensemble_relative_l2": float(
                np.mean([row["ensemble_relative_l2"] for row in field_rows])
            ),
            "mean_relative_gain_vs_base": float(
                np.mean([row["relative_gain_vs_base"] for row in field_rows])
            ),
            "field_better_fraction": float(
                np.mean([row["relative_gain_vs_base"] > 0.0 for row in field_rows])
            ),
        },
        "decision": "EXTENSION_GATE_PASS" if passed else "NO_DESIGN_LOCK_OPEN",
        "policy": policy,
        "claim_boundary": [
            "The new topologies and rigs were designed after v5k/v5l and remain development evidence.",
            "Existing checkpoints were not retrained, selected or calibrated on extension labels.",
            "The original design-lock rigs and families remain unconstructed and unopened.",
            "All data are synthetic and do not establish experimental or OERF performance.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "cells.csv", cell_rows)
    _write_csv(OUTPUT_DIR / "field_rows.csv", field_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "target": report["target_summary"],
                "field": report["field_truth_diagnostic"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
