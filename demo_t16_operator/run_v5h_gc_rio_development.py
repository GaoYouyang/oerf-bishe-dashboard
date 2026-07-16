#!/usr/bin/env python3
"""Train and freeze the development-only v5h GC-RIO candidates.

This runner deliberately removes the design-lock rigs before constructing the
dataset. It can select checkpoints on validation target residuals, but it
cannot produce a design-lock result or an experimental-data claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.data import build_dataset
    from demo_t16_operator.gc_rio.model import (
        GeometryConditionedResidualInverseOperator,
    )
    from demo_t16_operator.gc_rio.protocol import (
        make_development_config,
        sha256_file,
        sha256_json,
        sha256_state_dict,
        validate_full_protocol,
    )
    from demo_t16_operator.gc_rio.signed_probe_model import (
        SignedProbeResidualInverseOperator,
    )
    from demo_t16_operator.gc_rio.shared_field_model import (
        SharedFieldResidualInverseOperator,
    )
    from demo_t16_operator.gc_rio.training import (
        analytic_dc_prediction,
        cluster_mean_metric,
        predict_rows,
        predictor_tensors,
        row_whitened_rmse,
        train_validation_model,
    )
else:
    from .gc_rio.data import build_dataset
    from .gc_rio.model import GeometryConditionedResidualInverseOperator
    from .gc_rio.protocol import (
        make_development_config,
        sha256_file,
        sha256_json,
        sha256_state_dict,
        validate_full_protocol,
    )
    from .gc_rio.signed_probe_model import SignedProbeResidualInverseOperator
    from .gc_rio.shared_field_model import SharedFieldResidualInverseOperator
    from .gc_rio.training import (
        analytic_dc_prediction,
        cluster_mean_metric,
        predict_rows,
        predictor_tensors,
        row_whitened_rmse,
        train_validation_model,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5h_gc_rio_development.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5h_gc_rio_development"
DEFAULT_WORK = ROOT / "results" / "v5h_gc_rio_work"
CANDIDATE_METHODS = {
    "fisher": {
        "no_target": "gc_rio_no_target_geometry",
        "correct": "gc_rio_correct_target_geometry",
        "shuffled": "gc_rio_shuffled_target_geometry",
    },
    "signed_probe": {
        "no_target": "sp_gc_rio_no_target_geometry",
        "correct": "sp_gc_rio_correct_target_geometry",
        "shuffled": "sp_gc_rio_shuffled_target_geometry",
    },
    "shared_field": {
        "no_target": "sf_rio_adjoint_only",
        "correct": "sf_rio_krylov_stack",
        "shuffled": "sf_rio_shuffled_source_operator",
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _indices(bundle: Any, split: str) -> list[int]:
    return [int(row["row_index"]) for row in bundle.rows if row["split"] == split]


def _model(
    config: Mapping[str, Any],
    *,
    candidate: str,
    use_target_geometry: bool,
    seed: int,
):
    training = config["training"]
    torch.manual_seed(int(seed))
    if candidate == "shared_field":
        return SharedFieldResidualInverseOperator(
            (
                int(config["depth"]),
                int(config["grid_size"]),
                int(config["grid_size"]),
            ),
            hidden_channels=int(training["hidden_channels"]),
            residual_blocks=int(training["residual_blocks"]),
            maximum_base_fraction=0.55,
            ridge_lambda=float(training["model_ridge_lambda"]),
            data_consistency_step=0.0,
            krylov_steps=3,
            use_krylov_features=use_target_geometry,
        )
    model_class = {
        "fisher": GeometryConditionedResidualInverseOperator,
        "signed_probe": SignedProbeResidualInverseOperator,
    }.get(candidate)
    if model_class is None:
        raise ValueError(f"unknown candidate: {candidate}")
    return model_class(
        (int(config["depth"]), int(config["grid_size"]), int(config["grid_size"])),
        hidden_channels=int(training["hidden_channels"]),
        residual_blocks=int(training["residual_blocks"]),
        maximum_learned_fraction=float(training["maximum_learned_fraction"]),
        ridge_lambda=float(training["model_ridge_lambda"]),
        data_consistency_step=float(training["data_consistency_step"]),
        use_target_geometry=use_target_geometry,
    )


def _analytic_predictions(
    bundle: Any,
    indices: Sequence[int],
    config: Mapping[str, Any],
    *,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    training = config["training"]
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        tensors = predictor_tensors(bundle.predictor_batch(batch_indices), "cpu")
        with torch.no_grad():
            prediction = analytic_dc_prediction(
                tensors,
                ridge_lambda=float(training["model_ridge_lambda"]),
                step_fraction=float(training["data_consistency_step"]),
            )
        outputs.append(prediction.numpy())
    return np.concatenate(outputs, axis=0)


def _zero_predictions(bundle: Any, indices: Sequence[int]) -> np.ndarray:
    return np.stack(
        [np.zeros_like(bundle.rows[int(index)]["target_residual_label"]) for index in indices]
    )


def _target_geometry_derangement(bundle: Any, indices: Sequence[int]) -> np.ndarray:
    """Swap the two target queries of each field while retaining true decoding."""

    positions: dict[str, list[int]] = defaultdict(list)
    for position, row_index in enumerate(indices):
        positions[str(bundle.rows[int(row_index)]["field_uid"])].append(position)
    permutation = np.arange(len(indices), dtype=np.int64)
    for field_uid, group in positions.items():
        if len(group) != 2:
            raise ValueError(f"field {field_uid} does not have exactly two target rows")
        left, right = group
        permutation[left], permutation[right] = right, left
        left_operator = bundle.rows[int(indices[left])]["target_operator"]
        right_operator = bundle.rows[int(indices[right])]["target_operator"]
        if np.array_equal(left_operator, right_operator):
            raise ValueError(f"field {field_uid} target operators are not distinct")
    return permutation


def _source_geometry_derangement(bundle: Any, indices: Sequence[int]) -> np.ndarray:
    """Map every row to a deterministic source operator from another rig."""

    permutation = np.empty(len(indices), dtype=np.int64)
    for position, row_index in enumerate(indices):
        rig_id = str(bundle.rows[int(row_index)]["rig_id"])
        for offset in range(1, len(indices) + 1):
            candidate = (position + offset) % len(indices)
            candidate_rig = str(bundle.rows[int(indices[candidate])]["rig_id"])
            if candidate_rig != rig_id:
                permutation[position] = candidate
                break
        else:
            raise ValueError("source-geometry shuffle needs at least two rigs")
    return permutation


def _score_method(
    bundle: Any,
    indices: Sequence[int],
    prediction: np.ndarray,
    *,
    method: str,
    seed: int | str,
    split: str,
) -> tuple[list[dict[str, Any]], float, tuple[dict[str, Any], ...]]:
    labels = np.stack([bundle.rows[int(index)]["target_residual_label"] for index in indices])
    sigma = np.asarray([bundle.rows[int(index)]["target_sigma"] for index in indices])
    metric = row_whitened_rmse(prediction, labels, sigma)
    rows: list[dict[str, Any]] = []
    for position, row_index in enumerate(indices):
        row = bundle.rows[int(row_index)]
        rows.append(
            {
                "method": method,
                "model_seed": seed,
                "split": split,
                "row_index": int(row_index),
                "rig_id": row["rig_id"],
                "family": row["family"],
                "field_uid": row["field_uid"],
                "target_view": int(row["target_view"]),
                "whitened_target_residual_rmse": float(metric[position]),
            }
        )
    aggregate, cells = cluster_mean_metric(
        metric,
        [bundle.rows[int(index)]["rig_id"] for index in indices],
        [bundle.rows[int(index)]["family"] for index in indices],
    )
    return rows, aggregate, cells


def _mean_method_rows(
    rows: Sequence[Mapping[str, Any]], method: str, split: str
) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["split"] == split:
            values[int(row["row_index"])].append(
                float(row["whitened_target_residual_rmse"])
            )
    return {index: float(np.mean(items)) for index, items in values.items()}


def _development_decision(
    config: Mapping[str, Any],
    bundle: Any,
    score_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Mapping[str, str],
) -> dict[str, Any]:
    split = "validation"
    method_maps = {
        method: _mean_method_rows(score_rows, method, split)
        for method in (
            "zero_correction",
            "analytic_dc",
            methods["no_target"],
            methods["correct"],
            methods["shuffled"],
        )
    }
    shared = sorted(set.intersection(*(set(values) for values in method_maps.values())))
    if not shared:
        raise RuntimeError("validation methods have no shared rows")
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in shared:
        row = bundle.rows[index]
        cells[(str(row["rig_id"]), str(row["family"]))].append(index)
    cell_rows: list[dict[str, Any]] = []
    for (rig_id, family), indices in sorted(cells.items()):
        means = {
            method: float(np.mean([values[index] for index in indices]))
            for method, values in method_maps.items()
        }
        strongest_baseline_method = min(
            (
                "zero_correction",
                "analytic_dc",
                methods["no_target"],
                methods["shuffled"],
            ),
            key=lambda method: means[method],
        )
        baseline = means[strongest_baseline_method]
        gain = 1.0 - means[methods["correct"]] / max(baseline, 1e-12)
        cell_rows.append(
            {
                "rig_id": rig_id,
                "family": family,
                "row_count": len(indices),
                **means,
                "strongest_baseline_method": strongest_baseline_method,
                "relative_gain_vs_strongest_baseline": gain,
            }
        )
    gains = np.asarray(
        [float(row["relative_gain_vs_strongest_baseline"]) for row in cell_rows]
    )
    gate = config["development_gate"]
    overall_gain = float(np.mean(gains))
    positive_fraction = float(np.mean(gains > 0.0))
    worst_degradation = float(max(0.0, -float(np.min(gains))))
    passed = (
        overall_gain >= float(gate["minimum_relative_gain"])
        and positive_fraction >= float(gate["minimum_positive_rig_family_fraction"])
        and worst_degradation <= float(gate["maximum_cluster_degradation"])
    )
    return {
        "label": "PROVISIONAL_FREEZE_CANDIDATE" if passed else "NO_DESIGN_LOCK_OPEN",
        "development_only": True,
        "overall_cell_mean_relative_gain": overall_gain,
        "positive_rig_family_fraction": positive_fraction,
        "worst_cell_degradation": worst_degradation,
        "thresholds": gate,
        "cells": cell_rows,
        "claim_ceiling": config["claim_ceiling"],
        "candidate_methods": dict(methods),
    }


def run(
    config_path: Path,
    output_dir: Path,
    work_dir: Path,
    *,
    candidate: str = "fisher",
) -> dict[str, Any]:
    full_config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_full_protocol(full_config)
    development_config = make_development_config(full_config)
    bundle = build_dataset(development_config)
    if any(row["split"] == "design_lock" for row in bundle.rows):
        raise RuntimeError("design-lock row was constructed during development")
    train_indices = _indices(bundle, "train")
    validation_indices = _indices(bundle, "validation")
    training = full_config["training"]
    if candidate not in CANDIDATE_METHODS:
        raise ValueError(f"unknown candidate: {candidate}")
    methods = CANDIDATE_METHODS[candidate]
    batch_size = int(training["batch_size"])
    model_seeds = [int(seed) for seed in training["model_seeds"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    histories: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int | str, str], np.ndarray] = {}

    for method in (methods["no_target"], methods["correct"]):
        use_target_geometry = method == methods["correct"]
        for seed in model_seeds:
            model = _model(
                full_config,
                candidate=candidate,
                use_target_geometry=use_target_geometry,
                seed=seed,
            )
            checkpoint = work_dir / method / str(seed) / "best.pt"
            result = train_validation_model(
                model,
                bundle,
                train_indices,
                validation_indices,
                training_config=training,
                model_seed=seed,
                checkpoint_path=checkpoint,
                device="cpu",
            )
            training_records.append(
                {
                    "method": method,
                    "model_seed": seed,
                    "best_epoch": result.best_epoch,
                    "best_validation_metric": result.best_validation_metric,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_public": False,
                }
            )
            histories.extend(
                {"method": method, "model_seed": seed, **record}
                for record in result.history
            )
            for split, indices in (
                ("train", train_indices),
                ("validation", validation_indices),
            ):
                prediction = predict_rows(
                    model,
                    bundle,
                    indices,
                    batch_size=batch_size,
                    device="cpu",
                )
                predictions[(method, seed, split)] = prediction
                prediction_records.append(
                    {
                        "method": method,
                        "model_seed": seed,
                        "split": split,
                        "prediction_sha256": sha256_state_dict(
                            {"target_residual_prediction": prediction}
                        ),
                        "prediction_created_before_scoring": True,
                    }
                )
                rows, _, _ = _score_method(
                    bundle,
                    indices,
                    prediction,
                    method=method,
                    seed=seed,
                    split=split,
                )
                score_rows.extend(rows)
                if use_target_geometry:
                    if candidate == "shared_field":
                        permutation = _source_geometry_derangement(bundle, indices)
                        shuffled = predict_rows(
                            model,
                            bundle,
                            indices,
                            batch_size=batch_size,
                            device="cpu",
                            conditioning_source_permutation=permutation,
                        )
                    else:
                        permutation = _target_geometry_derangement(bundle, indices)
                        shuffled = predict_rows(
                            model,
                            bundle,
                            indices,
                            batch_size=batch_size,
                            device="cpu",
                            conditioning_permutation=permutation,
                        )
                    shuffled_method = methods["shuffled"]
                    prediction_records.append(
                        {
                            "method": shuffled_method,
                            "model_seed": seed,
                            "split": split,
                            "prediction_sha256": sha256_state_dict(
                                {"target_residual_prediction": shuffled}
                            ),
                            "prediction_created_before_scoring": True,
                        }
                    )
                    shuffled_rows, _, _ = _score_method(
                        bundle,
                        indices,
                        shuffled,
                        method=shuffled_method,
                        seed=seed,
                        split=split,
                    )
                    score_rows.extend(shuffled_rows)

    for split, indices in (("train", train_indices), ("validation", validation_indices)):
        zero = _zero_predictions(bundle, indices)
        prediction_records.append(
            {
                "method": "zero_correction",
                "model_seed": "deterministic",
                "split": split,
                "prediction_sha256": sha256_state_dict(
                    {"target_residual_prediction": zero}
                ),
                "prediction_created_before_scoring": True,
            }
        )
        zero_rows, _, _ = _score_method(
            bundle,
            indices,
            zero,
            method="zero_correction",
            seed="deterministic",
            split=split,
        )
        score_rows.extend(zero_rows)
        analytic = _analytic_predictions(
            bundle, indices, full_config, batch_size=batch_size
        )
        prediction_records.append(
            {
                "method": "analytic_dc",
                "model_seed": "deterministic",
                "split": split,
                "prediction_sha256": sha256_state_dict(
                    {"target_residual_prediction": analytic}
                ),
                "prediction_created_before_scoring": True,
            }
        )
        analytic_rows, _, _ = _score_method(
            bundle,
            indices,
            analytic,
            method="analytic_dc",
            seed="deterministic",
            split=split,
        )
        score_rows.extend(analytic_rows)

    decision = _development_decision(
        full_config, bundle, score_rows, methods=methods
    )
    method_summary: list[dict[str, Any]] = []
    for method in (
        "zero_correction",
        "analytic_dc",
        methods["no_target"],
        methods["correct"],
        methods["shuffled"],
    ):
        for split in ("train", "validation"):
            values = [
                float(row["whitened_target_residual_rmse"])
                for row in score_rows
                if row["method"] == method and row["split"] == split
            ]
            method_summary.append(
                {
                    "method": method,
                    "split": split,
                    "scored_rows_including_seeds": len(values),
                    "mean_whitened_target_residual_rmse": float(np.mean(values)),
                    "median_whitened_target_residual_rmse": float(np.median(values)),
                }
            )

    code_paths = {
        "data": ROOT / "gc_rio" / "data.py",
        "model": ROOT / "gc_rio" / "model.py",
        "protocol": ROOT / "gc_rio" / "protocol.py",
        "training": ROOT / "gc_rio" / "training.py",
        "runner": Path(__file__).resolve(),
    }
    if candidate == "signed_probe":
        code_paths["signed_probe_model"] = ROOT / "gc_rio" / "signed_probe_model.py"
    if candidate == "shared_field":
        code_paths["shared_field_model"] = ROOT / "gc_rio" / "shared_field_model.py"
    freeze_manifest = {
        "schema": "v5h-gc-rio-development-freeze-1",
        "candidate": candidate,
        "created_utc": datetime.now(UTC).isoformat(),
        "full_config_sha256": sha256_json(full_config),
        "development_dataset_manifest": bundle.manifest,
        "code_file_sha256": {
            name: sha256_file(path) for name, path in sorted(code_paths.items())
        },
        "checkpoint_records": training_records,
        "prediction_records": prediction_records,
        "design_lock_rows_constructed": 0,
        "design_lock_labels_read": False,
        "checkpoint_selection_split": "validation",
        "truth_field_used_for_fit_or_selection": False,
    }
    report = {
        "schema": "v5h-gc-rio-development-report-1",
        "candidate": candidate,
        "evidence_label": full_config["evidence_label"],
        "claim_ceiling": full_config["claim_ceiling"],
        "dataset": {
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "design_lock_rows_constructed": 0,
            "rigs": bundle.manifest["split_rig_sets"],
            "families": bundle.manifest["split_family_sets"],
            "non_oracle_noise_estimator": "paired flow-off difference MAD",
            "synthetic_generator_only": True,
        },
        "architecture": {
            "name": {
                "fisher": "GC-RIO v0",
                "signed_probe": "Signed-probe GC-RIO v1",
                "shared_field": "Shared-field Krylov RIO v2",
            }[candidate],
            "target_observation_at_inference": False,
            "target_operator_at_inference": True,
            "source_adjoint_and_fisher_statistics": True,
            "target_fisher_query": candidate in {"fisher", "signed_probe"},
            "signed_target_row_space_probes": candidate == "signed_probe",
            "source_krylov_stack": candidate == "shared_field",
            "zero_correction_initialization": candidate == "shared_field",
            "target_independent_shared_field": candidate == "shared_field",
            "explicit_target_physics_decoder": True,
            "fno_or_camera_token_attention": False,
        },
        "training_records": training_records,
        "method_summary": method_summary,
        "decision": decision,
        "freeze_manifest_sha256": sha256_json(freeze_manifest),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "limitations": [
            "Synthetic weak-deflection forward model only; this is not an OERF experiment.",
            "Development validation is reused for checkpoint selection, so it is not confirmatory evidence.",
            "Two design-lock rigs remain unopened and are insufficient for a publication-level rig claim.",
            "DeepONet, FNO/FFNO, NIO-style and learned iterative baselines are not yet in this v0 gate.",
        ],
    }
    _write_csv(output_dir / "training_history.csv", histories)
    _write_csv(output_dir / "score_rows.csv", score_rows)
    _write_csv(output_dir / "method_summary.csv", method_summary)
    _write_json(output_dir / "freeze_manifest.json", freeze_manifest)
    _write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", choices=sorted(CANDIDATE_METHODS), default="fisher")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    output = args.output or (
        DEFAULT_OUTPUT
        if args.candidate == "fisher"
        else (
            ROOT / "results" / "v5i_signed_probe_development"
            if args.candidate == "signed_probe"
            else ROOT / "results" / "v5k_shared_field_development"
        )
    )
    work = args.work or (
        DEFAULT_WORK
        if args.candidate == "fisher"
        else (
            ROOT / "results" / "v5i_signed_probe_work"
            if args.candidate == "signed_probe"
            else ROOT / "results" / "v5k_shared_field_work"
        )
    )
    report = run(args.config, output, work, candidate=args.candidate)
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
