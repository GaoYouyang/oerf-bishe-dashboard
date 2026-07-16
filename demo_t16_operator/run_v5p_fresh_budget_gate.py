#!/usr/bin/env python3
"""Preregistered fresh-development gate for low-budget SFIO-PAPBB.

This runner constructs validation rows only. It freezes the candidate and
primary PBB baseline before labels are scored, records exact executed source
forward/adjoint calls, and never constructs design-lock rows.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.data import build_dataset
    from demo_t16_operator.gc_rio.protocol import (
        sha256_file,
        sha256_json,
        sha256_state_dict,
    )
    from demo_t16_operator.gc_rio.shared_field_model import source_adjoint_fisher
    from demo_t16_operator.gc_rio.training import (
        cluster_mean_metric,
        predictor_tensors,
        row_whitened_rmse,
    )
    from demo_t16_operator.release_provenance import (
        relative_file_hashes,
        runtime_environment,
    )
    from demo_t16_operator.run_v5h_gc_rio_development import _model
    from demo_t16_operator.run_v5n_strong_classical_baselines import (
        WORK_DIR,
        _field_diagnostics,
        projected_bb_correction,
    )
    from demo_t16_operator.run_v5o_prior_anchored_frontier import (
        prior_anchored_bb_correction,
    )
else:
    from .gc_rio.data import build_dataset
    from .gc_rio.protocol import sha256_file, sha256_json, sha256_state_dict
    from .gc_rio.shared_field_model import source_adjoint_fisher
    from .gc_rio.training import (
        cluster_mean_metric,
        predictor_tensors,
        row_whitened_rmse,
    )
    from .release_provenance import relative_file_hashes, runtime_environment
    from .run_v5h_gc_rio_development import _model
    from .run_v5n_strong_classical_baselines import (
        WORK_DIR,
        _field_diagnostics,
        projected_bb_correction,
    )
    from .run_v5o_prior_anchored_frontier import prior_anchored_bb_correction


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5p_fresh_budget_gate.json"
OUTPUT_DIR = ROOT / "results" / "v5p_fresh_budget_gate"


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


def _write_checksums(output: Path, names: Sequence[str]) -> None:
    lines = []
    for name in names:
        digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (output / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def validate_preregistered_config(config: Mapping[str, Any]) -> None:
    candidate = config["candidate"]
    baseline = config["primary_baseline"]
    gate = config["gate"]
    if int(candidate["refinement_iterations"]) != 8:
        raise ValueError("v5p freezes exactly eight PAPBB refinement iterations")
    if not np.isclose(float(candidate["relative_anchor"]), 0.1):
        raise ValueError("v5p freezes relative_anchor=0.1")
    if int(baseline["iterations"]) != 9:
        raise ValueError("v5p freezes the primary baseline at PBB-9")
    if int(candidate["source_forward_calls_per_field"]) > int(
        baseline["source_forward_calls_per_field"]
    ) or int(candidate["source_adjoint_calls_per_field"]) > int(
        baseline["source_adjoint_calls_per_field"]
    ):
        raise ValueError("candidate source-call budget may not exceed the baseline")
    if not bool(candidate["reuse_source_statistics_across_seeds"]):
        raise ValueError("source statistics must be reused across ensemble seeds")
    if not bool(gate["evaluation_label_selection_forbidden"]):
        raise ValueError("fresh evaluation label selection must remain forbidden")
    if not bool(gate["design_lock_construction_forbidden"]):
        raise ValueError("design-lock construction must remain forbidden")
    if any(str(rig.get("split")) != "validation" for rig in config["rigs"]):
        raise ValueError("v5p may contain validation rigs only")


def fresh_dataset_config(
    base: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config["seed"] = int(preregistration["fresh_seed"])
    config["fields_per_family"] = int(preregistration["fields_per_family"])
    config["rigs"] = copy.deepcopy(preregistration["rigs"])
    config["splits"] = {
        "train": {"families": []},
        "validation": {"families": list(preregistration["families"])},
        "design_lock": {"families": []},
    }
    config["families"] = []
    for key, value in preregistration.get("dataset_overrides", {}).items():
        config[str(key)] = copy.deepcopy(value)
    return config


def _unique_field_indices(bundle: Any) -> list[int]:
    first: dict[str, int] = {}
    for row in bundle.rows:
        first.setdefault(str(row["field_uid"]), int(row["row_index"]))
    return [first[key] for key in sorted(first)]


def _load_models(
    base: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    source_report: Mapping[str, Any],
) -> tuple[list[torch.nn.Module], dict[str, str]]:
    recorded = {
        (row["method"], int(row["model_seed"])): row["checkpoint_sha256"]
        for row in source_report["training_records"]
    }
    models = []
    hashes = {}
    method = str(preregistration["source_method"])
    for raw_seed in preregistration["ensemble_seeds"]:
        seed = int(raw_seed)
        checkpoint = WORK_DIR / method / str(seed) / "best.pt"
        digest = sha256_file(checkpoint)
        if digest != recorded[(method, seed)]:
            raise RuntimeError(f"checkpoint drift for seed {seed}")
        model = _model(
            base,
            candidate="shared_field",
            use_target_geometry=False,
            seed=seed,
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.eval()
        models.append(model)
        hashes[str(seed)] = digest
    return models, hashes


def _shared_prior_map(
    bundle: Any,
    models: Sequence[torch.nn.Module],
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, float]]:
    indices = _unique_field_indices(bundle)
    counter: dict[str, int] = {}
    feature_seconds = 0.0
    cnn_seconds = 0.0
    output: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            tensors = predictor_tensors(bundle.predictor_batch(selected), "cpu")
            tick = time.perf_counter()
            statistics = source_adjoint_fisher(
                tensors["source_operator"],
                tensors["source_residual"],
                tensors["source_sigma"],
                operator_call_counter=counter,
            )
            feature_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            members = [
                model(
                    **tensors,
                    precomputed_source_statistics=statistics,
                ).correction.numpy()
                for model in models
            ]
            mean = np.mean(members, axis=0)
            cnn_seconds += time.perf_counter() - tick
            for position, index in enumerate(selected):
                output[str(bundle.rows[index]["field_uid"])] = mean[position]
    return output, counter, {
        "source_statistics_seconds": feature_seconds,
        "three_seed_cnn_seconds": cnn_seconds,
        "total_prior_seconds": feature_seconds + cnn_seconds,
    }


def _timed_correction_map(
    bundle: Any,
    solver: Callable[[Mapping[str, Any], MutableMapping[str, int]], np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, int], float]:
    counter: dict[str, int] = {}
    output = {}
    tick = time.perf_counter()
    for index in _unique_field_indices(bundle):
        row = bundle.rows[index]
        output[str(row["field_uid"])] = solver(row, counter)
    elapsed = time.perf_counter() - tick
    return output, counter, elapsed


def _target_prediction(
    bundle: Any, corrections: Mapping[str, np.ndarray]
) -> tuple[list[int], np.ndarray, float]:
    indices = [int(row["row_index"]) for row in bundle.rows]
    tick = time.perf_counter()
    prediction = np.stack(
        [
            bundle.rows[index]["target_operator"]
            @ corrections[str(bundle.rows[index]["field_uid"])]
            for index in indices
        ]
    )
    return indices, prediction, time.perf_counter() - tick


def _score_methods(
    bundle: Any,
    indices: Sequence[int],
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, tuple[dict[str, Any], ...]]]:
    labels = np.stack(
        [bundle.rows[int(index)]["target_residual_label"] for index in indices]
    )
    sigma = np.asarray(
        [bundle.rows[int(index)]["target_sigma"] for index in indices]
    )
    rigs = [str(bundle.rows[int(index)]["rig_id"]) for index in indices]
    families = [str(bundle.rows[int(index)]["family"]) for index in indices]
    aggregates = {}
    cells = {}
    for method, prediction in predictions.items():
        row_metric = row_whitened_rmse(prediction, labels, sigma)
        aggregates[method], cells[method] = cluster_mean_metric(
            row_metric, rigs, families
        )
    return aggregates, cells


def _call_summary(
    counter: Mapping[str, int], field_count: int
) -> dict[str, int | float]:
    return {
        "field_count": field_count,
        "source_forward_total": int(counter.get("source_forward", 0)),
        "source_adjoint_total": int(counter.get("source_adjoint", 0)),
        "source_forward_per_field": float(counter.get("source_forward", 0))
        / field_count,
        "source_adjoint_per_field": float(counter.get("source_adjoint", 0))
        / field_count,
    }


def run() -> dict[str, Any]:
    preregistration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_preregistered_config(preregistration)
    base = json.loads(
        (ROOT / preregistration["source_config"]).read_text(encoding="utf-8")
    )
    source_report = json.loads(
        (ROOT / preregistration["source_report"]).read_text(encoding="utf-8")
    )
    config = fresh_dataset_config(base, preregistration)
    bundle = build_dataset(config)
    if {row["split"] for row in bundle.rows} != {"validation"}:
        raise RuntimeError("v5p may construct fresh validation rows only")
    if bundle.manifest["split_rig_sets"]["design_lock"]:
        raise RuntimeError("v5p constructed forbidden design-lock rigs")
    if bundle.manifest["split_family_sets"]["design_lock"]:
        raise RuntimeError("v5p constructed forbidden design-lock families")

    models, checkpoint_hashes = _load_models(
        base, preregistration, source_report
    )
    field_count = len(_unique_field_indices(bundle))
    priors, prior_counter, prior_timing = _shared_prior_map(
        bundle,
        models,
        batch_size=int(base["training"]["batch_size"]),
    )
    candidate_config = preregistration["candidate"]
    candidate, refinement_counter, refinement_seconds = _timed_correction_map(
        bundle,
        lambda row, counter: prior_anchored_bb_correction(
            row,
            priors[str(row["field_uid"])],
            iterations=int(candidate_config["refinement_iterations"]),
            relative_anchor=float(candidate_config["relative_anchor"]),
            operator_call_counter=counter,
        ),
    )
    candidate_counter = {
        "source_forward": int(refinement_counter.get("source_forward", 0)),
        "source_adjoint": int(prior_counter.get("source_adjoint", 0))
        + int(refinement_counter.get("source_adjoint", 0)),
    }

    def pbb(count: int) -> tuple[dict[str, np.ndarray], dict[str, int], float]:
        return _timed_correction_map(
            bundle,
            lambda row, counter: projected_bb_correction(
                row, count, operator_call_counter=counter
            ),
        )

    pbb9, pbb9_counter, pbb9_seconds = pbb(9)
    pbb11, pbb11_counter, pbb11_seconds = pbb(11)
    pbb32, pbb32_counter, pbb32_seconds = pbb(32)
    zero = {
        str(bundle.rows[index]["field_uid"]): np.zeros_like(
            bundle.rows[index]["base_field"]
        )
        for index in _unique_field_indices(bundle)
    }
    corrections = {
        "zero_correction": zero,
        "shared_field_prior": priors,
        "source_only_pbb_9": pbb9,
        "source_only_pbb_11": pbb11,
        "source_only_pbb_32": pbb32,
        "sfio_papbb_budget_vector_v1": candidate,
    }

    predictions = {}
    decode_seconds = {}
    indices: list[int] | None = None
    for method, method_corrections in corrections.items():
        method_indices, prediction, elapsed = _target_prediction(
            bundle, method_corrections
        )
        if indices is None:
            indices = method_indices
        elif indices != method_indices:
            raise RuntimeError("prediction row order drift")
        predictions[method] = prediction
        decode_seconds[method] = elapsed
    assert indices is not None
    prediction_sha256_before_scoring = sha256_state_dict(
        {f"{method}_target_prediction": value for method, value in predictions.items()}
    )

    aggregates, cells = _score_methods(bundle, indices, predictions)
    candidate_name = str(candidate_config["id"])
    baseline_name = str(preregistration["primary_baseline"]["id"])
    cell_rows = []
    for candidate_cell, baseline_cell in zip(
        cells[candidate_name], cells[baseline_name], strict=True
    ):
        candidate_value = float(candidate_cell["mean_whitened_rmse"])
        baseline_value = float(baseline_cell["mean_whitened_rmse"])
        cell_rows.append(
            {
                "rig_id": candidate_cell["rig_id"],
                "family": candidate_cell["family"],
                "row_count": candidate_cell["row_count"],
                "candidate_target_standardized_rmse": candidate_value,
                "baseline_target_standardized_rmse": baseline_value,
                "candidate_gain_vs_pbb9": 1.0
                - candidate_value / max(baseline_value, 1e-12),
            }
        )
    gains = np.asarray([row["candidate_gain_vs_pbb9"] for row in cell_rows])
    ratio_gain = 1.0 - aggregates[candidate_name] / aggregates[baseline_name]

    field_rows = _field_diagnostics(bundle, corrections)
    field_summary = {}
    for method in corrections:
        values = [row for row in field_rows if row["method"] == method]
        field_summary[method] = {
            "mean_relative_l2": float(np.mean([row["relative_l2"] for row in values])),
            "mean_gain_vs_base": float(np.mean([row["gain_vs_base"] for row in values])),
            "better_fraction": float(
                np.mean([row["gain_vs_base"] > 0.0 for row in values])
            ),
        }

    candidate_calls = _call_summary(candidate_counter, field_count)
    pbb9_calls = _call_summary(pbb9_counter, field_count)
    expected_candidate = candidate_config
    expected_baseline = preregistration["primary_baseline"]
    if (
        candidate_calls["source_forward_per_field"]
        != float(expected_candidate["source_forward_calls_per_field"])
        or candidate_calls["source_adjoint_per_field"]
        != float(expected_candidate["source_adjoint_calls_per_field"])
        or pbb9_calls["source_forward_per_field"]
        != float(expected_baseline["source_forward_calls_per_field"])
        or pbb9_calls["source_adjoint_per_field"]
        != float(expected_baseline["source_adjoint_calls_per_field"])
    ):
        raise RuntimeError("executed source operator calls violate preregistration")

    gate = preregistration["gate"]
    passed = (
        ratio_gain >= float(gate["minimum_relative_gain"])
        and float(np.mean(gains > 0.0))
        >= float(gate["minimum_positive_rig_family_fraction"])
        and max(0.0, -float(np.min(gains)))
        <= float(gate["maximum_cell_degradation"])
    )
    report = {
        "schema": preregistration["schema"],
        "evidence_label": preregistration["evidence_label"],
        "claim_ceiling": preregistration["claim_ceiling"],
        "preregistration_config_sha256": sha256_json(preregistration),
        "base_config_sha256": sha256_json(base),
        "checkpoint_hashes": checkpoint_hashes,
        "source_provenance": {
            "direct_dependency_sha256": relative_file_hashes(
                ROOT,
                [
                    Path(__file__),
                    CONFIG_PATH,
                    ROOT / preregistration["source_config"],
                    ROOT / preregistration["source_report"],
                    ROOT / "release_provenance.py",
                    ROOT / "gc_rio" / "data.py",
                    ROOT / "gc_rio" / "protocol.py",
                    ROOT / "gc_rio" / "shared_field_model.py",
                    ROOT / "gc_rio" / "training.py",
                    ROOT / "run_v5h_gc_rio_development.py",
                    ROOT / "run_v5n_strong_classical_baselines.py",
                    ROOT / "run_v5o_prior_anchored_frontier.py",
                    *[
                        WORK_DIR
                        / str(preregistration["source_method"])
                        / str(seed)
                        / "best.pt"
                        for seed in preregistration["ensemble_seeds"]
                    ],
                ],
            ),
            "runtime_environment": runtime_environment(
                device="cpu", torch_version=torch.__version__
            ),
            "determinism_boundary": (
                "V5P is a CPU inference/evaluation replay. Wall-clock values are "
                "informational; predictions and decisions are hash-checked."
            ),
        },
        "prediction_sha256_before_scoring": prediction_sha256_before_scoring,
        "design_lock_rows_constructed": 0,
        "dataset_manifest": bundle.manifest,
        "sample_accounting": {
            "field_count": field_count,
            "target_row_count": len(indices),
            "rig_family_cell_count": len(cell_rows),
            "target_rows_per_field": len(indices) / field_count,
        },
        "source_operator_call_accounting": {
            "candidate": candidate_calls,
            "primary_baseline": pbb9_calls,
            "pbb_11": _call_summary(pbb11_counter, field_count),
            "pbb_32": _call_summary(pbb32_counter, field_count),
            "target_decode_calls_per_field_per_method": len(indices) / field_count,
        },
        "wall_clock_seconds": {
            **prior_timing,
            "candidate_refinement_seconds": refinement_seconds,
            "candidate_total_before_target_decode_seconds": prior_timing[
                "total_prior_seconds"
            ]
            + refinement_seconds,
            "pbb_9_seconds": pbb9_seconds,
            "pbb_11_seconds": pbb11_seconds,
            "pbb_32_seconds": pbb32_seconds,
            "target_decode_by_method": decode_seconds,
            "timing_boundary": "Single CPU run on the local Mac; checkpoint loading and dataset construction are excluded. No timing superiority claim is authorized.",
        },
        "target_summary": {
            "cluster_mean_target_standardized_rmse": aggregates,
            "candidate_ratio_of_cluster_means_gain_vs_pbb9": ratio_gain,
            "candidate_cell_mean_gain_vs_pbb9": float(np.mean(gains)),
            "candidate_positive_cell_fraction": float(np.mean(gains > 0.0)),
            "candidate_worst_cell_degradation": max(0.0, -float(np.min(gains))),
            "cells": cell_rows,
        },
        "field_truth_diagnostic": field_summary,
        "gate": gate,
        "decision": (
            "FRESH_DEVELOPMENT_GATE_PASS_NO_DESIGN_LOCK"
            if passed
            else "FRESH_DEVELOPMENT_NO_GO"
        ),
        "limitations": [
            "All fields and observations are synthetic weak-deflection proxies.",
            "The candidate hypothesis was generated after inspecting v5o extension results.",
            "Scalar camera sigma standardizes but does not fully whiten correlated noise.",
            "Three CNN forwards add compute that is not represented by source operator counts.",
            "Field truth is a post-prediction diagnostic and does not enter the primary gate.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "cells.csv", cell_rows)
    _write_csv(OUTPUT_DIR / "field_rows.csv", field_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(OUTPUT_DIR, ["cells.csv", "field_rows.csv", "report.json"])
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "sample_accounting": report["sample_accounting"],
                "calls": report["source_operator_call_accounting"],
                "timing": report["wall_clock_seconds"],
                "target": report["target_summary"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
