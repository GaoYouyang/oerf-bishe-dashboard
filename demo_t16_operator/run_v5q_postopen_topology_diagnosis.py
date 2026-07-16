#!/usr/bin/env python3
"""Post-open diagnosis of the topology dependence exposed by v5p.

This script does not tune SFIO-PAPBB, train a router, or alter the failed v5p
decision. It hashes a source-observable feature table before reading target
labels, then reports descriptive cross-rig associations for hypothesis
generation only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.gc_rio.protocol import sha256_json, sha256_state_dict
    from demo_t16_operator.gc_rio.shared_field_model import source_adjoint_fisher
    from demo_t16_operator.gc_rio.training import predictor_tensors
    from demo_t16_operator.release_provenance import (
        relative_file_hashes,
        runtime_environment,
    )
    from demo_t16_operator.run_v5n_strong_classical_baselines import (
        _source_system,
        projected_bb_correction,
    )
    from demo_t16_operator.run_v5o_prior_anchored_frontier import (
        prior_anchored_bb_correction,
    )
    from demo_t16_operator.run_v5p_fresh_budget_gate import (
        CONFIG_PATH,
        OUTPUT_DIR as V5P_OUTPUT_DIR,
        ROOT,
        _load_models,
        _score_methods,
        _target_prediction,
        _unique_field_indices,
        fresh_dataset_config,
    )
    from demo_t16_operator.gc_rio.data import build_dataset
else:
    from .gc_rio.data import build_dataset
    from .gc_rio.protocol import sha256_json, sha256_state_dict
    from .gc_rio.shared_field_model import source_adjoint_fisher
    from .gc_rio.training import predictor_tensors
    from .release_provenance import relative_file_hashes, runtime_environment
    from .run_v5n_strong_classical_baselines import (
        _source_system,
        projected_bb_correction,
    )
    from .run_v5o_prior_anchored_frontier import prior_anchored_bb_correction
    from .run_v5p_fresh_budget_gate import (
        CONFIG_PATH,
        OUTPUT_DIR as V5P_OUTPUT_DIR,
        ROOT,
        _load_models,
        _score_methods,
        _target_prediction,
        _unique_field_indices,
        fresh_dataset_config,
    )


OUTPUT_DIR = ROOT / "results" / "v5q_postopen_topology_diagnosis"
FEATURE_NAMES = (
    "source_residual_standardized_rms",
    "source_operator_effective_rank_fraction",
    "source_operator_log10_condition",
    "prior_rms_over_base",
    "prior_member_disagreement_over_base",
    "prior_pbb9_cosine",
    "candidate_pbb9_cosine",
    "candidate_minus_pbb9_rms_over_base",
    "candidate_minus_pbb9_source_visibility",
    "candidate_source_gain_vs_pbb9",
    "candidate_floor_fraction",
    "pbb9_floor_fraction",
    "base_gradient_alignment",
    "pbb9_gradient_alignment",
    "base_total_variation_over_rms",
    "pbb9_total_variation_over_rms",
    "pbb9_positive_volume_fraction",
)


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
    lines = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)) + 1e-18))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 1e-15 else 0.0


def _field_shape_features(
    field: np.ndarray,
    support: np.ndarray,
    *,
    depth: int,
    grid_size: int,
) -> dict[str, float]:
    volume = np.asarray(field, dtype=np.float64).reshape(
        depth, grid_size, grid_size
    )
    mask = np.asarray(support, dtype=bool).reshape(depth, grid_size, grid_size)
    gradients = np.gradient(volume)
    vectors = np.stack([component[mask] for component in gradients], axis=1)
    if vectors.size == 0:
        return {
            "gradient_alignment": 0.0,
            "total_variation_over_rms": 0.0,
            "positive_volume_fraction": 0.0,
        }
    structure = vectors.T @ vectors / max(len(vectors), 1)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(structure), 0.0))[::-1]
    alignment = float(
        (eigenvalues[0] - eigenvalues[1])
        / max(float(np.sum(eigenvalues)), 1e-15)
    )
    active = volume[mask]
    return {
        "gradient_alignment": alignment,
        "total_variation_over_rms": _rms(np.linalg.norm(vectors, axis=1))
        / max(_rms(active), 1e-15),
        "positive_volume_fraction": float(np.mean(active > 1e-8)),
    }


def compute_source_feature_row(
    row: Mapping[str, Any],
    member_priors: np.ndarray,
    prior: np.ndarray,
    candidate: np.ndarray,
    pbb9: np.ndarray,
    *,
    depth: int,
    grid_size: int,
) -> dict[str, Any]:
    """Compute one target-label-free diagnostic row."""

    support = np.asarray(row["support"], dtype=bool)
    base = np.asarray(row["base_field"], dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    pbb9 = np.asarray(pbb9, dtype=np.float64)
    members = np.asarray(member_priors, dtype=np.float64)
    if members.ndim != 2 or members.shape[1] != len(base):
        raise ValueError("member_priors must have shape [member, voxel]")
    if not all(value.shape == base.shape for value in (prior, candidate, pbb9)):
        raise ValueError("all corrections must match the base field")

    operator, residual, active_support = _source_system(row)
    if not np.array_equal(active_support, support):
        raise RuntimeError("support drift while constructing diagnostics")
    singular_values = np.linalg.svd(operator, compute_uv=False)
    positive_singular = singular_values[singular_values > 1e-12]
    effective_rank = float(
        np.square(np.sum(singular_values))
        / max(float(np.sum(np.square(singular_values))), 1e-15)
    )
    condition = (
        float(positive_singular[0] / positive_singular[-1])
        if len(positive_singular)
        else float("inf")
    )

    prior_active = prior[support]
    candidate_active = candidate[support]
    pbb9_active = pbb9[support]
    base_active = base[support]
    difference = candidate_active - pbb9_active
    source_prior = operator @ prior_active
    source_candidate = operator @ candidate_active
    source_pbb9 = operator @ pbb9_active
    source_candidate_rms = _rms(source_candidate - residual)
    source_pbb9_rms = _rms(source_pbb9 - residual)
    difference_rms = _rms(difference)
    base_rms = _rms(base_active)
    disagreement = _rms(members[:, support] - np.mean(members[:, support], axis=0))
    base_shape = _field_shape_features(
        base, support, depth=depth, grid_size=grid_size
    )
    pbb9_shape = _field_shape_features(
        base + pbb9, support, depth=depth, grid_size=grid_size
    )

    return {
        "field_uid": str(row["field_uid"]),
        "rig_id": str(row["rig_id"]),
        "family": str(row["family"]),
        "source_residual_standardized_rms": _rms(residual),
        "source_operator_effective_rank_fraction": effective_rank
        / max(float(min(operator.shape)), 1.0),
        "source_operator_log10_condition": math.log10(max(condition, 1.0)),
        "prior_rms_over_base": _rms(prior_active) / max(base_rms, 1e-15),
        "prior_member_disagreement_over_base": disagreement
        / max(base_rms, 1e-15),
        "prior_pbb9_cosine": _cosine(prior_active, pbb9_active),
        "candidate_pbb9_cosine": _cosine(candidate_active, pbb9_active),
        "candidate_minus_pbb9_rms_over_base": difference_rms
        / max(base_rms, 1e-15),
        "candidate_minus_pbb9_source_visibility": _rms(operator @ difference)
        / max(difference_rms, 1e-15),
        "candidate_source_gain_vs_pbb9": 1.0
        - source_candidate_rms / max(source_pbb9_rms, 1e-15),
        "candidate_floor_fraction": float(
            np.mean((base_active + candidate_active) <= 1e-8)
        ),
        "pbb9_floor_fraction": float(
            np.mean((base_active + pbb9_active) <= 1e-8)
        ),
        "base_gradient_alignment": base_shape["gradient_alignment"],
        "pbb9_gradient_alignment": pbb9_shape["gradient_alignment"],
        "base_total_variation_over_rms": base_shape[
            "total_variation_over_rms"
        ],
        "pbb9_total_variation_over_rms": pbb9_shape[
            "total_variation_over_rms"
        ],
        "pbb9_positive_volume_fraction": pbb9_shape[
            "positive_volume_fraction"
        ],
        "source_prior_standardized_rms": _rms(source_prior - residual),
        "source_candidate_standardized_rms": source_candidate_rms,
        "source_pbb9_standardized_rms": source_pbb9_rms,
    }


def _shared_prior_members(
    bundle: Any,
    models: Sequence[torch.nn.Module],
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    member_maps: dict[str, np.ndarray] = {}
    mean_map: dict[str, np.ndarray] = {}
    indices = _unique_field_indices(bundle)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            tensors = predictor_tensors(bundle.predictor_batch(selected), "cpu")
            statistics = source_adjoint_fisher(
                tensors["source_operator"],
                tensors["source_residual"],
                tensors["source_sigma"],
            )
            members = np.stack(
                [
                    model(
                        **tensors,
                        precomputed_source_statistics=statistics,
                    )
                    .correction.detach()
                    .cpu()
                    .numpy()
                    for model in models
                ],
                axis=0,
            )
            for position, index in enumerate(selected):
                field_uid = str(bundle.rows[index]["field_uid"])
                member_maps[field_uid] = members[:, position]
                mean_map[field_uid] = np.mean(members[:, position], axis=0)
    return member_maps, mean_map


def _correction_map(bundle: Any, solver: Any) -> dict[str, np.ndarray]:
    return {
        str(bundle.rows[index]["field_uid"]): solver(bundle.rows[index])
        for index in _unique_field_indices(bundle)
    }


def _finite_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def _target_gain_by_field(
    bundle: Any,
    candidate: Mapping[str, np.ndarray],
    pbb9: Mapping[str, np.ndarray],
) -> dict[str, float]:
    errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"candidate": [], "pbb9": []}
    )
    for row in bundle.rows:
        field_uid = str(row["field_uid"])
        label = np.asarray(row["target_residual_label"], dtype=np.float64)
        sigma = float(row["target_sigma"])
        operator = np.asarray(row["target_operator"], dtype=np.float64)
        errors[field_uid]["candidate"].append(
            _rms((operator @ candidate[field_uid] - label) / sigma)
        )
        errors[field_uid]["pbb9"].append(
            _rms((operator @ pbb9[field_uid] - label) / sigma)
        )
    return {
        field_uid: 1.0
        - float(np.mean(values["candidate"]))
        / max(float(np.mean(values["pbb9"])), 1e-15)
        for field_uid, values in errors.items()
    }


def _cell_rows(field_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in field_rows:
        grouped[(str(row["rig_id"]), str(row["family"]))].append(row)
    output = []
    for (rig_id, family), rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "rig_id": rig_id,
            "family": family,
            "field_count": len(rows),
            "candidate_target_gain_vs_pbb9": float(
                np.mean([float(row["candidate_target_gain_vs_pbb9"]) for row in rows])
            ),
        }
        for name in FEATURE_NAMES:
            record[name] = float(np.mean([float(row[name]) for row in rows]))
        output.append(record)
    return output


def _correlation_rows(
    field_rows: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rigs = sorted({str(row["rig_id"]) for row in field_rows})
    families = sorted({str(row["family"]) for row in field_rows})
    gain = [float(row["candidate_target_gain_vs_pbb9"]) for row in field_rows]
    output = []
    for name in FEATURE_NAMES:
        feature = [float(row[name]) for row in field_rows]
        cell_feature = [float(row[name]) for row in cell_rows]
        cell_gain = [float(row["candidate_target_gain_vs_pbb9"]) for row in cell_rows]
        centered_feature = []
        centered_gain = []
        for rig in rigs:
            for family in families:
                selected = [
                    row
                    for row in field_rows
                    if row["rig_id"] == rig and row["family"] == family
                ]
                mean_feature = float(np.mean([float(row[name]) for row in selected]))
                mean_gain = float(
                    np.mean(
                        [float(row["candidate_target_gain_vs_pbb9"]) for row in selected]
                    )
                )
                centered_feature.extend(
                    [float(row[name]) - mean_feature for row in selected]
                )
                centered_gain.extend(
                    [
                        float(row["candidate_target_gain_vs_pbb9"]) - mean_gain
                        for row in selected
                    ]
                )
        per_rig = []
        leave_one_rig_oriented = []
        for rig in rigs:
            held_out = [row for row in field_rows if row["rig_id"] == rig]
            remainder = [row for row in field_rows if row["rig_id"] != rig]
            test_corr = _finite_spearman(
                [float(row[name]) for row in held_out],
                [float(row["candidate_target_gain_vs_pbb9"]) for row in held_out],
            )
            train_corr = _finite_spearman(
                [float(row[name]) for row in remainder],
                [float(row["candidate_target_gain_vs_pbb9"]) for row in remainder],
            )
            if test_corr is not None:
                per_rig.append(test_corr)
            if test_corr is not None and train_corr is not None and train_corr != 0.0:
                leave_one_rig_oriented.append(
                    float(np.sign(train_corr) * test_corr)
                )
        output.append(
            {
                "feature": name,
                "field_spearman": _finite_spearman(feature, gain),
                "cell_spearman": _finite_spearman(cell_feature, cell_gain),
                "within_rig_family_centered_spearman": _finite_spearman(
                    centered_feature, centered_gain
                ),
                "per_rig_spearman_median": (
                    float(np.median(per_rig)) if per_rig else None
                ),
                "per_rig_same_sign_fraction": (
                    float(
                        np.mean(
                            np.sign(per_rig)
                            == np.sign(_finite_spearman(feature, gain) or 0.0)
                        )
                    )
                    if per_rig
                    else None
                ),
                "leave_one_rig_oriented_positive_fraction": (
                    float(np.mean(np.asarray(leave_one_rig_oriented) > 0.0))
                    if leave_one_rig_oriented
                    else None
                ),
                "leave_one_rig_oriented_median": (
                    float(np.median(leave_one_rig_oriented))
                    if leave_one_rig_oriented
                    else None
                ),
            }
        )
    return output


def _zero_threshold_sanity_check(
    bundle: Any,
    source_rows: Sequence[Mapping[str, Any]],
    target_gain: Mapping[str, float],
    candidate_prediction: np.ndarray,
    pbb9_prediction: np.ndarray,
) -> dict[str, Any]:
    """Score one natural post-open rule without authorizing it."""

    selected = {
        str(row["field_uid"]): float(row["candidate_source_gain_vs_pbb9"]) > 0.0
        for row in source_rows
    }
    indices = [int(row["row_index"]) for row in bundle.rows]
    gated_prediction = np.stack(
        [
            candidate_prediction[position]
            if selected[str(bundle.rows[index]["field_uid"])]
            else pbb9_prediction[position]
            for position, index in enumerate(indices)
        ]
    )
    aggregates, cells = _score_methods(
        bundle,
        indices,
        {
            "postopen_zero_threshold": gated_prediction,
            "source_only_pbb_9": pbb9_prediction,
        },
    )
    cell_gains = []
    for gate_cell, baseline_cell in zip(
        cells["postopen_zero_threshold"],
        cells["source_only_pbb_9"],
        strict=True,
    ):
        cell_gains.append(
            1.0
            - float(gate_cell["mean_whitened_rmse"])
            / max(float(baseline_cell["mean_whitened_rmse"]), 1e-15)
        )
    selected_uids = [field_uid for field_uid, value in selected.items() if value]
    per_rig = []
    for rig in sorted({str(row["rig_id"]) for row in source_rows}):
        rig_uids = [
            str(row["field_uid"])
            for row in source_rows
            if row["rig_id"] == rig and selected[str(row["field_uid"])]
        ]
        per_rig.append(
            {
                "rig_id": rig,
                "selected_fields": len(rig_uids),
                "selected_mean_target_gain": (
                    float(np.mean([target_gain[field_uid] for field_uid in rig_uids]))
                    if rig_uids
                    else None
                ),
                "selected_target_harm_fraction": (
                    float(np.mean([target_gain[field_uid] < 0.0 for field_uid in rig_uids]))
                    if rig_uids
                    else None
                ),
            }
        )
    return {
        "rule": "select candidate iff candidate source standardized residual RMS is below PBB-9",
        "status": "post-open descriptive sanity check; not preregistered or authorized",
        "field_coverage": len(selected_uids) / max(len(source_rows), 1),
        "selected_field_mean_target_gain": float(
            np.mean([target_gain[field_uid] for field_uid in selected_uids])
        ),
        "selected_field_target_harm_fraction": float(
            np.mean([target_gain[field_uid] < 0.0 for field_uid in selected_uids])
        ),
        "gated_ratio_of_cluster_means_gain_vs_pbb9": 1.0
        - aggregates["postopen_zero_threshold"]
        / max(aggregates["source_only_pbb_9"], 1e-15),
        "gated_cell_mean_gain_vs_pbb9": float(np.mean(cell_gains)),
        "gated_positive_cell_fraction": float(np.mean(np.asarray(cell_gains) > 0.0)),
        "gated_worst_cell_degradation": max(0.0, -float(np.min(cell_gains))),
        "per_rig": per_rig,
    }


def run() -> dict[str, Any]:
    preregistration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base = json.loads(
        (ROOT / preregistration["source_config"]).read_text(encoding="utf-8")
    )
    source_report = json.loads(
        (ROOT / preregistration["source_report"]).read_text(encoding="utf-8")
    )
    v5p_report = json.loads(
        (V5P_OUTPUT_DIR / "report.json").read_text(encoding="utf-8")
    )
    if v5p_report["decision"] != "FRESH_DEVELOPMENT_NO_GO":
        raise RuntimeError("v5q is defined only as a diagnosis of failed v5p")

    bundle = build_dataset(fresh_dataset_config(base, preregistration))
    models, checkpoint_hashes = _load_models(base, preregistration, source_report)
    members, priors = _shared_prior_members(
        bundle,
        models,
        batch_size=int(base["training"]["batch_size"]),
    )
    candidate_config = preregistration["candidate"]
    candidate = _correction_map(
        bundle,
        lambda row: prior_anchored_bb_correction(
            row,
            priors[str(row["field_uid"])],
            iterations=int(candidate_config["refinement_iterations"]),
            relative_anchor=float(candidate_config["relative_anchor"]),
        ),
    )
    pbb9 = _correction_map(
        bundle, lambda row: projected_bb_correction(row, 9)
    )
    pbb11 = _correction_map(
        bundle, lambda row: projected_bb_correction(row, 11)
    )
    pbb32 = _correction_map(
        bundle, lambda row: projected_bb_correction(row, 32)
    )
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
    predictions = {
        method: _target_prediction(bundle, values)[1]
        for method, values in corrections.items()
    }
    reproduced_prediction_sha256 = sha256_state_dict(
        {
            f"{method}_target_prediction": value
            for method, value in predictions.items()
        }
    )
    if reproduced_prediction_sha256 != v5p_report[
        "prediction_sha256_before_scoring"
    ]:
        raise RuntimeError("v5q failed to reproduce frozen v5p predictions")

    source_rows = []
    for index in _unique_field_indices(bundle):
        row = bundle.rows[index]
        field_uid = str(row["field_uid"])
        source_rows.append(
            compute_source_feature_row(
                row,
                members[field_uid],
                priors[field_uid],
                candidate[field_uid],
                pbb9[field_uid],
                depth=int(base["depth"]),
                grid_size=int(base["grid_size"]),
            )
        )
    source_feature_sha256_before_label_access = sha256_json(source_rows)

    target_gain = _target_gain_by_field(bundle, candidate, pbb9)
    field_rows = [
        {
            **row,
            "candidate_target_gain_vs_pbb9": target_gain[str(row["field_uid"])],
        }
        for row in source_rows
    ]
    cells = _cell_rows(field_rows)
    correlations = _correlation_rows(field_rows, cells)
    zero_threshold_sanity = _zero_threshold_sanity_check(
        bundle,
        source_rows,
        target_gain,
        predictions["sfio_papbb_budget_vector_v1"],
        predictions["source_only_pbb_9"],
    )
    ranked = sorted(
        correlations,
        key=lambda row: abs(float(row["cell_spearman"] or 0.0)),
        reverse=True,
    )
    family_summary = []
    for family in sorted({str(row["family"]) for row in field_rows}):
        selected = [row for row in field_rows if row["family"] == family]
        family_summary.append(
            {
                "family": family,
                "field_count": len(selected),
                "mean_target_gain_vs_pbb9": float(
                    np.mean(
                        [float(row["candidate_target_gain_vs_pbb9"]) for row in selected]
                    )
                ),
                "positive_field_fraction": float(
                    np.mean(
                        [float(row["candidate_target_gain_vs_pbb9"]) > 0.0 for row in selected]
                    )
                ),
            }
        )

    report = {
        "schema": "v5q-postopen-topology-diagnosis-1",
        "evidence_label": "post_open_hypothesis_generation_only",
        "claim_ceiling": "No gate, router, algorithm improvement, design lock, OERF, or publication superiority claim is authorized.",
        "chronology": {
            "consumes_failed_v5p": True,
            "model_or_hyperparameter_changed": False,
            "threshold_or_router_fitted": False,
            "target_labels_accessed_only_after_source_feature_hash": True,
        },
        "v5p_prediction_sha256": v5p_report[
            "prediction_sha256_before_scoring"
        ],
        "reproduced_prediction_sha256": reproduced_prediction_sha256,
        "source_feature_sha256_before_label_access": source_feature_sha256_before_label_access,
        "checkpoint_hashes": checkpoint_hashes,
        "source_provenance": {
            "direct_dependency_sha256": relative_file_hashes(
                ROOT,
                [
                    Path(__file__),
                    ROOT / "run_v5p_fresh_budget_gate.py",
                    CONFIG_PATH,
                    ROOT / preregistration["source_config"],
                    ROOT / preregistration["source_report"],
                    V5P_OUTPUT_DIR / "report.json",
                    ROOT / "release_provenance.py",
                    ROOT / "gc_rio" / "data.py",
                    ROOT / "gc_rio" / "protocol.py",
                    ROOT / "gc_rio" / "shared_field_model.py",
                    ROOT / "gc_rio" / "training.py",
                    ROOT / "run_v5h_gc_rio_development.py",
                    ROOT / "run_v5n_strong_classical_baselines.py",
                    ROOT / "run_v5o_prior_anchored_frontier.py",
                    *[
                        ROOT
                        / "results"
                        / "v5k_shared_field_work"
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
                "V5Q replays frozen CPU predictions and requires the V5P "
                "prediction hash to match before any post-open diagnosis."
            ),
        },
        "sample_accounting": {
            "fields": len(field_rows),
            "rigs": len({row["rig_id"] for row in field_rows}),
            "families": len({row["family"] for row in field_rows}),
            "rig_family_cells": len(cells),
        },
        "feature_definitions": {
            "candidate_source_gain_vs_pbb9": "One minus the ratio of source standardized residual RMS; observable at inference but not an independent target metric.",
            "candidate_minus_pbb9_source_visibility": "RMS of the whitened source projection of the candidate-PBB field difference divided by its active-field RMS.",
            "gradient_alignment": "Dominant-minus-second eigenvalue of the 3D gradient structure tensor divided by its trace; a source-reconstruction morphology proxy.",
            "prior_member_disagreement_over_base": "Voxel RMS spread across the three frozen prior members divided by active base-field RMS.",
        },
        "family_summary": family_summary,
        "correlations": correlations,
        "descriptive_top_cell_associations": ranked[:6],
        "postopen_zero_threshold_sanity_check": zero_threshold_sanity,
        "decision": "NO_RELIABILITY_GATE_AUTHORIZED_POSTOPEN_DIAGNOSIS_ONLY",
        "interpretation_rules": [
            "Cell and field correlations are exploratory because the same opened v5p labels generated this diagnosis.",
            "A high overall correlation can be caused entirely by family separation; within-rig-family-centered and leave-one-rig summaries must be inspected.",
            "No p-value or threshold authorizes a router on these data.",
            "Any reliability rule must be frozen and tested on genuinely new families or sessions with PBB fallback.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "field_features.csv", field_rows)
    _write_csv(OUTPUT_DIR / "cell_features.csv", cells)
    _write_csv(OUTPUT_DIR / "correlations.csv", correlations)
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(
        OUTPUT_DIR,
        ["field_features.csv", "cell_features.csv", "correlations.csv", "report.json"],
    )
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "sample_accounting": report["sample_accounting"],
                "family_summary": report["family_summary"],
                "top_associations": report["descriptive_top_cell_associations"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
