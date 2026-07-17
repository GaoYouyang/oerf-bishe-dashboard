#!/usr/bin/env python3
"""Run the opened M2.2 exact-nullspace headroom oracle on the 12^3 fixture.

The dense SVD is deliberately an oracle, not a deployable reconstruction
method and not a speed comparison.  It asks only whether an already trained
residual contains a useful component in the numerical kernel of the frozen
voxel inverse operator.  It does not imply invisibility to an independent
continuous renderer or to real finite-aperture BOST measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_t16_operator.jacru_m2_exact_nullspace_oracle import (
    ExactDenseNullspaceProjector,
    build_exact_dense_nullspace_projector,
)
from site_tools import run_jacru_m2_learned_residual_gate as m2


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/jacru_m2_2_exact_nullspace_oracle_postopen_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/jacru_m2_2_exact_nullspace_oracle_postopen_public"
)
SCHEMA = "jacru-m2-2-exact-nullspace-oracle-postopen-report-1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-limit", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_rows(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["method"], int(row["model_seed"]), row["case_id"]): row
            for row in csv.DictReader(handle)
        }


@torch.no_grad()
def _assemble_active_matrix_batched(
    operator,
    *,
    support: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return A[:, active] in CPU float64 with an explicit setup ledger."""

    if batch_size < 1:
        raise ValueError("assembly batch_size must be positive")
    mask = support.to(device=operator.support.device, dtype=torch.bool).reshape(-1)
    active = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if active.numel() < 1:
        raise ValueError("support must retain active voxels")
    operator.reset_call_counts()
    zero = torch.zeros(
        (1, 1, *operator.grid_shape),
        dtype=operator.support.dtype,
        device=operator.support.device,
    )
    zero_output = operator.forward(zero)
    zero_max = float(torch.max(torch.abs(zero_output)))
    if zero_max > 64.0 * torch.finfo(operator.support.dtype).eps:
        raise RuntimeError("dense oracle requires a linear zero-preserving operator")
    columns: list[torch.Tensor] = []
    voxel_count = int(np.prod(operator.grid_shape))
    for start in range(0, int(active.numel()), batch_size):
        indices = active[start : start + batch_size]
        basis = torch.zeros(
            (len(indices), voxel_count),
            dtype=operator.support.dtype,
            device=operator.support.device,
        )
        basis[torch.arange(len(indices), device=basis.device), indices] = 1.0
        projected = operator.forward(basis.reshape(len(indices), 1, *operator.grid_shape))
        columns.append(projected.reshape(len(indices), -1).detach().cpu())
    active_by_measurement = torch.cat(columns, dim=0)
    matrix = active_by_measurement.mT.to(torch.float64).contiguous()
    calls = operator.call_report()
    expected = 1 + math.ceil(int(active.numel()) / batch_size)
    if calls != {"forward_calls": expected, "adjoint_calls": 0}:
        raise RuntimeError(f"dense assembly call ledger drifted: {calls}")
    return matrix, {
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "active_voxel_count": int(active.numel()),
        "measurement_count": int(matrix.shape[0]),
        "setup_forward_calls": expected,
        "zero_forward_maximum_absolute": zero_max,
        "status": "DENSE_TOY_ORACLE_SETUP_NOT_RECONSTRUCTION_BUDGET",
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["method"]), int(row["model_seed"]), str(row["split"])),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    for (method, seed, split), values in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "model_seed": seed,
                "split": split,
                "case_count": len(values),
                "reference_field_relative_l2_mean": float(
                    np.mean([row["reference_field_relative_l2"] for row in values])
                ),
                "original_field_relative_l2_mean": float(
                    np.mean([row["original_field_relative_l2"] for row in values])
                ),
                "oracle_field_relative_l2_mean": float(
                    np.mean([row["field_relative_l2"] for row in values])
                ),
                "oracle_h1_relative_error_mean": float(
                    np.mean([row["h1_seminorm_relative_error"] for row in values])
                ),
                "field_gain_to_reference_mean": float(
                    np.mean([row["field_gain_to_reference"] for row in values])
                ),
                "h1_gain_to_reference_mean": float(
                    np.mean([row["h1_gain_to_reference"] for row in values])
                ),
                "original_gain_retention_mean": float(
                    np.mean([row["original_gain_retention"] for row in values])
                ),
                "measured_reprojection_ratio_to_reference_mean": float(
                    np.mean(
                        [row["measured_reprojection_ratio_to_reference"] for row in values]
                    )
                ),
                "clean_reprojection_ratio_to_reference_mean": float(
                    np.mean([row["clean_reprojection_ratio_to_reference"] for row in values])
                ),
                "null_correction_energy_fraction_mean": float(
                    np.mean([row["null_correction_energy_fraction"] for row in values])
                ),
                "row_correction_energy_fraction_mean": float(
                    np.mean([row["row_correction_energy_fraction"] for row in values])
                ),
                "visible_null_correction_fraction_maximum": float(
                    np.max([row["visible_null_correction_fraction"] for row in values])
                ),
                "internal_projection_residual_maximum": float(
                    np.max([row["internal_projection_residual"] for row in values])
                ),
                "field_gain_to_reference_minimum": float(
                    np.min([row["field_gain_to_reference"] for row in values])
                ),
                "field_harm_rate": float(
                    np.mean([row["field_harm_to_reference"] for row in values])
                ),
            }
        )
    return output


def _decisions(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
    gates: dict[str, Any],
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for method in methods:
        diagnostics: dict[str, Any] = {}
        checks: dict[str, bool] = {}
        selected = [row for row in rows if row["method"] == method]
        for split in ("development", "ood"):
            values = [row for row in selected if row["split"] == split]
            gains = np.asarray([row["field_gain_to_reference"] for row in values])
            h1_gains = np.asarray([row["h1_gain_to_reference"] for row in values])
            retention = np.asarray([row["original_gain_retention"] for row in values])
            ratios = np.asarray(
                [row["measured_reprojection_ratio_to_reference"] for row in values]
            )
            harm = gains < -float(gates["field_harm_threshold_fraction"])
            diagnostics[f"{split}_field_gain_mean"] = float(np.mean(gains))
            diagnostics[f"{split}_h1_gain_mean"] = float(np.mean(h1_gains))
            diagnostics[f"{split}_original_gain_retention_mean"] = float(
                np.mean(retention)
            )
            diagnostics[f"{split}_reprojection_ratio_mean"] = float(np.mean(ratios))
            diagnostics[f"{split}_field_harm_rate"] = float(np.mean(harm))
            diagnostics[f"{split}_worst_field_gain"] = float(np.min(gains))
            diagnostics[f"{split}_visible_null_fraction_maximum"] = float(
                np.max([row["visible_null_correction_fraction"] for row in values])
            )
            diagnostics[f"{split}_projection_residual_maximum"] = float(
                np.max([row["internal_projection_residual"] for row in values])
            )
            seed_means = [
                float(
                    np.mean(
                        [
                            row["field_gain_to_reference"]
                            for row in values
                            if int(row["model_seed"]) == seed
                        ]
                    )
                )
                for seed in sorted({int(row["model_seed"]) for row in values})
            ]
            diagnostics[f"{split}_per_seed_field_gain_means"] = seed_means
            checks[f"{split}_field_gain"] = float(np.mean(gains)) >= float(
                gates[f"{split}_field_gain_to_reference_minimum_fraction"]
            )
            checks[f"{split}_h1_gain"] = float(np.mean(h1_gains)) >= float(
                gates[f"{split}_h1_gain_to_reference_minimum_fraction"]
            )
            checks[f"{split}_retention"] = float(np.mean(retention)) >= float(
                gates[f"{split}_original_gain_retention_minimum_fraction"]
            )
            checks[f"{split}_reprojection"] = float(np.mean(ratios)) <= float(
                gates[f"{split}_reprojection_ratio_to_reference_maximum"]
            )
            checks[f"{split}_harm"] = float(np.mean(harm)) <= float(
                gates["field_harm_rate_maximum"]
            )
            checks[f"{split}_worst_case"] = float(np.min(gains)) >= float(
                gates["worst_field_gain_minimum_fraction"]
            )
            checks[f"{split}_all_seed_means_positive"] = (
                not bool(gates["require_all_model_seed_mean_field_gains_positive"])
                or all(value > 0.0 for value in seed_means)
            )
            checks[f"{split}_projection_residual"] = diagnostics[
                f"{split}_projection_residual_maximum"
            ] <= float(gates["maximum_internal_projection_residual"])
            checks[f"{split}_visible_null_fraction"] = diagnostics[
                f"{split}_visible_null_fraction_maximum"
            ] <= float(gates["maximum_visible_null_correction_fraction"])
        decisions[method] = {
            "checks": checks,
            "diagnostics": diagnostics,
            "passed_exact_oracle_headroom_gate": all(checks.values()),
        }
    return decisions


def _plot(path: Path, rows: list[dict[str, Any]], methods: list[str]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    colors = {"jacru_m2": "#146c94", "pooled_cnn": "#d95f59"}
    labels = {"jacru_m2": "JACRU-M2", "pooled_cnn": "Pooled CNN"}
    splits = ("development", "ood")
    x = np.arange(len(splits), dtype=np.float64)
    width = 0.16
    for method_index, method in enumerate(methods):
        values = [[row for row in rows if row["method"] == method and row["split"] == split] for split in splits]
        reference = [np.mean([row["reference_field_relative_l2"] for row in group]) for group in values]
        original = [np.mean([row["original_field_relative_l2"] for row in group]) for group in values]
        oracle = [np.mean([row["field_relative_l2"] for row in group]) for group in values]
        offset = (method_index - 0.5) * 3 * width
        axes[0, 0].bar(x + offset - width, reference, width, color="#8d99a6", alpha=0.75, label=f"{labels[method]} reference")
        axes[0, 0].bar(x + offset, original, width, color=colors[method], alpha=0.38, label=f"{labels[method]} raw")
        axes[0, 0].bar(x + offset + width, oracle, width, color=colors[method], label=f"{labels[method]} oracle")
        raw_reprojection = [np.mean([row["original_measured_reprojection_ratio_to_reference"] for row in group]) for group in values]
        projected_reprojection = [np.mean([row["measured_reprojection_ratio_to_reference"] for row in group]) for group in values]
        axes[0, 1].plot(x, raw_reprojection, marker="o", color=colors[method], linestyle="--", label=f"{labels[method]} raw")
        axes[0, 1].plot(x, projected_reprojection, marker="s", color=colors[method], label=f"{labels[method]} oracle")
        null_fraction = [np.mean([row["null_correction_energy_fraction"] for row in group]) for group in values]
        row_fraction = [np.mean([row["row_correction_energy_fraction"] for row in group]) for group in values]
        axes[1, 0].plot(x, null_fraction, marker="o", color=colors[method], label=f"{labels[method]} null")
        axes[1, 0].plot(x, row_fraction, marker="s", color=colors[method], linestyle="--", label=f"{labels[method]} row")
        retention = [np.mean([row["original_gain_retention"] for row in group]) for group in values]
        axes[1, 1].plot(x, retention, marker="o", linewidth=2, color=colors[method], label=labels[method])

    tick_labels = ("development", "exploratory OOD")
    for ax in axes.reshape(-1):
        ax.set_xticks(x, tick_labels)
    axes[0, 0].set_ylabel("field relative-L2")
    axes[0, 0].set_title("CGLS-24 reference, raw prior, exact-null oracle")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(1.0, color="#222", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("measured reprojection ratio to CGLS-24")
    axes[0, 1].set_title("projection consistency (log scale)")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylabel("correction norm fraction")
    axes[1, 0].set_title("where the learned correction lives")
    axes[1, 1].axhline(0.25, color="#222", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("retained fraction of raw field gain")
    axes[1, 1].set_title("headroom retained after exact projection")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("M2.2 exact dense null-space headroom oracle · opened synthetic T0", fontsize=15, fontweight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    source_config_path = ROOT / config["source_t0_config"]
    source_result_dir = ROOT / config["source_t0_results"]
    if _sha256(source_config_path) != config["source_t0_config_sha256"]:
        raise RuntimeError("source T0 config hash drifted")
    if _sha256(source_result_dir / "summary.json") != config["source_t0_summary_sha256"]:
        raise RuntimeError("source T0 summary hash drifted")
    source_config = _read_json(source_config_path)
    methods = [str(value) for value in config["methods"]]
    if not set(methods).issubset(set(source_config["methods"])):
        raise ValueError("oracle methods must be source T0 methods")
    if int(np.prod(source_config["fixture"]["grid_shape"])) > int(
        config["dense_oracle"]["maximum_grid_voxels"]
    ):
        raise ValueError("grid exceeds frozen dense-oracle limit")
    if args.seed_limit is not None:
        if args.seed_limit < 1:
            raise ValueError("seed-limit must be positive")
        source_config = json.loads(json.dumps(source_config))
        for split in source_config["splits"].values():
            split["base_seeds"] = split["base_seeds"][: args.seed_limit]
        source_config["training"]["model_seeds"] = source_config["training"]["model_seeds"][: args.seed_limit]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fixture = m2._fixture_config(source_config)
    records = m2._prepare_records(source_config, fixture)
    device = m2._choose_device(args.device or source_config["training"]["device"])
    trained: list[dict[str, Any]] = []
    for method in methods:
        for seed in source_config["training"]["model_seeds"]:
            trained.append(
                m2._train_one(
                    method=method,
                    seed=int(seed),
                    config=source_config,
                    records=records,
                    device=device,
                    epoch_override=args.epochs,
                )
            )
    source_rows = _source_rows(source_result_dir / "metric_rows.csv")

    projectors: dict[str, ExactDenseNullspaceProjector] = {}
    setup_ledger: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.split == "train":
            continue
        digest = record.case.inference.geometry.digest
        if digest in projectors:
            continue
        operator = record.case.inference.operator
        matrix, setup = _assemble_active_matrix_batched(
            operator,
            support=operator.support,
            batch_size=int(config["dense_oracle"]["assembly_batch_size"]),
        )
        factor_started = time.perf_counter()
        projector = build_exact_dense_nullspace_projector(
            support=operator.support,
            dense_matrix=matrix,
            rank_rtol=float(config["dense_oracle"]["rank_relative_tolerance"]),
            rank_atol=float(config["dense_oracle"]["rank_absolute_tolerance"]),
        )
        setup.update(
            {
                "rank": projector.rank,
                "nullity_lower_bound": projector.dense_active_matrix.shape[1] - projector.rank,
                "largest_singular_value": float(projector.singular_values[0]),
                "smallest_retained_singular_value": float(projector.singular_values[projector.rank - 1]) if projector.rank else None,
                "rank_tolerance": projector.rank_tolerance,
                "factorization_seconds": time.perf_counter() - factor_started,
            }
        )
        projectors[digest] = projector
        setup_ledger[digest] = setup

    reference_iterations = int(config["reference"]["iterations"])
    references: dict[str, dict[str, Any]] = {}
    reference_rows: list[dict[str, Any]] = []
    for record in records:
        if record.split == "train":
            continue
        operator = record.case.inference.operator
        forward, adjoint = m2._operator_maps(operator)
        operator.reset_call_counts()
        result = m2.cgls_baseline(
            record.case.inference.observations_uv[0],
            forward=forward,
            adjoint=adjoint,
            support=operator.support,
            spacing_xyz=m2._spacing(fixture),
            iterations=reference_iterations,
        )
        if result.forward_calls != reference_iterations or result.adjoint_calls != reference_iterations:
            raise RuntimeError("CGLS reference call budget drifted")
        scored = m2._score_prediction(
            record=record,
            method="cgls_24_reference",
            model_seed=-1,
            prediction=result.field,
            gate=None,
            correction_rms=None,
            optimization_forward_calls=reference_iterations,
            optimization_adjoint_calls=reference_iterations,
            grouped_adjoint_calls=0,
            neural_inference_seconds=0.0,
        )
        references[record.case.inference.case_id] = {"field": result.field.detach().clone(), "score": scored}
        reference_rows.append(scored)

    rows: list[dict[str, Any]] = []
    reproduction_deltas: list[dict[str, Any]] = []
    for run in trained:
        model = run["model"]
        model_device = next(model.parameters()).device
        for record in records:
            if record.split == "train":
                continue
            kwargs = m2._to_device(record.batch.model_kwargs(), model_device)
            with torch.no_grad():
                prediction, gate = model(**kwargs, return_gate=True)
            operator = record.case.inference.operator
            original = prediction[0, 0].detach().cpu().to(operator.support)
            original_score = m2._score_prediction(
                record=record,
                method=str(run["method"]),
                model_seed=int(run["model_seed"]),
                prediction=original,
                gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                correction_rms=float(torch.sqrt(torch.mean((original - record.batch.base_field[0, 0].to(original)).square()))),
                optimization_forward_calls=13,
                optimization_adjoint_calls=13,
                grouped_adjoint_calls=1,
                neural_inference_seconds=0.0,
            )
            source = source_rows[(str(run["method"]), int(run["model_seed"]), record.case.inference.case_id)]
            reproduction_deltas.append(
                {
                    "method": run["method"],
                    "model_seed": run["model_seed"],
                    "case_id": record.case.inference.case_id,
                    "field_absolute_delta": abs(float(original_score["field_relative_l2"]) - float(source["field_relative_l2"])),
                    "reprojection_absolute_delta": abs(float(original_score["measured_reprojection_relative_l2"]) - float(source["measured_reprojection_relative_l2"])),
                }
            )
            reference = references[record.case.inference.case_id]
            reference_field = reference["field"].to(original)
            correction = (original - reference_field) * operator.support
            projector = projectors[record.case.inference.geometry.digest]
            projected = projector.project(correction)
            oracle_field = reference_field + projected.null_space_correction.to(reference_field)
            oracle_score = m2._score_prediction(
                record=record,
                method=str(run["method"]),
                model_seed=int(run["model_seed"]),
                prediction=oracle_field,
                gate=float(gate[0, 0, 0, 0, 0].detach().cpu()),
                correction_rms=float(torch.sqrt(torch.mean(projected.null_space_correction.square()))),
                optimization_forward_calls=reference_iterations,
                optimization_adjoint_calls=reference_iterations,
                grouped_adjoint_calls=1,
                neural_inference_seconds=0.0,
            )
            reference_score = reference["score"]
            ref_field = float(reference_score["field_relative_l2"])
            ref_h1 = float(reference_score["h1_seminorm_relative_error"])
            original_gain = (ref_field - float(original_score["field_relative_l2"])) / ref_field
            oracle_gain = (ref_field - float(oracle_score["field_relative_l2"])) / ref_field
            correction_active = correction.reshape(-1)[projector.support_mask.reshape(-1)]
            null_active = projected.null_space_correction.reshape(-1)[projector.support_mask.reshape(-1)]
            row_norm = float(torch.linalg.vector_norm(projected.row_space_correction))
            null_norm = float(torch.linalg.vector_norm(projected.null_space_correction))
            correction_norm = float(torch.linalg.vector_norm(correction).clamp_min(1e-30))
            visible_denominator = float(
                torch.linalg.vector_norm(projector.dense_active_matrix @ correction_active).clamp_min(1e-30)
            )
            visible_fraction = float(
                torch.linalg.vector_norm(projector.dense_active_matrix @ null_active)
            ) / visible_denominator
            oracle_score.update(
                {
                    "reference_method": "cgls_24",
                    "reference_field_relative_l2": ref_field,
                    "reference_h1_relative_error": ref_h1,
                    "reference_measured_reprojection_relative_l2": float(reference_score["measured_reprojection_relative_l2"]),
                    "reference_clean_reprojection_relative_l2": float(reference_score["clean_reprojection_relative_l2"]),
                    "original_field_relative_l2": float(original_score["field_relative_l2"]),
                    "original_h1_relative_error": float(original_score["h1_seminorm_relative_error"]),
                    "original_measured_reprojection_ratio_to_reference": float(original_score["measured_reprojection_relative_l2"]) / max(float(reference_score["measured_reprojection_relative_l2"]), 1e-30),
                    "field_gain_to_reference": oracle_gain,
                    "h1_gain_to_reference": (ref_h1 - float(oracle_score["h1_seminorm_relative_error"])) / ref_h1,
                    "original_field_gain_to_reference": original_gain,
                    "original_gain_retention": oracle_gain / max(original_gain, 1e-30),
                    "measured_reprojection_ratio_to_reference": float(oracle_score["measured_reprojection_relative_l2"]) / max(float(reference_score["measured_reprojection_relative_l2"]), 1e-30),
                    "clean_reprojection_ratio_to_reference": float(oracle_score["clean_reprojection_relative_l2"]) / max(float(reference_score["clean_reprojection_relative_l2"]), 1e-30),
                    "row_correction_energy_fraction": row_norm / correction_norm,
                    "null_correction_energy_fraction": null_norm / correction_norm,
                    "visible_null_correction_fraction": visible_fraction,
                    "numerical_rank": projected.rank,
                    "active_voxel_count": projected.active_voxel_count,
                    "numerical_nullity_lower_bound": projected.active_voxel_count - projected.rank,
                    "internal_projection_residual": projected.internal_projection_residual,
                    "nullspace_residual": projected.nullspace_residual,
                    "field_harm_to_reference": int(float(oracle_score["field_relative_l2"]) > ref_field * (1.0 + float(config["decision_gates"]["field_harm_threshold_fraction"]))),
                    "oracle_setup_excluded_from_reconstruction_budget": True,
                }
            )
            rows.append(oracle_score)

    reproduction = {
        "row_count": len(reproduction_deltas),
        "maximum_field_absolute_delta": max(row["field_absolute_delta"] for row in reproduction_deltas),
        "maximum_reprojection_absolute_delta": max(row["reprojection_absolute_delta"] for row in reproduction_deltas),
    }
    reproduction["passed_1e_6"] = bool(
        reproduction["maximum_field_absolute_delta"] <= 1e-6
        and reproduction["maximum_reprojection_absolute_delta"] <= 1e-6
    )
    if args.epochs is None and args.seed_limit is None and not reproduction["passed_1e_6"]:
        raise RuntimeError(f"source prediction reproduction failed: {reproduction}")
    aggregate = _aggregate(rows)
    decisions = _decisions(rows, methods=methods, gates=config["decision_gates"])
    any_headroom = any(value["passed_exact_oracle_headroom_gate"] for value in decisions.values())
    summary = {
        "schema_version": SCHEMA,
        "status": "M2_2_EXACT_NULLSPACE_HEADROOM_FOUND_ORACLE_ONLY" if any_headroom else "M2_2_EXACT_NULLSPACE_ORACLE_NO_GO",
        "evidence_level": config["evidence_level"],
        "source_config_sha256": _sha256(config_path),
        "source_t0_config_sha256": _sha256(source_config_path),
        "source_t0_summary_sha256": _sha256(source_result_dir / "summary.json"),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "metric_row_count": len(rows),
        "reference_row_count": len(reference_rows),
        "zero_step_source_reproduction": reproduction,
        "dense_setup_ledger": setup_ledger,
        "training_runs": [{key: value for key, value in run.items() if key not in {"model", "history"}} for run in trained],
        "aggregate": aggregate,
        "decisions": decisions,
        "authorization": {
            "claim_deployable_algorithm": False,
            "claim_method_superiority": False,
            "claim_real_bost_generalization": False,
            "open_fresh_or_final": False,
            "continue_matrix_free_projection_research": any_headroom,
        },
        "claim_boundary": config["claim_boundary"],
        "public_export_policy": {
            "contains_model_checkpoints": False,
            "contains_restricted_papers": False,
            "contains_private_experimental_arrays": False,
        },
    }
    _write_csv(output / "metric_rows.csv", rows)
    _write_csv(output / "aggregate_rows.csv", aggregate)
    _write_csv(output / "reference_rows.csv", reference_rows)
    _write_csv(output / "zero_step_reproduction.csv", reproduction_deltas)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot(output / "diagnostic", rows, methods)
    readme = f"""# M2.2 exact dense null-space headroom oracle

Status: `{summary['status']}`

This opened synthetic T0 experiment uses a full CPU float64 SVD on each frozen
12-cubed voxel operator. It is an oracle and has no runtime, efficiency,
deployment, real-BOST, or fresh-data claim. The dense setup is excluded from
the reconstruction call budget and reported separately. Numerical `ker(A)`
belongs only to the approximate inverse operator; it is not the true optical
null space. Zero-step source reproduction passed: `{reproduction['passed_1e_6']}`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifacts = (
        "README.md",
        "aggregate_rows.csv",
        "diagnostic.pdf",
        "diagnostic.png",
        "metric_rows.csv",
        "reference_rows.csv",
        "summary.json",
        "zero_step_reproduction.csv",
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(output / name)}  {name}\n" for name in artifacts),
        encoding="ascii",
    )
    print(json.dumps({"status": summary["status"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
