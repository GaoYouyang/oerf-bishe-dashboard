#!/usr/bin/env python3
"""Run a toy conformance check for the V6B limited-query protocol.

This runner deliberately includes both an in-class truth and a misspecified
truth.  It verifies plumbing, accounting and failure visibility; it is not an
external renderer result, a fresh V6B gate, or evidence for an OERF claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.limited_query_calibration import (
        BudgetedForwardOracle,
        NominalLinearOperator,
        RayKernelChannelOperator,
        build_gate_design,
        collect_forward_observations,
        fit_channel_gates,
        sha256_arrays,
        voxel_kernel_offsets,
    )
else:
    from .limited_query_calibration import (
        BudgetedForwardOracle,
        NominalLinearOperator,
        RayKernelChannelOperator,
        build_gate_design,
        collect_forward_observations,
        fit_channel_gates,
        sha256_arrays,
        voxel_kernel_offsets,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v6b_protocol_conformance.json"
OUTPUT_DIR = ROOT / "results" / "v6b_protocol_conformance"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metrics table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _unit_columns(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    values = rng.normal(size=(rows, columns))
    values /= np.linalg.norm(values, axis=0, keepdims=True)
    return values


def _relative_action_error(
    candidate: np.ndarray, truth: np.ndarray, probes: np.ndarray
) -> float:
    target = truth @ probes
    return float(
        np.linalg.norm((candidate - truth) @ probes)
        / max(np.linalg.norm(target), np.finfo(float).tiny)
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.vdot(left, right) / max(denominator, np.finfo(float).tiny))


def _gradient_metrics(
    candidate: np.ndarray,
    truth: np.ndarray,
    target_fields: np.ndarray,
    iterates: np.ndarray,
) -> tuple[float, float]:
    cosines = []
    relative_errors = []
    for target_field, iterate in zip(target_fields, iterates, strict=True):
        observation = truth @ target_field
        truth_gradient = truth.T @ (truth @ iterate - observation)
        candidate_gradient = candidate.T @ (candidate @ iterate - observation)
        cosines.append(_cosine(candidate_gradient, truth_gradient))
        relative_errors.append(
            float(
                np.linalg.norm(candidate_gradient - truth_gradient)
                / max(np.linalg.norm(truth_gradient), np.finfo(float).tiny)
            )
        )
    return float(np.median(cosines)), float(np.median(relative_errors))


def _dot_defect(
    operator: RayKernelChannelOperator,
    rng: np.random.Generator,
    pairs: int,
) -> float:
    defects = []
    for _ in range(pairs):
        field = rng.normal(size=operator.input_size)
        residual = rng.normal(size=operator.output_size)
        left = float(np.vdot(operator.forward(field), residual))
        right = float(np.vdot(field, operator.adjoint(residual)))
        defects.append(abs(left - right) / max(abs(left), abs(right), 1e-15))
    return float(max(defects))


def _scaled_out_of_class_residual(
    rng: np.random.Generator, reference: np.ndarray, relative_norm: float
) -> np.ndarray:
    left = rng.normal(size=(reference.shape[0], 4))
    right = rng.normal(size=(4, reference.shape[1]))
    residual = left @ right
    residual *= (
        float(relative_norm)
        * np.linalg.norm(reference)
        / max(np.linalg.norm(residual), np.finfo(float).tiny)
    )
    return residual


def _minimum_norm_secant(
    nominal: np.ndarray,
    probes: np.ndarray,
    observations: np.ndarray,
    rcond: float,
) -> np.ndarray:
    residuals = observations - nominal @ probes
    return nominal + residuals @ np.linalg.pinv(probes, rcond=float(rcond))


def run(config_path: Path = CONFIG_PATH, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))
    shape = tuple(int(value) for value in config["input_shape"])
    input_size = int(np.prod(shape))
    output_size = int(config["output_size"])
    offsets = voxel_kernel_offsets(int(config["kernel_radius"]))
    budgets = [int(value) for value in config["query_budgets"]]
    max_budget = max(budgets)
    rows: list[dict[str, Any]] = []
    frozen_hash_rows: list[dict[str, Any]] = []

    for stratum in ("in_class", "out_of_class"):
        for rig_index in range(int(config["rigs_per_stratum"])):
            nominal_matrix = rng.normal(scale=0.15, size=(output_size, input_size))
            nominal = NominalLinearOperator.from_matrix(nominal_matrix)
            coefficients = rng.normal(
                scale=float(config["coefficient_scale"]),
                size=(output_size, len(offsets)),
            )
            truth_gates = rng.uniform(
                float(config["gate_low"]),
                float(config["gate_high"]),
                size=len(offsets),
            )
            in_class_truth = RayKernelChannelOperator(
                nominal,
                input_shape=shape,
                ray_coefficients=coefficients,
                gates=truth_gates,
                offsets=offsets,
            ).materialize()
            truth_matrix = in_class_truth.copy()
            if stratum == "out_of_class":
                truth_matrix += _scaled_out_of_class_residual(
                    rng,
                    truth_matrix,
                    float(config["out_of_class_relative_frobenius"]),
                )

            calibration_bank = _unit_columns(rng, input_size, max_budget)
            hidden_bank = _unit_columns(
                rng, input_size, int(config["hidden_probes"])
            )
            for budget in budgets:
                probes = calibration_bank[:, :budget]
                oracle = BudgetedForwardOracle(
                    lambda x, matrix=truth_matrix: matrix @ x,
                    input_size=input_size,
                    output_size=output_size,
                    budget=budget,
                )
                observations = collect_forward_observations(oracle, probes)
                design, target = build_gate_design(
                    nominal,
                    input_shape=shape,
                    ray_coefficients=coefficients,
                    offsets=offsets,
                    probes=probes,
                    observations=observations,
                )
                fitted_gates = fit_channel_gates(
                    design,
                    target,
                    relative_ridge=float(config["gate_relative_ridge"]),
                )
                candidate_operator = RayKernelChannelOperator(
                    nominal,
                    input_shape=shape,
                    ray_coefficients=coefficients,
                    gates=fitted_gates,
                    offsets=offsets,
                )
                candidate_matrix = candidate_operator.materialize()
                secant_matrix = _minimum_norm_secant(
                    nominal_matrix,
                    probes,
                    observations,
                    float(config["secant_rcond"]),
                )

                # These hashes are fixed before the evaluator computes hidden metrics.
                prediction_hash = sha256_arrays(fitted_gates, candidate_matrix)
                secant_hash = sha256_arrays(secant_matrix)
                frozen_hash_rows.append(
                    {
                        "stratum": stratum,
                        "rig": rig_index,
                        "K": budget,
                        "candidate_sha256": prediction_hash,
                        "secant_sha256": secant_hash,
                    }
                )
                _write_json(
                    output_dir / "prediction_hashes_before_scoring.json",
                    frozen_hash_rows,
                )

                gradient_points = int(config["gradient_points"])
                target_fields = rng.normal(size=(gradient_points, input_size))
                iterates = target_fields + 0.3 * rng.normal(
                    size=(gradient_points, input_size)
                )

                for method, matrix in (
                    ("nominal", nominal_matrix),
                    ("secant", secant_matrix),
                    ("channel_gate", candidate_matrix),
                ):
                    gradient_cosine, gradient_relative_error = _gradient_metrics(
                        matrix,
                        truth_matrix,
                        target_fields,
                        iterates,
                    )
                    rows.append(
                        {
                            "stratum": stratum,
                            "rig": rig_index,
                            "K": budget,
                            "method": method,
                            "hidden_action_relative_l2": _relative_action_error(
                                matrix, truth_matrix, hidden_bank
                            ),
                            "gradient_cosine_median": gradient_cosine,
                            "gradient_relative_l2_median": gradient_relative_error,
                            "query_count": oracle.query_count,
                            "gate_relative_l2": (
                                float(
                                    np.linalg.norm(fitted_gates - truth_gates)
                                    / np.linalg.norm(truth_gates)
                                )
                                if method == "channel_gate"
                                else ""
                            ),
                            "dot_defect_max": (
                                _dot_defect(
                                    candidate_operator,
                                    rng,
                                    int(config["dot_pairs"]),
                                )
                                if method == "channel_gate"
                                else ""
                            ),
                        }
                    )

    _write_csv(output_dir / "metrics.csv", rows)
    summary = _summarize(rows, config)
    _write_json(output_dir / "report.json", summary)
    _plot(rows, output_dir / "v6b_protocol_conformance.png")
    _write_checksums(output_dir)
    return summary


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    aggregates: list[dict[str, Any]] = []
    for stratum in ("in_class", "out_of_class"):
        for budget in [int(value) for value in config["query_budgets"]]:
            for method in ("nominal", "secant", "channel_gate"):
                selected = [
                    row
                    for row in rows
                    if row["stratum"] == stratum
                    and row["K"] == budget
                    and row["method"] == method
                ]
                aggregates.append(
                    {
                        "stratum": stratum,
                        "K": budget,
                        "method": method,
                        "rigs": len(selected),
                        "hidden_action_relative_l2_median": float(
                            np.median(
                                [row["hidden_action_relative_l2"] for row in selected]
                            )
                        ),
                        "gradient_cosine_median": float(
                            np.median([row["gradient_cosine_median"] for row in selected])
                        ),
                        "gradient_relative_l2_median": float(
                            np.median(
                                [row["gradient_relative_l2_median"] for row in selected]
                            )
                        ),
                    }
                )
    channel_rows = [row for row in rows if row["method"] == "channel_gate"]
    conformance_pass = (
        all(row["query_count"] == row["K"] for row in rows)
        and max(float(row["dot_defect_max"]) for row in channel_rows)
        <= float(config["dot_tolerance_float64"])
    )
    return {
        "schema_version": config["schema_version"],
        "decision": (
            "PASS_PROTOCOL_CONFORMANCE_ONLY"
            if conformance_pass
            else "FAIL_PROTOCOL_CONFORMANCE"
        ),
        "evidence_scope": config["evidence_scope"],
        "scientific_claims_unlocked": [],
        "fresh_v6b_status": "UNCONSTRUCTED",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "query_accounting": {
            "truth_forward_per_rig": [int(value) for value in config["query_budgets"]],
            "truth_adjoint": 0,
            "all_counts_exact": all(row["query_count"] == row["K"] for row in rows),
        },
        "dot_product": {
            "pairs_per_candidate": int(config["dot_pairs"]),
            "max_float64_defect": max(
                float(row["dot_defect_max"]) for row in channel_rows
            ),
            "tolerance": float(config["dot_tolerance_float64"]),
        },
        "aggregates": aggregates,
        "limitations": [
            "truth and candidate share the same toy nominal operator",
            "in-class truth is generated by the candidate family",
            "out-of-class residual is synthetic low rank",
            "this run cannot unlock inverse, external-renderer, real-BOS or OERF claims",
        ],
    }


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = {"nominal": "#68747f", "secant": "#e56b4f", "channel_gate": "#146b78"}
    labels = {"nominal": "Nominal A0", "secant": "K-query secant", "channel_gate": "27-gate candidate"}
    for axis, stratum in zip(axes, ("in_class", "out_of_class"), strict=True):
        for method in ("nominal", "secant", "channel_gate"):
            selected = [row for row in rows if row["stratum"] == stratum and row["method"] == method]
            budgets = sorted({int(row["K"]) for row in selected})
            medians = [
                float(
                    np.median(
                        [
                            row["hidden_action_relative_l2"]
                            for row in selected
                            if row["K"] == budget
                        ]
                    )
                )
                for budget in budgets
            ]
            axis.plot(budgets, medians, marker="o", color=colors[method], label=labels[method])
        axis.set_title("In-class truth" if stratum == "in_class" else "Misspecified truth")
        axis.set_xlabel("High-fidelity forward queries K")
        axis.set_ylabel("Hidden action relative L2")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("V6B protocol conformance only - not a fresh scientific result", fontsize=11)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_checksums(output_dir: Path) -> None:
    names = [
        "metrics.csv",
        "prediction_hashes_before_scoring.json",
        "report.json",
        "v6b_protocol_conformance.png",
    ]
    lines = []
    for name in names:
        digest = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
