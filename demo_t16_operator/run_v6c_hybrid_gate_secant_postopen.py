#!/usr/bin/env python3
"""Post-open toy diagnosis of a structured gate plus residual secant update."""

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
        HybridGateSecantOperator,
        LowRankSecantCorrection,
        NominalLinearOperator,
        RayKernelChannelOperator,
        build_gate_design,
        collect_forward_observations,
        fit_channel_gates,
        fit_residual_secant,
        sha256_arrays,
        voxel_kernel_offsets,
    )
    from demo_t16_operator.run_v6b_protocol_conformance import (
        _dot_defect,
        _gradient_metrics,
        _relative_action_error,
        _scaled_out_of_class_residual,
        _unit_columns,
    )
else:
    from .limited_query_calibration import (
        BudgetedForwardOracle,
        HybridGateSecantOperator,
        LowRankSecantCorrection,
        NominalLinearOperator,
        RayKernelChannelOperator,
        build_gate_design,
        collect_forward_observations,
        fit_channel_gates,
        fit_residual_secant,
        sha256_arrays,
        voxel_kernel_offsets,
    )
    from .run_v6b_protocol_conformance import (
        _dot_defect,
        _gradient_metrics,
        _relative_action_error,
        _scaled_out_of_class_residual,
        _unit_columns,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v6c_hybrid_gate_secant_postopen.json"
OUTPUT_DIR = ROOT / "results" / "v6c_hybrid_gate_secant_postopen"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _add_relative_noise(
    rng: np.random.Generator, observations: np.ndarray, relative_noise: float
) -> np.ndarray:
    scale = (
        float(relative_noise)
        * np.linalg.norm(observations, axis=0, keepdims=True)
        / np.sqrt(observations.shape[0])
    )
    return observations + scale * rng.normal(size=observations.shape)


def _aggregate(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    methods = ("nominal", "secant", "channel_gate", "gate_plus_secant")
    for stratum in ("in_class", "out_of_class"):
        for budget in config["query_budgets"]:
            for method in methods:
                selected = [
                    row
                    for row in rows
                    if row["stratum"] == stratum
                    and row["K"] == budget
                    and row["method"] == method
                ]
                output.append(
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
    return output


def run(config_path: Path = CONFIG_PATH, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))
    shape = tuple(int(value) for value in config["input_shape"])
    input_size = int(np.prod(shape))
    output_size = int(config["output_size"])
    budgets = [int(value) for value in config["query_budgets"]]
    max_budget = max(budgets)
    offsets = voxel_kernel_offsets(int(config["kernel_radius"]))
    rows: list[dict[str, Any]] = []
    frozen_hashes: list[dict[str, Any]] = []

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
            gate_truth = RayKernelChannelOperator(
                nominal,
                input_shape=shape,
                ray_coefficients=coefficients,
                gates=truth_gates,
                offsets=offsets,
            )
            truth_matrix = gate_truth.materialize()
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
                clean_observations = collect_forward_observations(oracle, probes)
                observations = _add_relative_noise(
                    rng,
                    clean_observations,
                    float(config["calibration_noise_relative"]),
                )

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
                gate_operator = RayKernelChannelOperator(
                    nominal,
                    input_shape=shape,
                    ray_coefficients=coefficients,
                    gates=fitted_gates,
                    offsets=offsets,
                )
                gate_matrix = gate_operator.materialize()

                nominal_residuals = observations - nominal_matrix @ probes
                secant_correction = LowRankSecantCorrection.fit(
                    probes,
                    nominal_residuals,
                    relative_ridge=float(config["secant_relative_ridge"]),
                )
                secant_matrix = nominal_matrix + secant_correction.materialize()
                hybrid_correction = fit_residual_secant(
                    gate_operator,
                    probes=probes,
                    observations=observations,
                    relative_ridge=float(config["secant_relative_ridge"]),
                )
                hybrid_operator = HybridGateSecantOperator(
                    gate_operator, hybrid_correction
                )
                hybrid_matrix = hybrid_operator.materialize()

                frozen_hashes.append(
                    {
                        "stratum": stratum,
                        "rig": rig_index,
                        "K": budget,
                        "channel_gate_sha256": sha256_arrays(
                            fitted_gates, gate_matrix
                        ),
                        "secant_sha256": sha256_arrays(secant_matrix),
                        "hybrid_sha256": sha256_arrays(hybrid_matrix),
                    }
                )
                _write_json(
                    output_dir / "prediction_hashes_before_scoring.json",
                    frozen_hashes,
                )

                points = int(config["gradient_points"])
                target_fields = rng.normal(size=(points, input_size))
                iterates = target_fields + 0.3 * rng.normal(
                    size=(points, input_size)
                )
                matrices = (
                    ("nominal", nominal_matrix),
                    ("secant", secant_matrix),
                    ("channel_gate", gate_matrix),
                    ("gate_plus_secant", hybrid_matrix),
                )
                for method, matrix in matrices:
                    gradient_cosine, gradient_relative = _gradient_metrics(
                        matrix, truth_matrix, target_fields, iterates
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
                            "gradient_relative_l2_median": gradient_relative,
                            "query_count": oracle.query_count,
                            "dot_defect_max": (
                                _dot_defect(
                                    hybrid_operator,
                                    rng,
                                    int(config["dot_pairs"]),
                                )
                                if method == "gate_plus_secant"
                                else ""
                            ),
                        }
                    )

    aggregates = _aggregate(rows, config)
    primary = max(budgets)
    primary_rows = [row for row in aggregates if row["K"] == primary]
    lookup = {
        (row["stratum"], row["method"]): row["hidden_action_relative_l2_median"]
        for row in primary_rows
    }
    hybrid_wins = all(
        lookup[(stratum, "gate_plus_secant")]
        < min(lookup[(stratum, "channel_gate")], lookup[(stratum, "secant")])
        for stratum in ("in_class", "out_of_class")
    )
    hybrid_rows = [row for row in rows if row["method"] == "gate_plus_secant"]
    report = {
        "schema_version": config["schema_version"],
        "decision": (
            "POST_OPEN_HYBRID_HYPOTHESIS_GENERATED"
            if hybrid_wins
            else "POST_OPEN_HYBRID_NOT_DOMINANT"
        ),
        "evidence_scope": config["evidence_scope"],
        "scientific_claims_unlocked": [],
        "fresh_status": "NOT_PREREGISTERED",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "method": {
            "name": "Structured Residual Calibration Operator (SRCO, working name)",
            "formula": "A_h = A_gate + E (X^T X + lambda I)^-1 X^T",
            "truth_forward_queries": budgets,
            "truth_adjoint_queries": 0,
        },
        "query_accounting_exact": all(row["query_count"] == row["K"] for row in rows),
        "dot_product": {
            "max_float64_defect": max(
                float(row["dot_defect_max"]) for row in hybrid_rows
            ),
            "tolerance": float(config["dot_tolerance_float64"]),
        },
        "primary_K": primary,
        "primary_hidden_action_medians": {
            f"{stratum}.{method}": value
            for (stratum, method), value in lookup.items()
        },
        "aggregates": aggregates,
        "limitations": [
            "the method was proposed after opening the V6B misspecification toy",
            "all rigs are synthetic and share the same generator family",
            "the low-rank discrepancy and calibration noise are hand specified",
            "no inverse, external renderer, real BOS or OERF claim is tested",
            "the working name is not a novelty claim and requires literature collision audit",
            "with origin-based input-output pairs this is operator calibration, not a strict quasi-Newton secant equation",
        ],
    }
    _write_csv(output_dir / "metrics.csv", rows)
    _write_json(output_dir / "report.json", report)
    _plot(rows, output_dir / "v6c_hybrid_gate_secant_postopen.png")
    _write_checksums(output_dir)
    return report


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    colors = {
        "nominal": "#68747f",
        "secant": "#e56b4f",
        "channel_gate": "#1a7782",
        "gate_plus_secant": "#317a4f",
    }
    labels = {
        "nominal": "Nominal A0",
        "secant": "K-query secant",
        "channel_gate": "27-gate",
        "gate_plus_secant": "Gate + residual secant",
    }
    for axis, stratum in zip(axes, ("in_class", "out_of_class"), strict=True):
        for method in labels:
            selected = [
                row
                for row in rows
                if row["stratum"] == stratum and row["method"] == method
            ]
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
            axis.plot(
                budgets,
                medians,
                marker="o",
                linewidth=2,
                color=colors[method],
                label=labels[method],
            )
        axis.set_title("In-class + noise" if stratum == "in_class" else "Misspecified + noise")
        axis.set_xlabel("High-fidelity forward queries K")
        axis.set_ylabel("Hidden action relative L2")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Post-open hybrid diagnosis - hypothesis generation only", fontsize=11)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_checksums(output_dir: Path) -> None:
    names = (
        "metrics.csv",
        "prediction_hashes_before_scoring.json",
        "report.json",
        "v6c_hybrid_gate_secant_postopen.png",
    )
    lines = [
        f"{hashlib.sha256((output_dir / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
