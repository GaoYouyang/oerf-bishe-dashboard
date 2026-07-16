#!/usr/bin/env python3
"""Post-open diagnosis of a noise-floor-gated structured residual update."""

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
        positive_part_residual_signal_fraction,
        ridge_effective_degrees_of_freedom,
        ridge_residual_noise_energy_diagonal,
        ridge_residual_noise_degrees_of_freedom,
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
        positive_part_residual_signal_fraction,
        ridge_effective_degrees_of_freedom,
        ridge_residual_noise_energy_diagonal,
        ridge_residual_noise_degrees_of_freedom,
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
CONFIG_PATH = ROOT / "configs" / "v6d_df_gated_srco_postopen.json"
OUTPUT_DIR = ROOT / "results" / "v6d_df_gated_srco_postopen"
METHODS = (
    "nominal",
    "secant",
    "channel_gate",
    "full_srco",
    "df_gated_srco",
)


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


def _noisy_observations(
    rng: np.random.Generator,
    clean: np.ndarray,
    relative_noise: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    standard_deviation = (
        float(relative_noise)
        * np.linalg.norm(clean, axis=0, keepdims=True)
        / np.sqrt(clean.shape[0])
    )
    noisy = clean + standard_deviation * rng.normal(size=clean.shape)
    column_variances = np.square(standard_deviation.reshape(-1))
    row_variances = np.repeat(column_variances, clean.shape[0])
    expected_energy = float(np.sum(row_variances))
    return noisy, expected_energy, row_variances


def _aggregate(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for stratum in ("in_class", "out_of_class"):
        for budget in config["query_budgets"]:
            for method in METHODS:
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


def _paired_summary(
    rows: list[dict[str, Any]],
    *,
    stratum: str,
    budget: int,
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    by_rig: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["stratum"] != stratum or row["K"] != budget:
            continue
        by_rig.setdefault(int(row["rig"]), {})[str(row["method"])] = float(
            row["hidden_action_relative_l2"]
        )
    relative_changes = np.array(
        [
            values[candidate] / values[baseline] - 1.0
            for values in by_rig.values()
        ],
        dtype=np.float64,
    )
    absolute_changes = np.array(
        [values[candidate] - values[baseline] for values in by_rig.values()],
        dtype=np.float64,
    )
    return {
        "stratum": stratum,
        "K": budget,
        "candidate": candidate,
        "baseline": baseline,
        "rigs": len(by_rig),
        "positive_rigs": int(np.sum(relative_changes < 0)),
        "median_relative_improvement": float(-np.median(relative_changes)),
        "p90_relative_degradation": float(np.percentile(relative_changes, 90)),
        "max_relative_degradation": float(np.max(relative_changes)),
        "max_absolute_change": float(np.max(absolute_changes)),
    }


def run(config_path: Path = CONFIG_PATH, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))
    shape = tuple(int(value) for value in config["input_shape"])
    input_size = int(np.prod(shape))
    output_size = int(config["output_size"])
    budgets = [int(value) for value in config["query_budgets"]]
    offsets = voxel_kernel_offsets(int(config["kernel_radius"]))
    rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
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
            truth_gate = RayKernelChannelOperator(
                nominal,
                input_shape=shape,
                ray_coefficients=coefficients,
                gates=truth_gates,
                offsets=offsets,
            )
            truth_matrix = truth_gate.materialize()
            if stratum == "out_of_class":
                truth_matrix += _scaled_out_of_class_residual(
                    rng,
                    truth_matrix,
                    float(config["out_of_class_relative_frobenius"]),
                )

            calibration_bank = _unit_columns(rng, input_size, max(budgets))
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
                clean = collect_forward_observations(oracle, probes)
                observations, expected_noise_energy, noise_variances = _noisy_observations(
                    rng, clean, float(config["calibration_noise_relative"])
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
                gate = RayKernelChannelOperator(
                    nominal,
                    input_shape=shape,
                    ray_coefficients=coefficients,
                    gates=fitted_gates,
                    offsets=offsets,
                )
                gate_matrix = gate.materialize()
                gate_residuals = observations - gate_matrix @ probes
                gate_df = ridge_effective_degrees_of_freedom(
                    design, relative_ridge=float(config["gate_relative_ridge"])
                )
                gate_residual_noise_df = ridge_residual_noise_degrees_of_freedom(
                    design, relative_ridge=float(config["gate_relative_ridge"])
                )
                gate_residual_noise_energy = ridge_residual_noise_energy_diagonal(
                    design,
                    noise_variances,
                    relative_ridge=float(config["gate_relative_ridge"]),
                )
                activation = positive_part_residual_signal_fraction(
                    gate_residuals,
                    expected_noise_energy=gate_residual_noise_energy,
                )

                secant = LowRankSecantCorrection.fit(
                    probes,
                    observations - nominal_matrix @ probes,
                    relative_ridge=float(config["secant_relative_ridge"]),
                )
                secant_matrix = nominal_matrix + secant.materialize()
                full_update = fit_residual_secant(
                    gate,
                    probes=probes,
                    observations=observations,
                    relative_ridge=float(config["secant_relative_ridge"]),
                )
                full_hybrid = HybridGateSecantOperator(gate, full_update)
                gated_hybrid = HybridGateSecantOperator(
                    gate, full_update.scaled(activation)
                )
                full_matrix = full_hybrid.materialize()
                gated_matrix = gated_hybrid.materialize()

                activation_rows.append(
                    {
                        "stratum": stratum,
                        "rig": rig_index,
                        "K": budget,
                        "activation": activation,
                        "gate_effective_df": gate_df,
                        "gate_residual_noise_df": gate_residual_noise_df,
                        "observed_gate_residual_energy": float(
                            np.vdot(gate_residuals, gate_residuals)
                        ),
                        "expected_noise_energy_before_df": expected_noise_energy,
                        "expected_noise_energy_after_smoother": gate_residual_noise_energy,
                    }
                )
                frozen_hashes.append(
                    {
                        "stratum": stratum,
                        "rig": rig_index,
                        "K": budget,
                        "activation": activation,
                        "gated_srco_sha256": sha256_arrays(gated_matrix),
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
                    ("full_srco", full_matrix),
                    ("df_gated_srco", gated_matrix),
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
                            "activation": activation if method == "df_gated_srco" else "",
                            "dot_defect_max": (
                                _dot_defect(
                                    gated_hybrid,
                                    rng,
                                    int(config["dot_pairs"]),
                                )
                                if method == "df_gated_srco"
                                else ""
                            ),
                        }
                    )

    aggregates = _aggregate(rows, config)
    primary = max(budgets)
    primary_lookup = {
        (row["stratum"], row["method"]): row["hidden_action_relative_l2_median"]
        for row in aggregates
        if row["K"] == primary
    }
    activation_summary = []
    for stratum in ("in_class", "out_of_class"):
        for budget in budgets:
            selected = [
                row["activation"]
                for row in activation_rows
                if row["stratum"] == stratum and row["K"] == budget
            ]
            activation_summary.append(
                {
                    "stratum": stratum,
                    "K": budget,
                    "median": float(np.median(selected)),
                    "p10": float(np.percentile(selected, 10)),
                    "p90": float(np.percentile(selected, 90)),
                }
            )
    gated_rows = [row for row in rows if row["method"] == "df_gated_srco"]
    paired_primary = [
        _paired_summary(
            rows,
            stratum=stratum,
            budget=primary,
            candidate="df_gated_srco",
            baseline=baseline,
        )
        for stratum in ("in_class", "out_of_class")
        for baseline in ("channel_gate", "secant", "full_srco")
    ]
    report = {
        "schema_version": config["schema_version"],
        "decision": "SECOND_STAGE_POST_OPEN_DIAGNOSIS",
        "evidence_scope": config["evidence_scope"],
        "scientific_claims_unlocked": [],
        "candidate_preregistration_authorized": False,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_provenance": {
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limited_query_calibration_sha256": hashlib.sha256(
                (ROOT / "limited_query_calibration.py").read_bytes()
            ).hexdigest(),
            "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "method": {
            "working_name": "DF-gated Structured Residual Calibration Operator",
            "activation": "alpha=max(0,1-tr((I-H) Sigma_diag (I-H)^T)/||Y-A_gate X||_F^2)",
            "toy_noise_model": "probe-dependent diagonal heteroscedastic covariance",
            "noise_source_required_for_real_use": "independent flow-off repeats",
            "gate_relative_ridge": float(config["gate_relative_ridge"]),
            "gate_ridge_penalty": "gate_relative_ridge * tr(G^T G) / Q",
            "secant_relative_ridge": float(config["secant_relative_ridge"]),
            "secant_ridge_penalty": "secant_relative_ridge * tr(X^T X) / K",
            "truth_forward_queries": budgets,
            "truth_adjoint_queries": 0,
        },
        "primary_K": primary,
        "primary_hidden_action_medians": {
            f"{stratum}.{method}": value
            for (stratum, method), value in primary_lookup.items()
        },
        "activation_summary": activation_summary,
        "paired_primary": paired_primary,
        "query_accounting_exact": all(row["query_count"] == row["K"] for row in rows),
        "dot_product": {
            "max_float64_defect": max(
                float(row["dot_defect_max"]) for row in gated_rows
            ),
            "tolerance": float(config["dot_tolerance_float64"]),
        },
        "aggregates": aggregates,
        "limitations": [
            "this formula was proposed after opening V6C and is not preregistered",
            "the expected noise energy is generator-known rather than estimated from flow-off repeats",
            "the toy uses the exact diagonal heteroscedastic covariance trace; correlated real BOS still needs whitening or a full covariance trace",
            "all operator discrepancies are synthetic",
            "no inverse, external renderer, real BOS or OERF claim is tested",
        ],
    }
    _write_csv(output_dir / "metrics.csv", rows)
    _write_csv(output_dir / "activation.csv", activation_rows)
    _write_json(output_dir / "report.json", report)
    _plot(rows, output_dir / "v6d_df_gated_srco_postopen.png")
    _write_checksums(output_dir)
    return report


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.7, 4.4), constrained_layout=True)
    colors = {
        "nominal": "#68747f",
        "secant": "#e56b4f",
        "channel_gate": "#1a7782",
        "full_srco": "#96733c",
        "df_gated_srco": "#317a4f",
    }
    labels = {
        "nominal": "Nominal A0",
        "secant": "K-query calibration",
        "channel_gate": "27-gate",
        "full_srco": "Always-on SRCO",
        "df_gated_srco": "DF-gated SRCO",
    }
    for axis, stratum in zip(axes[:2], ("in_class", "out_of_class"), strict=True):
        for method in METHODS:
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
    axes[0].legend(frameon=False, fontsize=7.7)
    activation_axis = axes[2]
    activation_colors = {"in_class": "#1a7782", "out_of_class": "#a34f43"}
    activation_labels = {
        "in_class": "In-class + noise",
        "out_of_class": "Misspecified + noise",
    }
    for stratum in ("in_class", "out_of_class"):
        selected = [
            row
            for row in rows
            if row["stratum"] == stratum and row["method"] == "df_gated_srco"
        ]
        budgets = sorted({int(row["K"]) for row in selected})
        groups = [
            np.asarray(
                [row["activation"] for row in selected if row["K"] == budget],
                dtype=np.float64,
            )
            for budget in budgets
        ]
        medians = [float(np.median(group)) for group in groups]
        lower = [float(np.percentile(group, 10)) for group in groups]
        upper = [float(np.percentile(group, 90)) for group in groups]
        color = activation_colors[stratum]
        activation_axis.plot(
            budgets,
            medians,
            marker="o",
            linewidth=2,
            color=color,
            label=activation_labels[stratum],
        )
        activation_axis.fill_between(budgets, lower, upper, color=color, alpha=0.14)
    activation_axis.set_title("Residual gate activation")
    activation_axis.set_xlabel("High-fidelity forward queries K")
    activation_axis.set_ylabel("Activation alpha (median, p10-p90)")
    activation_axis.set_ylim(-0.04, 1.04)
    activation_axis.grid(alpha=0.25)
    activation_axis.legend(frameon=False, fontsize=7.7, loc="center right")
    figure.suptitle("DF-gated SRCO - second-stage post-open diagnosis", fontsize=11)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_checksums(output_dir: Path) -> None:
    names = (
        "activation.csv",
        "metrics.csv",
        "prediction_hashes_before_scoring.json",
        "report.json",
        "v6d_df_gated_srco_postopen.png",
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
