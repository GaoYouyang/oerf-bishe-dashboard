#!/usr/bin/env python3
"""Run the preregistered synthetic cone-ray control-variate mechanism gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

try:
    from .aperture_control_variate import (
        concentric_square_to_disk,
        cross_fitted_control_variate,
        disk_product_quadrature,
        sample_uniform_disk_antithetic,
        sample_uniform_disk_iid,
        weighted_operator_mean,
    )
    from .finite_aperture_bost import (
        _disk_subrays,
        build_aperture_subray_operator_bank,
        finite_aperture_reference_scale,
    )
except ImportError:
    from aperture_control_variate import (
        concentric_square_to_disk,
        cross_fitted_control_variate,
        disk_product_quadrature,
        sample_uniform_disk_antithetic,
        sample_uniform_disk_iid,
        weighted_operator_mean,
    )
    from finite_aperture_bost import (
        _disk_subrays,
        build_aperture_subray_operator_bank,
        finite_aperture_reference_scale,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "n2_cvcr_n0_mechanism_prereg_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "n2_cvcr_n0_mechanism_gate_v1"
METHODS = (
    "iid_mc",
    "antithetic_mc",
    "scrambled_sobol",
    "sunflower_qmc",
    "product_quadrature",
    "cf_quadratic_cv",
)
COLORS = {
    "iid_mc": "#6b7280",
    "antithetic_mc": "#2563eb",
    "scrambled_sobol": "#0f766e",
    "sunflower_qmc": "#ca8a04",
    "product_quadrature": "#7c3aed",
    "cf_quadratic_cv": "#b42318",
}
LABELS = {
    "iid_mc": "IID MC",
    "antithetic_mc": "Antithetic MC",
    "scrambled_sobol": "Scrambled Sobol",
    "sunflower_qmc": "Sunflower QMC",
    "product_quadrature": "Disk product quadrature",
    "cf_quadratic_cv": "Cross-fitted quadratic CV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(int(base)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def sample_scrambled_sobol_disk(count: int, *, seed: int) -> np.ndarray:
    exponent = int(round(math.log2(int(count))))
    if int(count) != 2**exponent:
        raise ValueError("scrambled Sobol count must be a power of two")
    sampler = qmc.Sobol(d=2, scramble=True, seed=int(seed))
    return concentric_square_to_disk(sampler.random_base2(exponent))


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1e-15)
    return float(np.linalg.norm(candidate - reference) / denominator)


def operator_action_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    x_probes: np.ndarray,
    y_probes: np.ndarray,
) -> dict[str, float]:
    matrix = np.asarray(candidate, dtype=np.float64).reshape(-1, candidate.shape[-1])
    truth = np.asarray(reference, dtype=np.float64).reshape(-1, reference.shape[-1])
    forward_errors = []
    adjoint_errors = []
    normal_errors = []
    normal_cosines = []
    dot_errors = []
    for x, y in zip(x_probes, y_probes, strict=True):
        forward_errors.append(relative_l2(matrix @ x, truth @ x))
        adjoint_errors.append(relative_l2(matrix.T @ y, truth.T @ y))
        candidate_normal = matrix.T @ (matrix @ x)
        truth_normal = truth.T @ (truth @ x)
        normal_errors.append(relative_l2(candidate_normal, truth_normal))
        denominator = max(
            float(np.linalg.norm(candidate_normal) * np.linalg.norm(truth_normal)),
            1e-30,
        )
        normal_cosines.append(float(np.dot(candidate_normal, truth_normal) / denominator))
        lhs = float(np.dot(matrix @ x, y))
        rhs = float(np.dot(x, matrix.T @ y))
        dot_errors.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30))
    return {
        "forward_action_relative_l2": float(np.mean(forward_errors)),
        "adjoint_action_relative_l2": float(np.mean(adjoint_errors)),
        "normal_action_relative_l2": float(np.mean(normal_errors)),
        "normal_action_cosine": float(np.mean(normal_cosines)),
        "internal_adjoint_relative_error": float(np.max(dot_errors)),
    }


def build_bank(
    config: dict[str, Any],
    rig: dict[str, Any],
    points: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    return build_aperture_subray_operator_bank(
        int(config["grid_size"]),
        int(config["depth"]),
        np.asarray(rig["angles_degrees"], dtype=float),
        points,
        aperture_radius=float(rig["aperture_radius"]),
        path_samples=int(rig["path_samples"]),
        cone_u=float(rig["cone_u"]),
        cone_z=float(rig["cone_z"]),
        bend=float(rig["bend"]),
        normalization_scale=float(scale),
        dtype=np.float64,
    )


def _metric_row(
    *,
    rig: dict[str, Any],
    budget: int,
    replicate: int,
    method: str,
    estimate: np.ndarray,
    reference: np.ndarray,
    x_probes: np.ndarray,
    y_probes: np.ndarray,
    postprocess_seconds: float,
) -> dict[str, Any]:
    return {
        "rig_id": str(rig["id"]),
        "rig_role": str(rig["role"]),
        "budget": int(budget),
        "replicate": int(replicate),
        "method": method,
        "high_fidelity_subray_evaluations": int(budget),
        "trainable_parameters": 0,
        "operator_relative_l2": relative_l2(estimate, reference),
        **operator_action_metrics(estimate, reference, x_probes, y_probes),
        "postprocess_seconds": float(postprocess_seconds),
    }


def run_experiment(config: dict[str, Any]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    budgets = [int(value) for value in config["budgets"]]
    if budgets != sorted(budgets) or any(value < 16 or value & (value - 1) for value in budgets):
        raise ValueError("budgets must be sorted powers of two of at least 16")
    if tuple(config["methods"]) != METHODS:
        raise ValueError("method roster differs from the preregistered implementation")
    maximum_budget = max(budgets)
    replicates = int(config["replicates"])
    base_seed = int(config["seed"])
    reference_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    estimates: dict[tuple[str, int, str], list[np.ndarray]] = {}
    references: dict[str, np.ndarray] = {}
    build_ledger: list[dict[str, Any]] = []

    reference_config = config["reference_quadrature"]
    coarse_points, coarse_weights = disk_product_quadrature(
        int(reference_config["coarse_radial_order"]),
        int(reference_config["coarse_angular_order"]),
    )
    fine_points, fine_weights = disk_product_quadrature(
        int(reference_config["fine_radial_order"]),
        int(reference_config["fine_angular_order"]),
    )

    for rig_index, rig in enumerate(config["rigs"]):
        rig_id = str(rig["id"])
        angles = np.asarray(rig["angles_degrees"], dtype=float)
        scale = finite_aperture_reference_scale(
            int(config["grid_size"]),
            int(config["depth"]),
            angles,
            aperture_samples=1,
            path_samples=int(rig["path_samples"]),
            cone_u=float(rig["cone_u"]),
            cone_z=float(rig["cone_z"]),
            bend=float(rig["bend"]),
        )
        start = time.perf_counter()
        coarse_bank = build_bank(config, rig, coarse_points, scale=scale)
        coarse_reference = weighted_operator_mean(coarse_bank, coarse_weights)
        coarse_seconds = time.perf_counter() - start
        start = time.perf_counter()
        fine_bank = build_bank(config, rig, fine_points, scale=scale)
        fine_reference = weighted_operator_mean(fine_bank, fine_weights)
        fine_seconds = time.perf_counter() - start
        reference_difference = relative_l2(coarse_reference, fine_reference)
        references[rig_id] = fine_reference
        reference_rows.append(
            {
                "rig_id": rig_id,
                "rig_role": str(rig["role"]),
                "coarse_points": len(coarse_points),
                "fine_points": len(fine_points),
                "coarse_to_fine_relative_l2": reference_difference,
                "reference_pass": reference_difference
                <= float(reference_config["maximum_relative_difference"]),
                "coarse_build_seconds": coarse_seconds,
                "fine_build_seconds": fine_seconds,
                "normalization_scale": scale,
            }
        )
        del coarse_bank, fine_bank

        probe_rng = np.random.default_rng(
            stable_seed(base_seed + int(config["action_probes"]["seed_offset"]), rig_id)
        )
        flattened = fine_reference.reshape(-1, fine_reference.shape[-1])
        probe_count = int(config["action_probes"]["count"])
        x_probes = probe_rng.normal(size=(probe_count, flattened.shape[1]))
        y_probes = probe_rng.normal(size=(probe_count, flattened.shape[0]))

        iid_points = np.concatenate(
            [
                sample_uniform_disk_iid(
                    maximum_budget,
                    seed=stable_seed(base_seed, rig_id, replicate, "iid"),
                )
                for replicate in range(replicates)
            ],
            axis=0,
        )
        anti_points = np.concatenate(
            [
                sample_uniform_disk_antithetic(
                    maximum_budget,
                    seed=stable_seed(base_seed, rig_id, replicate, "antithetic"),
                )
                for replicate in range(replicates)
            ],
            axis=0,
        )
        sobol_points = np.concatenate(
            [
                sample_scrambled_sobol_disk(
                    maximum_budget,
                    seed=stable_seed(base_seed, rig_id, replicate, "sobol"),
                )
                for replicate in range(replicates)
            ],
            axis=0,
        )
        method_banks: dict[str, np.ndarray] = {}
        for method, points in (
            ("iid_mc", iid_points),
            ("antithetic_mc", anti_points),
            ("scrambled_sobol", sobol_points),
        ):
            start = time.perf_counter()
            bank = build_bank(config, rig, points, scale=scale).reshape(
                replicates, maximum_budget, *fine_reference.shape
            )
            elapsed = time.perf_counter() - start
            method_banks[method] = bank
            build_ledger.append(
                {
                    "rig_id": rig_id,
                    "method": method,
                    "replicates": replicates,
                    "maximum_budget": maximum_budget,
                    "operator_bank_bytes": int(bank.nbytes),
                    "measured_build_seconds": elapsed,
                }
            )

        for budget in budgets:
            sunflower_points = _disk_subrays(budget)
            start = time.perf_counter()
            sunflower = np.mean(
                build_bank(config, rig, sunflower_points, scale=scale), axis=0
            )
            sunflower_seconds = time.perf_counter() - start
            metric_rows.append(
                _metric_row(
                    rig=rig,
                    budget=budget,
                    replicate=-1,
                    method="sunflower_qmc",
                    estimate=sunflower,
                    reference=fine_reference,
                    x_probes=x_probes,
                    y_probes=y_probes,
                    postprocess_seconds=sunflower_seconds,
                )
            )
            estimates.setdefault((rig_id, budget, "sunflower_qmc"), []).append(sunflower)

            quadrature_orders = config["product_quadrature"][str(budget)]
            product_points, product_weights = disk_product_quadrature(
                int(quadrature_orders[0]), int(quadrature_orders[1])
            )
            if len(product_points) != budget:
                raise ValueError("product quadrature must use the declared subray budget")
            start = time.perf_counter()
            product_estimate = weighted_operator_mean(
                build_bank(config, rig, product_points, scale=scale), product_weights
            )
            product_seconds = time.perf_counter() - start
            metric_rows.append(
                _metric_row(
                    rig=rig,
                    budget=budget,
                    replicate=-1,
                    method="product_quadrature",
                    estimate=product_estimate,
                    reference=fine_reference,
                    x_probes=x_probes,
                    y_probes=y_probes,
                    postprocess_seconds=product_seconds,
                )
            )
            estimates.setdefault((rig_id, budget, "product_quadrature"), []).append(
                product_estimate
            )

            for replicate in range(replicates):
                iid_values = method_banks["iid_mc"][replicate, :budget]
                iid_estimate = np.mean(iid_values, axis=0)
                anti_estimate = np.mean(
                    method_banks["antithetic_mc"][replicate, :budget], axis=0
                )
                sobol_estimate = np.mean(
                    method_banks["scrambled_sobol"][replicate, :budget], axis=0
                )
                start = time.perf_counter()
                cv = cross_fitted_control_variate(
                    iid_points.reshape(replicates, maximum_budget, 2)[replicate, :budget],
                    iid_values,
                    basis=str(config["control_variate"]["basis"]),
                    ridge=float(config["control_variate"]["ridge"]),
                )
                cv_seconds = time.perf_counter() - start
                values = {
                    "iid_mc": (iid_estimate, 0.0),
                    "antithetic_mc": (anti_estimate, 0.0),
                    "scrambled_sobol": (sobol_estimate, 0.0),
                    "cf_quadratic_cv": (cv.estimate, cv_seconds),
                }
                method_errors: dict[str, float] = {}
                for method, (estimate, postprocess_seconds) in values.items():
                    row = _metric_row(
                        rig=rig,
                        budget=budget,
                        replicate=replicate,
                        method=method,
                        estimate=estimate,
                        reference=fine_reference,
                        x_probes=x_probes,
                        y_probes=y_probes,
                        postprocess_seconds=postprocess_seconds,
                    )
                    metric_rows.append(row)
                    method_errors[method] = float(row["operator_relative_l2"])
                    estimates.setdefault((rig_id, budget, method), []).append(estimate)
                paired_rows.append(
                    {
                        "rig_id": rig_id,
                        "rig_role": str(rig["role"]),
                        "budget": budget,
                        "replicate": replicate,
                        "iid_operator_relative_l2": method_errors["iid_mc"],
                        "cv_operator_relative_l2": method_errors["cf_quadratic_cv"],
                        "cv_improvement_over_iid": (
                            method_errors["iid_mc"]
                            - method_errors["cf_quadratic_cv"]
                        )
                        / max(method_errors["iid_mc"], 1e-15),
                        "cv_harms_iid": method_errors["cf_quadratic_cv"]
                        > method_errors["iid_mc"],
                    }
                )
        del method_banks

    aggregate_rows: list[dict[str, Any]] = []
    for rig in config["rigs"]:
        rig_id = str(rig["id"])
        reference = references[rig_id]
        denominator = max(float(np.linalg.norm(reference)), 1e-15)
        for budget in budgets:
            for method in METHODS:
                group = [
                    row
                    for row in metric_rows
                    if row["rig_id"] == rig_id
                    and row["budget"] == budget
                    and row["method"] == method
                ]
                estimate_group = estimates[(rig_id, budget, method)]
                errors = np.asarray(
                    [float(row["operator_relative_l2"]) for row in group]
                )
                rmse = float(np.sqrt(np.mean(errors * errors)))
                mean_estimate = np.mean(np.stack(estimate_group), axis=0)
                bias = float(np.linalg.norm(mean_estimate - reference) / denominator)
                aggregate_rows.append(
                    {
                        "rig_id": rig_id,
                        "rig_role": str(rig["role"]),
                        "budget": budget,
                        "method": method,
                        "replicate_count": len(group),
                        "operator_rmse_relative": rmse,
                        "operator_mean_relative_l2": float(np.mean(errors)),
                        "operator_p95_relative_l2": float(np.quantile(errors, 0.95)),
                        "operator_worst_relative_l2": float(np.max(errors)),
                        "operator_bias_relative": bias,
                        "bias_to_rmse_standard_error_ratio": bias
                        / max(rmse / math.sqrt(len(group)), 1e-15),
                        "forward_action_mean_relative_l2": float(
                            np.mean([row["forward_action_relative_l2"] for row in group])
                        ),
                        "adjoint_action_mean_relative_l2": float(
                            np.mean([row["adjoint_action_relative_l2"] for row in group])
                        ),
                        "normal_action_mean_relative_l2": float(
                            np.mean([row["normal_action_relative_l2"] for row in group])
                        ),
                        "normal_action_mean_cosine": float(
                            np.mean([row["normal_action_cosine"] for row in group])
                        ),
                        "maximum_internal_adjoint_relative_error": float(
                            np.max([row["internal_adjoint_relative_error"] for row in group])
                        ),
                        "mean_postprocess_seconds": float(
                            np.mean([row["postprocess_seconds"] for row in group])
                        ),
                    }
                )
    evidence = {
        "references": reference_rows,
        "build_ledger": build_ledger,
        "aggregate_rows": aggregate_rows,
    }
    return metric_rows, aggregate_rows, paired_rows, evidence


def evaluate_gates(
    config: dict[str, Any],
    reference_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = int(config["primary_budget"])
    audit_rigs = [str(rig["id"]) for rig in config["rigs"] if rig["role"] == "audit"]
    gates = config["gates"]

    def aggregate(rig_id: str, method: str) -> dict[str, Any]:
        return next(
            row
            for row in aggregate_rows
            if row["rig_id"] == rig_id
            and row["budget"] == primary
            and row["method"] == method
        )

    per_rig = []
    baseline_methods = METHODS[:-1]
    for rig_id in audit_rigs:
        candidate = aggregate(rig_id, "cf_quadratic_cv")
        baselines = [aggregate(rig_id, method) for method in baseline_methods]
        best = min(baselines, key=lambda row: row["operator_rmse_relative"])
        per_rig.append(
            {
                "rig_id": rig_id,
                "candidate_operator_rmse": candidate["operator_rmse_relative"],
                "best_baseline_method": best["method"],
                "best_baseline_operator_rmse": best["operator_rmse_relative"],
                "operator_rmse_improvement_over_best": (
                    float(best["operator_rmse_relative"])
                    - float(candidate["operator_rmse_relative"])
                )
                / max(float(best["operator_rmse_relative"]), 1e-15),
                "normal_action_error_ratio_to_best": float(
                    candidate["normal_action_mean_relative_l2"]
                )
                / max(float(best["normal_action_mean_relative_l2"]), 1e-15),
                "bias_to_rmse_standard_error_ratio": candidate[
                    "bias_to_rmse_standard_error_ratio"
                ],
            }
        )

    candidate_errors = np.asarray(
        [
            row["operator_relative_l2"]
            for row in metric_rows
            if row["rig_id"] in audit_rigs
            and row["budget"] == primary
            and row["method"] == "cf_quadratic_cv"
        ],
        dtype=float,
    )
    pooled_baselines = []
    for method in baseline_methods:
        errors = np.asarray(
            [
                row["operator_relative_l2"]
                for row in metric_rows
                if row["rig_id"] in audit_rigs
                and row["budget"] == primary
                and row["method"] == method
            ],
            dtype=float,
        )
        pooled_baselines.append(
            {
                "method": method,
                "operator_rmse": float(np.sqrt(np.mean(errors * errors))),
            }
        )
    best_pooled = min(pooled_baselines, key=lambda row: row["operator_rmse"])
    candidate_pooled_rmse = float(np.sqrt(np.mean(candidate_errors * candidate_errors)))
    pooled_improvement = (
        float(best_pooled["operator_rmse"]) - candidate_pooled_rmse
    ) / max(float(best_pooled["operator_rmse"]), 1e-15)
    primary_pairs = [
        row
        for row in paired_rows
        if row["rig_id"] in audit_rigs and row["budget"] == primary
    ]
    harm_fraction = float(np.mean([bool(row["cv_harms_iid"]) for row in primary_pairs]))
    candidate_adjoint = max(
        float(row["internal_adjoint_relative_error"])
        for row in metric_rows
        if row["rig_id"] in audit_rigs
        and row["budget"] == primary
        and row["method"] == "cf_quadratic_cv"
    )

    checks = {
        "reference_quadrature_converged": all(
            bool(row["reference_pass"]) for row in reference_rows
        ),
        "pooled_operator_rmse_improvement_pass": pooled_improvement
        >= float(gates["minimum_pooled_operator_rmse_improvement_over_best_baseline"]),
        "every_audit_rig_improvement_pass": all(
            row["operator_rmse_improvement_over_best"]
            >= float(
                gates[
                    "minimum_each_audit_rig_operator_rmse_improvement_over_best_baseline"
                ]
            )
            for row in per_rig
        ),
        "paired_harm_fraction_pass": harm_fraction
        <= float(gates["maximum_paired_harm_fraction_vs_iid"]),
        "bias_proxy_pass": all(
            row["bias_to_rmse_standard_error_ratio"]
            <= float(gates["maximum_bias_to_rmse_standard_error_ratio"])
            for row in per_rig
        ),
        "normal_action_error_pass": all(
            row["normal_action_error_ratio_to_best"]
            <= float(gates["maximum_normal_action_error_ratio_to_best_baseline"])
            for row in per_rig
        ),
        "internal_adjoint_pass": candidate_adjoint
        <= float(gates["maximum_internal_adjoint_relative_error"]),
    }
    if not checks["reference_quadrature_converged"]:
        decision = "HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED"
    elif not checks["internal_adjoint_pass"]:
        decision = "FAIL_INTERNAL_ADJOINT_CONSISTENCY"
    elif all(checks.values()):
        decision = "GO_CVCR_N1_LEARNED_ALLOCATION_PREREGISTRATION"
    else:
        decision = "NO_GO_STOP_CF_QUADRATIC_CV_CAPACITY_ESCALATION"
    return {
        "decision": decision,
        "primary_budget": primary,
        "checks": checks,
        "audit_rig_comparisons": per_rig,
        "candidate_pooled_operator_rmse": candidate_pooled_rmse,
        "best_pooled_baseline": best_pooled,
        "pooled_operator_rmse_improvement_over_best": pooled_improvement,
        "paired_harm_fraction_vs_iid": harm_fraction,
        "maximum_internal_adjoint_relative_error": candidate_adjoint,
    }


def write_figure(
    path: Path,
    config: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    gate_report: dict[str, Any],
) -> None:
    budgets = [int(value) for value in config["budgets"]]
    audit_rigs = {str(rig["id"]) for rig in config["rigs"] if rig["role"] == "audit"}
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    for method in METHODS:
        operator_values = []
        normal_values = []
        for budget in budgets:
            group = [
                row
                for row in aggregate_rows
                if row["rig_id"] in audit_rigs
                and row["budget"] == budget
                and row["method"] == method
            ]
            operator_values.append(
                float(np.sqrt(np.mean([row["operator_rmse_relative"] ** 2 for row in group])))
            )
            normal_values.append(
                float(np.mean([row["normal_action_mean_relative_l2"] for row in group]))
            )
        axes[0, 0].plot(
            budgets,
            operator_values,
            marker="o",
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[1, 0].plot(
            budgets,
            normal_values,
            marker="o",
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )
    for axis, ylabel in (
        (axes[0, 0], "Pooled operator relative RMSE"),
        (axes[1, 0], "Mean normal-action relative L2"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(budgets, labels=budgets)
        axis.set_xlabel("High-fidelity subrays per pixel")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].set_title("Matched-ray operator integration error")
    axes[1, 0].set_title("Impact on A^T A action")
    axes[0, 0].legend(frameon=False, fontsize=8)

    comparisons = gate_report["audit_rig_comparisons"]
    axes[0, 1].barh(
        [row["rig_id"] for row in comparisons],
        [100.0 * row["operator_rmse_improvement_over_best"] for row in comparisons],
        color=[
            "#0f766e" if row["operator_rmse_improvement_over_best"] >= 0.0 else "#b42318"
            for row in comparisons
        ],
    )
    axes[0, 1].axvline(
        100.0
        * float(
            config["gates"][
                "minimum_each_audit_rig_operator_rmse_improvement_over_best_baseline"
            ]
        ),
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="preregistered per-rig gate",
    )
    axes[0, 1].set_xlabel("CV improvement over each rig's best baseline (%)")
    axes[0, 1].set_title(f"Primary budget B={gate_report['primary_budget']}")
    axes[0, 1].grid(axis="x", alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 1].axis("off")
    lines = [
        gate_report["decision"],
        "",
        f"pooled gain: {100.0 * gate_report['pooled_operator_rmse_improvement_over_best']:.2f}%",
        f"best baseline: {gate_report['best_pooled_baseline']['method']}",
        f"paired harm vs IID: {100.0 * gate_report['paired_harm_fraction_vs_iid']:.1f}%",
        f"max adjoint error: {gate_report['maximum_internal_adjoint_relative_error']:.2e}",
        "",
        *[
            f"{'PASS' if passed else 'FAIL'}  {name}"
            for name, passed in gate_report["checks"].items()
        ],
        "",
        "Synthetic mechanism evidence only.",
        "No real-field or reconstruction claim.",
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.2,
        color="#111827",
    )
    figure.suptitle("N2-CVCR-N0 preregistered synthetic mechanism gate", fontsize=15)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_readme(path: Path, report: dict[str, Any]) -> None:
    gate = report["gate_report"]
    lines = [
        "# N2-CVCR-N0 result",
        "",
        f"Decision: `{gate['decision']}`",
        "",
        "This directory contains a preregistered synthetic operator-integration mechanism test.",
        "It is not a real BOST, OERF, 3-D reconstruction, learned-operator, or publication result.",
        "",
        "## Primary numbers",
        "",
        f"- Primary subray budget: `{gate['primary_budget']}`",
        f"- Pooled candidate operator RMSE: `{gate['candidate_pooled_operator_rmse']:.8g}`",
        f"- Best pooled baseline: `{gate['best_pooled_baseline']['method']}` / `{gate['best_pooled_baseline']['operator_rmse']:.8g}`",
        f"- Pooled improvement: `{100.0 * gate['pooled_operator_rmse_improvement_over_best']:.3f}%`",
        f"- Paired harm fraction versus IID: `{100.0 * gate['paired_harm_fraction_vs_iid']:.3f}%`",
        f"- Maximum internal adjoint error: `{gate['maximum_internal_adjoint_relative_error']:.3e}`",
        "",
        "See `report.json`, `aggregate_metrics.csv`, `replicate_metrics.csv`,",
        "`paired_metrics.csv`, `reference_convergence.csv`, and the diagnostic figure.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checksums(output: Path) -> None:
    targets = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{sha256(path)}  {path.name}" for path in targets]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, aggregate_rows, paired_rows, evidence = run_experiment(config)
    gate_report = evaluate_gates(
        config,
        evidence["references"],
        aggregate_rows,
        metric_rows,
        paired_rows,
    )
    report = {
        "schema": "n2-cvcr-n0-synthetic-mechanism-result-1",
        "evidence_label": config["evidence_label"],
        "claim_ceiling": config["claim_ceiling"],
        "gate_report": gate_report,
        "reference_convergence": evidence["references"],
        "build_ledger": evidence["build_ledger"],
        "authorizations": {
            "learned_allocation_preregistration": gate_report["decision"]
            == "GO_CVCR_N1_LEARNED_ALLOCATION_PREREGISTRATION",
            "formal_n2_algorithm_claim": False,
            "three_dimensional_reconstruction_claim": False,
            "experimental_or_oerf_claim": False,
            "generalization_claim": False,
            "publication_success_claim": False,
        },
        "stop_rules": config["stop_rules"],
    }
    write_csv(output / "replicate_metrics.csv", metric_rows)
    write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output / "paired_metrics.csv", paired_rows)
    write_csv(output / "reference_convergence.csv", evidence["references"])
    write_csv(output / "build_ledger.csv", evidence["build_ledger"])
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_figure(output / "n2_cvcr_n0_mechanism_gate.png", config, aggregate_rows, gate_report)
    write_readme(output / "README.md", report)
    write_checksums(output)
    print(json.dumps(gate_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
