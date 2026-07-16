#!/usr/bin/env python3
"""Run v3k-D strong non-learning controls before any learned step rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .adjoint_landweber import (
        feasibility_project,
        geometry_normalization,
        global_landweber_trajectory,
        landweber_trajectory,
        mask_key,
        quadratic_steepest_descent_trajectory,
    )
    from .counterfactual_geometry import CounterfactualInputFactory, schedule_balance
    from .direct_operator_data import ridge_reconstruction_matrix
    from .models import make_model
    from .run_v3k_a_counterfactual_supervision import (
        bootstrap_interval,
        load_private_dataset,
        precompute_base_predictions,
    )
    from .run_v3k_c_adjoint_landweber_gate import (
        find_pair,
        geometry_masks,
        make_dataset,
        metric_rows,
        relative_l2,
        sha256,
        source_cluster_values,
        source_indices,
        summarize,
        summarize_layouts,
        write_csv,
    )
    from .train_eval import choose_device
except ImportError:
    from adjoint_landweber import (
        feasibility_project,
        geometry_normalization,
        global_landweber_trajectory,
        landweber_trajectory,
        mask_key,
        quadratic_steepest_descent_trajectory,
    )
    from counterfactual_geometry import CounterfactualInputFactory, schedule_balance
    from direct_operator_data import ridge_reconstruction_matrix
    from models import make_model
    from run_v3k_a_counterfactual_supervision import (
        bootstrap_interval,
        load_private_dataset,
        precompute_base_predictions,
    )
    from run_v3k_c_adjoint_landweber_gate import (
        find_pair,
        geometry_masks,
        make_dataset,
        metric_rows,
        relative_l2,
        sha256,
        source_cluster_values,
        source_indices,
        summarize,
        summarize_layouts,
        write_csv,
    )
    from train_eval import choose_device


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v3k_d_strong_numerical_controls.json"
PUBLIC_FILES = [
    "v3k_d_pair_manifest.csv",
    "v3k_d_validation_screen.csv",
    "v3k_d_selection_commit.json",
    "v3k_d_sample_metrics.csv",
    "v3k_d_split_summary.csv",
    "v3k_d_pairwise_summary.csv",
    "v3k_d_layout_summary.csv",
    "v3k_d_quadratic_step_audit.csv",
    "v3k_d_operator_call_ledger.csv",
    "v3k_d_strong_controls_dashboard.json",
    "v3k_d_strong_controls_report.json",
    "t16_v3k_d_strong_numerical_controls.png",
]
LABELS = {
    "locked_fno_raw": "Locked FNO (raw)",
    "feasible_fno": "FNO + hard support",
    "ridge_raw": "Tuned ridge (unwindowed)",
    "feasible_ridge": "Tuned ridge + hard support",
    "fno_geometry": "FNO + geometry-normalized Landweber",
    "fno_global": "FNO + one global Landweber step",
    "fno_quadratic": "FNO + quadratic step then projection",
    "fno_lookup": "FNO + validation-tuned spectral lookup",
    "ridge_geometry": "Ridge + geometry-normalized Landweber",
    "ridge_global": "Ridge + one global Landweber step",
    "ridge_quadratic": "Ridge + quadratic step then projection",
    "ridge_lookup": "Ridge + validation-tuned spectral lookup",
}
START_KEYS = {"fno": "feasible_fno", "ridge": "feasible_ridge"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ridge_predictions(
    data: dict[str, np.ndarray],
    dataset,
    factory: CounterfactualInputFactory,
    ridge_relative: float,
) -> np.ndarray:
    """Reconstruct an unwindowed, signed Tikhonov field for every pair."""
    shape = (len(dataset), *data["field"].shape[1:])
    output = np.empty(shape, dtype=np.float64)
    for identifier in sorted({str(row["geometry_id"]) for row in dataset.pairs}):
        indices = np.asarray(
            [i for i, row in enumerate(dataset.pairs) if str(row["geometry_id"]) == identifier],
            dtype=np.int64,
        )
        sources = np.asarray(
            [int(dataset.pairs[i]["source_index"]) for i in indices], dtype=np.int64
        )
        mask = factory.mask(identifier)
        selected = np.flatnonzero(mask > 0.5)
        inverse = ridge_reconstruction_matrix(
            data["forward_matrix"], mask, float(ridge_relative)
        )
        observation = np.asarray(data["observation"][sources], dtype=np.float64)
        flattened = observation[:, :, selected, :].reshape(
            len(indices), observation.shape[1], -1
        )
        output[indices] = np.einsum(
            "bdm,pm->bdp", flattened, inverse, optimize=True
        ).reshape(len(indices), *shape[1:])
    return output


def field_score(
    prediction: np.ndarray,
    truth: np.ndarray,
    sources: np.ndarray,
) -> tuple[float, float, int]:
    _, collapsed = source_cluster_values(relative_l2(prediction, truth), sources)
    return float(np.mean(collapsed)), float(np.median(collapsed)), len(collapsed)


def screen_row(
    *,
    stage: str,
    start: str,
    method: str,
    regime: str,
    ridge_relative: float | str,
    step_fraction: float | str,
    iterations: int,
    a_calls: int,
    at_calls: int,
    score: tuple[float, float, int],
    layout_rows: int,
) -> dict[str, object]:
    return {
        "selection_split": "val",
        "screen_stage": stage,
        "start": start,
        "method": method,
        "spectral_regime": regime,
        "ridge_relative": ridge_relative,
        "step_fraction": step_fraction,
        "iterations": int(iterations),
        "operator_a_calls_per_sample": int(a_calls),
        "operator_at_calls_per_sample": int(at_calls),
        "independent_field_count": int(score[2]),
        "layout_rows": int(layout_rows),
        "source_field_mean_rel_l2": float(score[0]),
        "source_field_median_rel_l2": float(score[1]),
        "selection_uses_audit_or_reprojection": False,
    }


def spectral_regimes(
    factory: CounterfactualInputFactory,
    normalization: dict[str, dict[str, np.ndarray | float]],
) -> tuple[float, dict[str, str], list[dict[str, object]]]:
    values = []
    for identifier in sorted(factory.catalog):
        mask = factory.mask(identifier)
        spectral = float(normalization[mask_key(mask)]["spectral_constant"])
        values.append((identifier, spectral))
    ordered = sorted({spectral for _, spectral in values})
    gaps = np.diff(ordered)
    if not len(gaps) or float(np.max(gaps)) <= 0.0:
        raise RuntimeError("operator-only spectral regimes require a nonzero gap")
    gap_index = int(np.argmax(gaps))
    boundary = 0.5 * (ordered[gap_index] + ordered[gap_index + 1])
    mapping = {
        identifier: "low" if spectral <= boundary else "high"
        for identifier, spectral in values
    }
    manifest = [
        {
            "geometry_id": identifier,
            "geometry_partition": str(factory.catalog[identifier]["geometry_partition"]),
            "mask_bits": identifier.removeprefix("g_"),
            "spectral_constant": spectral,
            "spectral_regime": mapping[identifier],
            "operator_only_boundary": boundary,
        }
        for identifier, spectral in values
    ]
    return boundary, mapping, manifest


def sample_regimes(dataset, mapping: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [mapping[str(row["geometry_id"])] for row in dataset.pairs], dtype="U8"
    )


def spectral_per_sample(
    masks: np.ndarray,
    normalization: dict[str, dict[str, np.ndarray | float]],
) -> np.ndarray:
    return np.asarray(
        [float(normalization[mask_key(mask)]["spectral_constant"]) for mask in masks],
        dtype=np.float64,
    )


def choose(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise RuntimeError("empty validation selection cell")
    return dict(
        min(
            rows,
            key=lambda row: (
                float(row["source_field_mean_rel_l2"]),
                int(row["operator_a_calls_per_sample"])
                + int(row["operator_at_calls_per_sample"]),
                int(row["iterations"]),
                float(row["step_fraction"] or 0.0),
                float(row["ridge_relative"] or 0.0),
            ),
        )
    )


def validation_screen(
    config: dict,
    data: dict[str, np.ndarray],
    dataset,
    base: np.ndarray,
    factory: CounterfactualInputFactory,
    normalization: dict[str, dict[str, np.ndarray | float]],
    global_spectral: float,
    regime_mapping: dict[str, str],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    protocol = config["selection_protocol"]
    sources = source_indices(dataset)
    truth = np.asarray(data["field"][sources], dtype=np.float64)
    observation = np.asarray(data["observation"][sources], dtype=np.float64)
    masks = geometry_masks(factory, dataset)
    regimes = sample_regimes(dataset, regime_mapping)
    support = data["support"]
    rows: list[dict[str, object]] = []
    ridge_cache: dict[float, np.ndarray] = {}

    for ridge_relative in [float(v) for v in protocol["ridge_relative_grid"]]:
        raw = ridge_predictions(data, dataset, factory, ridge_relative)
        ridge_cache[ridge_relative] = raw
        feasible = feasibility_project(raw, support)
        for method, prediction in (("ridge_raw", raw), ("feasible_ridge", feasible)):
            rows.append(
                screen_row(
                    stage="ridge_lambda",
                    start="ridge",
                    method=method,
                    regime="all",
                    ridge_relative=ridge_relative,
                    step_fraction="",
                    iterations=0,
                    a_calls=0,
                    at_calls=0,
                    score=field_score(prediction, truth, sources),
                    layout_rows=len(dataset),
                )
            )
    ridge_raw_choice = choose([r for r in rows if r["method"] == "ridge_raw"])
    ridge_feasible_choice = choose(
        [r for r in rows if r["method"] == "feasible_ridge"]
    )
    raw_ridge = ridge_cache[float(ridge_raw_choice["ridge_relative"])]
    feasible_ridge = feasibility_project(
        ridge_cache[float(ridge_feasible_choice["ridge_relative"])], support
    )
    starts = {
        "fno": feasibility_project(np.asarray(base[:, 0], dtype=np.float64), support),
        "ridge": feasible_ridge,
    }
    checkpoints = [int(v) for v in protocol["iteration_counts"]]
    betas = [float(v) for v in protocol["step_fractions"]]

    for start_name, start in starts.items():
        for beta in betas:
            geometry_path = landweber_trajectory(
                start,
                observation,
                data["forward_matrix"],
                masks,
                support,
                beta,
                checkpoints,
                normalization,
                method="standard",
            )
            global_path = global_landweber_trajectory(
                start,
                observation,
                data["forward_matrix"],
                masks,
                support,
                beta,
                checkpoints,
                global_spectral,
            )
            for iteration in checkpoints:
                rows.append(
                    screen_row(
                        stage="fixed_step",
                        start=start_name,
                        method="geometry_landweber",
                        regime="all",
                        ridge_relative="",
                        step_fraction=beta,
                        iterations=iteration,
                        a_calls=iteration,
                        at_calls=iteration,
                        score=field_score(geometry_path[iteration], truth, sources),
                        layout_rows=len(dataset),
                    )
                )
                rows.append(
                    screen_row(
                        stage="fixed_step",
                        start=start_name,
                        method="global_landweber",
                        regime="all",
                        ridge_relative="",
                        step_fraction=beta,
                        iterations=iteration,
                        a_calls=iteration,
                        at_calls=iteration,
                        score=field_score(global_path[iteration], truth, sources),
                        layout_rows=len(dataset),
                    )
                )
                for regime in ("low", "high"):
                    selected = np.flatnonzero(regimes == regime)
                    if not len(selected):
                        raise RuntimeError(f"validation has no {regime} spectral layouts")
                    rows.append(
                        screen_row(
                            stage="spectral_lookup_cell",
                            start=start_name,
                            method="spectral_lookup",
                            regime=regime,
                            ridge_relative="",
                            step_fraction=beta,
                            iterations=iteration,
                            a_calls=iteration,
                            at_calls=iteration,
                            score=field_score(
                                geometry_path[iteration][selected],
                                truth[selected],
                                sources[selected],
                            ),
                            layout_rows=len(selected),
                        )
                    )

        quadratic_path, _ = quadratic_steepest_descent_trajectory(
            start,
            observation,
            data["forward_matrix"],
            masks,
            support,
            checkpoints,
            spectral_per_sample(masks, normalization),
        )
        for iteration in checkpoints:
            rows.append(
                screen_row(
                    stage="quadratic_step_before_projection",
                    start=start_name,
                    method="quadratic_step",
                    regime="all",
                    ridge_relative="",
                    step_fraction="",
                    iterations=iteration,
                    a_calls=2 * iteration + 1,
                    at_calls=iteration,
                    score=field_score(quadratic_path[iteration], truth, sources),
                    layout_rows=len(dataset),
                )
            )

    selection: dict[str, object] = {
        "ridge_raw": ridge_raw_choice,
        "feasible_ridge": ridge_feasible_choice,
        "methods": {},
    }
    for start_name in START_KEYS:
        for solver in ("geometry_landweber", "global_landweber", "quadratic_step"):
            winner = choose(
                [
                    row
                    for row in rows
                    if row["start"] == start_name
                    and row["method"] == solver
                    and row["spectral_regime"] == "all"
                ]
            )
            short = {
                "geometry_landweber": "geometry",
                "global_landweber": "global",
                "quadratic_step": "quadratic",
            }[solver]
            selection["methods"][f"{start_name}_{short}"] = winner
        table = {}
        for regime in ("low", "high"):
            table[regime] = choose(
                [
                    row
                    for row in rows
                    if row["start"] == start_name
                    and row["method"] == "spectral_lookup"
                    and row["spectral_regime"] == regime
                ]
            )
        selection["methods"][f"{start_name}_lookup"] = {
            "start": start_name,
            "method": "spectral_lookup",
            "regime_table": table,
        }
    return rows, selection, {
        "locked_fno_raw": np.asarray(base[:, 0], dtype=np.float64),
        "feasible_fno": starts["fno"],
        "ridge_raw": raw_ridge,
        "feasible_ridge": feasible_ridge,
    }


def lookup_prediction(
    start: np.ndarray,
    observation: np.ndarray,
    masks: np.ndarray,
    support: np.ndarray,
    operator: np.ndarray,
    normalization: dict[str, dict[str, np.ndarray | float]],
    regimes: np.ndarray,
    table: dict[str, dict[str, object]],
) -> np.ndarray:
    output = np.empty_like(start, dtype=np.float64)
    for regime in ("low", "high"):
        indices = np.flatnonzero(regimes == regime)
        if not len(indices):
            continue
        choice = table[regime]
        iteration = int(choice["iterations"])
        output[indices] = landweber_trajectory(
            start[indices],
            observation[indices],
            operator,
            masks[indices],
            support,
            float(choice["step_fraction"]),
            [iteration],
            normalization,
            method="standard",
        )[iteration]
    return output


def predictions_for_split(
    data: dict[str, np.ndarray],
    dataset,
    base: np.ndarray,
    factory: CounterfactualInputFactory,
    normalization: dict[str, dict[str, np.ndarray | float]],
    global_spectral: float,
    regime_mapping: dict[str, str],
    selection: dict[str, object],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, float]]:
    sources = source_indices(dataset)
    observation = np.asarray(data["observation"][sources], dtype=np.float64)
    masks = geometry_masks(factory, dataset)
    support = data["support"]
    raw_fno = np.asarray(base[:, 0], dtype=np.float64)
    feasible_fno = feasibility_project(raw_fno, support)
    start_time = time.perf_counter()
    raw_ridge = ridge_predictions(
        data, dataset, factory, float(selection["ridge_raw"]["ridge_relative"])
    )
    ridge_raw_seconds = time.perf_counter() - start_time
    start_time = time.perf_counter()
    feasible_ridge = feasibility_project(
        ridge_predictions(
            data,
            dataset,
            factory,
            float(selection["feasible_ridge"]["ridge_relative"]),
        ),
        support,
    )
    ridge_feasible_seconds = time.perf_counter() - start_time
    predictions = {
        "locked_fno_raw": raw_fno,
        "feasible_fno": feasible_fno,
        "ridge_raw": raw_ridge,
        "feasible_ridge": feasible_ridge,
    }
    starts = {"fno": feasible_fno, "ridge": feasible_ridge}
    regimes = sample_regimes(dataset, regime_mapping)
    spectral = spectral_per_sample(masks, normalization)
    diagnostics: list[dict[str, object]] = []
    runtimes = {
        "ridge_raw": ridge_raw_seconds,
        "feasible_ridge": ridge_feasible_seconds,
    }
    for method, choice in selection["methods"].items():
        start_name = str(choice["start"])
        start = starts[start_name]
        method_start = time.perf_counter()
        solver = str(choice["method"])
        if solver == "geometry_landweber":
            iteration = int(choice["iterations"])
            prediction = landweber_trajectory(
                start,
                observation,
                data["forward_matrix"],
                masks,
                support,
                float(choice["step_fraction"]),
                [iteration],
                normalization,
                method="standard",
            )[iteration]
        elif solver == "global_landweber":
            iteration = int(choice["iterations"])
            prediction = global_landweber_trajectory(
                start,
                observation,
                data["forward_matrix"],
                masks,
                support,
                float(choice["step_fraction"]),
                [iteration],
                global_spectral,
            )[iteration]
        elif solver == "quadratic_step":
            iteration = int(choice["iterations"])
            trajectory, detail = quadratic_steepest_descent_trajectory(
                start,
                observation,
                data["forward_matrix"],
                masks,
                support,
                [iteration],
                spectral,
            )
            prediction = trajectory[iteration]
            before = detail["objective_before"]
            after = detail["objective_after"]
            relative_increase = (after - before) / np.maximum(before, 1e-12)
            diagnostics.append(
                {
                    "source_split": str(dataset.pairs[0]["source_split"]),
                    "start": start_name,
                    "method": method,
                    "iterations": iteration,
                    "step_sample_count": int(detail["step_size"].size),
                    "normalized_step_min": float(np.min(detail["normalized_step_fraction"])),
                    "normalized_step_median": float(np.median(detail["normalized_step_fraction"])),
                    "normalized_step_max": float(np.max(detail["normalized_step_fraction"])),
                    "projection_change_median": float(np.median(detail["projection_change_relative_l2"])),
                    "projection_change_p95": float(np.quantile(detail["projection_change_relative_l2"], 0.95)),
                    "projected_objective_increase_fraction": float(np.mean(relative_increase > 1e-10)),
                    "maximum_projected_objective_relative_increase": float(np.max(relative_increase)),
                    "wording_boundary": "exact along unconstrained quadratic line before projection only",
                }
            )
        elif solver == "spectral_lookup":
            prediction = lookup_prediction(
                start,
                observation,
                masks,
                support,
                data["forward_matrix"],
                normalization,
                regimes,
                choice["regime_table"],
            )
        else:
            raise RuntimeError(f"unknown selected solver {solver!r}")
        predictions[method] = prediction
        runtimes[method] = time.perf_counter() - method_start
    return predictions, diagnostics, runtimes


def attach_validation_scores(
    selection: dict[str, object],
    predictions: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    dataset,
) -> tuple[str, dict[str, dict[str, float]], dict[str, object]]:
    sources = source_indices(dataset)
    truth = np.asarray(data["field"][sources], dtype=np.float64)
    scores = {}
    for method, prediction in predictions.items():
        mean, median, count = field_score(prediction, truth, sources)
        scores[method] = {
            "source_field_mean_rel_l2": mean,
            "source_field_median_rel_l2": median,
            "independent_field_count": count,
        }
    champion = min(
        scores,
        key=lambda method: (
            scores[method]["source_field_mean_rel_l2"],
            method_total_calls(selection, method),
            method,
        ),
    )
    oracle = truth_oracle_headroom(predictions, champion, truth, sources)
    selection["validation_scores"] = scores
    selection["champion_method"] = champion
    selection["truth_oracle_diagnostic"] = oracle
    selection["operator_call_frontier"] = operator_call_frontier(selection, scores)
    return champion, scores, oracle


def truth_oracle_headroom(
    predictions: dict[str, np.ndarray],
    champion: str,
    truth: np.ndarray,
    sources: np.ndarray,
) -> dict[str, object]:
    methods = list(predictions)
    identifiers = np.unique(sources)
    collapsed = []
    for method in methods:
        errors = relative_l2(predictions[method], truth)
        collapsed.append(
            [float(np.mean(errors[sources == source])) for source in identifiers]
        )
    matrix = np.asarray(collapsed, dtype=np.float64)
    champion_error = matrix[methods.index(champion)]
    best_index = np.argmin(matrix, axis=0)
    oracle_error = np.min(matrix, axis=0)
    gain = 100.0 * (champion_error - oracle_error) / np.maximum(champion_error, 1e-12)
    return {
        "role": "nondeployable validation-truth diagnostic; never a selectable method",
        "candidate_method_count": len(methods),
        "independent_field_count": len(identifiers),
        "mean_oracle_headroom_vs_champion_pct": float(np.mean(gain)),
        "median_oracle_headroom_vs_champion_pct": float(np.median(gain)),
        "nonchampion_best_field_fraction": float(
            np.mean(best_index != methods.index(champion))
        ),
        "best_method_field_counts": {
            method: int(np.sum(best_index == index))
            for index, method in enumerate(methods)
        },
    }


def validation_field_pair(
    predictions: dict[str, np.ndarray],
    candidate: str,
    comparator: str,
    data: dict[str, np.ndarray],
    dataset,
    config: dict,
) -> dict[str, float]:
    sources = source_indices(dataset)
    truth = np.asarray(data["field"][sources], dtype=np.float64)
    identifiers = np.unique(sources)
    values = {}
    for method in (candidate, comparator):
        errors = relative_l2(predictions[method], truth)
        values[method] = np.asarray(
            [float(np.mean(errors[sources == source])) for source in identifiers]
        )
    gain = 100.0 * (values[comparator] - values[candidate]) / np.maximum(
        values[comparator], 1e-12
    )
    interval = bootstrap_interval(
        gain,
        int(config["gate"]["bootstrap_seed"]),
        int(config["gate"]["bootstrap_replicates"]),
    )
    return {
        "mean_field_gain_pct": float(np.mean(gain)),
        "field_cluster_ci95_low_pct": float(interval[0]),
        "field_cluster_ci95_high_pct": float(interval[1]),
        "positive_field_fraction": float(np.mean(gain > 0.0)),
        "harm_rate_gt_1pct": float(np.mean(gain < -1.0)),
    }


def method_total_calls(selection: dict[str, object], method: str) -> int:
    if method not in selection.get("methods", {}):
        return 0
    choice = selection["methods"][method]
    if choice["method"] == "spectral_lookup":
        return max(
            2 * int(cell["iterations"])
            for cell in choice["regime_table"].values()
        )
    iteration = int(choice["iterations"])
    return 3 * iteration + 1 if choice["method"] == "quadratic_step" else 2 * iteration


def operator_call_frontier(
    selection: dict[str, object], scores: dict[str, dict[str, float]]
) -> list[dict[str, object]]:
    """Return best validation accuracy at each observed A/A^T call budget."""
    budgets = sorted({method_total_calls(selection, method) for method in scores})
    output = []
    for budget in budgets:
        eligible = [
            method for method in scores if method_total_calls(selection, method) <= budget
        ]
        winner = min(
            eligible,
            key=lambda method: (
                float(scores[method]["source_field_mean_rel_l2"]),
                method_total_calls(selection, method),
                method,
            ),
        )
        output.append(
            {
                "maximum_operator_calls_per_sample": budget,
                "best_method": winner,
                "best_method_label": LABELS[winner],
                "source_field_mean_rel_l2": float(
                    scores[winner]["source_field_mean_rel_l2"]
                ),
                "call_definition": "A plus A^T; initialization costs are separate in the ledger",
            }
        )
    return output


def correction_references() -> dict[str, str]:
    references = {}
    for method in LABELS:
        if method.startswith("fno_"):
            references[method] = "feasible_fno"
        elif method.startswith("ridge_"):
            references[method] = "feasible_ridge"
        elif method == "locked_fno_raw":
            references[method] = "feasible_fno"
        elif method == "feasible_fno":
            references[method] = "feasible_fno"
        elif method == "ridge_raw":
            references[method] = "feasible_ridge"
        elif method == "feasible_ridge":
            references[method] = "feasible_ridge"
        else:
            raise RuntimeError(f"missing correction reference for {method}")
    return references


def comparison_pairs(champion: str) -> list[tuple[str, str]]:
    pairs = [(champion, "feasible_fno")]
    pairs.extend(
        (champion, method) for method in LABELS if method != champion
    )
    pairs.extend([
        ("feasible_fno", "locked_fno_raw"),
        ("feasible_ridge", "ridge_raw"),
        ("ridge_raw", "feasible_fno"),
        ("feasible_ridge", "feasible_fno"),
    ])
    for method in LABELS:
        if method.startswith("fno_"):
            pairs.extend([(method, "feasible_fno")])
        elif method.startswith("ridge_"):
            pairs.extend([(method, "feasible_ridge"), (method, "feasible_fno")])
    output = []
    for pair in pairs:
        if pair not in output:
            output.append(pair)
    return output


def call_ledger_rows(
    split: str,
    dataset,
    selection: dict[str, object],
    runtimes: dict[str, float],
    fno_seconds: float,
    batch_size: int,
) -> list[dict[str, object]]:
    geometry_count = len({str(row["geometry_id"]) for row in dataset.pairs})
    rows = []
    for method in LABELS:
        uses_fno = method in {"locked_fno_raw", "feasible_fno"} or method.startswith("fno_")
        uses_ridge = method in {"ridge_raw", "feasible_ridge"} or method.startswith("ridge_")
        a_calls = 0
        at_calls = 0
        worst_iterations = 0
        if method in selection["methods"]:
            choice = selection["methods"][method]
            if choice["method"] == "spectral_lookup":
                cells = choice["regime_table"].values()
                worst_iterations = max(int(cell["iterations"]) for cell in cells)
                a_calls = worst_iterations
                at_calls = worst_iterations
            else:
                worst_iterations = int(choice["iterations"])
                if choice["method"] == "quadratic_step":
                    a_calls = 2 * worst_iterations + 1
                    at_calls = worst_iterations
                else:
                    a_calls = worst_iterations
                    at_calls = worst_iterations
        if method == "ridge_raw":
            initialization_seconds = float(runtimes.get("ridge_raw", 0.0))
        elif uses_ridge:
            initialization_seconds = float(runtimes.get("feasible_ridge", 0.0))
        elif uses_fno:
            initialization_seconds = fno_seconds
        else:
            initialization_seconds = 0.0
        refinement_seconds = (
            float(runtimes.get(method, 0.0))
            if method in selection["methods"]
            else 0.0
        )
        rows.append(
            {
                "source_split": split,
                "method": method,
                "method_label": LABELS[method],
                "sample_count": len(dataset),
                "geometry_count": geometry_count,
                "maximum_iterations_per_sample": worst_iterations,
                "operator_a_calls_per_sample": a_calls,
                "operator_at_calls_per_sample": at_calls,
                "fno_sample_forwards": len(dataset) if uses_fno else 0,
                "fno_actual_batch_calls": math.ceil(len(dataset) / batch_size) if uses_fno else 0,
                "ridge_inverse_solve_count": geometry_count if uses_ridge else 0,
                "ridge_matvec_count": len(dataset) if uses_ridge else 0,
                "initialization_runtime_seconds": initialization_seconds,
                "refinement_runtime_seconds": refinement_seconds,
                "evaluation_metric_forward_calls_excluded": True,
            }
        )
    return rows


def quadratic_gate_rows(
    diagnostics: list[dict[str, object]], config: dict
) -> tuple[float, bool]:
    fraction = max(
        float(row["projected_objective_increase_fraction"]) for row in diagnostics
    )
    return fraction, fraction <= float(
        config["gate"]["maximum_quadratic_step_projected_objective_increase_fraction"]
    )


def plot_results(
    path: Path,
    selection: dict[str, object],
    pairwise: list[dict[str, object]],
    summary: list[dict[str, object]],
    call_rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    champion = str(selection["champion_method"])
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.5), constrained_layout=True)
    scores = selection["validation_scores"]
    ordered = sorted(scores, key=lambda method: float(scores[method]["source_field_mean_rel_l2"]))
    axes[0].barh(
        [LABELS[method] for method in ordered],
        [float(scores[method]["source_field_mean_rel_l2"]) for method in ordered],
        color=["#287271" if method == champion else "#9aa7a7" for method in ordered],
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Validation-selected absolute error")
    axes[0].set_xlabel("source-field mean relative L2")
    axes[0].tick_params(axis="y", labelsize=6)

    val_calls = {row["method"]: row for row in call_rows if row["source_split"] == "val"}
    axes[1].scatter(
        [int(val_calls[m]["operator_a_calls_per_sample"]) + int(val_calls[m]["operator_at_calls_per_sample"]) for m in ordered],
        [float(scores[m]["source_field_mean_rel_l2"]) for m in ordered],
        c=["#287271" if m == champion else "#d08c60" for m in ordered],
    )
    for method in ordered:
        row = val_calls[method]
        axes[1].annotate(
            method,
            (int(row["operator_a_calls_per_sample"]) + int(row["operator_at_calls_per_sample"]), float(scores[method]["source_field_mean_rel_l2"])),
            fontsize=5,
        )
    axes[1].set_title("Error versus A/A^T calls")
    axes[1].set_xlabel("calls per sample")
    axes[1].set_ylabel("validation field error")

    split_order = ["val", "test_iid", "test_noise_ood", "test_family_ood", "test_joint_ood"]
    selected = [find_pair(pairwise, split, champion, "feasible_fno") for split in split_order]
    means = np.asarray([float(row["mean_field_gain_pct"]) for row in selected])
    lower = means - np.asarray([float(row["field_cluster_ci95_low_pct"]) for row in selected])
    upper = np.asarray([float(row["field_cluster_ci95_high_pct"]) for row in selected]) - means
    axes[2].errorbar(range(len(split_order)), means, yerr=[lower, upper], fmt="o", color="#287271", capsize=3)
    axes[2].axhline(0.0, color="#58666e", linewidth=1)
    axes[2].set_xticks(range(len(split_order)), [s.replace("test_", "") for s in split_order], rotation=28, ha="right")
    axes[2].set_title("Champion vs feasible FNO")
    axes[2].set_ylabel("field gain (%) with 95% CI")

    quad = [row for row in diagnostics if row["source_split"] == "val"]
    axes[3].bar(
        [str(row["start"]) for row in quad],
        [float(row["normalized_step_median"]) for row in quad],
        color=["#4f6d7a", "#d08c60"],
    )
    axes[3].set_title("Quadratic candidate step audit")
    axes[3].set_ylabel("median alpha * L(mask)")
    axes[3].set_xlabel("initialization")
    fig.suptitle("v3k-D: strong non-learning numerical controls", fontsize=14)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_checksums(output_dir: Path) -> None:
    lines = [f"{sha256(output_dir / name)}  {name}" for name in PUBLIC_FILES]
    (output_dir / "v3k_d_strong_controls_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    dataset_config = read_json(ROOT / "configs" / str(config["dataset_config"]))
    private_path = ROOT / "results" / str(config["private_dataset_npz"])
    checkpoint_path = ROOT / "results" / str(config["base_checkpoint"])
    data = load_private_dataset(private_path)
    checkpoint_hash_before = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = choose_device(args.device or "cpu")
    design = config["pair_design"]
    factory = CounterfactualInputFactory(
        data, float(design["frozen_fno_input_ridge_relative"])
    )
    all_masks = np.asarray(
        [factory.mask(identifier) for identifier in sorted(factory.catalog)],
        dtype=np.float64,
    )
    normalization = geometry_normalization(data["forward_matrix"], all_masks)
    global_spectral = max(
        float(value["spectral_constant"]) for value in normalization.values()
    )
    boundary, regime_mapping, regime_manifest = spectral_regimes(factory, normalization)

    model = make_model(
        "fno",
        dataset_config["models"]["fno"],
        int(data["inputs"].shape[1]),
        residual=True,
    )
    model.load_state_dict(checkpoint, strict=True)
    selection_split = str(config["selection_split"])
    selection_dataset, selection_pairs = make_dataset(
        data,
        factory,
        selection_split,
        str(design["evaluation_geometry_partition_by_split"][selection_split]),
        design,
    )
    started = time.perf_counter()
    selection_base = precompute_base_predictions(model, selection_dataset, device, batch_size=16)
    selection_fno_seconds = time.perf_counter() - started
    screen, selection, _ = validation_screen(
        config,
        data,
        selection_dataset,
        selection_base,
        factory,
        normalization,
        global_spectral,
        regime_mapping,
    )
    validation_predictions, validation_diagnostics, validation_runtimes = predictions_for_split(
        data,
        selection_dataset,
        selection_base,
        factory,
        normalization,
        global_spectral,
        regime_mapping,
        selection,
    )
    champion, _, oracle = attach_validation_scores(
        selection, validation_predictions, data, selection_dataset
    )
    validation_pair = validation_field_pair(
        validation_predictions,
        champion,
        "feasible_fno",
        data,
        selection_dataset,
        config,
    )
    output_dir = ROOT / "results" / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_commit = {
        "experiment": config["name"],
        "selection_split": selection_split,
        "selection_metric": config["selection_protocol"]["metric"],
        "selection_tie_break": config["selection_protocol"]["tie_break"],
        "global_spectral_constant": global_spectral,
        "operator_only_spectral_boundary": boundary,
        "spectral_regime_manifest": regime_manifest,
        "selected": selection,
        "validation_champion_vs_feasible_fno": validation_pair,
        "independent_selection_fields": len(np.unique(source_indices(selection_dataset))),
        "selection_layout_rows": len(selection_dataset),
        "selection_sample_seed_sha256": hashlib.sha256(
            np.asarray(data["sample_seed"][source_indices(selection_dataset)], dtype=np.int64).tobytes()
        ).hexdigest(),
        "selection_geometry_sha256": hashlib.sha256(
            "\n".join(str(row["geometry_id"]) for row in selection_pairs).encode("utf-8")
        ).hexdigest(),
        "audit_camera_used_for_selection": False,
        "audit_or_reprojection_metrics_computed_by_selection_function": False,
        "test_field_or_metric_rows_present_at_commit": False,
        "truth_oracle_is_non_deployable": True,
    }
    (output_dir / PUBLIC_FILES[2]).write_text(
        json.dumps(selection_commit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    datasets = {selection_split: selection_dataset}
    bases = {selection_split: selection_base}
    fno_seconds = {selection_split: selection_fno_seconds}
    pair_manifest = [
        {**row, "schedule_role": "validation_selection"} for row in selection_pairs
    ]
    schedule_audit = {selection_split: schedule_balance(selection_pairs)}
    for split, partition in design["evaluation_geometry_partition_by_split"].items():
        split = str(split)
        if split == selection_split:
            continue
        dataset, pairs = make_dataset(data, factory, split, str(partition), design)
        datasets[split] = dataset
        started = time.perf_counter()
        bases[split] = precompute_base_predictions(model, dataset, device, batch_size=16)
        fno_seconds[split] = time.perf_counter() - started
        pair_manifest.extend(
            {**row, "schedule_role": "post_selection_development_audit"}
            for row in pairs
        )
        schedule_audit[split] = schedule_balance(pairs)

    sample_rows: list[dict[str, object]] = []
    quadratic_rows: list[dict[str, object]] = []
    call_rows: list[dict[str, object]] = []
    for split, dataset in datasets.items():
        if split == selection_split:
            predictions = validation_predictions
            diagnostics = validation_diagnostics
            runtimes = validation_runtimes
        else:
            predictions, diagnostics, runtimes = predictions_for_split(
                data,
                dataset,
                bases[split],
                factory,
                normalization,
                global_spectral,
                regime_mapping,
                selection,
            )
        quadratic_rows.extend(diagnostics)
        sample_rows.extend(
            metric_rows(
                predictions,
                dataset,
                data,
                factory,
                labels=LABELS,
                correction_reference=correction_references(),
            )
        )
        call_rows.extend(
            call_ledger_rows(
                split, dataset, selection, runtimes, fno_seconds[split], batch_size=16
            )
        )

    comparisons = comparison_pairs(champion)
    summary, pairwise = summarize(
        sample_rows, config, labels=LABELS, comparisons=comparisons
    )
    layout_summary = summarize_layouts(sample_rows, champion, "feasible_fno")
    quadratic_fraction, quadratic_ok = quadratic_gate_rows(quadratic_rows, config)
    dev_splits = [split for split in datasets if split != selection_split]
    champion_pairs = {
        split: find_pair(pairwise, split, champion, "feasible_fno")
        for split in datasets
    }
    gate = config["gate"]
    conditioning_signal = (
        float(oracle["mean_oracle_headroom_vs_champion_pct"])
        >= float(gate["minimum_truth_oracle_headroom_for_future_conditioning_pct"])
        and float(oracle["nonchampion_best_field_fraction"])
        >= float(gate["minimum_truth_oracle_nonchampion_fraction"])
    )
    gate_checks = {
        "quadratic_projected_objective_stability": quadratic_ok,
        "champion_validation_mean_gain": float(validation_pair["mean_field_gain_pct"])
        >= float(gate["minimum_champion_validation_gain_vs_feasible_fno_pct"]),
        "champion_validation_ci": float(validation_pair["field_cluster_ci95_low_pct"])
        > float(gate["minimum_champion_validation_ci95_low_pct"]),
        "champion_validation_coverage": float(validation_pair["positive_field_fraction"])
        >= float(gate["minimum_champion_positive_field_fraction"]),
        "champion_validation_tail": float(validation_pair["harm_rate_gt_1pct"])
        <= float(gate["maximum_champion_harm_rate_gt_1pct"]),
        "all_development_domains_field_no_material_harm": all(
            float(champion_pairs[split]["mean_field_gain_pct"])
            >= float(gate["minimum_each_development_domain_mean_gain_pct"])
            for split in dev_splits
        ),
        "all_development_domains_audit_no_material_harm": all(
            float(champion_pairs[split]["mean_audit_gain_pct"])
            >= float(gate["minimum_each_development_domain_audit_gain_pct"])
            for split in dev_splits
        ),
    }
    controls_complete = all(gate_checks.values())
    status = (
        "STRONG_NUMERICAL_CONTROLS_COMPLETE_BB_AND_FRESH_LOCK_STILL_REQUIRED"
        if controls_complete
        else "STRONG_NUMERICAL_CONTROL_GATE_FAILED"
    )

    write_csv(output_dir / PUBLIC_FILES[0], pair_manifest)
    write_csv(output_dir / PUBLIC_FILES[1], screen)
    write_csv(output_dir / PUBLIC_FILES[3], sample_rows)
    write_csv(output_dir / PUBLIC_FILES[4], summary)
    write_csv(output_dir / PUBLIC_FILES[5], pairwise)
    write_csv(output_dir / PUBLIC_FILES[6], layout_summary)
    write_csv(output_dir / PUBLIC_FILES[7], quadratic_rows)
    write_csv(output_dir / PUBLIC_FILES[8], call_rows)
    plot_results(
        output_dir / PUBLIC_FILES[11], selection, pairwise, summary, call_rows, quadratic_rows
    )
    dashboard = {
        "experiment": config["name"],
        "scientific_status": status,
        "strong_numerical_controls_complete": controls_complete,
        "champion_method": champion,
        "champion_label": LABELS[champion],
        "selected_hyperparameters": selection,
        "operator_call_frontier": selection["operator_call_frontier"],
        "validation_champion_vs_feasible_fno": validation_pair,
        "gate_checks": gate_checks,
        "gate_thresholds": gate,
        "quadratic_step_audit": {
            "maximum_objective_increase_fraction": quadratic_fraction,
            "exactness_boundary": "closed form is exact before projection, not for the constrained path",
        },
        "truth_oracle_diagnostic": oracle,
        "future_conditioning_signal_present": conditioning_signal,
        "schedule_audit": schedule_audit,
        "split_summary": summary,
        "pairwise_summary": pairwise,
        "layout_summary": layout_summary,
        "worst_layout": min(layout_summary, key=lambda row: float(row["mean_field_gain_pct"])),
        "next_decision": {
            "barzilai_borwein_control_required_next": True,
            "fresh_locked_fields_and_layouts_required_before_confirmation": True,
            "learned_scalar_development_training_authorized": False,
            "learned_scalar_confirmatory_training_authorized": False,
            "reason": "BB/spectral-gradient control and a fresh untouched lock are still missing",
            "conditional_scalar_worth_revisiting_after_bb": conditioning_signal,
        },
        "novelty_boundary": {
            "landweber_is_baseline_not_innovation": True,
            "global_step_is_baseline_not_innovation": True,
            "quadratic_step_is_baseline_not_innovation": True,
            "spectral_lookup_is_validation_tuned_non_neural_baseline": True,
            "ridge_warm_start_is_baseline_not_innovation": True,
        },
        "claims_boundary": config["claims_boundary"],
    }
    (output_dir / PUBLIC_FILES[9]).write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": status,
        "dashboard": dashboard,
        "protocol": {
            "selection": "validation source-field relative L2 after collapsing layouts; no audit/reprojection selection",
            "initializations": "locked FNO and validation-tuned unwindowed ridge, each followed by the same hard support projection",
            "controls": "geometry-normalized fixed step, one global absolute step, unconstrained quadratic closed-form step then projection, and validation-tuned two-regime lookup",
            "lookup_boundary": "largest gap in exact spectral constants over all 28 frozen operators; table values use validation truth",
            "quadratic_call_contract": "cached implementation uses 2T+1 A calls and T A^T calls",
            "statistical_unit": "source field after layout collapse",
            "test_read_timing": "development-domain metrics constructed only after selection commit was written",
        },
        "provenance": {
            "config_sha256": sha256(args.config),
            "selection_commit_sha256": sha256(output_dir / PUBLIC_FILES[2]),
            "private_dataset_sha256": sha256(private_path),
            "base_checkpoint_sha256_before": checkpoint_hash_before,
            "base_checkpoint_sha256_after": sha256(checkpoint_path),
            "base_checkpoint_drift": int(checkpoint_hash_before != sha256(checkpoint_path)),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "public_assets": PUBLIC_FILES,
        "private_assets": {
            "private_dataset_published": False,
            "base_checkpoint_published": False,
            "new_checkpoint_count": 0,
        },
    }
    (output_dir / PUBLIC_FILES[10]).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_checksums(output_dir)
    print(
        json.dumps(
            {
                "status": status,
                "champion": champion,
                "validation_pair": validation_pair,
                "truth_oracle": oracle,
                "gate_checks": gate_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
