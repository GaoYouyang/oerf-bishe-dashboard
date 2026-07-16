#!/usr/bin/env python3
"""Run the v3k-E safeguarded projected-BB mechanism gate."""

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
        forward_project,
        geometry_normalization,
        projected_bb_trajectory,
    )
    from .counterfactual_geometry import CounterfactualInputFactory, schedule_balance
    from .models import make_model
    from .run_v3k_a_counterfactual_supervision import (
        load_private_dataset,
        precompute_base_predictions,
    )
    from . import run_v3k_d_strong_numerical_controls as v3d
    from .train_eval import choose_device
except ImportError:
    from adjoint_landweber import (
        forward_project,
        geometry_normalization,
        projected_bb_trajectory,
    )
    from counterfactual_geometry import CounterfactualInputFactory, schedule_balance
    from models import make_model
    from run_v3k_a_counterfactual_supervision import (
        load_private_dataset,
        precompute_base_predictions,
    )
    import run_v3k_d_strong_numerical_controls as v3d
    from train_eval import choose_device


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v3k_e_projected_bb_gate.json"
PUBLIC_FILES = [
    "v3k_e_pair_manifest.csv",
    "v3k_e_validation_screen.csv",
    "v3k_e_selection_commit.json",
    "v3k_e_sample_metrics.csv",
    "v3k_e_split_summary.csv",
    "v3k_e_pairwise_summary.csv",
    "v3k_e_layout_summary.csv",
    "v3k_e_bb_step_audit.csv",
    "v3k_e_operator_call_ledger.csv",
    "v3k_e_projected_bb_dashboard.json",
    "v3k_e_projected_bb_report.json",
    "t16_v3k_e_projected_bb_gate.png",
]
LABELS = {
    "feasible_fno": "FNO + hard support",
    "fno_geometry": "FNO + fixed Landweber",
    "fno_quadratic": "FNO + quadratic step",
    "fno_pbb_strict": "FNO + projected BB (strict)",
    "fno_pbb_wide": "FNO + projected BB (wide)",
    "feasible_ridge": "Ridge + hard support",
    "ridge_geometry": "Ridge + fixed Landweber",
    "ridge_quadratic": "Ridge + quadratic step",
    "ridge_pbb_strict": "Ridge + projected BB (strict)",
    "ridge_pbb_wide": "Ridge + projected BB (wide)",
}
START_METHODS = {"fno": "feasible_fno", "ridge": "feasible_ridge"}
INHERITED_METHODS = {
    "feasible_fno",
    "fno_geometry",
    "fno_quadratic",
    "feasible_ridge",
    "ridge_geometry",
    "ridge_quadratic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def screen_row(
    *,
    start: str,
    mode: str,
    variant: str,
    lower: float,
    upper: float,
    iteration: int,
    score: tuple[float, float, int],
    diagnostics: dict[str, np.ndarray],
    layout_rows: int,
) -> dict[str, object]:
    steps = slice(0, iteration)
    source = diagnostics["step_source_code"][steps]
    objective = diagnostics["objective_before"][steps]
    transitions = np.diff(objective, axis=0)
    previous = objective[:-1]
    increase = transitions / np.maximum(previous, 1e-12)
    return {
        "selection_split": "val",
        "start": start,
        "bb_mode": mode,
        "bb_variant": variant,
        "initial_step_fraction": float(
            diagnostics["normalized_step_fraction"][0, 0]
        ),
        "normalized_step_min_bound": float(lower),
        "normalized_step_max_bound": float(upper),
        "normalized_step_observed_min": float(
            np.min(diagnostics["normalized_step_fraction"][steps])
        ),
        "normalized_step_observed_max": float(
            np.max(diagnostics["normalized_step_fraction"][steps])
        ),
        "iterations": int(iteration),
        "operator_a_calls_per_sample": int(iteration),
        "operator_at_calls_per_sample": int(iteration),
        "independent_field_count": int(score[2]),
        "layout_rows": int(layout_rows),
        "source_field_mean_rel_l2": float(score[0]),
        "source_field_median_rel_l2": float(score[1]),
        "bb1_step_fraction": float(np.mean(source == 1)),
        "bb2_step_fraction": float(np.mean(source == 2)),
        "fallback_step_fraction": float(np.mean(source == 3)),
        "clipped_low_fraction": float(
            np.mean(diagnostics["clipped_low"][steps])
        ),
        "clipped_high_fraction": float(
            np.mean(diagnostics["clipped_high"][steps])
        ),
        "observed_objective_transition_count": int(increase.size),
        "observed_objective_increase_fraction": float(
            np.mean(increase > 1e-10) if increase.size else 0.0
        ),
        "selection_uses_audit_or_reprojection": False,
    }


def choose(
    rows: list[dict[str, object]], variants: list[str]
) -> dict[str, object]:
    if not rows:
        raise RuntimeError("empty projected-BB selection cell")
    order = {variant: index for index, variant in enumerate(variants)}
    return dict(
        min(
            rows,
            key=lambda row: (
                float(row["source_field_mean_rel_l2"]),
                int(row["operator_a_calls_per_sample"])
                + int(row["operator_at_calls_per_sample"]),
                int(row["iterations"]),
                order[str(row["bb_variant"])],
                float(row["normalized_step_max_bound"]),
            ),
        )
    )


def validation_screen(
    config: dict,
    data: dict[str, np.ndarray],
    dataset,
    baseline: dict[str, np.ndarray],
    factory: CounterfactualInputFactory,
    normalization: dict[str, dict[str, np.ndarray | float]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    protocol = config["selection_protocol"]
    sources = v3d.source_indices(dataset)
    truth = np.asarray(data["field"][sources], dtype=np.float64)
    observation = np.asarray(data["observation"][sources], dtype=np.float64)
    masks = v3d.geometry_masks(factory, dataset)
    spectral = v3d.spectral_per_sample(masks, normalization)
    checkpoints = [int(value) for value in protocol["iteration_counts"]]
    variants = [str(value) for value in protocol["variants"]]
    rows: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}

    for start_name, start_method in START_METHODS.items():
        start = baseline[start_method]
        for mode, upper_values in (
            ("strict", [float(protocol["strict_normalized_step_max"])]),
            (
                "wide",
                [float(value) for value in protocol["wide_normalized_step_max_grid"]],
            ),
        ):
            for variant in variants:
                for upper in upper_values:
                    trajectory, diagnostics = projected_bb_trajectory(
                        start,
                        observation,
                        data["forward_matrix"],
                        masks,
                        data["support"],
                        checkpoints,
                        spectral,
                        variant,
                        float(protocol["initial_step_fraction"]),
                        float(protocol["normalized_step_min"]),
                        upper,
                        float(protocol["curvature_floor_relative"]),
                    )
                    for iteration in checkpoints:
                        rows.append(
                            screen_row(
                                start=start_name,
                                mode=mode,
                                variant=variant,
                                lower=float(protocol["normalized_step_min"]),
                                upper=upper,
                                iteration=iteration,
                                score=v3d.field_score(
                                    trajectory[iteration], truth, sources
                                ),
                                diagnostics=diagnostics,
                                layout_rows=len(dataset),
                            )
                        )
            winner = choose(
                [
                    row
                    for row in rows
                    if row["start"] == start_name and row["bb_mode"] == mode
                ],
                variants,
            )
            winner["public_method"] = f"{start_name}_pbb_{mode}"
            selected[f"{start_name}_{mode}"] = winner
    return rows, selected


def baseline_predictions(
    data: dict[str, np.ndarray],
    dataset,
    base: np.ndarray,
    factory: CounterfactualInputFactory,
    normalization: dict[str, dict[str, np.ndarray | float]],
    global_spectral: float,
    regime_mapping: dict[str, str],
    baseline_selection: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    predictions, _, runtimes = v3d.predictions_for_split(
        data,
        dataset,
        base,
        factory,
        normalization,
        global_spectral,
        regime_mapping,
        baseline_selection,
    )
    return (
        {method: predictions[method] for method in LABELS if method in INHERITED_METHODS},
        runtimes,
    )


def pbb_predictions(
    config: dict,
    data: dict[str, np.ndarray],
    dataset,
    predictions: dict[str, np.ndarray],
    factory: CounterfactualInputFactory,
    normalization: dict[str, dict[str, np.ndarray | float]],
    selected: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    protocol = config["selection_protocol"]
    sources = v3d.source_indices(dataset)
    observation = np.asarray(data["observation"][sources], dtype=np.float64)
    masks = v3d.geometry_masks(factory, dataset)
    spectral = v3d.spectral_per_sample(masks, normalization)
    split = str(dataset.pairs[0]["source_split"])
    audits: list[dict[str, object]] = []
    runtimes: dict[str, float] = {}

    for choice in selected.values():
        start_name = str(choice["start"])
        method = str(choice["public_method"])
        iteration = int(choice["iterations"])
        started = time.perf_counter()
        trajectory, detail = projected_bb_trajectory(
            predictions[START_METHODS[start_name]],
            observation,
            data["forward_matrix"],
            masks,
            data["support"],
            [iteration],
            spectral,
            str(choice["bb_variant"]),
            float(protocol["initial_step_fraction"]),
            float(protocol["normalized_step_min"]),
            float(choice["normalized_step_max_bound"]),
            float(protocol["curvature_floor_relative"]),
        )
        prediction = trajectory[iteration]
        runtimes[method] = time.perf_counter() - started
        predictions[method] = prediction

        weight = masks[:, None, :, None]
        final_residual = (
            forward_project(prediction, data["forward_matrix"]) - observation
        ) * weight
        final_objective = 0.5 * np.sum(final_residual * final_residual, axis=(1, 2, 3))
        objective = np.concatenate(
            [detail["objective_before"], final_objective[None]], axis=0
        )
        relative_increase = np.diff(objective, axis=0) / np.maximum(
            objective[:-1], 1e-12
        )
        source_codes = detail["step_source_code"]
        audits.append(
            {
                "source_split": split,
                "method": method,
                "method_label": LABELS[method],
                "start": start_name,
                "bb_mode": str(choice["bb_mode"]),
                "bb_variant": str(choice["bb_variant"]),
                "iterations": iteration,
                "layout_count": len(dataset),
                "step_sample_count": int(detail["step_size"].size),
                "initial_step_fraction": float(protocol["initial_step_fraction"]),
                "normalized_step_min_bound": float(protocol["normalized_step_min"]),
                "normalized_step_max_bound": float(
                    choice["normalized_step_max_bound"]
                ),
                "normalized_step_observed_min": float(
                    np.min(detail["normalized_step_fraction"])
                ),
                "normalized_step_observed_median": float(
                    np.median(detail["normalized_step_fraction"])
                ),
                "normalized_step_observed_max": float(
                    np.max(detail["normalized_step_fraction"])
                ),
                "bb1_step_fraction": float(np.mean(source_codes == 1)),
                "bb2_step_fraction": float(np.mean(source_codes == 2)),
                "initial_step_fraction_of_steps": float(np.mean(source_codes == 0)),
                "fallback_step_fraction": float(np.mean(source_codes == 3)),
                "clipped_low_fraction": float(np.mean(detail["clipped_low"])),
                "clipped_high_fraction": float(np.mean(detail["clipped_high"])),
                "objective_transition_count": int(relative_increase.size),
                "objective_increase_fraction": float(
                    np.mean(relative_increase > 1e-10)
                ),
                "maximum_objective_relative_increase": float(
                    np.max(relative_increase)
                ),
                "projection_change_median": float(
                    np.median(detail["projected_change_relative_l2"])
                ),
                "projection_change_p95": float(
                    np.quantile(detail["projected_change_relative_l2"], 0.95)
                ),
                "final_objective_forward_is_diagnostic_only": True,
                "convergence_claimed": False,
            }
        )
    return audits, runtimes


def correction_references() -> dict[str, str]:
    return {
        method: "feasible_fno" if method.startswith("fno_") or method == "feasible_fno" else "feasible_ridge"
        for method in LABELS
    }


def comparison_pairs() -> list[tuple[str, str]]:
    return [
        ("fno_pbb_strict", "fno_geometry"),
        ("fno_pbb_wide", "fno_geometry"),
        ("fno_pbb_wide", "fno_quadratic"),
        ("fno_pbb_wide", "fno_pbb_strict"),
        ("ridge_pbb_strict", "ridge_geometry"),
        ("ridge_pbb_wide", "ridge_geometry"),
        ("ridge_pbb_wide", "ridge_quadratic"),
        ("ridge_pbb_wide", "ridge_pbb_strict"),
        ("fno_pbb_wide", "ridge_pbb_wide"),
    ]


def baseline_calls(
    method: str, baseline_selection: dict[str, object]
) -> tuple[int, int, int]:
    if method in {"feasible_fno", "feasible_ridge"}:
        return 0, 0, 0
    choice = baseline_selection["methods"][method]
    iteration = int(choice["iterations"])
    if str(choice["method"]) == "quadratic_step":
        return iteration, 2 * iteration + 1, iteration
    return iteration, iteration, iteration


def call_ledger_rows(
    split: str,
    dataset,
    baseline_selection: dict[str, object],
    pbb_selection: dict[str, dict[str, object]],
    baseline_runtimes: dict[str, float],
    pbb_runtimes: dict[str, float],
    fno_seconds: float,
    batch_size: int,
) -> list[dict[str, object]]:
    geometry_count = len({str(row["geometry_id"]) for row in dataset.pairs})
    pbb_by_method = {
        str(choice["public_method"]): choice for choice in pbb_selection.values()
    }
    rows = []
    for method in LABELS:
        if method in pbb_by_method:
            iteration = int(pbb_by_method[method]["iterations"])
            a_calls = at_calls = iteration
            selection_origin = "v3k_e_validation"
        else:
            iteration, a_calls, at_calls = baseline_calls(method, baseline_selection)
            selection_origin = "inherited_v3k_d_validation"
        uses_fno = method == "feasible_fno" or method.startswith("fno_")
        uses_ridge = method == "feasible_ridge" or method.startswith("ridge_")
        rows.append(
            {
                "source_split": split,
                "method": method,
                "method_label": LABELS[method],
                "selection_origin": selection_origin,
                "sample_count": len(dataset),
                "geometry_count": geometry_count,
                "maximum_iterations_per_sample": iteration,
                "operator_a_calls_per_sample": a_calls,
                "operator_at_calls_per_sample": at_calls,
                "total_refinement_operator_calls_per_sample": a_calls + at_calls,
                "fno_sample_forwards": len(dataset) if uses_fno else 0,
                "fno_actual_batch_calls": math.ceil(len(dataset) / batch_size)
                if uses_fno
                else 0,
                "ridge_inverse_solve_count": geometry_count if uses_ridge else 0,
                "ridge_matvec_count": len(dataset) if uses_ridge else 0,
                "initialization_runtime_seconds": fno_seconds
                if uses_fno
                else float(baseline_runtimes.get("feasible_ridge", 0.0)),
                "refinement_runtime_seconds": float(
                    pbb_runtimes.get(method, baseline_runtimes.get(method, 0.0))
                ),
                "metric_and_final_objective_forward_calls_excluded": True,
            }
        )
    return rows


def validation_scores(
    predictions: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    dataset,
) -> dict[str, dict[str, float]]:
    sources = v3d.source_indices(dataset)
    truth = np.asarray(data["field"][sources], dtype=np.float64)
    output = {}
    for method in LABELS:
        mean, median, count = v3d.field_score(predictions[method], truth, sources)
        output[method] = {
            "source_field_mean_rel_l2": mean,
            "source_field_median_rel_l2": median,
            "independent_field_count": count,
        }
    return output


def selected_total_calls(
    method: str,
    baseline_selection: dict[str, object],
    pbb_selection: dict[str, dict[str, object]],
) -> int:
    for choice in pbb_selection.values():
        if str(choice["public_method"]) == method:
            return 2 * int(choice["iterations"])
    _, a_calls, at_calls = baseline_calls(method, baseline_selection)
    return a_calls + at_calls


def plot_results(
    path: Path,
    scores: dict[str, dict[str, float]],
    pairwise: list[dict[str, object]],
    audits: list[dict[str, object]],
    call_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17.4, 4.6), constrained_layout=True)
    methods = [
        "fno_geometry",
        "fno_quadratic",
        "fno_pbb_strict",
        "fno_pbb_wide",
    ]
    axes[0].barh(
        [LABELS[method] for method in methods],
        [scores[method]["source_field_mean_rel_l2"] for method in methods],
        color=["#8a9a9a", "#d08c60", "#557a95", "#287271"],
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Validation field error")
    axes[0].set_xlabel("source-field mean relative L2")
    axes[0].tick_params(axis="y", labelsize=7)

    validation_calls = {
        str(row["method"]): row
        for row in call_rows
        if str(row["source_split"]) == "val"
    }
    axes[1].scatter(
        [validation_calls[method]["total_refinement_operator_calls_per_sample"] for method in methods],
        [scores[method]["source_field_mean_rel_l2"] for method in methods],
        c=["#8a9a9a", "#d08c60", "#557a95", "#287271"],
    )
    for method in methods:
        axes[1].annotate(
            method.replace("fno_", ""),
            (
                validation_calls[method]["total_refinement_operator_calls_per_sample"],
                scores[method]["source_field_mean_rel_l2"],
            ),
            fontsize=7,
        )
    axes[1].set_title("Accuracy at declared call budget")
    axes[1].set_xlabel("A + A^T calls per sample")
    axes[1].set_ylabel("validation field error")

    split_order = [
        "val",
        "test_iid",
        "test_noise_ood",
        "test_family_ood",
        "test_joint_ood",
    ]
    gains = [
        v3d.find_pair(pairwise, split, "fno_pbb_wide", "fno_geometry")
        for split in split_order
    ]
    axes[2].bar(
        range(len(split_order)),
        [float(row["mean_field_gain_pct"]) for row in gains],
        color=[
            "#287271" if float(row["mean_field_gain_pct"]) >= 0 else "#b8564f"
            for row in gains
        ],
    )
    axes[2].axhline(0.0, color="#58666e", linewidth=1)
    axes[2].set_xticks(
        range(len(split_order)),
        [value.replace("test_", "") for value in split_order],
        rotation=28,
        ha="right",
    )
    axes[2].set_title("Wide PBB vs fixed Landweber")
    axes[2].set_ylabel("mean field gain (%)")

    selected_audits = [
        row
        for row in audits
        if str(row["source_split"]) == "val"
        and str(row["method"]).startswith("fno_")
    ]
    axes[3].bar(
        [str(row["bb_mode"]) for row in selected_audits],
        [float(row["clipped_high_fraction"]) for row in selected_audits],
        color=["#557a95", "#d08c60"],
    )
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_title("Selected PBB high-clip rate")
    axes[3].set_ylabel("fraction of all scalar steps")
    fig.suptitle("v3k-E: projected BB is fast but noise-unsafe", fontsize=14)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_checksums(output_dir: Path) -> None:
    lines = [f"{v3d.sha256(output_dir / name)}  {name}" for name in PUBLIC_FILES]
    (output_dir / "v3k_e_projected_bb_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    dataset_config = read_json(ROOT / "configs" / str(config["dataset_config"]))
    private_path = ROOT / "results" / str(config["private_dataset_npz"])
    checkpoint_path = ROOT / "results" / str(config["base_checkpoint"])
    baseline_commit_path = ROOT / "results" / str(config["baseline_selection_commit"])
    baseline_commit = read_json(baseline_commit_path)
    baseline_selection = baseline_commit["selected"]
    data = load_private_dataset(private_path)
    checkpoint_hash_before = v3d.sha256(checkpoint_path)
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
    _, regime_mapping, _ = v3d.spectral_regimes(factory, normalization)

    model = make_model(
        "fno",
        dataset_config["models"]["fno"],
        int(data["inputs"].shape[1]),
        residual=True,
    )
    model.load_state_dict(checkpoint, strict=True)
    selection_split = str(config["selection_split"])
    selection_dataset, selection_pairs = v3d.make_dataset(
        data,
        factory,
        selection_split,
        str(design["evaluation_geometry_partition_by_split"][selection_split]),
        design,
    )
    started = time.perf_counter()
    selection_base = precompute_base_predictions(
        model, selection_dataset, device, batch_size=16
    )
    selection_fno_seconds = time.perf_counter() - started
    validation_predictions, validation_baseline_runtimes = baseline_predictions(
        data,
        selection_dataset,
        selection_base,
        factory,
        normalization,
        global_spectral,
        regime_mapping,
        baseline_selection,
    )
    screen, selected = validation_screen(
        config,
        data,
        selection_dataset,
        validation_predictions,
        factory,
        normalization,
    )
    validation_audits, validation_pbb_runtimes = pbb_predictions(
        config,
        data,
        selection_dataset,
        validation_predictions,
        factory,
        normalization,
        selected,
    )
    scores = validation_scores(
        validation_predictions, data, selection_dataset
    )
    champion = min(
        LABELS,
        key=lambda method: (
            scores[method]["source_field_mean_rel_l2"],
            selected_total_calls(method, baseline_selection, selected),
            method,
        ),
    )

    output_dir = ROOT / "results" / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_commit = {
        "experiment": config["name"],
        "selection_split": selection_split,
        "selection_metric": config["selection_protocol"]["metric"],
        "selection_tie_break": config["selection_protocol"]["tie_break"],
        "selected_projected_bb": selected,
        "validation_scores": scores,
        "validation_champion": champion,
        "validation_champion_label": LABELS[champion],
        "inherited_baseline_selection_commit_sha256": v3d.sha256(
            baseline_commit_path
        ),
        "independent_selection_fields": len(
            np.unique(v3d.source_indices(selection_dataset))
        ),
        "selection_layout_rows": len(selection_dataset),
        "selection_sample_seed_sha256": hashlib.sha256(
            np.asarray(
                data["sample_seed"][v3d.source_indices(selection_dataset)],
                dtype=np.int64,
            ).tobytes()
        ).hexdigest(),
        "selection_geometry_sha256": hashlib.sha256(
            "\n".join(str(row["geometry_id"]) for row in selection_pairs).encode(
                "utf-8"
            )
        ).hexdigest(),
        "audit_camera_used_for_selection": False,
        "audit_or_reprojection_metrics_computed_by_selection_function": False,
        "test_field_or_metric_rows_present_at_commit": False,
        "strict_and_wide_selected_independently": True,
    }
    (output_dir / PUBLIC_FILES[2]).write_text(
        json.dumps(selection_commit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    datasets = {selection_split: selection_dataset}
    bases = {selection_split: selection_base}
    fno_seconds = {selection_split: selection_fno_seconds}
    pair_manifest = [
        {**row, "schedule_role": "validation_selection"}
        for row in selection_pairs
    ]
    schedule_audit = {selection_split: schedule_balance(selection_pairs)}
    for split, partition in design["evaluation_geometry_partition_by_split"].items():
        split = str(split)
        if split == selection_split:
            continue
        dataset, pairs = v3d.make_dataset(data, factory, split, str(partition), design)
        datasets[split] = dataset
        started = time.perf_counter()
        bases[split] = precompute_base_predictions(
            model, dataset, device, batch_size=16
        )
        fno_seconds[split] = time.perf_counter() - started
        pair_manifest.extend(
            {**row, "schedule_role": "post_selection_development_audit"}
            for row in pairs
        )
        schedule_audit[split] = schedule_balance(pairs)

    sample_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    call_rows: list[dict[str, object]] = []
    for split, dataset in datasets.items():
        if split == selection_split:
            predictions = validation_predictions
            baseline_runtimes = validation_baseline_runtimes
            pbb_runtimes = validation_pbb_runtimes
            audits = validation_audits
        else:
            predictions, baseline_runtimes = baseline_predictions(
                data,
                dataset,
                bases[split],
                factory,
                normalization,
                global_spectral,
                regime_mapping,
                baseline_selection,
            )
            audits, pbb_runtimes = pbb_predictions(
                config,
                data,
                dataset,
                predictions,
                factory,
                normalization,
                selected,
            )
        audit_rows.extend(audits)
        sample_rows.extend(
            v3d.metric_rows(
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
                split,
                dataset,
                baseline_selection,
                selected,
                baseline_runtimes,
                pbb_runtimes,
                fno_seconds[split],
                batch_size=16,
            )
        )

    comparisons = comparison_pairs()
    summary, pairwise = v3d.summarize(
        sample_rows, config, labels=LABELS, comparisons=comparisons
    )
    layout_summary = v3d.summarize_layouts(
        sample_rows, "fno_pbb_wide", "fno_geometry"
    )
    wide_vs_geometry = {
        split: v3d.find_pair(
            pairwise, split, "fno_pbb_wide", "fno_geometry"
        )
        for split in datasets
    }
    val_wide_vs_quadratic = v3d.find_pair(
        pairwise, "val", "fno_pbb_wide", "fno_quadratic"
    )
    strict_val_audit = next(
        row
        for row in audit_rows
        if row["source_split"] == "val" and row["method"] == "fno_pbb_strict"
    )
    wide_val_audit = next(
        row
        for row in audit_rows
        if row["source_split"] == "val" and row["method"] == "fno_pbb_wide"
    )
    gate = config["gate"]
    development_splits = [split for split in datasets if split != selection_split]
    geometry_calls = selected_total_calls(
        "fno_geometry", baseline_selection, selected
    )
    wide_calls = selected_total_calls("fno_pbb_wide", baseline_selection, selected)
    gate_checks = {
        "wide_validation_mean_gain_vs_geometry": float(
            wide_vs_geometry["val"]["mean_field_gain_pct"]
        )
        >= float(gate["minimum_wide_validation_mean_gain_vs_geometry_pct"]),
        "wide_validation_mean_gain_vs_quadratic": float(
            val_wide_vs_quadratic["mean_field_gain_pct"]
        )
        >= float(gate["minimum_wide_validation_mean_gain_vs_quadratic_pct"]),
        "wide_validation_tail_safe": float(
            wide_vs_geometry["val"]["harm_rate_gt_1pct"]
        )
        <= float(gate["maximum_wide_validation_harm_rate_gt_1pct"]),
        "all_development_domains_mean_safe": all(
            float(wide_vs_geometry[split]["mean_field_gain_pct"])
            >= float(
                gate["minimum_each_development_domain_mean_gain_vs_geometry_pct"]
            )
            for split in development_splits
        ),
        "selected_wide_objective_stable": float(
            wide_val_audit["objective_increase_fraction"]
        )
        <= float(gate["maximum_selected_objective_increase_fraction"]),
        "strict_bound_degeneracy_observed": float(
            strict_val_audit["clipped_high_fraction"]
        )
        >= float(gate["minimum_strict_high_clip_fraction_for_degeneracy_diagnosis"]),
        "same_call_budget_as_fixed_landweber": wide_calls == geometry_calls,
    }
    publishable_gate = all(gate_checks.values())
    status = (
        "PROJECTED_BB_CONTROL_PASSES_FRESH_LOCK_REQUIRED"
        if publishable_gate
        else "PBB_MEAN_ACCELERATION_BUT_NOISE_OOD_AND_TAIL_UNSAFE_DISCREPANCY_CONTROL_REQUIRED"
    )

    v3d.write_csv(output_dir / PUBLIC_FILES[0], pair_manifest)
    v3d.write_csv(output_dir / PUBLIC_FILES[1], screen)
    v3d.write_csv(output_dir / PUBLIC_FILES[3], sample_rows)
    v3d.write_csv(output_dir / PUBLIC_FILES[4], summary)
    v3d.write_csv(output_dir / PUBLIC_FILES[5], pairwise)
    v3d.write_csv(output_dir / PUBLIC_FILES[6], layout_summary)
    v3d.write_csv(output_dir / PUBLIC_FILES[7], audit_rows)
    v3d.write_csv(output_dir / PUBLIC_FILES[8], call_rows)
    plot_results(
        output_dir / PUBLIC_FILES[11], scores, pairwise, audit_rows, call_rows
    )

    dashboard = {
        "experiment": config["name"],
        "scientific_status": status,
        "publishable_mechanism_gate_passed": publishable_gate,
        "validation_champion": champion,
        "validation_champion_label": LABELS[champion],
        "selected_projected_bb": selected,
        "validation_scores": scores,
        "gate_checks": gate_checks,
        "gate_thresholds": gate,
        "same_call_budget": {
            "fno_geometry_total_calls": geometry_calls,
            "fno_pbb_wide_total_calls": wide_calls,
            "metric_and_final_objective_calls_excluded": True,
        },
        "wide_pbb_vs_fixed_landweber": wide_vs_geometry,
        "wide_pbb_vs_quadratic_validation": val_wide_vs_quadratic,
        "selected_step_audit": {
            "strict_validation": strict_val_audit,
            "wide_validation": wide_val_audit,
        },
        "schedule_audit": schedule_audit,
        "split_summary": summary,
        "pairwise_summary": pairwise,
        "layout_summary": layout_summary,
        "worst_layout": min(
            layout_summary, key=lambda row: float(row["mean_field_gain_pct"])
        ),
        "interpretation": {
            "positive_result": "wide safeguarded PBB improves validation mean and family/joint OOD at the same A/A^T budget",
            "failure_result": "the same rule has a large validation harm tail and severe noise-OOD regression",
            "mechanism_diagnosis": "data-fidelity descent can continue after truth-field error begins to rise; objective descent alone is not a noise stopping rule",
            "strict_result": "the classical sub-2/L cap clips nearly every spectral proposal and collapses toward fixed Landweber",
        },
        "next_decision": {
            "learned_scalar_development_training_authorized": False,
            "learned_scalar_confirmatory_training_authorized": False,
            "next_control": "noise-aware discrepancy or residual-whiteness stopping on the same PBB trajectory",
            "second_control": "literature-aligned nonmonotone SPG with all line-search A calls charged",
            "fresh_locked_fields_and_layouts_required_before_claim": True,
            "reason": "mean acceleration is real, but the current rule is not tail-safe or noise-OOD safe",
        },
        "novelty_boundary": {
            "bb1988_is_baseline_not_innovation": True,
            "clipped_projected_bb_is_baseline_not_innovation": True,
            "spg_convergence_cannot_be_transferred_to_no_line_search_pbb": True,
            "potential_algorithmic_contribution": "operator-conditioned noise-aware stopping or safeguarded spectral policy, only after deterministic controls and fresh-lock confirmation",
        },
        "claims_boundary": config["claims_boundary"],
    }
    (output_dir / PUBLIC_FILES[9]).write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": status,
        "dashboard": dashboard,
        "protocol": {
            "selection": "strict and wide PBB selected independently on validation source-field relative L2 after layout collapse",
            "secant": "BB secants use successive projected feasible iterates and their masked data gradients",
            "safeguard": "invalid curvature falls back to the inherited initial step, then all steps are clipped in alpha times L(mask)",
            "call_contract": "each PBB iteration uses one A and one A^T; final-objective and evaluation forwards are diagnostic and excluded",
            "statistical_unit": "source field after four-layout collapse",
            "test_read_timing": "development-domain datasets and metrics constructed only after selection commit was written",
        },
        "provenance": {
            "config_sha256": v3d.sha256(args.config),
            "selection_commit_sha256": v3d.sha256(output_dir / PUBLIC_FILES[2]),
            "baseline_selection_commit_sha256": v3d.sha256(baseline_commit_path),
            "private_dataset_sha256": v3d.sha256(private_path),
            "base_checkpoint_sha256_before": checkpoint_hash_before,
            "base_checkpoint_sha256_after": v3d.sha256(checkpoint_path),
            "base_checkpoint_drift": int(
                checkpoint_hash_before != v3d.sha256(checkpoint_path)
            ),
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
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_checksums(output_dir)
    print(
        json.dumps(
            {
                "status": status,
                "validation_champion": champion,
                "selected_projected_bb": selected,
                "wide_pbb_vs_fixed_landweber": wide_vs_geometry,
                "gate_checks": gate_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
