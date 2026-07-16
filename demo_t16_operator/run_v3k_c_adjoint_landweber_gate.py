#!/usr/bin/env python3
"""Run the v3k-C zero-learning adjoint and Landweber gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .adjoint_landweber import (
        adjoint_inner_product_relative_error,
        feasibility_project,
        finite_difference_gradient_relative_error,
        forward_project,
        geometry_normalization,
        landweber_trajectory,
    )
    from .counterfactual_geometry import (
        CounterfactualGeometryDataset,
        CounterfactualInputFactory,
        build_pair_schedule,
        schedule_balance,
    )
    from .models import make_model
    from .run_v3k_a_counterfactual_supervision import (
        bootstrap_interval,
        load_private_dataset,
        precompute_base_predictions,
    )
    from .train_eval import choose_device
except ImportError:
    from adjoint_landweber import (
        adjoint_inner_product_relative_error,
        feasibility_project,
        finite_difference_gradient_relative_error,
        forward_project,
        geometry_normalization,
        landweber_trajectory,
    )
    from counterfactual_geometry import (
        CounterfactualGeometryDataset,
        CounterfactualInputFactory,
        build_pair_schedule,
        schedule_balance,
    )
    from models import make_model
    from run_v3k_a_counterfactual_supervision import (
        bootstrap_interval,
        load_private_dataset,
        precompute_base_predictions,
    )
    from train_eval import choose_device


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v3k_c_adjoint_landweber_gate.json"
PUBLIC_FILES = [
    "v3k_c_pair_manifest.csv",
    "v3k_c_adjoint_checks.csv",
    "v3k_c_validation_screen.csv",
    "v3k_c_selection_commit.json",
    "v3k_c_sample_metrics.csv",
    "v3k_c_split_summary.csv",
    "v3k_c_pairwise_summary.csv",
    "v3k_c_adjoint_landweber_dashboard.json",
    "v3k_c_adjoint_landweber_report.json",
    "t16_v3k_c_adjoint_landweber_gate.png",
    "v3k_c_layout_summary.csv",
]
LABELS = {
    "locked_fno_raw": "Locked FNO (raw)",
    "feasible_fno": "FNO + nonnegative support",
    "landweber_standard": "Projected Landweber",
    "landweber_jacobi": "Jacobi-projected Landweber",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(output_dir: Path) -> None:
    rows = [f"{sha256(output_dir / name)}  {name}" for name in PUBLIC_FILES]
    (output_dir / "v3k_c_adjoint_landweber_checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def source_indices(dataset: CounterfactualGeometryDataset) -> np.ndarray:
    return np.asarray(
        [int(row["source_index"]) for row in dataset.pairs], dtype=np.int64
    )


def source_cluster_values(
    values: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    identifiers = np.unique(sources)
    collapsed = np.asarray(
        [float(np.mean(values[sources == source])) for source in identifiers],
        dtype=np.float64,
    )
    return identifiers, collapsed


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm((prediction - reference).reshape(len(prediction), -1), axis=1)
    denominator = np.maximum(
        np.linalg.norm(reference.reshape(len(reference), -1), axis=1), 1e-12
    )
    return numerator / denominator


def direct_projection_error(
    prediction: np.ndarray,
    reference: np.ndarray,
    operator: np.ndarray,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    projected = forward_project(prediction, operator)
    difference = projected - np.asarray(reference, dtype=np.float64)
    denominator = np.asarray(reference, dtype=np.float64)
    if masks is not None:
        weight = np.asarray(masks, dtype=np.float64)[:, None, :, None]
        difference = difference * weight
        denominator = denominator * weight
    return np.linalg.norm(difference.reshape(len(prediction), -1), axis=1) / np.maximum(
        np.linalg.norm(denominator.reshape(len(prediction), -1), axis=1), 1e-12
    )


def audit_error(
    prediction: np.ndarray,
    clean_observation: np.ndarray,
    operator: np.ndarray,
    audit_index: int,
) -> np.ndarray:
    projected = np.einsum(
        "bdp,np->bdn",
        prediction.reshape(len(prediction), prediction.shape[1], -1),
        np.asarray(operator, dtype=np.float64)[int(audit_index)],
        optimize=True,
    )
    return relative_l2(projected, clean_observation[:, :, int(audit_index)])


def geometry_masks(
    factory: CounterfactualInputFactory, dataset: CounterfactualGeometryDataset
) -> np.ndarray:
    return np.asarray(
        [factory.mask(str(row["geometry_id"])) for row in dataset.pairs],
        dtype=np.float64,
    )


def make_dataset(
    data: dict[str, np.ndarray],
    factory: CounterfactualInputFactory,
    split: str,
    partition: str,
    design: dict,
) -> tuple[CounterfactualGeometryDataset, list[dict[str, object]]]:
    pairs = build_pair_schedule(
        data,
        str(split),
        str(partition),
        "evaluation",
        int(design["repeats_per_source"]),
        int(design["assignment_seed"]),
        int(design["counterfactual_stride"]),
    )
    return CounterfactualGeometryDataset(factory, pairs), pairs


def numerical_checks(
    data: dict[str, np.ndarray],
    factory: CounterfactualInputFactory,
    normalization: dict[str, dict[str, np.ndarray | float]],
) -> list[dict[str, object]]:
    rows = []
    depth, height, width = data["field"].shape[1:]
    views, detector, _ = data["forward_matrix"].shape
    for index, identifier in enumerate(sorted(factory.catalog)):
        rng = np.random.default_rng(20261601 + index)
        volume = rng.normal(size=(1, depth, height, width))
        direction = rng.normal(size=volume.shape)
        projection = rng.normal(size=(1, depth, views, detector))
        observation = rng.normal(size=projection.shape)
        mask = factory.mask(identifier).astype(np.float64)
        values = normalization["".join(str(int(value)) for value in mask)]
        rows.append(
            {
                "geometry_id": identifier,
                "geometry_partition": str(
                    factory.catalog[identifier]["geometry_partition"]
                ),
                "mask_bits": identifier.removeprefix("g_"),
                "active_camera_count": int(mask.sum()),
                "audit_camera_active": int(mask[factory.audit_query_index] > 0.5),
                "adjoint_inner_product_relative_error": adjoint_inner_product_relative_error(
                    volume, projection, data["forward_matrix"], mask
                ),
                "finite_difference_gradient_relative_error": finite_difference_gradient_relative_error(
                    volume,
                    direction,
                    observation,
                    data["forward_matrix"],
                    mask,
                ),
                "spectral_constant": float(values["spectral_constant"]),
                "jacobi_floor": float(values["jacobi_floor"]),
                "preconditioned_spectral_constant": float(
                    values["preconditioned_spectral_constant"]
                ),
            }
        )
    return rows


def validation_screen(
    config: dict,
    data: dict[str, np.ndarray],
    dataset: CounterfactualGeometryDataset,
    base: np.ndarray,
    normalization: dict[str, dict[str, np.ndarray | float]],
    factory: CounterfactualInputFactory,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    protocol = config["numerical_protocol"]
    sources = source_indices(dataset)
    truth = np.asarray(data["field"][sources], dtype=np.float64)
    observations = np.asarray(data["observation"][sources], dtype=np.float64)
    masks = geometry_masks(factory, dataset)
    feasible = feasibility_project(base[:, 0], data["support"])
    rows: list[dict[str, object]] = []
    checkpoints = [int(value) for value in protocol["iteration_counts"]]
    for method in [str(value) for value in config["methods"]]:
        for beta in [float(value) for value in protocol["step_fractions"]]:
            trajectory = landweber_trajectory(
                feasible,
                observations,
                data["forward_matrix"],
                masks,
                data["support"],
                beta,
                checkpoints,
                normalization,
                method=method,
            )
            for iteration in checkpoints:
                prediction = trajectory[iteration]
                _, field = source_cluster_values(relative_l2(prediction, truth), sources)
                rows.append(
                    {
                        "selection_split": str(config["selection_split"]),
                        "method": method,
                        "step_fraction": beta,
                        "iterations": int(iteration),
                        "independent_field_count": int(len(field)),
                        "layout_rows": int(len(dataset)),
                        "source_field_mean_rel_l2": float(np.mean(field)),
                        "source_field_median_rel_l2": float(np.median(field)),
                    }
                )
    selected: dict[str, dict[str, object]] = {}
    for method in [str(value) for value in config["methods"]]:
        candidates = [row for row in rows if str(row["method"]) == method]
        winner = min(
            candidates,
            key=lambda row: (
                float(row["source_field_mean_rel_l2"]),
                int(row["iterations"]),
                float(row["step_fraction"]),
            ),
        )
        selected[method] = dict(winner)
    return rows, selected


def predictions_for_split(
    config: dict,
    data: dict[str, np.ndarray],
    dataset: CounterfactualGeometryDataset,
    base: np.ndarray,
    selected: dict[str, dict[str, object]],
    normalization: dict[str, dict[str, np.ndarray | float]],
    factory: CounterfactualInputFactory,
) -> dict[str, np.ndarray]:
    sources = source_indices(dataset)
    observations = np.asarray(data["observation"][sources], dtype=np.float64)
    masks = geometry_masks(factory, dataset)
    feasible = feasibility_project(base[:, 0], data["support"])
    output = {
        "locked_fno_raw": np.asarray(base[:, 0], dtype=np.float64),
        "feasible_fno": feasible,
    }
    for method, label in (
        ("standard", "landweber_standard"),
        ("jacobi", "landweber_jacobi"),
    ):
        choice = selected[method]
        iteration = int(choice["iterations"])
        output[label] = landweber_trajectory(
            feasible,
            observations,
            data["forward_matrix"],
            masks,
            data["support"],
            float(choice["step_fraction"]),
            [iteration],
            normalization,
            method=method,
        )[iteration]
    return output


def metric_rows(
    predictions: dict[str, np.ndarray],
    dataset: CounterfactualGeometryDataset,
    data: dict[str, np.ndarray],
    factory: CounterfactualInputFactory,
    labels: dict[str, str] | None = None,
    correction_reference: str | dict[str, str] = "feasible_fno",
) -> list[dict[str, object]]:
    sources = source_indices(dataset)
    truth = np.asarray(data["field"][sources], dtype=np.float64)
    observations = np.asarray(data["observation"][sources], dtype=np.float64)
    clean = np.asarray(data["clean_observation"][sources], dtype=np.float64)
    masks = geometry_masks(factory, dataset)
    method_labels = LABELS if labels is None else labels
    support = np.asarray(data["support"], dtype=np.float64)
    outside = support < 0.02
    rows: list[dict[str, object]] = []
    for method, prediction in predictions.items():
        reference_name = (
            correction_reference
            if isinstance(correction_reference, str)
            else correction_reference[method]
        )
        feasible = predictions[reference_name]
        field = relative_l2(prediction, truth)
        audit = audit_error(
            prediction,
            clean,
            data["forward_matrix"],
            int(data["audit_query_index"]),
        )
        used_noisy = direct_projection_error(
            prediction, observations, data["forward_matrix"], masks
        )
        used_clean = direct_projection_error(
            prediction, clean, data["forward_matrix"], masks
        )
        correction = np.linalg.norm(
            (prediction - feasible).reshape(len(prediction), -1), axis=1
        ) / np.maximum(
            np.linalg.norm(feasible.reshape(len(feasible), -1), axis=1), 1e-12
        )
        outside_norm = np.linalg.norm(
            (prediction * outside).reshape(len(prediction), -1), axis=1
        ) / np.maximum(
            np.linalg.norm(prediction.reshape(len(prediction), -1), axis=1), 1e-12
        )
        for index, pair in enumerate(dataset.pairs):
            source = int(pair["source_index"])
            rows.append(
                {
                    "method": method,
                    "method_label": method_labels[method],
                    "source_split": str(pair["source_split"]),
                    "pair_index": int(pair["pair_index"]),
                    "source_index": source,
                    "sample_seed": int(data["sample_seed"][source]),
                    "geometry_id": str(pair["geometry_id"]),
                    "geometry_partition": str(pair["geometry_partition"]),
                    "family_id": int(data["family_id"][source]),
                    "noise_level": float(data["noise_level"][source]),
                    "field_rel_l2": float(field[index]),
                    "audit_reprojection_rel_l2": float(audit[index]),
                    "used_noisy_reprojection_rel_l2": float(used_noisy[index]),
                    "used_clean_reprojection_rel_l2": float(used_clean[index]),
                    "correction_vs_feasible_relative_l2": float(correction[index]),
                    "outside_support_relative_l2": float(outside_norm[index]),
                    "negative_voxel_fraction": float(
                        np.mean(prediction[index] < -1e-8)
                    ),
                }
            )
    return rows


def summarize(
    rows: list[dict[str, object]],
    config: dict,
    labels: dict[str, str] | None = None,
    comparisons: list[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    method_labels = LABELS if labels is None else labels
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), str(row["source_split"]), int(row["source_index"]))
        ].append(row)
    collapsed: dict[tuple[str, str, int], dict[str, float]] = {}
    metric_names = [
        "field_rel_l2",
        "audit_reprojection_rel_l2",
        "used_noisy_reprojection_rel_l2",
        "used_clean_reprojection_rel_l2",
        "correction_vs_feasible_relative_l2",
        "outside_support_relative_l2",
        "negative_voxel_fraction",
    ]
    for key, subset in grouped.items():
        collapsed[key] = {
            metric: float(np.mean([float(row[metric]) for row in subset]))
            for metric in metric_names
        }
    splits = sorted({str(row["source_split"]) for row in rows})
    methods = list(method_labels)
    summary = []
    for split in splits:
        sources = sorted(
            {int(row["source_index"]) for row in rows if str(row["source_split"]) == split}
        )
        for method in methods:
            values = [collapsed[(method, split, source)] for source in sources]
            field = np.asarray([value["field_rel_l2"] for value in values])
            summary.append(
                {
                    "method": method,
                    "method_label": method_labels[method],
                    "source_split": split,
                    "independent_field_count": len(sources),
                    "layouts_per_field": len(
                        [
                            row
                            for row in rows
                            if str(row["source_split"]) == split
                            and str(row["method"]) == method
                        ]
                    )
                    // len(sources),
                    "mean_field_rel_l2": float(np.mean(field)),
                    "median_field_rel_l2": float(np.median(field)),
                    "p90_field_rel_l2": float(np.quantile(field, 0.9)),
                    **{
                        f"mean_{metric}": float(
                            np.mean([value[metric] for value in values])
                        )
                        for metric in metric_names[1:]
                    },
                }
            )

    if comparisons is None:
        comparisons = [
            ("feasible_fno", "locked_fno_raw"),
            ("landweber_standard", "feasible_fno"),
            ("landweber_jacobi", "feasible_fno"),
            ("landweber_standard", "landweber_jacobi"),
        ]
    pairwise = []
    gate = config["gate"]
    for split in splits:
        sources = sorted(
            {int(row["source_index"]) for row in rows if str(row["source_split"]) == split}
        )
        for comparison_index, (candidate, comparator) in enumerate(comparisons):
            field_gains = np.asarray(
                [
                    100.0
                    * (
                        collapsed[(comparator, split, source)]["field_rel_l2"]
                        - collapsed[(candidate, split, source)]["field_rel_l2"]
                    )
                    / max(
                        collapsed[(comparator, split, source)]["field_rel_l2"],
                        1e-12,
                    )
                    for source in sources
                ]
            )
            audit_gains = np.asarray(
                [
                    100.0
                    * (
                        collapsed[(comparator, split, source)][
                            "audit_reprojection_rel_l2"
                        ]
                        - collapsed[(candidate, split, source)][
                            "audit_reprojection_rel_l2"
                        ]
                    )
                    / max(
                        collapsed[(comparator, split, source)][
                            "audit_reprojection_rel_l2"
                        ],
                        1e-12,
                    )
                    for source in sources
                ]
            )
            seed = int(gate["bootstrap_seed"]) + 10 * splits.index(split) + comparison_index
            field_ci = bootstrap_interval(
                field_gains, seed, int(gate["bootstrap_replicates"])
            )
            audit_ci = bootstrap_interval(
                audit_gains, seed + 1000, int(gate["bootstrap_replicates"])
            )
            pairwise.append(
                {
                    "source_split": split,
                    "candidate": candidate,
                    "comparator": comparator,
                    "independent_field_count": len(sources),
                    "mean_field_gain_pct": float(np.mean(field_gains)),
                    "field_cluster_ci95_low_pct": field_ci[0],
                    "field_cluster_ci95_high_pct": field_ci[1],
                    "median_field_gain_pct": float(np.median(field_gains)),
                    "p10_field_gain_pct": float(np.quantile(field_gains, 0.1)),
                    "positive_field_fraction": float(np.mean(field_gains > 0.0)),
                    "harm_rate_gt_1pct": float(np.mean(field_gains < -1.0)),
                    "mean_audit_gain_pct": float(np.mean(audit_gains)),
                    "audit_cluster_ci95_low_pct": audit_ci[0],
                    "audit_cluster_ci95_high_pct": audit_ci[1],
                }
            )
    return summary, pairwise


def find_pair(
    pairwise: list[dict[str, object]], split: str, candidate: str, comparator: str
) -> dict[str, object]:
    return next(
        row
        for row in pairwise
        if str(row["source_split"]) == split
        and str(row["candidate"]) == candidate
        and str(row["comparator"]) == comparator
    )


def summarize_layouts(
    rows: list[dict[str, object]],
    candidate: str,
    comparator: str = "feasible_fno",
) -> list[dict[str, object]]:
    indexed = {
        (
            str(row["method"]),
            str(row["source_split"]),
            str(row["geometry_id"]),
            int(row["source_index"]),
        ): row
        for row in rows
    }
    output = []
    groups = sorted(
        {
            (str(row["source_split"]), str(row["geometry_id"]))
            for row in rows
            if str(row["method"]) == candidate
        }
    )
    for split, geometry in groups:
        sources = sorted(
            {
                int(row["source_index"])
                for row in rows
                if str(row["method"]) == candidate
                and str(row["source_split"]) == split
                and str(row["geometry_id"]) == geometry
            }
        )
        field_gains = []
        audit_gains = []
        for source in sources:
            selected = indexed[(candidate, split, geometry, source)]
            control = indexed[(comparator, split, geometry, source)]
            field_gains.append(
                100.0
                * (float(control["field_rel_l2"]) - float(selected["field_rel_l2"]))
                / max(float(control["field_rel_l2"]), 1e-12)
            )
            audit_gains.append(
                100.0
                * (
                    float(control["audit_reprojection_rel_l2"])
                    - float(selected["audit_reprojection_rel_l2"])
                )
                / max(float(control["audit_reprojection_rel_l2"]), 1e-12)
            )
        output.append(
            {
                "source_split": split,
                "geometry_id": geometry,
                "candidate": candidate,
                "comparator": comparator,
                "source_field_count": len(sources),
                "mean_field_gain_pct": float(np.mean(field_gains)),
                "minimum_field_gain_pct": float(np.min(field_gains)),
                "positive_field_fraction": float(np.mean(np.asarray(field_gains) > 0.0)),
                "harm_rate_gt_1pct": float(np.mean(np.asarray(field_gains) < -1.0)),
                "mean_audit_gain_pct": float(np.mean(audit_gains)),
            }
        )
    return output


def gate_result(
    config: dict,
    checks: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    selected: dict[str, dict[str, object]],
) -> tuple[dict[str, bool], str, str]:
    gate = config["gate"]
    preferred = min(
        selected,
        key=lambda method: float(selected[method]["source_field_mean_rel_l2"]),
    )
    candidate = f"landweber_{preferred}"
    val = find_pair(pairwise, "val", candidate, "feasible_fno")
    geometry = find_pair(pairwise, "test_iid", candidate, "feasible_fno")
    joint = find_pair(pairwise, "test_joint_ood", candidate, "feasible_fno")
    results = {
        "adjoint_identity": max(
            float(row["adjoint_inner_product_relative_error"]) for row in checks
        )
        <= float(gate["maximum_adjoint_relative_error"]),
        "finite_difference_gradient": max(
            float(row["finite_difference_gradient_relative_error"]) for row in checks
        )
        <= float(gate["maximum_gradient_relative_error"]),
        "stable_normalized_step": float(selected[preferred]["step_fraction"])
        <= float(gate["maximum_normalized_step_fraction"])
        < 2.0,
        "validation_mean_gain": float(val["mean_field_gain_pct"])
        >= float(gate["minimum_validation_gain_vs_feasible_fno_pct"]),
        "validation_cluster_ci": float(val["field_cluster_ci95_low_pct"])
        > float(gate["minimum_validation_field_cluster_ci95_low_pct"]),
        "validation_field_coverage": float(val["positive_field_fraction"])
        >= float(gate["minimum_validation_positive_field_fraction"]),
        "validation_tail_harm": float(val["harm_rate_gt_1pct"])
        <= float(gate["maximum_validation_harm_rate_gt_1pct"]),
        "validation_audit": float(val["mean_audit_gain_pct"])
        >= float(gate["minimum_validation_audit_gain_pct"]),
        "geometry_ood_field": float(geometry["mean_field_gain_pct"])
        >= float(gate["minimum_geometry_ood_gain_vs_feasible_fno_pct"]),
        "geometry_ood_audit": float(geometry["mean_audit_gain_pct"])
        >= float(gate["minimum_geometry_ood_audit_gain_pct"]),
        "joint_ood_no_material_harm": float(joint["mean_field_gain_pct"])
        >= float(gate["minimum_joint_ood_gain_vs_feasible_fno_pct"]),
    }
    passed = all(results.values())
    status = (
        "ADJOINT_DIRECTION_VALID_FIXED_LANDWEBER_REQUIRED_BASELINE"
        if passed
        else "ADJOINT_DIRECTION_GATE_FAIL_STOP_LEARNING"
    )
    return results, status, preferred


def plot_results(
    output_path: Path,
    config: dict,
    checks: list[dict[str, object]],
    screen: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    summary: list[dict[str, object]],
    selected: dict[str, dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.5), constrained_layout=True)
    maximum = [
        max(float(row["adjoint_inner_product_relative_error"]) for row in checks),
        max(float(row["finite_difference_gradient_relative_error"]) for row in checks),
    ]
    axes[0].bar(["adjoint", "gradient"], maximum, color=["#287271", "#d08c60"])
    axes[0].set_yscale("log")
    axes[0].axhline(
        float(config["gate"]["maximum_gradient_relative_error"]),
        color="#a44a3f",
        linestyle="--",
        label="gradient gate",
    )
    axes[0].set_title("Operator checks")
    axes[0].set_ylabel("maximum relative error")
    axes[0].legend(fontsize=7)

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(config["numerical_protocol"]["step_fractions"])))
    for color, beta in zip(colors, config["numerical_protocol"]["step_fractions"]):
        rows = sorted(
            [
                row
                for row in screen
                if str(row["method"]) == "standard"
                and float(row["step_fraction"]) == float(beta)
            ],
            key=lambda row: int(row["iterations"]),
        )
        axes[1].plot(
            [int(row["iterations"]) for row in rows],
            [float(row["source_field_mean_rel_l2"]) for row in rows],
            marker="o",
            linewidth=1.1,
            markersize=3,
            color=color,
            label=f"beta={beta:g}",
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Validation-only Landweber screen")
    axes[1].set_xlabel("iterations")
    axes[1].set_ylabel("source-field mean L2")
    axes[1].legend(fontsize=6, ncol=2)

    preferred = min(
        selected,
        key=lambda method: float(selected[method]["source_field_mean_rel_l2"]),
    )
    candidate = f"landweber_{preferred}"
    split_order = ["val", "test_iid", "test_noise_ood", "test_family_ood", "test_joint_ood"]
    selected_pairs = [find_pair(pairwise, split, candidate, "feasible_fno") for split in split_order]
    means = np.asarray([float(row["mean_field_gain_pct"]) for row in selected_pairs])
    lower = means - np.asarray([float(row["field_cluster_ci95_low_pct"]) for row in selected_pairs])
    upper = np.asarray([float(row["field_cluster_ci95_high_pct"]) for row in selected_pairs]) - means
    axes[2].errorbar(range(len(split_order)), means, yerr=[lower, upper], fmt="o", color="#287271", capsize=3)
    axes[2].axhline(0.0, color="#58666e", linewidth=1)
    axes[2].set_xticks(range(len(split_order)), [value.replace("test_", "") for value in split_order], rotation=28, ha="right")
    axes[2].set_title(f"{LABELS[candidate]} vs feasible FNO")
    axes[2].set_ylabel("field gain (%) with 95% CI")

    width = 0.19
    x = np.arange(len(split_order))
    for method_index, method in enumerate(LABELS):
        values = [
            next(
                float(row["mean_field_rel_l2"])
                for row in summary
                if str(row["source_split"]) == split and str(row["method"]) == method
            )
            for split in split_order
        ]
        axes[3].bar(
            x + (method_index - 1.5) * width,
            values,
            width,
            label=LABELS[method],
        )
    axes[3].set_xticks(x, [value.replace("test_", "") for value in split_order], rotation=28, ha="right")
    axes[3].set_title("Absolute field error stays visible")
    axes[3].set_ylabel("source-field mean relative L2")
    axes[3].legend(fontsize=6)
    fig.suptitle("v3k-C: zero-learning adjoint direction gate", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    dataset_config = read_json(ROOT / "configs" / str(config["dataset_config"]))
    private_path = ROOT / "results" / str(config["private_dataset_npz"])
    checkpoint_path = ROOT / "results" / str(config["base_checkpoint"])
    data = load_private_dataset(private_path)
    base_hash_before = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = choose_device(args.device or "cpu")
    design = config["pair_design"]
    factory = CounterfactualInputFactory(data, float(design["ridge_relative"]))
    all_masks = np.asarray(
        [factory.mask(identifier) for identifier in sorted(factory.catalog)],
        dtype=np.float64,
    )
    normalization = geometry_normalization(
        data["forward_matrix"],
        all_masks,
        float(config["numerical_protocol"]["diagonal_floor_relative"]),
    )
    checks = numerical_checks(data, factory, normalization)

    base_model = make_model(
        "fno",
        dataset_config["models"]["fno"],
        int(data["inputs"].shape[1]),
        residual=True,
    )
    base_model.load_state_dict(checkpoint, strict=True)
    selection_split = str(config["selection_split"])
    selection_dataset, selection_pairs = make_dataset(
        data,
        factory,
        selection_split,
        str(design["evaluation_geometry_partition_by_split"][selection_split]),
        design,
    )
    selection_base = precompute_base_predictions(
        base_model, selection_dataset, device, batch_size=16
    )
    screen, selected = validation_screen(
        config,
        data,
        selection_dataset,
        selection_base,
        normalization,
        factory,
    )

    output_dir = ROOT / "results" / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_commit = {
        "experiment": config["name"],
        "selection_split": selection_split,
        "selection_metric": config["numerical_protocol"]["selection_metric"],
        "selection_tie_break": config["numerical_protocol"]["selection_tie_break"],
        "grid": {
            "methods": config["methods"],
            "step_fractions": config["numerical_protocol"]["step_fractions"],
            "iteration_counts": config["numerical_protocol"]["iteration_counts"],
        },
        "selected": selected,
        "independent_selection_fields": len(np.unique(source_indices(selection_dataset))),
        "selection_layout_rows": len(selection_dataset),
        "selection_sample_seed_sha256": hashlib.sha256(
            np.asarray(
                data["sample_seed"][source_indices(selection_dataset)], dtype=np.int64
            ).tobytes()
        ).hexdigest(),
        "selection_geometry_sha256": hashlib.sha256(
            "\n".join(str(row["geometry_id"]) for row in selection_pairs).encode("utf-8")
        ).hexdigest(),
        "audit_camera_used_for_selection": False,
        "audit_or_reprojection_metrics_computed_by_selection_function": False,
        "test_field_or_metric_rows_present_at_commit": False,
    }
    (output_dir / PUBLIC_FILES[3]).write_text(
        json.dumps(selection_commit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    datasets = {selection_split: selection_dataset}
    bases = {selection_split: selection_base}
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
        bases[split] = precompute_base_predictions(base_model, dataset, device, batch_size=16)
        pair_manifest.extend({**row, "schedule_role": "post_selection_development_audit"} for row in pairs)
        schedule_audit[split] = schedule_balance(pairs)

    sample_rows: list[dict[str, object]] = []
    for split, dataset in datasets.items():
        predictions = predictions_for_split(
            config,
            data,
            dataset,
            bases[split],
            selected,
            normalization,
            factory,
        )
        sample_rows.extend(metric_rows(predictions, dataset, data, factory))
    summary, pairwise = summarize(sample_rows, config)
    gate_checks, status, preferred = gate_result(config, checks, pairwise, selected)
    preferred_label = f"landweber_{preferred}"
    layout_summary = summarize_layouts(sample_rows, preferred_label)
    support_pair = find_pair(pairwise, "val", "feasible_fno", "locked_fno_raw")
    primary_pair = find_pair(pairwise, "val", preferred_label, "feasible_fno")
    jacobi_pair = find_pair(
        pairwise, "val", "landweber_standard", "landweber_jacobi"
    )

    write_csv(output_dir / PUBLIC_FILES[0], pair_manifest)
    write_csv(output_dir / PUBLIC_FILES[1], checks)
    write_csv(output_dir / PUBLIC_FILES[2], screen)
    write_csv(output_dir / PUBLIC_FILES[4], sample_rows)
    write_csv(output_dir / PUBLIC_FILES[5], summary)
    write_csv(output_dir / PUBLIC_FILES[6], pairwise)
    write_csv(output_dir / PUBLIC_FILES[10], layout_summary)
    plot_results(
        output_dir / PUBLIC_FILES[9],
        config,
        checks,
        screen,
        pairwise,
        summary,
        selected,
    )

    dashboard = {
        "experiment": config["name"],
        "scientific_status": status,
        "zero_learning_gate_pass": all(gate_checks.values()),
        "preferred_numerical_method": preferred,
        "selected_hyperparameters": selected,
        "gate_checks": gate_checks,
        "gate_thresholds": config["gate"],
        "operator_checks": {
            "geometry_count": len(checks),
            "maximum_adjoint_inner_product_relative_error": max(
                float(row["adjoint_inner_product_relative_error"]) for row in checks
            ),
            "maximum_finite_difference_gradient_relative_error": max(
                float(row["finite_difference_gradient_relative_error"]) for row in checks
            ),
            "spectral_constant_min": min(
                float(row["spectral_constant"]) for row in checks
            ),
            "spectral_constant_max": max(
                float(row["spectral_constant"]) for row in checks
            ),
            "audit_camera_excluded_from_all_geometries": not any(
                int(row["audit_camera_active"]) for row in checks
            ),
        },
        "mechanism_diagnosis": {
            "free_feasibility_projection_gain_vs_raw_fno_pct": float(
                support_pair["mean_field_gain_pct"]
            ),
            "preferred_landweber_gain_vs_feasible_fno_pct": float(
                primary_pair["mean_field_gain_pct"]
            ),
            "preferred_landweber_validation_ci95": [
                float(primary_pair["field_cluster_ci95_low_pct"]),
                float(primary_pair["field_cluster_ci95_high_pct"]),
            ],
            "standard_gain_vs_jacobi_pct": float(jacobi_pair["mean_field_gain_pct"]),
            "known_support_projection_is_a_required_future_baseline": True,
            "fixed_landweber_is_a_required_future_baseline": all(
                gate_checks.values()
            ),
        },
        "schedule_audit": schedule_audit,
        "split_summary": summary,
        "pairwise_summary": pairwise,
        "layout_summary": layout_summary,
        "worst_layout": min(
            layout_summary, key=lambda row: float(row["mean_field_gain_pct"])
        ),
        "next_decision": {
            "if_pass": "authorize only a bounded learned conditional-scalar development prototype after adding closed-form line-search and global-step controls",
            "if_fail": "stop learned adjoint-residual work and use numerical refinement only",
            "learned_scalar_development_prototype_authorized": all(
                gate_checks.values()
            ),
            "learned_scalar_confirmatory_training_authorized": False,
            "learned_voxel_preconditioner_authorized": False,
            "per_view_set_aggregation_authorized": False,
            "superiority_training_authorized": False,
            "blind_final_opened": False,
        },
        "known_limits": {
            "development_splits_were_read_by_earlier_experiments": True,
            "fresh_blind_fields_and_layouts_required_for_confirmation": True,
            "support_is_a_hard_mask_derived_from_a_known_synthetic_taper": True,
            "real_bost_support_is_not_assumed_available": True,
            "step_is_geometry_normalized_beta_over_exact_mask_spectral_constant": True,
            "one_global_absolute_step_control_tested": False,
            "closed_form_line_search_control_tested": False,
            "frozen_v3i_coordinate_channel_naming_is_inherited": True,
            "coordinate_channel_contract_regenerated_and_retrained": False,
            "audit_noise_scale_has_weak_full_view_generator_coupling": True,
        },
        "claims_boundary": config["claims_boundary"],
    }
    (output_dir / PUBLIC_FILES[7]).write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": status,
        "dashboard": dashboard,
        "protocol": {
            "initialization": "locked FNO, followed by a separately reported nonnegative known-support projection",
            "candidate": "geometry-normalized projected Landweber from the feasible FNO initialization",
            "control": "Jacobi-preconditioned projected Landweber under the same validation grid",
            "selection": "validation source-field mean after collapsing four unseen layouts; Q_audit excluded",
            "statistical_unit": "source field after layout collapse; no model-seed pseudo-replication",
            "test_read_timing": "four development test domains evaluated only after v3k_c_selection_commit.json was written",
            "novelty_boundary": "support projection, adjoint gradients, Landweber, and Jacobi preconditioning are numerical baselines, not contributions",
            "step_contract": "beta is normalized by each mask operator's exact spectral constant; this is not one global absolute alpha",
            "selection_artifact_boundary": "the selection function computes field error only; Q_audit and reprojection values first appear after the commit",
        },
        "provenance": {
            "config_sha256": sha256(args.config),
            "selection_commit_sha256": sha256(output_dir / PUBLIC_FILES[3]),
            "private_dataset_sha256": sha256(private_path),
            "base_checkpoint_sha256_before": base_hash_before,
            "base_checkpoint_sha256_after": sha256(checkpoint_path),
            "base_checkpoint_drift": int(base_hash_before != sha256(checkpoint_path)),
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
    (output_dir / PUBLIC_FILES[8]).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_checksums(output_dir)
    print(
        json.dumps(
            {
                "status": status,
                "preferred_method": preferred,
                "selected": selected,
                "gate_checks": gate_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
