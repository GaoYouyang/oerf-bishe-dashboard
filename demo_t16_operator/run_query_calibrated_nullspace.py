#!/usr/bin/env python3
"""Audit query-camera calibration of a learned support-nullspace correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .bost_physics import forward_volume
    from .data import BOSTDataset, generate_dataset, load_npz, split_indices
    from .models import count_parameters, make_model
    from .run_ablations import SPLIT_ORDER, t_interval
    from .run_support_nullspace_corrector import (
        TorchSupportNullProjector,
        corrected_prediction,
        corrector_input,
        load_base_model,
        read_json,
        support_fit_state,
    )
    from .train_eval import _gradient_relative, _masked_projection_relative, _relative_norm
except ImportError:
    from bost_physics import forward_volume
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from models import count_parameters, make_model
    from run_ablations import SPLIT_ORDER, t_interval
    from run_support_nullspace_corrector import (
        TorchSupportNullProjector,
        corrected_prediction,
        corrector_input,
        load_base_model,
        read_json,
        support_fit_state,
    )
    from train_eval import _gradient_relative, _masked_projection_relative, _relative_norm


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "query_calibrated_nullspace.json"
DEFAULT_OUTPUT = ROOT / "results" / "query_calibrated_nullspace"
CORRECTOR_WORK = ROOT / "results" / "support_nullspace_corrector_work"
METHODS = [
    "support_fit_base",
    "full_null_correction",
    "query_binary_one",
    "query_line_search_one",
    "query_line_search_all",
    "field_oracle_line_search",
]
LABELS = {
    "support_fit_base": "support fit",
    "full_null_correction": "full learned null",
    "query_binary_one": "one-query accept/reject",
    "query_line_search_one": "one-query line search",
    "query_line_search_all": "all-query line search",
    "field_oracle_line_search": "field-oracle line search",
}
METRICS = [
    "alpha",
    "rel_l2",
    "gradient_rel_l2",
    "support_reprojection_rel_l2",
    "heldout_reprojection_rel_l2",
    "selected_query_noisy_residual",
    "all_query_noisy_residual",
    "support_correction_leakage",
    "field_improvement_vs_base_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


def clipped_line_search_alpha(
    direction: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> float:
    """Solve min_alpha ||mask * (residual - alpha * direction)||_2."""
    active = np.asarray(mask, dtype=np.float64)[None, :, None]
    direction_active = np.asarray(direction, dtype=np.float64) * active
    residual_active = np.asarray(residual, dtype=np.float64) * active
    denominator = float(np.sum(direction_active**2))
    if denominator <= 1e-14:
        return 0.0
    alpha = float(np.sum(direction_active * residual_active) / denominator)
    return float(np.clip(alpha, alpha_min, alpha_max))


def informative_query_mask(
    direction_projection: np.ndarray,
    query_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Choose the withheld view most sensitive to the proposed correction."""
    available = np.flatnonzero(np.asarray(query_mask) > 0.5)
    if len(available) == 0:
        raise ValueError("query calibration requires at least one withheld view")
    energies = np.sum(np.asarray(direction_projection)[:, available, :] ** 2, axis=(0, 2))
    selected = int(available[int(np.argmax(energies))])
    mask = np.zeros_like(query_mask, dtype=np.float64)
    mask[selected] = 1.0
    return mask, selected


def masked_noisy_residual(
    prediction: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
) -> float:
    active = np.asarray(mask, dtype=np.float64)[None, :, None]
    numerator = float(np.sum(((prediction - observed) * active) ** 2))
    denominator = float(np.sum((observed * active) ** 2))
    return float(np.sqrt(numerator / max(denominator, 1e-14)))


def field_line_search_alpha(
    direction: np.ndarray,
    target_minus_base: np.ndarray,
    alpha_min: float,
    alpha_max: float,
) -> float:
    denominator = float(np.sum(direction**2))
    if denominator <= 1e-14:
        return 0.0
    alpha = float(np.sum(direction * target_minus_base) / denominator)
    return float(np.clip(alpha, alpha_min, alpha_max))


def load_null_corrector(
    config: dict,
    input_channels: int,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    model = make_model(
        "fno",
        config["model"],
        in_channels=input_channels,
        residual=False,
    ).to(device)
    checkpoint = CORRECTOR_WORK / str(seed) / "nullspace_correction.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing learned nullspace checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def prediction_metrics(
    prediction: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    clean: np.ndarray,
    observed: np.ndarray,
    support_mask: np.ndarray,
    query_mask: np.ndarray,
    selected_query_mask: np.ndarray,
    operator: np.ndarray,
) -> dict[str, float]:
    projected = forward_volume(prediction, operator)
    correction_projection = forward_volume(prediction - base, operator)
    base_error = _relative_norm(base - target, target)
    field_error = _relative_norm(prediction - target, target)
    return {
        "rel_l2": field_error,
        "gradient_rel_l2": _gradient_relative(prediction, target),
        "support_reprojection_rel_l2": _masked_projection_relative(
            projected, clean, support_mask
        ),
        "heldout_reprojection_rel_l2": _masked_projection_relative(
            projected, clean, query_mask
        ),
        "selected_query_noisy_residual": masked_noisy_residual(
            projected, observed, selected_query_mask
        ),
        "all_query_noisy_residual": masked_noisy_residual(
            projected, observed, query_mask
        ),
        "support_correction_leakage": _relative_norm(
            correction_projection * support_mask[None, :, None],
            clean * support_mask[None, :, None],
        ),
        "field_improvement_vs_base_pct": 100.0
        * (base_error - field_error)
        / (base_error + 1e-12),
    }


def evaluate_seed(
    seed: int,
    base_model: torch.nn.Module,
    corrector: torch.nn.Module,
    data: dict[str, np.ndarray],
    dataset_config: dict,
    corrector_config: dict,
    experiment_config: dict,
    device: torch.device,
) -> list[dict[str, object]]:
    operator_torch = torch.from_numpy(data["forward_matrix"]).to(device)
    operator_numpy = np.asarray(data["forward_matrix"], dtype=np.float64)
    projector = TorchSupportNullProjector(operator_numpy)
    rows: list[dict[str, object]] = []
    alpha_min = float(experiment_config["alpha_min"])
    alpha_max = float(experiment_config["alpha_max"])

    for split, indices in split_indices(data).items():
        if split not in SPLIT_ORDER:
            continue
        loader = DataLoader(
            BOSTDataset(data, indices),
            batch_size=int(dataset_config["training"]["batch_size"]),
        )
        local_offset = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                observation_torch = batch["observation"].to(device)
                support_mask_torch = batch["view_mask"].to(device)
                base, residual, absolute, support_weight = support_fit_state(
                    base_model,
                    x,
                    observation_torch,
                    support_mask_torch,
                    operator_torch,
                )
                inputs = corrector_input(x, base, residual, absolute)
                full_prediction, correction, _ = corrected_prediction(
                    "nullspace_correction",
                    corrector,
                    inputs,
                    base,
                    support_mask_torch,
                    projector,
                    float(corrector_config["correction_cap_ratio"]),
                )

                for batch_index in range(x.shape[0]):
                    sample_index = int(indices[local_offset + batch_index])
                    base_np = base[batch_index, 0].cpu().numpy().astype(np.float64)
                    full_np = full_prediction[batch_index, 0].cpu().numpy().astype(np.float64)
                    correction_np = correction[batch_index, 0].cpu().numpy().astype(np.float64)
                    target = np.asarray(data["field"][sample_index], dtype=np.float64)
                    clean = np.asarray(data["clean_observation"][sample_index], dtype=np.float64)
                    observed = np.asarray(data["observation"][sample_index], dtype=np.float64)
                    support_mask = np.asarray(data["view_mask"][sample_index], dtype=np.float64)
                    query_mask = 1.0 - support_mask
                    base_projection = forward_volume(base_np, operator_numpy)
                    correction_projection = forward_volume(correction_np, operator_numpy)
                    one_query_mask, query_index = informative_query_mask(
                        correction_projection, query_mask
                    )
                    observation_residual = observed - base_projection
                    alpha_one = clipped_line_search_alpha(
                        correction_projection,
                        observation_residual,
                        one_query_mask,
                        alpha_min,
                        alpha_max,
                    )
                    alpha_all = clipped_line_search_alpha(
                        correction_projection,
                        observation_residual,
                        query_mask,
                        alpha_min,
                        alpha_max,
                    )
                    alpha_field = field_line_search_alpha(
                        correction_np,
                        target - base_np,
                        alpha_min,
                        alpha_max,
                    )
                    base_one = masked_noisy_residual(base_projection, observed, one_query_mask)
                    full_one = masked_noisy_residual(
                        base_projection + correction_projection, observed, one_query_mask
                    )
                    alpha_binary = float(full_one < base_one)
                    alphas = {
                        "support_fit_base": 0.0,
                        "full_null_correction": 1.0,
                        "query_binary_one": alpha_binary,
                        "query_line_search_one": alpha_one,
                        "query_line_search_all": alpha_all,
                        "field_oracle_line_search": alpha_field,
                    }
                    assert np.allclose(full_np, base_np + correction_np, atol=1e-6)
                    for method, alpha in alphas.items():
                        prediction = base_np + alpha * correction_np
                        row: dict[str, object] = {
                            "seed": seed,
                            "split": split,
                            "sample_index": sample_index,
                            "sample_seed": int(data["sample_seed"][sample_index]),
                            "family_id": int(data["family_id"][sample_index]),
                            "view_count": int(data["view_count"][sample_index]),
                            "noise_level": float(data["noise_level"][sample_index]),
                            "support_fit_residual_weight": float(
                                support_weight[batch_index, 0, 0, 0, 0].cpu()
                            ),
                            "selected_query_index": query_index,
                            "available_query_views": int(np.sum(query_mask)),
                            "method": method,
                            "alpha": alpha,
                        }
                        row.update(
                            prediction_metrics(
                                prediction,
                                base_np,
                                target,
                                clean,
                                observed,
                                support_mask,
                                query_mask,
                                one_query_mask,
                                operator_numpy,
                            )
                        )
                        rows.append(row)
                local_offset += x.shape[0]
    return rows


def make_run_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seeds = sorted({int(row["seed"]) for row in sample_rows})
    for seed in seeds:
        for split in SPLIT_ORDER:
            for method in METHODS:
                subset = [
                    row
                    for row in sample_rows
                    if int(row["seed"]) == seed
                    and row["split"] == split
                    and row["method"] == method
                ]
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "method": method,
                        "sample_count": len(subset),
                        **{
                            f"{metric}_mean": float(
                                np.mean([float(row[metric]) for row in subset])
                            )
                            for metric in METRICS
                        },
                        "field_better_than_base_fraction": float(
                            np.mean(
                                [
                                    float(row["field_improvement_vs_base_pct"]) > 0.0
                                    for row in subset
                                ]
                            )
                        ),
                    }
                )
    return rows


def summarize_runs(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_names = [f"{metric}_mean" for metric in METRICS] + [
        "field_better_than_base_fraction"
    ]
    for method in METHODS:
        for split in SPLIT_ORDER:
            subset = [
                row
                for row in run_rows
                if row["method"] == method and row["split"] == split
            ]
            output: dict[str, object] = {
                "method": method,
                "split": split,
                "seed_count": len(subset),
            }
            for metric in metric_names:
                mean, standard_deviation, interval = t_interval(
                    float(row[metric]) for row in subset
                )
                output[metric] = mean
                output[f"{metric}_seed_std"] = standard_deviation
                output[f"{metric}_seed_ci95_t"] = interval
            rows.append(output)
    return rows


def aggregate_findings(sample_rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    findings: dict[str, dict[str, float]] = {}
    for method in METHODS:
        subset = [row for row in sample_rows if row["method"] == method]
        improvements = np.asarray(
            [float(row["field_improvement_vs_base_pct"]) for row in subset]
        )
        findings[method] = {
            "sample_seed_cells": len(subset),
            "mean_field_improvement_pct": float(np.mean(improvements)),
            "p10_field_improvement_pct": float(np.percentile(improvements, 10)),
            "field_better_than_base_fraction": float(np.mean(improvements > 0.0)),
            "mean_alpha": float(np.mean([float(row["alpha"]) for row in subset])),
            "zero_alpha_fraction": float(
                np.mean([float(row["alpha"]) <= 1e-8 for row in subset])
            ),
        }
    return findings


def summary_value(
    summary: list[dict[str, object]], method: str, split: str, metric: str
) -> float:
    return float(
        next(
            row[metric]
            for row in summary
            if row["method"] == method and row["split"] == split
        )
    )


def plot_field_heatmap(summary: list[dict[str, object]], path: Path) -> None:
    display = METHODS[:-1]
    matrix = np.asarray(
        [
            [summary_value(summary, method, split, "rel_l2_mean") for split in SPLIT_ORDER]
            for method in display
        ]
    )
    fig, ax = plt.subplots(figsize=(11.2, 5.1), constrained_layout=True)
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(
        np.arange(len(SPLIT_ORDER)),
        [value.replace("test_", "").replace("_", "\n") for value in SPLIT_ORDER],
    )
    ax.set_yticks(np.arange(len(display)), [LABELS[value] for value in display])
    ax.set_title("Query calibration preserves support consistency")
    threshold = 0.62 * float(matrix.max())
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "#263238",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, label="field relative L2")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_improvement(summary: list[dict[str, object]], path: Path) -> None:
    methods = [
        "full_null_correction",
        "query_binary_one",
        "query_line_search_one",
        "query_line_search_all",
    ]
    colors = ["#9c4a38", "#81651a", "#2b669f", "#176b5d"]
    x = np.arange(len(SPLIT_ORDER))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12.2, 5.4), constrained_layout=True)
    for index, (method, color) in enumerate(zip(methods, colors)):
        values = [
            summary_value(summary, method, split, "field_improvement_vs_base_pct_mean")
            for split in SPLIT_ORDER
        ]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            color=color,
            label=LABELS[method],
        )
    ax.axhline(0.0, color="#444444", linewidth=1)
    ax.set_xticks(
        x,
        [value.replace("test_", "").replace("_", "\n") for value in SPLIT_ORDER],
    )
    ax.set_ylabel("field improvement over support fit (%)")
    ax.set_title("Can a reserved query camera rescue the learned null direction?")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.24)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_alpha(sample_rows: list[dict[str, object]], path: Path) -> None:
    methods = ["query_line_search_one", "query_line_search_all", "field_oracle_line_search"]
    colors = ["#2b669f", "#176b5d", "#8a6412"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), constrained_layout=True)
    for ax, method, color in zip(axes, methods, colors):
        values = [
            [
                float(row["alpha"])
                for row in sample_rows
                if row["method"] == method and row["split"] == split
            ]
            for split in SPLIT_ORDER
        ]
        box = ax.boxplot(values, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_xticks(
            np.arange(1, len(SPLIT_ORDER) + 1),
            [value.replace("test_", "").replace("_", "\n") for value in SPLIT_ORDER],
        )
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel("calibrated alpha")
        ax.set_title(LABELS[method])
        ax.grid(True, axis="y", alpha=0.22)
    fig.suptitle("Only the field-oracle panel uses three-dimensional truth")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "query_calibrated_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def main() -> None:
    args = parse_args()
    experiment_config = read_json(args.config)
    base_path = args.config.parent / str(experiment_config["base_experiment_config"])
    corrector_path = args.config.parent / str(
        experiment_config["corrector_experiment_config"]
    )
    base_config = read_json(base_path)
    corrector_config = read_json(corrector_path)
    dataset_config = read_json(base_path.parent / str(base_config["dataset_config"]))
    device = torch.device(args.device or str(experiment_config["device"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT / "results" / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, dataset_path)
    data = load_npz(dataset_path)

    sample_rows: list[dict[str, object]] = []
    for seed in [int(value) for value in base_config["training_seeds"]]:
        base_model = load_base_model(base_config, dataset_config, data, seed, device)
        corrector = load_null_corrector(
            corrector_config,
            int(data["inputs"].shape[1]) + 2,
            seed,
            device,
        )
        seed_rows = evaluate_seed(
            seed,
            base_model,
            corrector,
            data,
            dataset_config,
            corrector_config,
            experiment_config,
            device,
        )
        sample_rows.extend(seed_rows)
        print(f"seed={seed}: evaluated {len(seed_rows)} method-sample rows", flush=True)

    run_rows = make_run_rows(sample_rows)
    summary = summarize_runs(run_rows)
    findings = aggregate_findings(sample_rows)
    write_csv(args.output_dir / "query_calibrated_samples.csv", sample_rows)
    write_csv(args.output_dir / "query_calibrated_runs.csv", run_rows)
    write_csv(args.output_dir / "query_calibrated_summary.csv", summary)
    plot_field_heatmap(summary, args.output_dir / "t16_query_calibrated_field_heatmap.png")
    plot_improvement(summary, args.output_dir / "t16_query_calibrated_improvement.png")
    plot_alpha(sample_rows, args.output_dir / "t16_query_calibrated_alpha.png")

    method_rows = [row for row in sample_rows if row["method"] == "query_line_search_all"]
    report = {
        "status": "completed_query_calibrated_support_nullspace_audit",
        "experiment": experiment_config["name"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "protocol": {
            "support_rule": "original declared sample views reconstruct the base field",
            "one_query_rule": "choose the withheld view with maximum predicted correction-projection energy",
            "amplitude_rule": "closed-form clipped least squares against noisy query observations",
            "alpha_interval": [
                float(experiment_config["alpha_min"]),
                float(experiment_config["alpha_max"]),
            ],
            "field_truth_in_deployed_methods": False,
            "field_oracle_is_retrospective_only": True,
        },
        "design": {
            "test_sample_seed_cells": int(len(sample_rows) / len(METHODS)),
            "method_sample_rows": len(sample_rows),
            "seed_count": len(base_config["training_seeds"]),
            "methods": METHODS,
            "corrector_parameters": count_parameters(
                load_null_corrector(
                    corrector_config,
                    int(data["inputs"].shape[1]) + 2,
                    int(base_config["training_seeds"][0]),
                    device,
                )
            ),
        },
        "method_findings": findings,
        "key_findings": {
            "one_query_line_search_mean_field_improvement_pct": findings[
                "query_line_search_one"
            ]["mean_field_improvement_pct"],
            "one_query_line_search_better_fraction": findings["query_line_search_one"][
                "field_better_than_base_fraction"
            ],
            "all_query_line_search_mean_field_improvement_pct": findings[
                "query_line_search_all"
            ]["mean_field_improvement_pct"],
            "all_query_line_search_better_fraction": findings["query_line_search_all"][
                "field_better_than_base_fraction"
            ],
            "field_oracle_directional_headroom_pct": findings[
                "field_oracle_line_search"
            ]["mean_field_improvement_pct"],
            "maximum_support_correction_leakage": float(
                np.max([float(row["support_correction_leakage"]) for row in method_rows])
            ),
        },
        "claims_boundary": [
            "The query-calibrated methods consume additional camera measurements at inference and are not zero-extra-view methods.",
            "The query cameras are synthetic canonical views with the same noise model, not OERF hardware measurements.",
            "The selected one-query angle is chosen from already available withheld synthetic views; physical camera placement remains separate.",
            "Closed-form amplitude calibration relies on the declared linear forward operator and a fixed learned correction direction.",
            "The field-oracle line search uses ground truth only as a retrospective directional upper bound.",
            "The support-nullspace guarantee applies to the current linear operator and numerical precision only.",
            "Three optimization seeds and 8x16x16 fields are a closure test, not a publication-scale claim.",
        ],
    }
    report_path = args.output_dir / "query_calibrated_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_checksums(
        args.output_dir,
        [
            "query_calibrated_samples.csv",
            "query_calibrated_runs.csv",
            "query_calibrated_summary.csv",
            "query_calibrated_report.json",
        ],
    )
    print(json.dumps(report["key_findings"], indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
