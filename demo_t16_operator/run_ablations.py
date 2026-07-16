#!/usr/bin/env python3
"""Run the multi-seed causal ablations for the T16 3D operator checkpoint."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
import torch

try:
    from .data import generate_dataset, load_npz
    from .models import make_model
    from .train_eval import evaluate_methods, set_seed, train_model
except ImportError:
    from data import generate_dataset, load_npz
    from models import make_model
    from train_eval import evaluate_methods, set_seed, train_model


ROOT = Path(__file__).resolve().parent
SPLIT_ORDER = ["test_iid", "test_view_ood", "test_noise_ood", "test_joint_ood", "test_family_ood"]
METRICS = {
    "field_rel_l2": "rel_l2_mean",
    "gradient_rel_l2": "gradient_rel_l2_mean",
    "observed_reprojection_rel_l2": "observed_reprojection_rel_l2_mean",
    "heldout_reprojection_rel_l2": "heldout_reprojection_rel_l2_mean",
    "mass_relative_error": "mass_relative_error_mean",
    "centroid_error": "centroid_error_mean",
}
COLORS = {
    "physics_lift": "#8b9494",
    "residual_fno": "#16806a",
    "absolute_fno": "#b56a2b",
    "fno_no_reprojection": "#7257a3",
    "matched_unet": "#2e6f9e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ablations.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force-data", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def t_interval(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, 0.0, 0.0
    standard_deviation = float(array.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=len(array) - 1))
    return mean, standard_deviation, critical * standard_deviation / np.sqrt(len(array))


def summarize_runs(run_rows: list[dict], physics_rows: dict[str, dict]) -> list[dict]:
    summary_rows = []
    experiment_ids = sorted({str(row["experiment"]) for row in run_rows})
    for experiment in ["physics_lift", *experiment_ids]:
        for split in SPLIT_ORDER:
            if experiment == "physics_lift":
                rows = [physics_rows[split]]
                parameters = 0.0
                train_seconds = 0.0
                seed_count = 1
            else:
                rows = [row for row in run_rows if row["experiment"] == experiment and row["split"] == split]
                parameters = float(np.mean([row["parameters"] for row in rows]))
                train_seconds = float(np.mean([row["train_seconds"] for row in rows]))
                seed_count = len(rows)
            summary = {
                "experiment": experiment,
                "split": split,
                "seed_count": seed_count,
                "parameters_mean": parameters,
                "train_seconds_mean": train_seconds,
            }
            for public_name, aggregate_name in METRICS.items():
                mean, seed_std, seed_ci = t_interval(float(row[aggregate_name]) for row in rows)
                summary[f"{public_name}_mean"] = mean
                summary[f"{public_name}_seed_std"] = seed_std
                summary[f"{public_name}_seed_ci95_t"] = seed_ci
            summary_rows.append(summary)
    return summary_rows


def paired_deltas(run_rows: list[dict], baseline: str = "residual_fno") -> list[dict]:
    lookup = {
        (str(row["experiment"]), int(row["seed"]), str(row["split"])): row
        for row in run_rows
    }
    challengers = sorted({str(row["experiment"]) for row in run_rows if row["experiment"] != baseline})
    seeds = sorted({int(row["seed"]) for row in run_rows if row["experiment"] == baseline})
    rows = []
    for challenger in challengers:
        for split in SPLIT_ORDER:
            paired = {
                "challenger": challenger,
                "baseline": baseline,
                "split": split,
                "seed_count": len(seeds),
            }
            for public_name, aggregate_name in METRICS.items():
                values = [
                    float(lookup[(challenger, seed, split)][aggregate_name])
                    - float(lookup[(baseline, seed, split)][aggregate_name])
                    for seed in seeds
                ]
                mean, seed_std, seed_ci = t_interval(values)
                paired[f"{public_name}_delta_mean"] = mean
                paired[f"{public_name}_delta_seed_std"] = seed_std
                paired[f"{public_name}_delta_ci95_t"] = seed_ci
                paired[f"{public_name}_better_seed_fraction"] = float(np.mean(np.asarray(values) < 0.0))
            rows.append(paired)
    return rows


def summary_lookup(summary_rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(str(row["experiment"]), str(row["split"])): row for row in summary_rows}


def plot_iid(summary_rows: list[dict], labels: dict[str, str], output_dir: Path) -> None:
    lookup = summary_lookup(summary_rows)
    experiments = ["physics_lift", "residual_fno", "absolute_fno", "fno_no_reprojection", "matched_unet"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "field_rel_l2", "IID field reconstruction"),
        (axes[1], "heldout_reprojection_rel_l2", "IID held-out reprojection"),
    ):
        values = [lookup[(experiment, "test_iid")][f"{metric}_mean"] for experiment in experiments]
        errors = [lookup[(experiment, "test_iid")][f"{metric}_seed_ci95_t"] for experiment in experiments]
        axis.bar(
            np.arange(len(experiments)),
            values,
            yerr=errors,
            capsize=4,
            color=[COLORS[experiment] for experiment in experiments],
        )
        axis.set_xticks(np.arange(len(experiments)), [labels[experiment] for experiment in experiments], rotation=22, ha="right")
        axis.set_ylabel("Relative L2, lower is better")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.23)
    fig.savefig(output_dir / "t16_ablation_iid.png", dpi=180)
    plt.close(fig)


def plot_field_heatmap(summary_rows: list[dict], labels: dict[str, str], output_dir: Path) -> None:
    lookup = summary_lookup(summary_rows)
    experiments = ["physics_lift", "residual_fno", "absolute_fno", "fno_no_reprojection", "matched_unet"]
    values = np.asarray(
        [[lookup[(experiment, split)]["field_rel_l2_mean"] for split in SPLIT_ORDER] for experiment in experiments]
    )
    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    image = axis.imshow(values, cmap="YlGnBu", aspect="auto", vmin=float(values.min()), vmax=float(values.max()))
    axis.set_xticks(
        np.arange(len(SPLIT_ORDER)),
        [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER],
    )
    axis.set_yticks(np.arange(len(experiments)), [labels[experiment] for experiment in experiments])
    axis.set_title("Field relative L2 across condition and family shifts")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            threshold = 0.58 * float(values.max())
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="white" if values[row_index, column_index] > threshold else "#17212b",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Relative L2")
    fig.savefig(output_dir / "t16_ablation_ood_heatmap.png", dpi=180)
    plt.close(fig)


def plot_causal_deltas(delta_rows: list[dict], labels: dict[str, str], output_dir: Path) -> None:
    lookup = {(row["challenger"], row["split"]): row for row in delta_rows}
    challengers = ["absolute_fno", "fno_no_reprojection", "matched_unet"]
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.0), constrained_layout=True)
    x = np.arange(len(SPLIT_ORDER))
    for axis, metric, title in (
        (axes[0], "field_rel_l2", "Field error delta against residual FNO"),
        (axes[1], "heldout_reprojection_rel_l2", "Held-out error delta against residual FNO"),
    ):
        for challenger in challengers:
            values = [lookup[(challenger, split)][f"{metric}_delta_mean"] for split in SPLIT_ORDER]
            errors = [lookup[(challenger, split)][f"{metric}_delta_ci95_t"] for split in SPLIT_ORDER]
            axis.errorbar(
                x,
                values,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=1.8,
                label=labels[challenger],
                color=COLORS[challenger],
            )
        axis.axhline(0.0, color="#31363b", linewidth=1.0, linestyle="--")
        axis.set_ylabel("Challenger - residual FNO")
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(ncol=3, fontsize=8.5)
    axes[1].set_xticks(x, [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER])
    axes[0].set_xticks(x, [""] * len(SPLIT_ORDER))
    fig.savefig(output_dir / "t16_ablation_causal_deltas.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ablation_config = read_json(args.config)
    dataset_config_path = args.config.parent / str(ablation_config["dataset_config"])
    dataset_config = read_json(dataset_config_path)
    if args.device is not None:
        dataset_config["training"]["device"] = args.device
    if args.epochs is not None:
        dataset_config["training"]["epochs"] = args.epochs

    results_root = ROOT / "results"
    output_dir = results_root / "ablations"
    work_root = results_root / "ablation_work"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    dataset_path = results_root / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, dataset_path, force=args.force_data)
    data = load_npz(dataset_path)
    in_channels = int(data["inputs"].shape[1])
    seeds = [int(seed) for seed in ablation_config["training_seeds"]]
    experiments = list(ablation_config["experiments"])
    labels = {"physics_lift": "Physics lift", **{str(item["id"]): str(item["label"]) for item in experiments}}

    run_rows: list[dict] = []
    physics_rows: dict[str, dict] = {}
    experiment_metadata = []
    for experiment in experiments:
        experiment_id = str(experiment["id"])
        backbone = str(experiment["backbone"])
        model_config = deepcopy(experiment.get("model", dataset_config["models"][backbone]))
        parameter_count = None
        for seed in seeds:
            run_config = deepcopy(dataset_config)
            run_config["seed"] = seed
            run_config["training"]["lambda_reprojection"] = float(experiment["lambda_reprojection"])
            set_seed(seed)
            model = make_model(
                backbone,
                model_config,
                in_channels,
                residual=bool(experiment["residual"]),
            )
            run_dir = work_root / experiment_id / str(seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            record = train_model(experiment_id, model, data, run_config, run_dir)
            parameter_count = int(record["parameters"])
            aggregate_rows, _, _ = evaluate_methods(
                data,
                {experiment_id: record},
                run_config,
                run_dir,
            )
            for row in aggregate_rows:
                if row["method"] == "physics_lift":
                    physics_rows.setdefault(str(row["split"]), row)
                    continue
                run_rows.append(
                    {
                        "experiment": experiment_id,
                        "label": str(experiment["label"]),
                        "seed": seed,
                        "split": str(row["split"]),
                        "parameters": int(record["parameters"]),
                        "epochs_ran": int(record["epochs_ran"]),
                        "best_epoch": int(record["best_epoch"]),
                        "best_val_rel_l2": float(record["best_val_rel_l2"]),
                        "train_seconds": float(record["train_seconds"]),
                        **{aggregate_name: float(row[aggregate_name]) for aggregate_name in METRICS.values()},
                    }
                )
            print(
                f"{experiment_id} seed={seed}: params={record['parameters']:,}, "
                f"best val={record['best_val_rel_l2']:.4f}, epochs={record['epochs_ran']}"
            )
        experiment_metadata.append(
            {
                **experiment,
                "model_config": model_config,
                "parameters": parameter_count,
            }
        )

    summary_rows = summarize_runs(run_rows, physics_rows)
    delta_rows = paired_deltas(run_rows)
    write_csv(output_dir / "ablation_runs.csv", run_rows)
    write_csv(output_dir / "ablation_summary.csv", summary_rows)
    write_csv(output_dir / "ablation_paired_deltas.csv", delta_rows)
    plot_iid(summary_rows, labels, output_dir)
    plot_field_heatmap(summary_rows, labels, output_dir)
    plot_causal_deltas(delta_rows, labels, output_dir)

    report = {
        "status": "completed_multi_seed_ablation",
        "name": ablation_config["name"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device_request": dataset_config["training"]["device"],
        },
        "dataset": {
            "config": dataset_config["name"],
            "samples": int(len(data["field"])),
            "shape": list(data["field"].shape[1:]),
            "fixed_across_training_seeds": True,
        },
        "training_seeds": seeds,
        "experiments": experiment_metadata,
        "summary": summary_rows,
        "paired_deltas_against_residual_fno": delta_rows,
        "interpretation_contract": {
            "negative_delta": "The challenger has lower error than residual FNO.",
            "uncertainty": "Student-t 95% intervals across three optimization seeds; low power and not a dataset bootstrap.",
            "absolute_fno": "Ablates the residual skip only. It still receives the calibrated physics lift as channel zero.",
            "matched_unet": "Closer parameter budget, but not an architecture-identical control.",
        },
        "claims_boundary": [
            "The dataset is fixed; seed intervals measure optimization variation, not phantom-population uncertainty.",
            "Three seeds provide a stability check but weak inferential power.",
            "No raw-projection encoder is implemented, so this does not compare against a direct measurement-to-volume operator.",
            "The synthetic linear stack remains a controlled surrogate rather than the full OERF BOST geometry.",
            "No temporal or four-dimensional operator is trained in this ablation.",
        ],
    }
    with (output_dir / "ablation_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    iid_lookup = summary_lookup(summary_rows)
    print("T16 multi-seed ablation complete")
    for experiment in ["physics_lift", "residual_fno", "absolute_fno", "fno_no_reprojection", "matched_unet"]:
        row = iid_lookup[(experiment, "test_iid")]
        print(
            f"{experiment}: field={row['field_rel_l2_mean']:.4f}, "
            f"heldout={row['heldout_reprojection_rel_l2_mean']:.4f}, "
            f"params={row['parameters_mean']:,.0f}"
        )
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
