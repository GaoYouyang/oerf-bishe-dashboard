#!/usr/bin/env python3
"""Test reliability-aware physics-lift gates across the T16 condition domains."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .bost_physics import forward_volume
    from .data import generate_dataset, load_npz, split_indices
    from .models import make_model
    from .run_ablations import METRICS, SPLIT_ORDER, read_json, t_interval, write_csv
    from .train_eval import evaluate_methods, set_seed, train_model
except ImportError:
    from bost_physics import forward_volume
    from data import generate_dataset, load_npz, split_indices
    from models import make_model
    from run_ablations import METRICS, SPLIT_ORDER, read_json, t_interval, write_csv
    from train_eval import evaluate_methods, set_seed, train_model


ROOT = Path(__file__).resolve().parent
COLORS = {
    "residual_reference": "#16806a",
    "absolute_reference": "#b56a2b",
    "fixed_view_gate": "#8b6f47",
    "learned_metadata_gate": "#7257a3",
    "quality_channel_residual": "#2e6f9e",
    "learned_observable_gate": "#b44f5f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "reliability_gates.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force-data", action="store_true")
    return parser.parse_args()


def observed_lift_residual(data: dict[str, np.ndarray]) -> np.ndarray:
    operator = data["forward_matrix"]
    values = []
    for index in range(len(data["lift"])):
        projected = forward_volume(data["lift"][index], operator)
        observation = data["observation"][index]
        mask = data["view_mask"][index][None, :, None]
        numerator = np.linalg.norm(((projected - observation) * mask).reshape(-1))
        denominator = np.linalg.norm((observation * mask).reshape(-1)) + 1e-8
        values.append(float(numerator / denominator))
    return np.asarray(values, dtype=np.float32)


def append_observable_quality(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    raw = observed_lift_residual(data)
    indices = split_indices(data)
    train_values = np.log(raw[indices["train"]] + 1e-6)
    train_mean = float(train_values.mean())
    train_std = float(train_values.std(ddof=1))
    normalized = np.clip((np.log(raw + 1e-6) - train_mean) / max(train_std, 1e-6), -4.0, 4.0) / 4.0
    sample_count, _, depth, height, width = data["inputs"].shape
    quality = np.broadcast_to(
        normalized[:, None, None, None, None],
        (sample_count, 1, depth, height, width),
    ).astype(np.float32)
    augmented = dict(data)
    augmented["inputs"] = np.concatenate([data["inputs"], quality], axis=1)
    audit = {
        "definition": "Relative observed-view error between calibrated physics lift reprojection and noisy input observation.",
        "uses_field_truth": False,
        "uses_clean_observation": False,
        "normalization": "log residual, train-only mean/std, clipped to +/-4 sigma and divided by four",
        "train_log_mean": train_mean,
        "train_log_std": train_std,
        "raw_by_split": {
            split: {
                "mean": float(raw[split_values].mean()),
                "std": float(raw[split_values].std(ddof=1)) if len(split_values) > 1 else 0.0,
            }
            for split, split_values in indices.items()
        },
    }
    return augmented, audit


def collect_gate_values(model: torch.nn.Module, data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> np.ndarray:
    if not hasattr(model, "gate_values"):
        return np.full(len(indices), np.nan, dtype=np.float32)
    values = []
    model.eval()
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            x = torch.from_numpy(data["inputs"][selected]).to(device)
            alpha = model.gate_values(x)
            if alpha is None:
                values.extend([np.nan] * len(selected))
            else:
                values.extend(alpha.detach().cpu().numpy().reshape(-1).tolist())
    return np.asarray(values, dtype=np.float32)


def summarize_runs(run_rows: list[dict]) -> list[dict]:
    rows = []
    for experiment in sorted({str(row["experiment"]) for row in run_rows}):
        for split in SPLIT_ORDER:
            selected = [row for row in run_rows if row["experiment"] == experiment and row["split"] == split]
            summary = {
                "experiment": experiment,
                "split": split,
                "seed_count": len(selected),
                "parameters_mean": float(np.mean([row["parameters"] for row in selected])),
                "train_seconds_mean": float(np.mean([row["train_seconds"] for row in selected])),
            }
            for public_name, aggregate_name in METRICS.items():
                mean, seed_std, seed_ci = t_interval(float(row[aggregate_name]) for row in selected)
                summary[f"{public_name}_mean"] = mean
                summary[f"{public_name}_seed_std"] = seed_std
                summary[f"{public_name}_seed_ci95_t"] = seed_ci
            finite_alpha = [float(row["gate_alpha_mean"]) for row in selected if math.isfinite(float(row["gate_alpha_mean"]))]
            if finite_alpha:
                alpha_mean, alpha_std, alpha_ci = t_interval(finite_alpha)
            else:
                alpha_mean, alpha_std, alpha_ci = math.nan, math.nan, math.nan
            summary["gate_alpha_mean"] = alpha_mean
            summary["gate_alpha_seed_std"] = alpha_std
            summary["gate_alpha_seed_ci95_t"] = alpha_ci
            rows.append(summary)
    return rows


def paired_deltas(run_rows: list[dict], baselines: list[str]) -> list[dict]:
    lookup = {
        (str(row["experiment"]), int(row["seed"]), str(row["split"])): row
        for row in run_rows
    }
    experiments = sorted({str(row["experiment"]) for row in run_rows})
    seeds = sorted({int(row["seed"]) for row in run_rows})
    rows = []
    for baseline in baselines:
        for challenger in experiments:
            if challenger == baseline:
                continue
            for split in SPLIT_ORDER:
                row = {"challenger": challenger, "baseline": baseline, "split": split, "seed_count": len(seeds)}
                for public_name, aggregate_name in METRICS.items():
                    values = [
                        float(lookup[(challenger, seed, split)][aggregate_name])
                        - float(lookup[(baseline, seed, split)][aggregate_name])
                        for seed in seeds
                    ]
                    mean, seed_std, seed_ci = t_interval(values)
                    row[f"{public_name}_delta_mean"] = mean
                    row[f"{public_name}_delta_seed_std"] = seed_std
                    row[f"{public_name}_delta_ci95_t"] = seed_ci
                    row[f"{public_name}_better_seed_fraction"] = float(np.mean(np.asarray(values) < 0.0))
                rows.append(row)
    return rows


def oracle_regret(run_rows: list[dict]) -> list[dict]:
    lookup = {
        (str(row["experiment"]), int(row["seed"]), str(row["split"])): row
        for row in run_rows
    }
    experiments = sorted({str(row["experiment"]) for row in run_rows})
    seeds = sorted({int(row["seed"]) for row in run_rows})
    rows = []
    for experiment in experiments:
        for metric_name, aggregate_name in (
            ("field_rel_l2", "rel_l2_mean"),
            ("heldout_reprojection_rel_l2", "heldout_reprojection_rel_l2_mean"),
        ):
            values = []
            wins = 0
            for seed in seeds:
                for split in SPLIT_ORDER:
                    residual = float(lookup[("residual_reference", seed, split)][aggregate_name])
                    absolute = float(lookup[("absolute_reference", seed, split)][aggregate_name])
                    oracle = min(residual, absolute)
                    candidate = float(lookup[(experiment, seed, split)][aggregate_name])
                    values.append((candidate - oracle) / max(oracle, 1e-8))
                    wins += int(candidate <= oracle + 1e-10)
            mean, seed_cell_std, normal_ci = t_interval(values)
            rows.append(
                {
                    "experiment": experiment,
                    "metric": metric_name,
                    "cell_count": len(values),
                    "mean_relative_regret": mean,
                    "cell_std": seed_cell_std,
                    "normal_style_ci95": normal_ci,
                    "p95_relative_regret": float(np.percentile(values, 95)),
                    "max_relative_regret": float(np.max(values)),
                    "oracle_or_better_fraction": wins / len(values),
                }
            )
    return rows


def plot_field_heatmap(summary_rows: list[dict], labels: dict[str, str], output_dir: Path) -> None:
    experiments = list(labels)
    lookup = {(row["experiment"], row["split"]): row for row in summary_rows}
    values = np.asarray(
        [[lookup[(experiment, split)]["field_rel_l2_mean"] for split in SPLIT_ORDER] for experiment in experiments]
    )
    fig, axis = plt.subplots(figsize=(11.2, 6.0), constrained_layout=True)
    image = axis.imshow(values, cmap="YlGnBu", aspect="auto", vmin=float(values.min()), vmax=float(values.max()))
    axis.set_xticks(np.arange(len(SPLIT_ORDER)), [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER])
    axis.set_yticks(np.arange(len(experiments)), [labels[experiment] for experiment in experiments])
    axis.set_title("Reliability-gate field error across five domains")
    threshold = 0.62 * float(values.max())
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "#17212b",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Field relative L2")
    fig.savefig(output_dir / "t16_gate_field_heatmap.png", dpi=180)
    plt.close(fig)


def plot_gate_alpha(summary_rows: list[dict], labels: dict[str, str], output_dir: Path) -> None:
    gated = ["fixed_view_gate", "learned_metadata_gate", "learned_observable_gate"]
    lookup = {(row["experiment"], row["split"]): row for row in summary_rows}
    fig, axis = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    x = np.arange(len(SPLIT_ORDER))
    for experiment in gated:
        values = [lookup[(experiment, split)]["gate_alpha_mean"] for split in SPLIT_ORDER]
        errors = [lookup[(experiment, split)]["gate_alpha_seed_ci95_t"] for split in SPLIT_ORDER]
        axis.errorbar(
            x,
            values,
            yerr=errors,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=COLORS[experiment],
            label=labels[experiment],
        )
    axis.axhline(1.0, color="#31363b", linestyle="--", linewidth=1.0, label="Residual alpha=1")
    axis.axhline(0.0, color="#b56a2b", linestyle=":", linewidth=1.0, label="Absolute alpha=0")
    axis.set_xticks(x, [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER])
    axis.set_ylabel("Mean physics-lift weight alpha")
    axis.set_title("What reliability weight does each gate apply?")
    axis.grid(alpha=0.22)
    axis.legend(ncol=2)
    fig.savefig(output_dir / "t16_gate_alpha.png", dpi=180)
    plt.close(fig)


def plot_oracle_regret(regret_rows: list[dict], labels: dict[str, str], output_dir: Path) -> None:
    experiments = list(labels)
    lookup = {(row["experiment"], row["metric"]): row for row in regret_rows}
    x = np.arange(len(experiments))
    width = 0.34
    fig, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    for offset, (metric, title, color) in enumerate(
        (
            ("field_rel_l2", "Field", "#16806a"),
            ("heldout_reprojection_rel_l2", "Held-out", "#2e6f9e"),
        )
    ):
        values = [100.0 * lookup[(experiment, metric)]["mean_relative_regret"] for experiment in experiments]
        axis.bar(x + (offset - 0.5) * width, values, width=width, color=color, alpha=0.86, label=title)
    axis.axhline(0.0, color="#31363b", linewidth=1.0)
    axis.set_xticks(x, [labels[experiment] for experiment in experiments], rotation=20, ha="right")
    axis.set_ylabel("Mean regret to per-cell Residual/Absolute oracle (%)")
    axis.set_title("Does one reliability model replace condition-wise oracle selection?")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    fig.savefig(output_dir / "t16_gate_oracle_regret.png", dpi=180)
    plt.close(fig)


def plot_condition_deltas(delta_rows: list[dict], labels: dict[str, str], output_dir: Path) -> None:
    candidates = ["fixed_view_gate", "learned_metadata_gate", "quality_channel_residual", "learned_observable_gate"]
    lookup = {
        (row["challenger"], row["baseline"], row["split"]): row
        for row in delta_rows
    }
    x = np.arange(len(SPLIT_ORDER))
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), constrained_layout=True)
    for axis, baseline, title in (
        (axes[0], "residual_reference", "Field delta against Residual FNO"),
        (axes[1], "absolute_reference", "Field delta against Absolute-output FNO"),
    ):
        for candidate in candidates:
            values = [lookup[(candidate, baseline, split)]["field_rel_l2_delta_mean"] for split in SPLIT_ORDER]
            axis.plot(x, values, marker="o", linewidth=1.7, color=COLORS[candidate], label=labels[candidate])
        axis.axhline(0.0, color="#31363b", linestyle="--", linewidth=1.0)
        axis.set_ylabel("Candidate - baseline")
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(ncol=2, fontsize=8.5)
    axes[0].set_xticks(x, [""] * len(SPLIT_ORDER))
    axes[1].set_xticks(x, [split.replace("test_", "").replace("_", "\n") for split in SPLIT_ORDER])
    fig.savefig(output_dir / "t16_gate_condition_deltas.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    gate_config = read_json(args.config)
    dataset_config = read_json(args.config.parent / str(gate_config["dataset_config"]))
    if args.device is not None:
        dataset_config["training"]["device"] = args.device
    if args.epochs is not None:
        dataset_config["training"]["epochs"] = args.epochs

    results_root = ROOT / "results"
    output_dir = results_root / "reliability_gates"
    work_root = results_root / "gate_work"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    dataset_path = results_root / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, dataset_path, force=args.force_data)
    base_data = load_npz(dataset_path)
    quality_data, quality_audit = append_observable_quality(base_data)
    data_variants = {"base": base_data, "observable_quality": quality_data}
    seeds = [int(seed) for seed in gate_config["training_seeds"]]
    experiments = list(gate_config["experiments"])
    labels = {str(item["id"]): str(item["label"]) for item in experiments}

    run_rows = []
    experiment_metadata = []
    for experiment in experiments:
        experiment_id = str(experiment["id"])
        data = data_variants[str(experiment["input_variant"])]
        model_config = deepcopy(dataset_config["models"]["fno"])
        parameter_count = None
        for seed in seeds:
            run_config = deepcopy(dataset_config)
            run_config["seed"] = seed
            set_seed(seed)
            model = make_model(
                "fno",
                model_config,
                int(data["inputs"].shape[1]),
                residual=bool(experiment["residual"]),
                gate_config=experiment.get("gate"),
            )
            run_dir = work_root / experiment_id / str(seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            record = train_model(experiment_id, model, data, run_config, run_dir)
            parameter_count = int(record["parameters"])
            aggregate_rows, _, _ = evaluate_methods(data, {experiment_id: record}, run_config, run_dir)
            indices = split_indices(data)
            gate_by_split = {
                split: collect_gate_values(record["model"], data, selected, record["device"])
                for split, selected in indices.items()
            }
            for row in aggregate_rows:
                if row["method"] != experiment_id:
                    continue
                alpha = gate_by_split[str(row["split"])]
                finite = alpha[np.isfinite(alpha)]
                run_rows.append(
                    {
                        "experiment": experiment_id,
                        "label": labels[experiment_id],
                        "seed": seed,
                        "split": str(row["split"]),
                        "input_variant": str(experiment["input_variant"]),
                        "parameters": int(record["parameters"]),
                        "epochs_ran": int(record["epochs_ran"]),
                        "best_epoch": int(record["best_epoch"]),
                        "best_val_rel_l2": float(record["best_val_rel_l2"]),
                        "train_seconds": float(record["train_seconds"]),
                        "gate_alpha_mean": float(finite.mean()) if len(finite) else math.nan,
                        "gate_alpha_std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else math.nan,
                        **{aggregate_name: float(row[aggregate_name]) for aggregate_name in METRICS.values()},
                    }
                )
            print(
                f"{experiment_id} seed={seed}: params={record['parameters']:,}, "
                f"best val={record['best_val_rel_l2']:.4f}, epochs={record['epochs_ran']}"
            )
        experiment_metadata.append({**experiment, "parameters": parameter_count})

    summary_rows = summarize_runs(run_rows)
    delta_rows = paired_deltas(run_rows, ["residual_reference", "absolute_reference"])
    regret_rows = oracle_regret(run_rows)
    write_csv(output_dir / "gate_runs.csv", run_rows)
    write_csv(output_dir / "gate_summary.csv", summary_rows)
    write_csv(output_dir / "gate_paired_deltas.csv", delta_rows)
    write_csv(output_dir / "gate_oracle_regret.csv", regret_rows)
    plot_field_heatmap(summary_rows, labels, output_dir)
    plot_gate_alpha(summary_rows, labels, output_dir)
    plot_oracle_regret(regret_rows, labels, output_dir)
    plot_condition_deltas(delta_rows, labels, output_dir)

    report = {
        "status": "completed_reliability_gate_benchmark",
        "name": gate_config["name"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device_request": dataset_config["training"]["device"],
        },
        "dataset": {
            "config": dataset_config["name"],
            "samples": int(len(base_data["field"])),
            "shape": list(base_data["field"].shape[1:]),
            "fixed_across_training_seeds": True,
        },
        "observable_quality_channel": quality_audit,
        "training_seeds": seeds,
        "experiments": experiment_metadata,
        "summary": summary_rows,
        "paired_deltas": delta_rows,
        "oracle_regret": regret_rows,
        "interpretation_contract": {
            "oracle": "Per seed-domain minimum of Residual FNO and Absolute-output FNO; uses truth metrics for audit only.",
            "gate_inputs": "View fraction, declared quality metadata, and optional observed-view lift reprojection residual.",
            "fixed_gate": "Alpha equals one for 5/7-view training cells and 0.6 for three-view OOD.",
            "learned_gate_initialization": "Alpha starts at one and is jointly trained with the FNO correction backbone.",
        },
        "claims_boundary": [
            "No gate input uses field truth, clean observation, or held-out-view labels.",
            "The oracle is available only for retrospective synthetic audit and cannot select a model on real data.",
            "The learned gates see only 5/7-view training conditions, so three-view behavior is extrapolation.",
            "Three optimization seeds are a stability screen, not a population-level uncertainty estimate.",
            "The linear synthetic forward model is not the full OERF BOST geometry.",
            "This benchmark remains framewise 3D and does not train a temporal or four-dimensional operator.",
        ],
    }
    with (output_dir / "gate_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    iid = {(row["experiment"], row["split"]): row for row in summary_rows}
    print("T16 reliability-gate benchmark complete")
    for experiment in labels:
        row = iid[(experiment, "test_iid")]
        print(
            f"{experiment}: field={row['field_rel_l2_mean']:.4f}, "
            f"heldout={row['heldout_reprojection_rel_l2_mean']:.4f}, "
            f"alpha={row['gate_alpha_mean']:.3f}, params={row['parameters_mean']:,.0f}"
        )
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
