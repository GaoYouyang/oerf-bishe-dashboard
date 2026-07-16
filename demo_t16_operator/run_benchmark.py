#!/usr/bin/env python3
"""Run the T16 paired-data, residual FNO, and U-Net benchmark end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .data import generate_dataset, load_npz
    from .models import make_model
    from .train_eval import (
        evaluate_methods,
        plot_accuracy_cost,
        plot_reconstruction_panel,
        plot_split_metrics,
        plot_training,
        set_seed,
        train_model,
        write_report,
    )
except ImportError:
    from data import generate_dataset, load_npz
    from models import make_model
    from train_eval import (
        evaluate_methods,
        plot_accuracy_cost,
        plot_reconstruction_panel,
        plot_split_metrics,
        plot_training,
        set_seed,
        train_model,
        write_report,
    )


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.json")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        config = json.load(handle)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device is not None:
        config["training"]["device"] = args.device
    set_seed(int(config["seed"]))

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = results_dir / f"{config['name']}_dataset.npz"
    generate_dataset(config, dataset_path, force=args.force_data)
    data = load_npz(dataset_path)
    in_channels = int(data["inputs"].shape[1])

    trained = {}
    for name in ("unet", "fno"):
        model = make_model(name, config["models"][name], in_channels)
        trained[name] = train_model(name, model, data, config, results_dir)

    aggregate_rows, sample_rows, predictions = evaluate_methods(data, trained, config, results_dir)
    plot_training(trained, results_dir)
    plot_split_metrics(aggregate_rows, results_dir)
    plot_reconstruction_panel(data, predictions, results_dir)
    plot_accuracy_cost(aggregate_rows, results_dir)
    report = write_report(config, data, trained, aggregate_rows, sample_rows, results_dir)

    print("T16 operator benchmark complete")
    print(f"dataset samples: {report['dataset']['samples']}")
    for name, model in report["models"].items():
        print(
            f"{name}: params={model['parameters']:,}, best val L2={model['best_val_rel_l2']:.4f}, "
            f"epochs={model['epochs_ran']}, train={model['train_seconds']:.1f}s"
        )
    for row in aggregate_rows:
        if row["split"] == "test_iid":
            print(
                f"{row['method']}: IID L2={row['rel_l2_mean']:.4f}, "
                f"heldout={row['heldout_reprojection_rel_l2_mean']:.4f}"
            )
    print(f"results: {results_dir}")


if __name__ == "__main__":
    main()
