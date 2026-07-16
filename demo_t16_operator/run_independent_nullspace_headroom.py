#!/usr/bin/env python3
"""Measure oracle nullspace headroom above the independent support-fit model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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
    from .models import make_dual_branch_model
    from .run_ablations import SPLIT_ORDER
    from .run_dual_branch_query import closed_form_support_weight
    from .run_nullspace_identifiability_audit import (
        nullspace_decomposition,
        project_to_nullspace,
    )
    from .train_eval import _masked_projection_relative, _relative_norm, project_torch
except ImportError:
    from bost_physics import forward_volume
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from models import make_dual_branch_model
    from run_ablations import SPLIT_ORDER
    from run_dual_branch_query import closed_form_support_weight
    from run_nullspace_identifiability_audit import nullspace_decomposition, project_to_nullspace
    from train_eval import _masked_projection_relative, _relative_norm, project_torch


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "independent_dual_support.json"
DEFAULT_OUTPUT = ROOT / "results" / "independent_nullspace_headroom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(
    experiment_config: dict,
    dataset_config: dict,
    data: dict[str, np.ndarray],
    seed: int,
) -> torch.nn.Module:
    model = make_dual_branch_model(
        dataset_config["models"]["fno"],
        in_channels=int(data["inputs"].shape[1]),
        router_features=len(experiment_config["router_features"]),
        router_hidden=int(experiment_config["router_hidden"]),
        expert_sharing="independent",
    )
    work_dir = ROOT / "results" / str(experiment_config["work_dir"]) / str(seed)
    checkpoint = work_dir / "dual_branch.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing independent checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model


def collect_support_fit_predictions(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(BOSTDataset(data, indices), batch_size=batch_size)
    operator = torch.from_numpy(data["forward_matrix"])
    predictions = []
    weights = []
    disagreements = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"]
            residual, absolute = model.experts(x)
            projected_residual = project_torch(residual, operator)
            projected_absolute = project_torch(absolute, operator)
            weight = closed_form_support_weight(
                projected_residual,
                projected_absolute,
                batch["observation"],
                batch["view_mask"],
            )
            mixture = model.combine(residual, absolute, weight)
            difference = torch.linalg.vector_norm(
                (residual - absolute).flatten(start_dim=1), dim=1
            )
            scale = 0.5 * (
                torch.linalg.vector_norm(residual.flatten(start_dim=1), dim=1)
                + torch.linalg.vector_norm(absolute.flatten(start_dim=1), dim=1)
            )
            predictions.append(mixture[:, 0].cpu().numpy())
            weights.append(weight[:, 0, 0, 0, 0].cpu().numpy())
            disagreements.append((difference / scale.clamp_min(1e-8)).cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(weights),
        np.concatenate(disagreements),
    )


def audit_sample(
    seed: int,
    split: str,
    sample_index: int,
    base: np.ndarray,
    weight: float,
    disagreement: float,
    data: dict[str, np.ndarray],
    decomposition: dict[str, object],
) -> dict[str, object]:
    target = np.asarray(data["field"][sample_index], dtype=np.float64)
    mask = np.asarray(data["view_mask"][sample_index], dtype=np.float64)
    query_mask = 1.0 - mask
    clean = np.asarray(data["clean_observation"][sample_index], dtype=np.float64)
    operator = np.asarray(data["forward_matrix"], dtype=np.float64)
    error = target - base
    correction = project_to_nullspace(error, np.asarray(decomposition["null_basis"]))
    oracle = base + correction
    base_projection = forward_volume(base, operator)
    oracle_projection = forward_volume(oracle, operator)
    correction_projection = forward_volume(correction, operator)
    base_field = _relative_norm(base - target, target)
    oracle_field = _relative_norm(oracle - target, target)
    base_query = _masked_projection_relative(base_projection, clean, query_mask)
    oracle_query = _masked_projection_relative(oracle_projection, clean, query_mask)
    error_energy = float(np.sum(error**2)) + 1e-12
    correction_energy = float(np.sum(correction**2))
    return {
        "seed": seed,
        "split": split,
        "sample_index": sample_index,
        "sample_seed": int(data["sample_seed"][sample_index]),
        "family_id": int(data["family_id"][sample_index]),
        "view_count": int(data["view_count"][sample_index]),
        "noise_level": float(data["noise_level"][sample_index]),
        "support_fit_residual_weight": weight,
        "branch_disagreement": disagreement,
        "support_rank": int(decomposition["rank"]),
        "support_nullity": int(decomposition["nullity"]),
        "base_field_rel_l2": base_field,
        "oracle_null_field_rel_l2": oracle_field,
        "field_improvement_pct": 100.0 * (base_field - oracle_field) / (base_field + 1e-12),
        "error_energy_in_support_nullspace": correction_energy / error_energy,
        "support_projection_change_rel_to_clean": _relative_norm(
            correction_projection * mask[None, :, None],
            clean * mask[None, :, None],
        ),
        "base_query_clean_reprojection": base_query,
        "oracle_null_query_clean_reprojection": oracle_query,
        "query_reprojection_improvement_pct": 100.0
        * (base_query - oracle_query)
        / (base_query + 1e-12),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["seed"]), str(row["split"]))].append(row)
    output = []
    for (seed, split), subset in sorted(groups.items()):
        improvements = np.asarray([float(row["field_improvement_pct"]) for row in subset])
        output.append(
            {
                "seed": seed,
                "split": split,
                "sample_count": len(subset),
                "mean_base_field_rel_l2": float(
                    np.mean([float(row["base_field_rel_l2"]) for row in subset])
                ),
                "mean_oracle_null_field_rel_l2": float(
                    np.mean([float(row["oracle_null_field_rel_l2"]) for row in subset])
                ),
                "mean_field_improvement_pct": float(np.mean(improvements)),
                "p10_field_improvement_pct": float(np.percentile(improvements, 10)),
                "field_improved_fraction": float(np.mean(improvements > 0.0)),
                "mean_error_energy_in_support_nullspace": float(
                    np.mean([float(row["error_energy_in_support_nullspace"]) for row in subset])
                ),
                "mean_query_reprojection_improvement_pct": float(
                    np.mean([float(row["query_reprojection_improvement_pct"]) for row in subset])
                ),
                "max_support_projection_change": float(
                    np.max([float(row["support_projection_change_rel_to_clean"]) for row in subset])
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict[str, object]], path: Path) -> None:
    splits = SPLIT_ORDER
    seeds = sorted({int(row["seed"]) for row in summary})
    means = []
    errors = []
    for split in splits:
        values = [
            float(row["mean_field_improvement_pct"])
            for row in summary
            if row["split"] == split
        ]
        means.append(float(np.mean(values)))
        errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.bar(x, means, yerr=errors, capsize=4, color="#196e63")
    ax.set_xticks(x, [split.replace("test_", "").replace("_", "\n") for split in splits])
    ax.set_ylabel("oracle-null field improvement (%)")
    ax.set_title(
        f"Headroom above independent support-fit ({len(seeds)} optimization seeds)"
    )
    ax.grid(True, axis="y", alpha=0.24)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_checksums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        digest = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (output_dir / "independent_nullspace_checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def main() -> None:
    args = parse_args()
    experiment_config = read_json(args.config)
    dataset_config = read_json(args.config.parent / experiment_config["dataset_config"])
    dataset_path = ROOT / "results" / f"{dataset_config['name']}_dataset.npz"
    generate_dataset(dataset_config, dataset_path)
    data = load_npz(dataset_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decompositions: dict[tuple[int, ...], dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for seed in [int(value) for value in experiment_config["training_seeds"]]:
        model = load_model(experiment_config, dataset_config, data, seed)
        for split, indices in split_indices(data).items():
            if split not in SPLIT_ORDER:
                continue
            predictions, weights, disagreements = collect_support_fit_predictions(
                model,
                data,
                indices,
                int(dataset_config["training"]["batch_size"]),
            )
            for local_index, sample_index in enumerate(indices):
                mask = np.asarray(data["view_mask"][sample_index])
                key = tuple(int(value) for value in mask.tolist())
                if key not in decompositions:
                    decompositions[key] = nullspace_decomposition(
                        data["forward_matrix"], mask
                    )
                rows.append(
                    audit_sample(
                        seed,
                        split,
                        int(sample_index),
                        np.asarray(predictions[local_index], dtype=np.float64),
                        float(weights[local_index]),
                        float(disagreements[local_index]),
                        data,
                        decompositions[key],
                    )
                )

    summary = summarize(rows)
    write_csv(args.output_dir / "independent_nullspace_samples.csv", rows)
    write_csv(args.output_dir / "independent_nullspace_summary.csv", summary)
    plot_summary(summary, args.output_dir / "t16_independent_nullspace_headroom.png")
    split_means = {
        split: float(
            np.mean(
                [
                    float(row["mean_field_improvement_pct"])
                    for row in summary
                    if row["split"] == split
                ]
            )
        )
        for split in SPLIT_ORDER
    }
    report = {
        "status": "completed_independent_support_fit_oracle_headroom",
        "experiment": experiment_config["name"],
        "sample_seed_cells": len(rows),
        "optimization_seeds": experiment_config["training_seeds"],
        "split_mean_field_improvement_pct": split_means,
        "key_findings": {
            "mean_field_improvement_pct": float(
                np.mean([float(row["field_improvement_pct"]) for row in rows])
            ),
            "minimum_split_mean_field_improvement_pct": min(split_means.values()),
            "mean_error_energy_in_support_nullspace": float(
                np.mean([float(row["error_energy_in_support_nullspace"]) for row in rows])
            ),
            "maximum_support_projection_change": float(
                np.max([float(row["support_projection_change_rel_to_clean"]) for row in rows])
            ),
            "all_sample_seed_cells_improved": bool(
                all(float(row["field_improvement_pct"]) > 0.0 for row in rows)
            ),
        },
        "decision": "Train a bounded support-nullspace neural correction only if headroom survives every split and seed.",
        "claims_boundary": [
            "The correction uses synthetic field truth and is an oracle upper bound, not an inference method.",
            "Independent experts double operator capacity and still require matched-capacity controls.",
            "The projector uses the linear synthetic support operator, not OERF ray geometry.",
            "Headroom does not imply that query-supervised learning will recover the null component.",
        ],
    }
    report_path = args.output_dir / "independent_nullspace_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_checksums(
        args.output_dir,
        [
            "independent_nullspace_samples.csv",
            "independent_nullspace_summary.csv",
            "independent_nullspace_report.json",
        ],
    )
    print(json.dumps(report["key_findings"], indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
