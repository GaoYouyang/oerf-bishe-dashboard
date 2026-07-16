#!/usr/bin/env python3
"""Compare continued FNO and a zero-init adapter on disjoint K=6 dev2 fields."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .data import BOSTDataset, generate_dataset, load_npz, split_indices
    from .direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from .models import make_model
    from .own_algorithm_data import append_ray_view_channels
    from .run_direct_operator_pilot import (
        build_domain_rows,
        collapse_model_seeds,
        domain_equal_weights,
        prediction_metrics,
        relative_field_error,
        stratified_bootstrap_interval,
        summarize_clusters,
        test_variant_indices,
        tune_classical_baselines,
        weighted_quantile,
        write_checksums,
    )
    from .run_v3c_protocol_gate import build_adapter
    from .train_eval import choose_device, collect_predictions, set_seed, train_model
except ImportError:
    from data import BOSTDataset, generate_dataset, load_npz, split_indices
    from direct_operator_data import prepare_direct_operator_data, replace_lift_with_ridge
    from models import make_model
    from own_algorithm_data import append_ray_view_channels
    from run_direct_operator_pilot import (
        build_domain_rows,
        collapse_model_seeds,
        domain_equal_weights,
        prediction_metrics,
        relative_field_error,
        stratified_bootstrap_interval,
        summarize_clusters,
        test_variant_indices,
        tune_classical_baselines,
        weighted_quantile,
        write_checksums,
    )
    from run_v3c_protocol_gate import build_adapter
    from train_eval import choose_device, collect_predictions, set_seed, train_model


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_ROOT / "v3c_k6_dev2_pilot.json"
METHODS = ["base_fno", "continued_fno", "zero_init_adapter"]
LABELS = {
    "base_fno": "base ridge residual FNO",
    "continued_fno": "same-checkpoint continued FNO",
    "zero_init_adapter": "frozen-FNO zero-init ray-set adapter",
}
COMPARISONS = {
    "adapter_vs_continued_fno": ("zero_init_adapter", "continued_fno"),
    "adapter_vs_base_fno": ("zero_init_adapter", "base_fno"),
    "continued_fno_vs_base_fno": ("continued_fno", "base_fno"),
}
CHECKSUM_FILES = [
    "v3c_k6_baseline_tuning.csv",
    "v3c_k6_training.csv",
    "v3c_k6_initialization.csv",
    "v3c_k6_samples.csv",
    "v3c_k6_clusters.csv",
    "v3c_k6_summary.csv",
    "v3c_k6_domains.csv",
    "v3c_k6_pairwise.csv",
    "v3c_k6_seed_summary.csv",
    "v3c_k6_pairwise_summary.csv",
    "v3c_k6_dashboard.json",
    "v3c_k6_report.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--base-epochs", type=int)
    parser.add_argument("--adapt-epochs", type=int)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_output_checksums(output_dir: Path) -> None:
    write_checksums(output_dir, CHECKSUM_FILES)
    (output_dir / "v3c_k6_checksums.sha256").write_text(
        (output_dir / "direct_operator_checksums.sha256").read_text(encoding="ascii"),
        encoding="ascii",
    )
    (output_dir / "direct_operator_checksums.sha256").unlink()


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def maximum_state_drift(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
) -> float:
    values = []
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            values.append(float(torch.max(torch.abs(value.detach().cpu() - reference[name]))))
    return max(values, default=0.0)


def total_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def training_config(
    dataset_config: dict,
    seed: int,
    epochs: int,
    device: str,
    learning_rate: float | None = None,
    fixed_epochs: bool = False,
) -> dict:
    config = copy.deepcopy(dataset_config)
    config["seed"] = int(seed)
    config["training"]["device"] = str(device)
    config["training"]["epochs"] = int(epochs)
    if learning_rate is not None:
        config["training"]["learning_rate"] = float(learning_rate)
    if fixed_epochs:
        config["training"]["early_stop_patience"] = int(epochs)
    return config


def evaluate_seed(
    outer_seed: int,
    data: dict[str, np.ndarray],
    records: dict[str, dict],
    dataset_config: dict,
    ridge_relative: float,
    audit_query_index: int,
) -> list[dict[str, object]]:
    indices = test_variant_indices(data)
    mapped: dict[str, dict[int, np.ndarray]] = {}
    inference_ms: dict[str, float] = {}
    for method, record in records.items():
        values, elapsed = collect_predictions(
            record["model"],
            BOSTDataset(data, indices),
            record["device"],
            int(dataset_config["training"]["batch_size"]),
        )
        mapped[method] = {
            int(index): values[local_index, 0]
            for local_index, index in enumerate(indices)
        }
        inference_ms[method] = float(elapsed)

    split_names = [str(value) for value in data["split_names"].tolist()]
    audit_mask = np.zeros(len(data["angles"]), dtype=np.float32)
    audit_mask[int(audit_query_index)] = 1.0
    rows: list[dict[str, object]] = []
    for index in indices:
        ridge_prediction = data["lift"][index]
        ridge_error = relative_field_error(ridge_prediction, data["field"][index])
        for method in METHODS:
            row = {
                "model_seed": int(outer_seed),
                "variant_index": int(index),
                "source_index": int(data["source_index"][index]),
                "sample_seed": int(data["sample_seed"][index]),
                "source_split": split_names[int(data["split_id"][index])],
                "family_id": int(data["family_id"][index]),
                "noise_level": float(data["noise_level"][index]),
                "total_budget": int(data["total_budget"][index]),
                "method": method,
                "classical_champion": "ridge",
                "ridge_relative": float(ridge_relative),
                "inference_ms_per_sample": inference_ms[method],
            }
            row.update(
                prediction_metrics(
                    mapped[method][int(index)],
                    data["field"][index],
                    data["clean_observation"][index],
                    data["view_mask"][index],
                    audit_mask,
                    data["forward_matrix"],
                    ridge_error,
                )
            )
            rows.append(row)
    return rows


def build_pairwise_rows(
    cluster_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in cluster_rows:
        grouped[(int(row["source_index"]), str(row["source_split"]))][str(row["method"])] = row
    output = []
    for (source_index, split), methods in sorted(grouped.items()):
        for comparison, (candidate, comparator) in COMPARISONS.items():
            candidate_row = methods[candidate]
            comparator_row = methods[comparator]
            candidate_field = float(candidate_row["field_rel_l2"])
            comparator_field = float(comparator_row["field_rel_l2"])
            candidate_audit = float(candidate_row["audit_reprojection_rel_l2"])
            comparator_audit = float(comparator_row["audit_reprojection_rel_l2"])
            output.append(
                {
                    "comparison": comparison,
                    "candidate": candidate,
                    "comparator": comparator,
                    "source_index": source_index,
                    "sample_seed": int(candidate_row["sample_seed"]),
                    "source_split": split,
                    "family_id": int(candidate_row["family_id"]),
                    "noise_level": float(candidate_row["noise_level"]),
                    "model_seed_count": int(candidate_row["model_seed_count"]),
                    "field_superiority_pct": 100.0
                    * (comparator_field - candidate_field)
                    / (comparator_field + 1e-12),
                    "audit_superiority_pct": 100.0
                    * (comparator_audit - candidate_audit)
                    / (comparator_audit + 1e-12),
                }
            )
    return output


def build_seed_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (int(row["model_seed"]), int(row["source_index"]), str(row["method"])): row
        for row in sample_rows
    }
    output = []
    seeds = sorted({int(row["model_seed"]) for row in sample_rows})
    sources = sorted({int(row["source_index"]) for row in sample_rows})
    for seed in seeds:
        for comparison, (candidate, comparator) in COMPARISONS.items():
            rows = []
            for source in sources:
                candidate_row = indexed[(seed, source, candidate)]
                comparator_row = indexed[(seed, source, comparator)]
                comparator_error = float(comparator_row["field_rel_l2"])
                rows.append(
                    {
                        "source_split": str(candidate_row["source_split"]),
                        "field_superiority_pct": 100.0
                        * (comparator_error - float(candidate_row["field_rel_l2"]))
                        / (comparator_error + 1e-12),
                    }
                )
            weights = domain_equal_weights(rows)
            output.append(
                {
                    "model_seed": seed,
                    "comparison": comparison,
                    "independent_field_count": len(rows),
                    "domain_equal_mean_field_superiority_pct": float(
                        np.sum(
                            weights
                            * np.asarray(
                                [float(row["field_superiority_pct"]) for row in rows]
                            )
                        )
                    ),
                }
            )
    return output


def summarize_pairwise(
    pairwise_rows: list[dict[str, object]],
    seed_rows: list[dict[str, object]],
    config: dict,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    gate = config["development_gate"]
    output = []
    for comparison in COMPARISONS:
        subset = [row for row in pairwise_rows if row["comparison"] == comparison]
        weights = domain_equal_weights(subset)
        field = np.asarray([float(row["field_superiority_pct"]) for row in subset])
        audit = np.asarray([float(row["audit_superiority_pct"]) for row in subset])
        ci_low, ci_high = stratified_bootstrap_interval(
            subset,
            "field_superiority_pct",
            rng,
            int(config["bootstrap_replicates"]),
        )
        domain_means = {
            domain: float(
                np.mean(
                    [
                        float(row["field_superiority_pct"])
                        for row in subset
                        if str(row["source_split"]) == domain
                    ]
                )
            )
            for domain in sorted({str(row["source_split"]) for row in subset})
        }
        comparison_seed_rows = [
            row for row in seed_rows if row["comparison"] == comparison
        ]
        every_seed_positive = all(
            float(row["domain_equal_mean_field_superiority_pct"]) > 0.0
            for row in comparison_seed_rows
        )
        mean_field = float(np.sum(weights * field))
        mean_audit = float(np.sum(weights * audit))
        p10 = weighted_quantile(field, weights, 0.1)
        harm = float(np.sum(weights * (field < -1.0)))
        development_pass = (
            comparison == "adapter_vs_continued_fno"
            and ci_low > float(gate["minimum_ci95_low_pct"])
            and p10 >= float(gate["minimum_p10_superiority_pct"])
            and harm <= float(gate["maximum_harm_rate_gt_1pct"])
            and mean_audit >= float(gate["minimum_clean_audit_superiority_pct"])
            and (
                every_seed_positive
                or not bool(gate["require_every_seed_mean_positive"])
            )
        )
        candidate, comparator = COMPARISONS[comparison]
        output.append(
            {
                "comparison": comparison,
                "candidate": candidate,
                "comparator": comparator,
                "independent_field_count": len(subset),
                "model_seed_count": int(subset[0]["model_seed_count"]),
                "mean_field_superiority_pct": mean_field,
                "field_superiority_ci95_low": ci_low,
                "field_superiority_ci95_high": ci_high,
                "p10_field_superiority_pct": p10,
                "harm_rate_gt_1pct": harm,
                "mean_clean_audit_superiority_pct": mean_audit,
                "every_seed_mean_positive": every_seed_positive,
                "every_domain_mean_nonnegative": all(value >= 0.0 for value in domain_means.values()),
                "domain_means": domain_means,
                "development_gate_pass": development_pass,
            }
        )
    return output


def build_cost_summary(
    training_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
) -> dict[str, object]:
    train_seconds = {
        method: float(
            np.mean(
                [
                    float(row["train_seconds"])
                    for row in training_rows
                    if str(row["method"]) == method
                ]
            )
        )
        for method in METHODS
    }
    inference_by_seed: dict[tuple[int, str], float] = {}
    for row in sample_rows:
        inference_by_seed[(int(row["model_seed"]), str(row["method"]))] = float(
            row["inference_ms_per_sample"]
        )
    inference_ms = {
        method: float(
            np.mean(
                [
                    value
                    for (_, current_method), value in inference_by_seed.items()
                    if current_method == method
                ]
            )
        )
        for method in METHODS
    }
    return {
        "mean_training_seconds": train_seconds,
        "mean_inference_ms_per_sample": inference_ms,
        "adapter_to_continued_training_time_ratio": train_seconds["zero_init_adapter"]
        / train_seconds["continued_fno"],
        "adapter_to_continued_inference_time_ratio": inference_ms["zero_init_adapter"]
        / inference_ms["continued_fno"],
    }


def build_decision(pairwise_summary: list[dict[str, object]]) -> dict[str, object]:
    indexed = {str(row["comparison"]): row for row in pairwise_summary}
    adapter_primary = indexed["adapter_vs_continued_fno"]
    continuation = indexed["continued_fno_vs_base_fno"]
    return {
        "stop_current_frozen_per_view_adapter": float(
            adapter_primary["field_superiority_ci95_high"]
        )
        < 0.0,
        "base_fno_training_horizon_not_saturated": float(
            continuation["field_superiority_ci95_low"]
        )
        > 0.0,
        "blind_final_remains_closed": True,
    }


def plot_results(
    summary: list[dict[str, object]],
    pairwise_summary: list[dict[str, object]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    method_rows = {str(row["method"]): row for row in summary}
    colors = ["#315f93", "#886719", "#126e64"]
    values = [float(method_rows[method]["mean_field_rel_l2"]) for method in METHODS]
    axes[0].bar(range(len(METHODS)), values, color=colors)
    axes[0].set_xticks(range(len(METHODS)), ["base FNO", "continued FNO", "zero-init adapter"])
    axes[0].set_ylabel("domain-equal field relative L2")
    axes[0].set_title("K=6 dev2 absolute error (lower is better)")
    axes[0].grid(axis="y", alpha=0.25)

    pair_values = [float(row["mean_field_superiority_pct"]) for row in pairwise_summary]
    lower = [
        value - float(row["field_superiority_ci95_low"])
        for value, row in zip(pair_values, pairwise_summary)
    ]
    upper = [
        float(row["field_superiority_ci95_high"]) - value
        for value, row in zip(pair_values, pairwise_summary)
    ]
    axes[1].errorbar(
        range(len(pairwise_summary)),
        pair_values,
        yerr=np.asarray([lower, upper]),
        fmt="o",
        color="#126e64",
        capsize=5,
    )
    axes[1].axhline(0.0, color="#8f4b3b", linewidth=1)
    comparison_labels = {
        "adapter_vs_continued_fno": "adapter vs\ncontinued FNO",
        "adapter_vs_base_fno": "adapter vs\nbase FNO",
        "continued_fno_vs_base_fno": "continued vs\nbase FNO",
    }
    axes[1].set_xticks(
        range(len(pairwise_summary)),
        [comparison_labels[str(row["comparison"])] for row in pairwise_summary],
    )
    axes[1].set_ylabel("paired field superiority (%)")
    axes[1].set_title("Field-cluster bootstrap interval")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def reanalyze_existing(output_dir: Path, experiment: dict) -> None:
    training_rows = read_csv(output_dir / "v3c_k6_training.csv")
    sample_rows = read_csv(output_dir / "v3c_k6_samples.csv")
    cluster_rows = read_csv(output_dir / "v3c_k6_clusters.csv")
    summary = read_csv(output_dir / "v3c_k6_summary.csv")
    pairwise_rows = build_pairwise_rows(cluster_rows)
    seed_rows = build_seed_rows(sample_rows)
    pairwise_summary = summarize_pairwise(pairwise_rows, seed_rows, experiment)
    cost_summary = build_cost_summary(training_rows, sample_rows)
    decision = build_decision(pairwise_summary)
    adapter_verdict = next(
        row for row in pairwise_summary if row["comparison"] == "adapter_vs_continued_fno"
    )
    status = (
        "V3C_K6_DEV2_ADAPTER_GATE_PASS"
        if bool(adapter_verdict["development_gate_pass"])
        else "V3C_K6_DEV2_ADAPTER_GATE_FAIL"
    )
    write_csv(output_dir / "v3c_k6_pairwise.csv", pairwise_rows)
    write_csv(output_dir / "v3c_k6_seed_summary.csv", seed_rows)
    write_csv(output_dir / "v3c_k6_pairwise_summary.csv", pairwise_summary)
    plot_results(summary, pairwise_summary, output_dir / "t16_v3c_k6_dev2.png")
    dashboard_path = output_dir / "v3c_k6_dashboard.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    dashboard["scientific_status"] = status
    dashboard["pairwise_summary"] = pairwise_summary
    dashboard["seed_summary"] = seed_rows
    dashboard["cost_summary"] = cost_summary
    dashboard["decision"] = decision
    dashboard_path.write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "v3c_k6_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["scientific_status"] = status
    report["pairwise_summary"] = pairwise_summary
    report["cost_summary"] = cost_summary
    report["decision"] = decision
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_output_checksums(output_dir)
    print(json.dumps({"status": status, "pairwise_summary": pairwise_summary}, indent=2))


def main() -> None:
    args = parse_args()
    experiment = read_json(args.config)
    dataset_config_path = CONFIG_ROOT / experiment["dataset_config"]
    protocol_config_path = CONFIG_ROOT / experiment["protocol_gate_config"]
    dataset_config = read_json(dataset_config_path)
    protocol_config = read_json(protocol_config_path)
    device = args.device or str(dataset_config["training"]["device"])
    runtime_device = choose_device(device)
    base_epochs = int(args.base_epochs or experiment["base_epochs"])
    adaptation_epochs = int(args.adapt_epochs or experiment["adaptation_epochs"])
    seeds = [int(value) for value in experiment["training_seeds"]]
    if args.seed_limit is not None:
        seeds = seeds[: int(args.seed_limit)]
    output_dir = args.output_dir or ROOT / "results" / experiment["output_dir"]
    work_dir = args.work_dir or ROOT / "results" / experiment["work_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.analysis_only:
        reanalyze_existing(output_dir, experiment)
        return

    dataset_path = ROOT / "results" / "t16_v3c_dev2_dataset.npz"
    generate_dataset(dataset_config, dataset_path, force=bool(args.force_data))
    base_data = load_npz(dataset_path)
    budget = int(experiment["total_budget"])
    direct = prepare_direct_operator_data(
        base_data,
        [budget],
        int(experiment["fixed_query_index"]),
        int(experiment["audit_query_index"]),
    )
    selected_ridge, champions, tuning_rows = tune_classical_baselines(
        direct,
        [budget],
        [float(value) for value in experiment["ridge_relative_grid"]],
    )
    if champions[budget] != "ridge":
        raise RuntimeError("v3c K=6 requires validation-locked ridge as the anchor")
    data = append_ray_view_channels(replace_lift_with_ridge(direct, selected_ridge))
    write_csv(output_dir / "v3c_k6_baseline_tuning.csv", tuning_rows)

    training_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    initialization_rows: list[dict[str, object]] = []
    for outer_seed in seeds:
        base_seed = outer_seed + 101
        set_seed(base_seed)
        base_model = make_model(
            "fno",
            dataset_config["models"]["fno"],
            int(data["inputs"].shape[1]),
            residual=True,
        )
        base_config = training_config(
            dataset_config,
            base_seed,
            base_epochs,
            device,
            fixed_epochs=False,
        )
        base_dir = work_dir / str(outer_seed) / "base_fno"
        base_record = train_model("base_fno", base_model, data, base_config, base_dir)
        base_state = cpu_state(base_record["model"])
        base_hash = state_sha256(base_state)

        continuation_seed = outer_seed + 7_001
        set_seed(continuation_seed)
        continued_model = make_model(
            "fno",
            dataset_config["models"]["fno"],
            int(data["inputs"].shape[1]),
            residual=True,
        )
        continued_model.load_state_dict(base_state, strict=True)
        continued_start_hash = state_sha256(cpu_state(continued_model))

        set_seed(outer_seed + 3_001)
        adapter_base, adapter_model = build_adapter(
            data,
            dataset_config,
            protocol_config["adapter"],
        )
        adapter_base.load_state_dict(base_state, strict=True)
        adapter_start_hash = state_sha256(cpu_state(adapter_base))
        val_indices = split_indices(data)["val"][:4]
        x_val = torch.from_numpy(data["inputs"][val_indices]).to(runtime_device)
        continued_model = continued_model.to(runtime_device)
        adapter_model = adapter_model.to(runtime_device)
        continued_model.eval()
        adapter_model.eval()
        with torch.no_grad():
            initial_output_difference = float(
                torch.max(torch.abs(continued_model(x_val) - adapter_model(x_val)))
            )

        adapt_config = training_config(
            dataset_config,
            continuation_seed,
            adaptation_epochs,
            device,
            learning_rate=float(experiment["adaptation_learning_rate"]),
            fixed_epochs=True,
        )
        continued_dir = work_dir / str(outer_seed) / "continued_fno"
        continued_record = train_model(
            "continued_fno",
            continued_model,
            data,
            adapt_config,
            continued_dir,
        )
        adapter_dir = work_dir / str(outer_seed) / "zero_init_adapter"
        adapter_record = train_model(
            "zero_init_adapter",
            adapter_model,
            data,
            adapt_config,
            adapter_dir,
        )
        frozen_base_drift = maximum_state_drift(
            adapter_record["model"].base_operator,
            base_state,
        )
        continued_drift = maximum_state_drift(continued_record["model"], base_state)
        if initial_output_difference != 0.0 or frozen_base_drift != 0.0:
            raise RuntimeError("zero-init or frozen-base contract failed during K=6 training")
        if int(continued_record["epochs_ran"]) != adaptation_epochs:
            raise RuntimeError("continued FNO did not receive the locked extra epochs")
        if int(adapter_record["epochs_ran"]) != adaptation_epochs:
            raise RuntimeError("adapter did not receive the locked extra epochs")

        records = {
            "base_fno": base_record,
            "continued_fno": continued_record,
            "zero_init_adapter": adapter_record,
        }
        sample_rows.extend(
            evaluate_seed(
                outer_seed,
                data,
                records,
                dataset_config,
                selected_ridge[budget],
                int(experiment["audit_query_index"]),
            )
        )
        initialization_rows.append(
            {
                "model_seed": outer_seed,
                "base_checkpoint_sha256": base_hash,
                "continued_start_sha256": continued_start_hash,
                "adapter_base_start_sha256": adapter_start_hash,
                "continued_and_adapter_same_start": (
                    base_hash == continued_start_hash == adapter_start_hash
                ),
                "maximum_initial_output_difference": initial_output_difference,
                "maximum_adapter_base_drift_after_training": frozen_base_drift,
                "maximum_continued_fno_drift_after_training": continued_drift,
            }
        )
        for method, record, phase in (
            ("base_fno", base_record, "base"),
            ("continued_fno", continued_record, "additional"),
            ("zero_init_adapter", adapter_record, "additional"),
        ):
            training_rows.append(
                {
                    "model_seed": outer_seed,
                    "method": method,
                    "phase": phase,
                    "trainable_parameters": int(record["parameters"]),
                    "total_parameters": total_parameters(record["model"]),
                    "epochs_ran": int(record["epochs_ran"]),
                    "best_epoch": int(record["best_epoch"]),
                    "best_val_rel_l2": float(record["best_val_rel_l2"]),
                    "train_seconds": float(record["train_seconds"]),
                    "device": str(record["device"]),
                }
            )
        print(f"seed={outer_seed}: K=6 base/continued/adapter evaluated", flush=True)

    cluster_rows = collapse_model_seeds(sample_rows)
    summary = summarize_clusters(
        cluster_rows,
        int(experiment["bootstrap_seed"]),
        int(experiment["bootstrap_replicates"]),
    )
    domain_rows = build_domain_rows(cluster_rows)
    pairwise_rows = build_pairwise_rows(cluster_rows)
    seed_rows = build_seed_rows(sample_rows)
    pairwise_summary = summarize_pairwise(pairwise_rows, seed_rows, experiment)
    cost_summary = build_cost_summary(training_rows, sample_rows)
    decision = build_decision(pairwise_summary)
    adapter_verdict = next(
        row for row in pairwise_summary if row["comparison"] == "adapter_vs_continued_fno"
    )
    full_protocol = (
        len(seeds) == len(experiment["training_seeds"])
        and base_epochs == int(experiment["base_epochs"])
        and adaptation_epochs == int(experiment["adaptation_epochs"])
    )
    if not full_protocol:
        status = "V3C_K6_SMOKE_ONLY"
    elif bool(adapter_verdict["development_gate_pass"]):
        status = "V3C_K6_DEV2_ADAPTER_GATE_PASS"
    else:
        status = "V3C_K6_DEV2_ADAPTER_GATE_FAIL"

    write_csv(output_dir / "v3c_k6_training.csv", training_rows)
    write_csv(output_dir / "v3c_k6_initialization.csv", initialization_rows)
    write_csv(output_dir / "v3c_k6_samples.csv", sample_rows)
    write_csv(output_dir / "v3c_k6_clusters.csv", cluster_rows)
    write_csv(output_dir / "v3c_k6_summary.csv", summary)
    write_csv(output_dir / "v3c_k6_domains.csv", domain_rows)
    write_csv(output_dir / "v3c_k6_pairwise.csv", pairwise_rows)
    write_csv(output_dir / "v3c_k6_seed_summary.csv", seed_rows)
    write_csv(output_dir / "v3c_k6_pairwise_summary.csv", pairwise_summary)
    plot_results(summary, pairwise_summary, output_dir / "t16_v3c_k6_dev2.png")

    dashboard = {
        "experiment": experiment["name"],
        "scientific_status": status,
        "full_protocol": full_protocol,
        "development_fields_only": True,
        "blind_final_opened": False,
        "total_budget": budget,
        "methods": METHODS,
        "labels": LABELS,
        "independent_test_field_count": 128,
        "model_seed_count": len(seeds),
        "base_epochs": base_epochs,
        "adaptation_epochs": adaptation_epochs,
        "summary": summary,
        "domains": domain_rows,
        "pairwise_summary": pairwise_summary,
        "seed_summary": seed_rows,
        "cost_summary": cost_summary,
        "decision": decision,
        "training": training_rows,
        "initialization": initialization_rows,
    }
    (output_dir / "v3c_k6_dashboard.json").write_text(
        json.dumps(dashboard, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "completed" if full_protocol else "smoke_only",
        "scientific_status": status,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": device,
        },
        "protocol": {
            "same_K6_inputs_and_ridge_anchor": True,
            "same_base_checkpoint_per_seed": all(
                bool(row["continued_and_adapter_same_start"])
                for row in initialization_rows
            ),
            "same_additional_epochs": True,
            "same_additional_learning_rate_loss_and_batch_order": True,
            "adapter_base_frozen": all(
                float(row["maximum_adapter_base_drift_after_training"]) == 0.0
                for row in initialization_rows
            ),
            "matched_additional_epochs_not_FLOPs": True,
            "q_audit_used_for_training_or_selection": False,
            "test_unit": "three-dimensional dev2 field after collapsing model seeds",
            "blind_final_opened": False,
        },
        "training": training_rows,
        "initialization": initialization_rows,
        "pairwise_summary": pairwise_summary,
        "cost_summary": cost_summary,
        "decision": decision,
        "claims_boundary": [
            "This is a K=6 development experiment on inspected linear synthetic dev2 fields.",
            "The continuation and adapter phases match extra epochs, data order, optimizer settings and losses, not FLOPs.",
            "A positive result does not open or replace the preregistered blind final evaluation.",
            "Independent forward models, numerical baselines, NeRIF refinement and real BOST remain required.",
        ],
    }
    (output_dir / "v3c_k6_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_output_checksums(output_dir)
    print(json.dumps({"status": status, "adapter_verdict": adapter_verdict}, indent=2))
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
