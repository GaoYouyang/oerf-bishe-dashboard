#!/usr/bin/env python3
"""Validate the v3d FNO optimizer-protocol audit without private checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v3d_fno_optimizer_audit"
AUDIT_CONFIG = ROOT / "configs" / "v3d_fno_optimizer_audit.json"
DATASET_CONFIG = ROOT / "configs" / "v3c_dev2_dataset.json"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    experiment = json.loads(AUDIT_CONFIG.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
    dashboard = json.loads(
        (RESULTS / "v3d_optimizer_dashboard.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (RESULTS / "v3d_optimizer_report.json").read_text(encoding="utf-8")
    )
    history = rows("v3d_optimizer_history.csv")
    checkpoints = rows("v3d_optimizer_checkpoints.csv")
    validation = rows("v3d_optimizer_validation_summary.csv")
    strategy_summary = rows("v3d_optimizer_strategy_summary.csv")
    samples = rows("v3d_optimizer_samples.csv")
    clusters = rows("v3d_optimizer_clusters.csv")
    test_summary = rows("v3d_optimizer_test_summary.csv")
    pairwise_summary = rows("v3d_optimizer_pairwise_summary.csv")
    seed_summary = rows("v3d_optimizer_seed_summary.csv")

    strategies = {str(row["id"]) for row in experiment["strategies"]}
    methods = {"base_24", *strategies}
    seeds = {int(value) for value in experiment["training_seeds"]}
    base_epoch = int(experiment["base_epochs"])
    max_epoch = int(experiment["max_total_epochs"])
    block = int(experiment["continuation_block_epochs"])
    epochs = set(range(base_epoch, max_epoch + 1, block))

    split_indices: dict[str, set[int]] = {}
    cursor = 0
    for split_name, spec in dataset["splits"].items():
        count = int(spec["count"])
        split_indices[split_name] = set(range(cursor, cursor + count))
        cursor += count
    dev_splits = {
        name: values
        for name, values in split_indices.items()
        if name not in {"train", "val"}
    }
    dev_indices = set().union(*dev_splits.values())

    assert dashboard["full_protocol"] is True
    assert dashboard["development_fields_only"] is True
    assert dashboard["blind_final_opened"] is False
    assert dashboard["geometry_gate_resolved"] is False
    assert dashboard["model_seed_count"] == len(seeds)
    assert dashboard["independent_test_field_count"] == len(dev_indices) == 128
    assert set(dashboard["strategy_ids"]) == strategies
    assert dashboard["validation_champion"] in strategies
    assert report["validation_champion"] == dashboard["validation_champion"]
    assert report["scientific_status"] == dashboard["scientific_status"]
    assert report["protocol"]["dev2_computed_after_all_validation_decisions"] is True
    assert report["protocol"]["test_or_q_audit_used_for_strategy_or_checkpoint_selection"] is False
    assert report["protocol"]["blind_final_opened"] is False
    assert report["protocol"]["base_optimizer_state_carried_into_continuation"] is False
    assert report["protocol"]["carry_adam_means_across_continuation_blocks_only"] is True
    assert report["protocol"]["batch_order_contract_hash_recorded_per_block"] is True
    assert report["protocol"]["batch_order_contract_binds_actual_train_indices"] is True
    assert report["protocol"]["validation_metric_aggregated_per_sample"] is True
    assert report["provenance"] == dashboard["provenance"]
    assert report["provenance"]["dataset_npz_public"] is False
    assert report["provenance"]["checkpoint_weights_public"] is False
    assert all(
        isinstance(report["provenance"][key], str)
        and len(report["provenance"][key]) == 64
        for key in (
            "experiment_config_sha256",
            "dataset_config_sha256",
            "training_script_sha256",
            "train_eval_script_sha256",
            "data_script_sha256",
            "dataset_npz_sha256",
        )
    )

    checkpoint_keys = {
        (
            str(row["strategy"]),
            int(row["model_seed"]),
            int(row["cumulative_epochs"]),
        )
        for row in checkpoints
    }
    assert len(checkpoint_keys) == len(checkpoints)
    assert checkpoint_keys == {
        (strategy, seed, epoch)
        for strategy in strategies
        for seed in seeds
        for epoch in epochs
    }
    for seed in seeds:
        base_rows = [
            row
            for row in checkpoints
            if int(row["model_seed"]) == seed
            and int(row["cumulative_epochs"]) == base_epoch
        ]
        assert len({row["selected_checkpoint_sha256"] for row in base_rows}) == 1
        assert len({row["endpoint_checkpoint_sha256"] for row in base_rows}) == 1
    assert all(
        int(row["block_seed"])
        == (
            int(row["model_seed"]) + 101
            if int(row["block_index"]) == 0
            else int(row["model_seed"]) + 10_000 + int(row["block_index"])
        )
        for row in checkpoints
    )
    for seed in seeds:
        for epoch in epochs:
            hashes = {
                row["batch_order_contract_sha256"]
                for row in checkpoints
                if int(row["model_seed"]) == seed
                and int(row["cumulative_epochs"]) == epoch
            }
            assert len(hashes) == 1
    assert all(float(row["relative_validation_improvement_pct"]) >= 0.0 for row in checkpoints)

    base_history_keys = {
        (int(row["model_seed"]), int(row["cumulative_epoch"]))
        for row in history
        if row["strategy"] == "shared_base"
    }
    assert base_history_keys == {
        (seed, epoch) for seed in seeds for epoch in range(1, base_epoch + 1)
    }
    continuation_history_keys = {
        (
            str(row["strategy"]),
            int(row["model_seed"]),
            int(row["cumulative_epoch"]),
        )
        for row in history
        if row["strategy"] != "shared_base"
    }
    assert continuation_history_keys == {
        (strategy, seed, epoch)
        for strategy in strategies
        for seed in seeds
        for epoch in range(base_epoch + 1, max_epoch + 1)
    }
    assert len(base_history_keys) + len(continuation_history_keys) == len(history)

    validation_keys = {
        (str(row["strategy"]), int(row["cumulative_epochs"]))
        for row in validation
    }
    assert len(validation_keys) == len(validation)
    assert validation_keys == {
        (strategy, epoch) for strategy in strategies for epoch in epochs
    }
    assert {row["strategy"] for row in strategy_summary} == strategies
    final_validation = {
        str(row["strategy"]): float(row["mean_validation_rel_l2"])
        for row in validation
        if int(row["cumulative_epochs"]) == max_epoch
    }
    recomputed_champion = min(final_validation, key=final_validation.get)
    assert recomputed_champion == dashboard["validation_champion"]

    sample_keys = {
        (
            str(row["strategy"]),
            int(row["model_seed"]),
            int(row["source_index"]),
        )
        for row in samples
    }
    assert len(sample_keys) == len(samples)
    assert sample_keys == {
        (method, seed, source_index)
        for method in methods
        for seed in seeds
        for source_index in dev_indices
    }
    assert all(
        int(row["source_index"]) in dev_splits[str(row["source_split"])]
        for row in samples
    )
    assert dev_indices.isdisjoint(split_indices["train"])
    assert dev_indices.isdisjoint(split_indices["val"])
    cluster_keys = {
        (str(row["strategy"]), int(row["source_index"])) for row in clusters
    }
    assert len(cluster_keys) == len(clusters)
    assert cluster_keys == {
        (method, source_index)
        for method in methods
        for source_index in dev_indices
    }
    assert {row["strategy"] for row in test_summary} == methods
    assert {row["candidate"] for row in pairwise_summary} == {
        dashboard["validation_champion"]
    }
    assert {row["comparator"] for row in pairwise_summary} == (
        strategies - {dashboard["validation_champion"]}
    )
    assert all(int(row["independent_field_count"]) == 128 for row in pairwise_summary)
    assert {
        (str(row["strategy"]), int(row["model_seed"])) for row in seed_summary
    } == {(method, seed) for method in methods for seed in seeds}

    rule = dashboard["validation_plateau_rule"]
    for strategy in strategies:
        ordered = sorted(
            [row for row in validation if row["strategy"] == strategy],
            key=lambda row: int(row["cumulative_epochs"]),
        )
        trailing = 0
        for row in reversed(ordered[1:]):
            plateau = (
                float(row["mean_relative_validation_improvement_pct"])
                <= float(rule["maximum_mean_relative_improvement_pct_per_block"])
                and float(row["max_seed_relative_validation_improvement_pct"])
                <= float(rule["maximum_seed_relative_improvement_pct_per_block"])
            )
            if not plateau:
                break
            trailing += 1
        summary = next(row for row in strategy_summary if row["strategy"] == strategy)
        assert int(summary["observed_consecutive_final_plateau_blocks"]) == trailing
        assert (summary["plateau_reached"] == "True") is (
            trailing >= int(rule["required_consecutive_final_plateau_blocks"])
        )

    assert not any(
        path.suffix in {".pt", ".pth", ".npz"} for path in RESULTS.iterdir()
    )
    for line in (RESULTS / "v3d_optimizer_checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines():
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest() == digest
    print("T16 v3d FNO optimizer-protocol audit validation passed")
    print(f"history_rows={len(history)}")
    print(f"sample_rows={len(samples)}")
    print(f"validation_champion={dashboard['validation_champion']}")
    print(f"scientific_status={dashboard['scientific_status']}")


if __name__ == "__main__":
    main()
