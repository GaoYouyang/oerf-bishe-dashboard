#!/usr/bin/env python3
"""Validate the complete v3d K=6 FNO validation-plateau audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v3d_fno_saturation_audit"
AUDIT_CONFIG = ROOT / "configs" / "v3d_fno_saturation_audit.json"
DATASET_CONFIG = ROOT / "configs" / "v3c_dev2_dataset.json"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    audit_config = json.loads(AUDIT_CONFIG.read_text(encoding="utf-8"))
    dataset_config = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
    dashboard = json.loads(
        (RESULTS / "v3d_fno_saturation_dashboard.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (RESULTS / "v3d_fno_saturation_report.json").read_text(encoding="utf-8")
    )
    history = rows("v3d_fno_history.csv")
    checkpoints = rows("v3d_fno_checkpoints.csv")
    validation = rows("v3d_fno_validation_summary.csv")
    samples = rows("v3d_fno_samples.csv")
    clusters = rows("v3d_fno_clusters.csv")
    test_summary = rows("v3d_fno_test_summary.csv")
    seed_summary = rows("v3d_fno_seed_summary.csv")

    expected_epochs = set(
        range(
            int(audit_config["base_epochs"]),
            int(audit_config["max_total_epochs"]) + 1,
            int(audit_config["continuation_block_epochs"]),
        )
    )
    expected_model_seeds = {
        int(value) for value in audit_config["training_seeds"]
    }
    split_indices: dict[str, set[int]] = {}
    cursor = 0
    for split_name, spec in dataset_config["splits"].items():
        count = int(spec["count"])
        split_indices[split_name] = set(range(cursor, cursor + count))
        cursor += count
    development_splits = {
        name: indices
        for name, indices in split_indices.items()
        if name not in {"train", "val"}
    }
    development_indices = set().union(*development_splits.values())
    assert dashboard["full_protocol"] is True
    assert dashboard["development_fields_only"] is True
    assert dashboard["blind_final_opened"] is False
    assert dashboard["scientific_status"] in {
        "V3D_FNO_VALIDATION_PLATEAU_REACHED",
        "V3D_FNO_VALIDATION_PLATEAU_NOT_REACHED_BY_MAX_EPOCH",
    }
    assert dashboard["model_seed_count"] == 3
    assert dashboard["independent_test_field_count"] == 128
    assert dashboard["max_total_epochs"] == 96
    assert {int(row["cumulative_epochs"]) for row in checkpoints} == expected_epochs
    assert len(checkpoints) == 3 * len(expected_epochs)
    assert len(validation) == len(expected_epochs)
    assert len(samples) == 3 * len(expected_epochs) * 128
    assert len(clusters) == len(expected_epochs) * 128
    assert len(test_summary) == len(expected_epochs)
    assert len(seed_summary) == 3 * len(expected_epochs)
    assert len(history) == 3 * 96
    checkpoint_keys = {
        (int(row["model_seed"]), int(row["cumulative_epochs"]))
        for row in checkpoints
    }
    assert len(checkpoint_keys) == len(checkpoints)
    assert {seed for seed, _ in checkpoint_keys} == expected_model_seeds
    assert checkpoint_keys == {
        (seed, epoch) for seed in expected_model_seeds for epoch in expected_epochs
    }
    history_keys = {
        (int(row["model_seed"]), int(row["cumulative_epoch"])) for row in history
    }
    assert len(history_keys) == len(history)
    assert history_keys == {
        (seed, epoch)
        for seed in expected_model_seeds
        for epoch in range(1, int(audit_config["max_total_epochs"]) + 1)
    }
    sample_keys = {
        (
            int(row["model_seed"]),
            int(row["cumulative_epochs"]),
            int(row["source_index"]),
        )
        for row in samples
    }
    assert len(sample_keys) == len(samples)
    assert sample_keys == {
        (seed, epoch, source_index)
        for seed in expected_model_seeds
        for epoch in expected_epochs
        for source_index in development_indices
    }
    assert all(
        int(row["source_index"])
        in development_splits[str(row["source_split"])]
        for row in samples
    )
    assert development_indices.isdisjoint(split_indices["train"])
    assert development_indices.isdisjoint(split_indices["val"])
    source_seed_pairs = {
        (int(row["source_index"]), int(row["sample_seed"])) for row in samples
    }
    assert len(source_seed_pairs) == len(development_indices)
    cluster_keys = {
        (int(row["cumulative_epochs"]), int(row["source_index"]))
        for row in clusters
    }
    assert len(cluster_keys) == len(clusters)
    assert cluster_keys == {
        (epoch, source_index)
        for epoch in expected_epochs
        for source_index in development_indices
    }
    seed_summary_keys = {
        (int(row["model_seed"]), int(row["cumulative_epochs"]))
        for row in seed_summary
    }
    assert len(seed_summary_keys) == len(seed_summary)
    assert seed_summary_keys == checkpoint_keys
    assert all(int(row["epochs_ran_in_block"]) == 12 for row in checkpoints if row["phase"] == "continuation")
    assert all(float(row["relative_validation_improvement_pct"]) >= 0.0 for row in checkpoints)
    assert all(row["retained_previous_checkpoint"] == "False" for row in checkpoints)
    assert all(
        int(row["retained_previous_checkpoint_count"]) == 0 for row in validation
    )
    assert all(
        row["retained_previous_checkpoint"] is False
        for row in dashboard["checkpoint_rows"]
    )
    assert report["protocol"]["plateau_selection_uses_validation_only"] is True
    assert report["protocol"]["validation_metric_aggregated_per_sample"] is True
    assert report["protocol"]["test_or_q_audit_used_for_plateau_selection"] is False
    assert report["protocol"]["blind_final_opened"] is False
    assert (
        report["protocol"][
            "optimizer_and_scheduler_restarted_for_each_continuation_block"
        ]
        is True
    )
    assert report["scientific_status"] == dashboard["scientific_status"]
    assert report["plateau_decision"] == dashboard["plateau_decision"]

    rule = dashboard["validation_plateau_rule"]
    validation_by_epoch = {
        int(row["cumulative_epochs"]): row for row in validation
    }
    assert len(validation_by_epoch) == len(validation)
    plateau_blocks = []
    for epoch in sorted(expected_epochs - {int(audit_config["base_epochs"])}):
        row = validation_by_epoch[epoch]
        plateau_blocks.append(
            float(row["mean_relative_validation_improvement_pct"])
            <= float(rule["maximum_mean_relative_improvement_pct_per_block"])
            and float(row["max_seed_relative_validation_improvement_pct"])
            <= float(rule["maximum_seed_relative_improvement_pct_per_block"])
        )
    trailing_plateau_blocks = 0
    for is_plateau in reversed(plateau_blocks):
        if not is_plateau:
            break
        trailing_plateau_blocks += 1
    recomputed_plateau = trailing_plateau_blocks >= int(
        rule["required_consecutive_final_plateau_blocks"]
    )
    assert dashboard["plateau_decision"]["plateau_reached"] is recomputed_plateau
    assert (
        dashboard["plateau_decision"][
            "observed_consecutive_final_plateau_blocks"
        ]
        == trailing_plateau_blocks
    )

    for line in (RESULTS / "v3d_fno_checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines():
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest() == digest
    print("T16 v3d FNO validation-plateau audit validation passed")
    print(f"history_rows={len(history)}")
    print(f"sample_rows={len(samples)}")
    print(f"scientific_status={dashboard['scientific_status']}")


if __name__ == "__main__":
    main()
