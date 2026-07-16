#!/usr/bin/env python3
"""Reject incomplete, misaligned or overstated v3f frontier exports."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v3f_deeponet_frontier"
FNO_RESULTS = ROOT / "results" / "v3d_fno_optimizer_audit"
COMPUTE_RESULTS = ROOT / "results" / "v3e_compute_accounting"
EXPECTED_CHECKSUM_FILES = {
    "v3f_deeponet_baseline_tuning.csv",
    "v3f_deeponet_lr_tuning.csv",
    "v3f_deeponet_history.csv",
    "v3f_deeponet_checkpoints.csv",
    "v3f_deeponet_validation_summary.csv",
    "v3f_deeponet_strategy_summary.csv",
    "v3f_deeponet_samples.csv",
    "v3f_deeponet_clusters.csv",
    "v3f_deeponet_test_summary.csv",
    "v3f_deeponet_pairwise_summary.csv",
    "v3f_deeponet_seed_summary.csv",
    "v3f_architecture_frontier.csv",
    "v3f_time_to_target.csv",
    "v3f_cross_architecture_pairwise.csv",
    "v3f_cross_architecture_domains.csv",
    "v3f_cross_architecture_seeds.csv",
    "v3f_selection_commit.json",
    "v3f_deeponet_frontier_dashboard.json",
    "v3f_deeponet_frontier_report.json",
    "t16_v3f_deeponet_fno_frontier.png",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    dashboard = json.loads(
        (RESULTS / "v3f_deeponet_frontier_dashboard.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (RESULTS / "v3f_deeponet_frontier_report.json").read_text(
            encoding="utf-8"
        )
    )
    lr_rows = read_csv(RESULTS / "v3f_deeponet_lr_tuning.csv")
    history = read_csv(RESULTS / "v3f_deeponet_history.csv")
    checkpoints = read_csv(RESULTS / "v3f_deeponet_checkpoints.csv")
    validation = read_csv(RESULTS / "v3f_deeponet_validation_summary.csv")
    strategies = read_csv(RESULTS / "v3f_deeponet_strategy_summary.csv")
    samples = read_csv(RESULTS / "v3f_deeponet_samples.csv")
    clusters = read_csv(RESULTS / "v3f_deeponet_clusters.csv")
    frontier = read_csv(RESULTS / "v3f_architecture_frontier.csv")
    targets = read_csv(RESULTS / "v3f_time_to_target.csv")
    cross = read_csv(RESULTS / "v3f_cross_architecture_pairwise.csv")
    domains = read_csv(RESULTS / "v3f_cross_architecture_domains.csv")
    seeds = read_csv(RESULTS / "v3f_cross_architecture_seeds.csv")
    selection_commit = json.loads(
        (RESULTS / "v3f_selection_commit.json").read_text(encoding="utf-8")
    )

    expected_status = (
        "MATCHED_DEVELOPMENT_FRONTIER_COMPLETE_CONFIRMATORY_SUPERIORITY_LOCKED"
    )
    assert dashboard["scientific_status"] == expected_status
    assert report["scientific_status"] == expected_status
    assert dashboard["confirmatory_superiority_eligible"] is False
    assert dashboard["blind_final_opened"] is False
    assert dashboard["matched_architectures"] == ["deeponet", "fno"]
    assert dashboard["fixed_epoch_checkpoints"] == [60, 120, 180, 240]
    assert len(lr_rows) == 4 * 3
    assert len(history) == 3 * 24 + 3 * 3 * (240 - 24)
    assert len(checkpoints) == 3 * 3 * (1 + (240 - 24) // 12)
    assert len(strategies) == 3
    assert len(samples) == 4 * 3 * 128
    assert len(clusters) == 4 * 128
    assert len(frontier) == 2 * 4
    assert len(targets) == 2 * 4
    assert len(cross) == 1
    assert len(domains) == 4
    assert len(seeds) == 3

    selected_lr = float(dashboard["selected_base_learning_rate"])
    grouped_lr: dict[float, list[float]] = defaultdict(list)
    for row in lr_rows:
        grouped_lr[float(row["learning_rate"])].append(
            float(row["best_validation_rel_l2"])
        )
    assert selected_lr == min(
        grouped_lr,
        key=lambda value: (sum(grouped_lr[value]) / len(grouped_lr[value]), value),
    )
    assert sum(row["global_validation_champion"] == "True" for row in lr_rows) == 3
    assert float(dashboard["discarded_learning_rate_search_seconds"]) > 0.0
    assert float(dashboard["learning_rate_screen_total_seconds"]) > float(
        dashboard["discarded_learning_rate_search_seconds"]
    )

    commit_digest = selection_commit.pop("selection_commit_sha256")
    reconstructed_digest = hashlib.sha256(
        json.dumps(
            selection_commit,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert reconstructed_digest == commit_digest
    assert dashboard["selection_commit_sha256"] == commit_digest
    assert report["selection_commit_sha256"] == commit_digest
    assert selection_commit["selection_scope"] == "validation_only"
    assert selection_commit["validation_aggregation"] == "sample_weighted_field_mean"
    assert selection_commit["dev2_or_q_audit_metric_present_in_selection_payload"] is False
    assert "reused synthetic development diagnostic" in selection_commit[
        "post_selection_dataset_role"
    ]
    assert {row["selection_commit_sha256"] for row in samples} == {commit_digest}

    deep_champion = dashboard["deeponet_validation_champion"]
    fno_champion = dashboard["fno_validation_champion"]
    assert sum(row["validation_champion"] == "True" for row in strategies) == 1
    assert next(
        row["strategy"] for row in strategies if row["validation_champion"] == "True"
    ) == deep_champion
    final_checkpoint_hashes = {
        row["model_seed"]: row["selected_checkpoint_sha256"]
        for row in checkpoints
        if row["strategy"] == deep_champion
        and int(row["cumulative_epochs"]) == 240
    }
    assert final_checkpoint_hashes == selection_commit[
        "deeponet_final_checkpoint_sha256_by_seed"
    ]
    final_validation = {
        architecture: float(value)
        for architecture, value in dashboard["final_validation_rel_l2"].items()
    }
    assert dashboard["validation_champion_architecture"] == min(
        final_validation, key=lambda name: (final_validation[name], name)
    )
    assert cross[0]["candidate"] == dashboard["validation_champion_architecture"]
    assert cross[0]["selection_basis"].startswith("lowest mean validation")
    assert cross[0]["confirmatory_superiority_eligible"] == "False"
    dominance = dashboard["dominance_diagnostic"]
    assert dominance["first_fno_endpoint_better_than_deeponet_final_epoch"] == 24
    assert dominance["observed_deeponet_fixed_checkpoint_count"] == 4
    assert dominance["observed_pareto_dominated_deeponet_fixed_checkpoint_count"] == 4
    assert float(dominance["deeponet_forward_flops_v1_to_fno_ratio"]) < 1.0
    assert float(dominance["deeponet_training_step_p50_to_fno_ratio"]) < 1.0

    expected_epochs = {60, 120, 180, 240}
    assert {row["architecture"] for row in frontier} == {"deeponet", "fno"}
    for architecture in ("deeponet", "fno"):
        subset = [row for row in frontier if row["architecture"] == architecture]
        assert {int(row["cumulative_epochs"]) for row in subset} == expected_epochs
        assert all(float(row["mean_validation_rel_l2"]) > 0.0 for row in subset)
        assert all(float(row["mean_cumulative_train_seconds"]) > 0.0 for row in subset)
        assert all(float(row["forward_estimated_flops_v1"]) > 0.0 for row in subset)

    profiles = {
        row["method"]: row
        for row in read_csv(COMPUTE_RESULTS / "v3e_compute_profiles.csv")
    }
    for row in frontier:
        source = profiles[row["method"]]
        assert int(row["total_parameters"]) == int(source["total_parameters"])
        assert float(row["forward_estimated_flops_v1"]) == float(
            source["forward_estimated_flops_v1"]
        )

    deep_contracts = {
        (int(row["model_seed"]), int(row["block_index"])): row[
            "batch_order_contract_sha256"
        ]
        for row in checkpoints
        if row["strategy"] == deep_champion
    }
    fno_checkpoints = read_csv(FNO_RESULTS / "v3d_optimizer_checkpoints.csv")
    fno_contracts = {
        (int(row["model_seed"]), int(row["block_index"])): row[
            "batch_order_contract_sha256"
        ]
        for row in fno_checkpoints
        if row["strategy"] == fno_champion
    }
    assert deep_contracts == fno_contracts
    for key, digest in deep_contracts.items():
        assert len(digest) == 64, key

    protocol = report["protocol"]
    assert protocol["architecture_selected_by_final_validation_only"] is True
    assert protocol["dev2_computed_after_all_deeponet_validation_decisions"] is True
    assert protocol["dev2_cannot_change_validation_winner"] is True
    assert protocol["matched_attempted_epochs_not_equal_flops_or_equal_wall_time"] is True
    assert protocol["batch_order_contract_binds_actual_train_indices"] is True
    assert protocol["validation_metric_aggregated_per_sample"] is True
    assert report["provenance"] == dashboard["provenance"]
    assert report["provenance"]["dataset_npz_public"] is False
    assert report["provenance"]["checkpoint_weights_public"] is False

    public_files = [path for path in RESULTS.rglob("*") if path.is_file()]
    assert not any(path.suffix in {".pt", ".pth", ".npz", ".pdf"} for path in public_files)
    checksums = {}
    for line in (RESULTS / "v3f_deeponet_frontier_checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines():
        digest, filename = line.split("  ", 1)
        checksums[filename] = digest
    assert set(checksums) == EXPECTED_CHECKSUM_FILES
    for filename, expected in checksums.items():
        actual = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
    print(
        "T16 v3f DeepONet/FNO frontier validation passed: "
        f"lr={selected_lr:g}, frontier={len(frontier)}, "
        f"winner={dashboard['validation_champion_architecture']}, "
        f"status={dashboard['scientific_status']}"
    )


if __name__ == "__main__":
    main()
