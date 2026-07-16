#!/usr/bin/env python3
"""Reject incomplete, leaked or overstated v3g DeepONet capacity exports."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v3g_deeponet_capacity_audit"
FNO_RESULTS = ROOT / "results" / "v3d_fno_optimizer_audit"
EXPECTED_CHECKSUM_FILES = {
    "v3g_deeponet_baseline_tuning.csv",
    "v3g_variant_manifest.csv",
    "v3g_screen.csv",
    "v3g_screen_summary.csv",
    "v3g_history.csv",
    "v3g_checkpoints.csv",
    "v3g_validation_summary.csv",
    "v3g_strategy_summary.csv",
    "v3g_samples.csv",
    "v3g_clusters.csv",
    "v3g_test_summary.csv",
    "v3g_pairwise_summary.csv",
    "v3g_seed_summary.csv",
    "v3g_validation_comparison.csv",
    "v3g_cross_baseline_pairwise.csv",
    "v3g_cross_baseline_domains.csv",
    "v3g_cross_baseline_seeds.csv",
    "v3g_selection_commit.json",
    "v3g_deeponet_capacity_dashboard.json",
    "v3g_deeponet_capacity_report.json",
    "t16_v3g_deeponet_capacity_audit.png",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    dashboard = json.loads(
        (RESULTS / "v3g_deeponet_capacity_dashboard.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (RESULTS / "v3g_deeponet_capacity_report.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = read_csv(RESULTS / "v3g_variant_manifest.csv")
    screen = read_csv(RESULTS / "v3g_screen.csv")
    screen_summary = read_csv(RESULTS / "v3g_screen_summary.csv")
    history = read_csv(RESULTS / "v3g_history.csv")
    checkpoints = read_csv(RESULTS / "v3g_checkpoints.csv")
    validation = read_csv(RESULTS / "v3g_validation_summary.csv")
    strategies = read_csv(RESULTS / "v3g_strategy_summary.csv")
    samples = read_csv(RESULTS / "v3g_samples.csv")
    clusters = read_csv(RESULTS / "v3g_clusters.csv")
    comparison = read_csv(RESULTS / "v3g_validation_comparison.csv")
    cross = read_csv(RESULTS / "v3g_cross_baseline_pairwise.csv")
    domains = read_csv(RESULTS / "v3g_cross_baseline_domains.csv")
    seeds = read_csv(RESULTS / "v3g_cross_baseline_seeds.csv")
    selection_commit = json.loads(
        (RESULTS / "v3g_selection_commit.json").read_text(encoding="utf-8")
    )

    expected_status = "BOUNDED_DEEPONET_CAPACITY_AUDIT_COMPLETE"
    assert dashboard["scientific_status"] == expected_status
    assert report["scientific_status"] == expected_status
    assert dashboard["confirmatory_superiority_eligible"] is False
    assert dashboard["blind_final_opened"] is False
    assert len(manifest) == 10
    screened_ids = {row["variant_id"] for row in manifest if row["screen"] == "True"}
    excluded_ids = {row["variant_id"] for row in manifest if row["screen"] == "False"}
    assert len(screened_ids) == 8
    assert len(excluded_ids) == 2
    assert all(row["within_parameter_cap"] == "True" for row in manifest if row["screen"] == "True")
    assert all(row["within_parameter_cap"] == "False" for row in manifest if row["screen"] == "False")
    assert {row["variant_id"] for row in screen} == screened_ids
    assert not ({row["variant_id"] for row in screen} & excluded_ids)
    assert len(screen) == 8 * 3 * 3
    assert len(screen_summary) == 8 * 3
    assert len(history) == 3 * 24 + 3 * 3 * (240 - 24)
    assert len(checkpoints) == 3 * 3 * (1 + (240 - 24) // 12)
    assert len(strategies) == 3
    assert len(samples) == 4 * 3 * 128
    assert len(clusters) == 4 * 128
    assert len(comparison) == 3 * 5
    assert len(cross) == 2
    assert len(domains) == 2 * 4
    assert len(seeds) == 2 * 3

    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in screen:
        grouped[(row["variant_id"], float(row["learning_rate"]))].append(row)
    assert set(len(values) for values in grouped.values()) == {3}
    selected_key = min(
        grouped,
        key=lambda key: (
            sum(float(row["best_validation_rel_l2"]) for row in grouped[key]) / 3,
            int(grouped[key][0]["parameter_count"]),
            key[0],
            key[1],
        ),
    )
    assert dashboard["selected_variant"] == selected_key[0]
    assert float(dashboard["selected_learning_rate"]) == selected_key[1]
    assert sum(row["global_validation_champion"] == "True" for row in screen) == 3
    assert sum(
        row["global_validation_champion"] == "True" for row in screen_summary
    ) == 1
    assert float(dashboard["screen_total_seconds"]) > float(
        dashboard["discarded_screen_seconds"]
    ) > 0.0
    contracts_by_seed: dict[int, set[str]] = defaultdict(set)
    for row in screen:
        contracts_by_seed[int(row["model_seed"])].add(
            row["batch_order_contract_sha256"]
        )
    assert all(len(values) == 1 for values in contracts_by_seed.values())
    assert all(len(next(iter(values))) == 64 for values in contracts_by_seed.values())

    commit_digest = selection_commit.pop("selection_commit_sha256")
    reconstructed = hashlib.sha256(
        json.dumps(
            selection_commit,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert reconstructed == commit_digest
    assert dashboard["selection_commit_sha256"] == commit_digest
    assert report["selection_commit_sha256"] == commit_digest
    assert selection_commit["selection_scope"] == "validation_only"
    assert selection_commit["validation_aggregation"] == "sample_weighted_field_mean"
    assert selection_commit["dev2_or_q_audit_metric_present_in_selection_payload"] is False
    assert "reused synthetic development diagnostic" in selection_commit[
        "post_selection_dataset_role"
    ]
    assert {row["selection_commit_sha256"] for row in samples} == {commit_digest}

    champion = dashboard["selected_optimizer_strategy"]
    assert sum(row["validation_champion"] == "True" for row in strategies) == 1
    assert next(
        row["strategy"] for row in strategies if row["validation_champion"] == "True"
    ) == champion
    final_hashes = {
        row["model_seed"]: row["selected_checkpoint_sha256"]
        for row in checkpoints
        if row["strategy"] == champion and int(row["cumulative_epochs"]) == 240
    }
    assert final_hashes == selection_commit[
        "selected_final_checkpoint_sha256_by_seed"
    ]
    assert {row["architecture"] for row in comparison} == {
        "v3g_selected_deeponet",
        "v3f_reference_deeponet",
        "fno",
    }
    assert {int(row["cumulative_epochs"]) for row in comparison} == {
        24,
        60,
        120,
        180,
        240,
    }
    assert {row["comparator"] for row in cross} == {
        "v3f_reference_deeponet",
        "fno",
    }
    assert all(row["candidate"] == "v3g_selected_deeponet" for row in cross)
    assert all(row["confirmatory_superiority_eligible"] == "False" for row in cross)
    assert all(
        row["uncertainty_unit"] == "field cluster after model-seed collapse"
        for row in cross
    )

    fno_dashboard = json.loads(
        (FNO_RESULTS / "v3d_optimizer_dashboard.json").read_text(encoding="utf-8")
    )
    fno_champion = fno_dashboard["validation_champion"]
    v3g_contracts = {
        (int(row["model_seed"]), int(row["block_index"])): row[
            "batch_order_contract_sha256"
        ]
        for row in checkpoints
        if row["strategy"] == champion
    }
    fno_checkpoints = read_csv(FNO_RESULTS / "v3d_optimizer_checkpoints.csv")
    fno_contracts = {
        (int(row["model_seed"]), int(row["block_index"])): row[
            "batch_order_contract_sha256"
        ]
        for row in fno_checkpoints
        if row["strategy"] == fno_champion
    }
    assert v3g_contracts == fno_contracts

    protocol = report["protocol"]
    assert protocol["pre_registered_bounded_variant_set"] is True
    assert protocol["capacity_cap_enforced_before_training"] is True
    assert protocol[
        "architecture_and_learning_rate_selected_by_three_seed_mean_validation"
    ] is True
    assert protocol["screen_batch_order_contract_binds_actual_train_indices"] is True
    assert protocol["validation_metric_aggregated_per_sample"] is True
    assert protocol["reused_dev2_computed_after_selection_commit"] is True
    assert protocol["dev2_cannot_change_architecture_learning_rate_or_optimizer"] is True
    assert report["provenance"] == dashboard["provenance"]
    assert report["provenance"]["dataset_npz_public"] is False
    assert report["provenance"]["checkpoint_weights_public"] is False

    public_files = [path for path in RESULTS.rglob("*") if path.is_file()]
    assert not any(path.suffix in {".pt", ".pth", ".npz", ".pdf"} for path in public_files)
    checksums = {}
    for line in (RESULTS / "v3g_deeponet_capacity_checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines():
        digest, filename = line.split("  ", 1)
        checksums[filename] = digest
    assert set(checksums) == EXPECTED_CHECKSUM_FILES
    for filename, expected in checksums.items():
        actual = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
    assert (RESULTS / "t16_v3g_deeponet_capacity_audit.png").stat().st_size > 20_000
    print(
        "T16 v3g DeepONet capacity audit validation passed: "
        f"screen={len(screen)}, selected={dashboard['selected_variant']}, "
        f"strategy={champion}, status={dashboard['scientific_status']}"
    )


if __name__ == "__main__":
    main()

