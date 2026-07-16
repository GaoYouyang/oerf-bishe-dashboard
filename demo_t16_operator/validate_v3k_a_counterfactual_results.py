#!/usr/bin/env python3
"""Independently validate the v3k-A equal-exposure mechanism result."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "v3k_a_counterfactual_supervision.json"
ALLOWED_STATUSES = {
    "COUNTERFACTUAL_DATA_MECHANISM_GATE_PASS_CONFIRMATION_NOT_AUTHORIZED",
    "GLOBAL_GEOMETRY_MODULATION_MECHANISM_FAIL_STOP_CAPACITY_SEARCH",
    "COUNTERFACTUAL_MECHANISM_INCONCLUSIVE_DO_NOT_SCALE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(table: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matched = [
        row
        for row in table
        if all(str(row[key]) == str(value) for key, value in criteria.items())
    ]
    assert len(matched) == 1, (criteria, len(matched))
    return matched[0]


def recompute_gate(
    config: dict,
    pairwise: list[dict[str, str]],
    interactions: list[dict[str, str]],
    swap_summary: list[dict],
) -> tuple[dict[str, bool], str]:
    gate = config["mechanism_gate"]
    delta = float(gate["minimum_mean_gain_pct"])
    floor = float(gate["minimum_cluster_ci95_low_pct"])
    required = int(gate["required_positive_seed_count"])
    primary = str(gate["primary_split"])
    held_out = str(gate["held_out_geometry_split"])
    joint = str(gate["joint_ood_split"])
    val_shuffled = one(
        pairwise,
        training_arm="m4_counterfactual",
        source_split=primary,
        comparator="shuffled_geometry",
    )
    val_static = one(
        pairwise,
        training_arm="m4_counterfactual",
        source_split=primary,
        comparator="static",
    )
    held_out_shuffled = one(
        pairwise,
        training_arm="m4_counterfactual",
        source_split=held_out,
        comparator="shuffled_geometry",
    )
    joint_shuffled = one(
        pairwise,
        training_arm="m4_counterfactual",
        source_split=joint,
        comparator="shuffled_geometry",
    )
    interaction = one(
        interactions, source_split=primary, comparator="shuffled_geometry"
    )
    swap = next(
        row for row in swap_summary if row["training_arm"] == "m4_counterfactual"
    )

    def gain_pass(row: dict[str, str], minimum: float = delta) -> bool:
        return (
            float(row["mean_field_gain_pct"]) >= minimum
            and float(row["field_cluster_ci95_low_pct"]) > floor
            and int(row["positive_seed_count"]) >= required
        )

    checks = {
        "m4_correct_vs_shuffled_validation": gain_pass(val_shuffled),
        "m4_correct_vs_static_validation": gain_pass(val_static),
        "m4_minus_m1_shuffled_interaction": (
            float(interaction["mean_interaction_gain_pct"])
            >= float(gate["minimum_interaction_gain_pct"])
            and float(interaction["field_cluster_ci95_low_pct"]) > floor
            and int(interaction["positive_seed_count"]) >= required
        ),
        "m4_correct_vs_shuffled_geometry_held_out": gain_pass(held_out_shuffled),
        "joint_ood_no_material_harm": float(joint_shuffled["mean_field_gain_pct"])
        >= float(gate["minimum_joint_ood_gain_pct"]),
        "same_model_descriptor_swap_propagates": (
            float(swap["mean_correct_descriptor_field_gain_pct"])
            >= float(gate["minimum_swap_field_gain_pct"])
            and float(swap["field_cluster_ci95_low_pct"]) > floor
            and int(swap["positive_seed_count"]) >= required
            and float(swap["mean_correction_swap_relative_pct"])
            >= float(gate["minimum_correction_swap_relative_pct"])
        ),
    }
    if all(checks.values()):
        status = "COUNTERFACTUAL_DATA_MECHANISM_GATE_PASS_CONFIRMATION_NOT_AUTHORIZED"
    else:
        upper = [
            float(val_shuffled["field_cluster_ci95_high_pct"]),
            float(val_static["field_cluster_ci95_high_pct"]),
            float(interaction["field_cluster_ci95_high_pct"]),
            float(swap["field_cluster_ci95_high_pct"]),
        ]
        status = (
            "GLOBAL_GEOMETRY_MODULATION_MECHANISM_FAIL_STOP_CAPACITY_SEARCH"
            if all(value < delta for value in upper)
            else "COUNTERFACTUAL_MECHANISM_INCONCLUSIVE_DO_NOT_SCALE"
        )
    return checks, status


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / "results" / str(config["output_dir"])
    work = ROOT / "results" / str(config["work_dir"])
    dashboard = json.loads(
        (output / "v3k_a_counterfactual_dashboard.json").read_text()
    )
    report = json.loads(
        (output / "v3k_a_counterfactual_report.json").read_text()
    )
    manifest = rows(output / "v3k_a_pair_manifest.csv")
    derangement = rows(output / "v3k_a_geometry_derangement.csv")
    history = rows(output / "v3k_a_training_history.csv")
    sample = rows(output / "v3k_a_sample_metrics.csv")
    summary = rows(output / "v3k_a_split_summary.csv")
    pairwise = rows(output / "v3k_a_pairwise_mechanism.csv")
    interactions = rows(output / "v3k_a_exposure_interaction.csv")
    swap = rows(output / "v3k_a_descriptor_swap.csv")

    assert dashboard["scientific_status"] in ALLOWED_STATUSES
    assert report["status"] == dashboard["scientific_status"]
    assert dashboard["development_only"] is True
    assert dashboard["training_arms"] == config["training_arms"]
    assert dashboard["model_seeds"] == config["model_seeds"]
    assert dashboard["training_epochs"] == 24
    assert dashboard["methods"] == ["locked_fno", *config["methods"]]
    assert len(dashboard["training_records"]) == 24
    assert len(history) == 2 * 4 * 3 * 24
    assert len(sample) == 2 * 5 * 3 * (160 + 4 * 128)
    assert len(summary) == 2 * 5 * 5
    assert len(pairwise) == 2 * 5 * 4
    assert len(interactions) == 5 * 4
    assert len(swap) == 2 * 3 * 160
    assert len(derangement) == 28
    assert all(row["fixed_point"] == "False" for row in derangement)
    assert all(
        row["correct_geometry_id"] != row["wrong_geometry_id"]
        for row in derangement
    )

    training = [row for row in manifest if row["schedule_role"] == "training"]
    checkpoint = [
        row for row in manifest if row["schedule_role"] == "checkpoint_selection"
    ]
    audit = [row for row in manifest if row["schedule_role"] == "development_audit"]
    assert len(training) == 1280
    assert len(checkpoint) == 160
    assert len(audit) == 512
    train_ids = {row["geometry_id"] for row in training}
    val_ids = {row["geometry_id"] for row in checkpoint}
    audit_ids = {row["geometry_id"] for row in audit}
    assert len(train_ids) == 16
    assert len(val_ids) == 4
    assert len(audit_ids) == 8
    assert train_ids.isdisjoint(val_ids | audit_ids)
    assert val_ids.isdisjoint(audit_ids)
    for row in manifest:
        bits = row["mask_bits"]
        assert len(bits) == 9
        assert bits.count("1") == 6
        assert bits[3] == "0"

    by_arm = defaultdict(list)
    for row in training:
        by_arm[row["training_arm"]].append(row)
    assert set(by_arm) == {"m1_repeat", "m4_counterfactual"}
    for arm in by_arm:
        rows_arm = by_arm[arm]
        assert len(rows_arm) == 640
        assert len({row["source_index"] for row in rows_arm}) == 160
        assert set(Counter(row["geometry_id"] for row in rows_arm).values()) == {40}
        exposure = Counter(row["source_index"] for row in rows_arm)
        assert set(exposure.values()) == {4}
    m1_geometries = defaultdict(set)
    m4_geometries = defaultdict(set)
    m4_slot_zero = {}
    for row in by_arm["m1_repeat"]:
        m1_geometries[row["source_index"]].add(row["geometry_id"])
    for row in by_arm["m4_counterfactual"]:
        m4_geometries[row["source_index"]].add(row["geometry_id"])
        if row["geometry_slot"] == "0":
            m4_slot_zero[row["source_index"]] = row["geometry_id"]
    assert all(len(values) == 1 for values in m1_geometries.values())
    assert all(len(values) == 4 for values in m4_geometries.values())
    assert all(next(iter(values)) == m4_slot_zero[source] for source, values in m1_geometries.items())

    sample_keys = {
        (
            row["training_arm"],
            row["method"],
            row["model_seed"],
            row["source_split"],
            row["pair_index"],
        )
        for row in sample
    }
    assert len(sample_keys) == len(sample)
    history_counts = Counter(
        (row["training_arm"], row["method"], row["model_seed"])
        for row in history
    )
    assert set(history_counts.values()) == {24}
    assert len(history_counts) == 24

    recomputed_checks, recomputed_status = recompute_gate(
        config, pairwise, interactions, dashboard["same_model_descriptor_swap"]
    )
    assert dashboard["gate_checks"] == recomputed_checks
    assert dashboard["scientific_status"] == recomputed_status
    assert dashboard["counterfactual_data_mechanism_gate_pass"] == all(
        recomputed_checks.values()
    )
    assert dashboard["parameter_contract"] == {
        "adapter_total_parameters": [45226],
        "adapter_trainable_parameters": [1023],
        "parameter_matched": True,
        "base_checkpoint_frozen": True,
        "base_predictions_geometry_specific": True,
    }
    contract = dashboard["equal_exposure_contract"]
    assert contract["rows_per_training_arm"] == 640
    assert contract["independent_training_fields"] == 160
    assert contract["m1_unique_layouts_per_field"] == 1
    assert contract["m4_unique_layouts_per_field"] == 4
    assert all(
        record["checkpoint_public"] is False
        for record in dashboard["training_records"]
    )
    assert dashboard["next_decision"]["superiority_training_authorized"] is False
    assert dashboard["next_decision"]["blind_final_opened"] is False
    assert dashboard["claims_boundary"]["superiority_tested"] is False
    assert dashboard["claims_boundary"]["real_bost_geometry_present"] is False

    base_checkpoint = torch.load(
        ROOT / "results" / str(config["base_checkpoint"]),
        map_location="cpu",
        weights_only=True,
    )
    record_lookup = {
        (
            str(record["training_arm"]),
            str(record["method"]),
            int(record["model_seed"]),
        ): record
        for record in dashboard["training_records"]
    }
    for arm in config["training_arms"]:
        for method in config["methods"]:
            for seed in config["model_seeds"]:
                path = work / arm / str(seed) / "checkpoints" / f"{method}.pt"
                assert path.is_file()
                assert sha256(path) == record_lookup[(arm, method, seed)][
                    "checkpoint_sha256"
                ]
                state = torch.load(path, map_location="cpu", weights_only=True)
                for name, value in base_checkpoint.items():
                    if torch.is_tensor(value):
                        assert torch.equal(state[f"base_operator.{name}"], value)

    assert not list(output.glob("*.pt"))
    assert not list(output.glob("*.pth"))
    assert not list(output.glob("*.npz"))
    assert not list(output.glob("*.pdf"))
    assert (output / "t16_v3k_a_counterfactual_supervision.png").stat().st_size > 20_000
    for line in (output / "v3k_a_counterfactual_checksums.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha256(output / name.strip()) == expected
    assert report["provenance"]["config_sha256"] == sha256(CONFIG)
    print(
        json.dumps(
            {
                "status": "PASS",
                "scientific_result": dashboard["scientific_status"],
                "training_runs": 24,
                "training_history_rows": len(history),
                "sample_metric_rows": len(sample),
                "descriptor_swap_rows": len(swap),
                "base_checkpoint_drift": 0,
                "superiority_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
