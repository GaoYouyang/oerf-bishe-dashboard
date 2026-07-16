#!/usr/bin/env python3
"""Independently validate the v3j negative mechanism result and privacy boundary."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "v3j_gc_sro_functional_pilot.json"
EXPECTED_STATUS = "GC_SRO_FUNCTIONAL_MECHANISM_GATE_FAIL_STOP_OR_REDESIGN"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / "results" / str(config["output_dir"])
    work = ROOT / "results" / str(config["work_dir"])
    dashboard = json.loads(
        (output / "v3j_gc_sro_functional_dashboard.json").read_text()
    )
    history = rows(output / "v3j_training_history.csv")
    sample = rows(output / "v3j_sample_metrics.csv")
    summary = rows(output / "v3j_split_summary.csv")
    pairwise = rows(output / "v3j_pairwise_mechanism.csv")
    swap = rows(output / "v3j_same_model_descriptor_swap.csv")
    derangement = rows(output / "v3j_geometry_derangement.csv")

    assert dashboard["scientific_status"] == EXPECTED_STATUS
    assert dashboard["functional_mechanism_gate_pass"] is False
    assert dashboard["development_only"] is True
    assert dashboard["model_seeds"] == config["model_seeds"]
    assert dashboard["training_epochs"] == 24
    assert dashboard["methods"] == ["locked_fno", *config["methods"]]
    assert dashboard["parameter_contract"] == {
        "adapter_total_parameters": [45226],
        "adapter_trainable_parameters": [1023],
        "parameter_matched": True,
        "base_checkpoint_frozen": True,
        "base_predictions_precomputed": True,
    }
    assert dashboard["derangement_contract"] == {
        "mapping_rows": 28,
        "fixed_points": 0,
        "within_partition": True,
        "batch_order_independent": True,
    }
    assert len(dashboard["training_records"]) == 12
    assert all(record["checkpoint_public"] is False for record in dashboard["training_records"])
    assert len(history) == 4 * 3 * 24
    assert len(sample) == 5 * 3 * 328
    assert len(summary) == 5 * 6
    assert len(pairwise) == 4 * 6
    assert len(swap) == 3 * 40
    assert len(derangement) == 28
    assert all(row["fixed_point"] == "False" for row in derangement)
    assert all(row["correct_geometry_id"] != row["wrong_geometry_id"] for row in derangement)
    assert len({(row["method"], row["model_seed"], row["source_index"]) for row in sample}) == len(sample)
    assert Counter((row["method"], row["model_seed"]) for row in history) == {
        (method, str(seed)): 24
        for method in config["methods"]
        for seed in config["model_seeds"]
    }

    gate_rows = dashboard["primary_gate_rows"]
    assert len(gate_rows) == 3
    assert {row["comparator"] for row in gate_rows} == set(
        config["functional_gate"]["required_comparators"]
    )
    assert all(row["functional_gate_pass"] is False for row in gate_rows)
    assert all(row["positive_seed_count"] == 1 for row in gate_rows)
    assert all(row["field_cluster_ci95_low_pct"] < 0 < row["field_cluster_ci95_high_pct"] for row in gate_rows)
    assert dashboard["generic_static_adapter_validation_gain_vs_locked_fno_pct"] > 3.0
    sensitivity = dashboard["same_model_descriptor_swap"]
    assert sensitivity["mean_embedding_swap_l2"] > 0.0
    assert sensitivity["mean_modulation_swap_l2"] > 0.0
    assert 0.0 < sensitivity["mean_correction_swap_relative_pct"] < 1.0
    assert abs(sensitivity["mean_correct_descriptor_field_gain_pct"]) < 0.01
    assert sensitivity["positive_seed_count"] == 1
    assert sensitivity["geometry_encoded_but_not_usefully_propagated"] is True
    assert dashboard["next_decision"]["continue_gc_sro_architecture"] is False
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
        (str(record["method"]), int(record["model_seed"])): record
        for record in dashboard["training_records"]
    }
    for method in config["methods"]:
        for seed in config["model_seeds"]:
            checkpoint_path = work / str(seed) / "checkpoints" / f"{method}.pt"
            assert checkpoint_path.is_file()
            assert sha256(checkpoint_path) == record_lookup[(method, seed)]["checkpoint_sha256"]
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            for name, value in base_checkpoint.items():
                if torch.is_tensor(value):
                    assert torch.equal(state[f"base_operator.{name}"], value)

    assert not list(output.glob("*.pt"))
    assert not list(output.glob("*.pth"))
    assert not list(output.glob("*.npz"))
    assert not list(output.glob("*.pdf"))
    assert (output / "t16_v3j_gc_sro_functional_pilot.png").stat().st_size > 20_000
    for line in (output / "v3j_gc_sro_functional_checksums.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha256(output / name.strip()) == expected
    print(
        json.dumps(
            {
                "status": "PASS",
                "scientific_result": "MECHANISM_GATE_FAIL",
                "training_runs": 12,
                "sample_metric_rows": len(sample),
                "same_model_swap_rows": len(swap),
                "base_checkpoint_drift": 0,
                "superiority_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
