#!/usr/bin/env python3
"""Validate the complete three-seed v3c K=6 dev2 pilot."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v3c_k6_dev2_pilot"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    dashboard = json.loads((RESULTS / "v3c_k6_dashboard.json").read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "v3c_k6_report.json").read_text(encoding="utf-8"))
    training = rows("v3c_k6_training.csv")
    initialization = rows("v3c_k6_initialization.csv")
    samples = rows("v3c_k6_samples.csv")
    clusters = rows("v3c_k6_clusters.csv")
    domains = rows("v3c_k6_domains.csv")
    pairwise = rows("v3c_k6_pairwise.csv")
    seed_summary = rows("v3c_k6_seed_summary.csv")
    pairwise_summary = rows("v3c_k6_pairwise_summary.csv")

    assert dashboard["full_protocol"] is True
    assert dashboard["development_fields_only"] is True
    assert dashboard["blind_final_opened"] is False
    assert dashboard["scientific_status"] in {
        "V3C_K6_DEV2_ADAPTER_GATE_PASS",
        "V3C_K6_DEV2_ADAPTER_GATE_FAIL",
    }
    assert dashboard["total_budget"] == 6
    assert dashboard["model_seed_count"] == 3
    assert dashboard["independent_test_field_count"] == 128
    assert dashboard["base_epochs"] == 24
    assert dashboard["adaptation_epochs"] == 12
    assert dashboard["decision"]["stop_current_frozen_per_view_adapter"] is True
    assert dashboard["decision"]["base_fno_training_horizon_not_saturated"] is True
    assert dashboard["decision"]["blind_final_remains_closed"] is True
    assert dashboard["cost_summary"]["adapter_to_continued_training_time_ratio"] > 1.0
    assert dashboard["cost_summary"]["adapter_to_continued_inference_time_ratio"] > 1.0
    assert len(training) == 9
    assert len(initialization) == 3
    assert len(samples) == 3 * 128 * 3
    assert len(clusters) == 128 * 3
    assert len(domains) == 4 * 3
    assert len(pairwise) == 128 * 3
    assert len(seed_summary) == 3 * 3
    assert len(pairwise_summary) == 3
    assert {row["method"] for row in samples} == {
        "base_fno",
        "continued_fno",
        "zero_init_adapter",
    }
    parameter_sets = {
        method: {
            (int(row["trainable_parameters"]), int(row["total_parameters"]))
            for row in training
            if row["method"] == method
        }
        for method in {row["method"] for row in training}
    }
    assert parameter_sets == {
        "base_fno": {(44203, 44203)},
        "continued_fno": {(44203, 44203)},
        "zero_init_adapter": {(4988, 49191)},
    }
    assert all(int(row["epochs_ran"]) == 12 for row in training if row["phase"] == "additional")
    assert all(row["continued_and_adapter_same_start"] == "True" for row in initialization)
    assert all(float(row["maximum_initial_output_difference"]) == 0.0 for row in initialization)
    assert all(float(row["maximum_adapter_base_drift_after_training"]) == 0.0 for row in initialization)
    assert all(float(row["maximum_continued_fno_drift_after_training"]) > 0.0 for row in initialization)
    assert report["protocol"]["same_K6_inputs_and_ridge_anchor"] is True
    assert report["protocol"]["same_base_checkpoint_per_seed"] is True
    assert report["protocol"]["same_additional_epochs"] is True
    assert report["protocol"]["matched_additional_epochs_not_FLOPs"] is True
    assert report["protocol"]["q_audit_used_for_training_or_selection"] is False
    assert report["protocol"]["blind_final_opened"] is False

    for line in (RESULTS / "v3c_k6_checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest() == digest
    print("T16 v3c K=6 dev2 pilot validation passed")
    print(f"sample_rows={len(samples)}")
    print(f"cluster_rows={len(clusters)}")
    print(f"scientific_status={dashboard['scientific_status']}")


if __name__ == "__main__":
    main()
