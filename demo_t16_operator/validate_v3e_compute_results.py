#!/usr/bin/env python3
"""Reject incomplete or overstated v3e compute-accounting exports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v3e_compute_accounting"
EXPECTED = {
    "ridge_unet_aug": (94193, 94193),
    "ridge_fno_aug": (44203, 44203),
    "ridge_deeponet": (49313, 49313),
    "ray_set_operator": (45973, 45973),
    "zero_init_ray_set_adapter": (49191, 4988),
}
CHECKSUM_FILES = {
    "v3e_compute_trials.csv",
    "v3e_compute_profiles.csv",
    "v3e_fno_error_compute_checkpoints.csv",
    "v3e_fno_time_to_target.csv",
    "v3e_compute_readiness.csv",
    "v3e_compute_dashboard.json",
    "v3e_compute_report.json",
    "t16_v3e_compute_accounting.png",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    dashboard = json.loads(
        (RESULTS / "v3e_compute_dashboard.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (RESULTS / "v3e_compute_report.json").read_text(encoding="utf-8")
    )
    profiles = read_csv("v3e_compute_profiles.csv")
    trials = read_csv("v3e_compute_trials.csv")
    readiness = read_csv("v3e_compute_readiness.csv")
    targets = read_csv("v3e_fno_time_to_target.csv")
    checkpoints = read_csv("v3e_fno_error_compute_checkpoints.csv")

    assert dashboard["scientific_status"] == (
        "COST_SCHEMA_COMPLETE_CROSS_ARCHITECTURE_SUPERIORITY_LOCKED"
    )
    assert dashboard["cross_architecture_superiority_gate_pass"] is False
    assert set(row["method"] for row in profiles) == set(EXPECTED)
    assert len(trials) == len(EXPECTED) * int(dashboard["worker_repeats"])
    assert len({int(row["worker_pid"]) for row in trials}) == len(trials)
    for row in profiles:
        total, trainable = EXPECTED[row["method"]]
        assert int(row["total_parameters"]) == total
        assert int(row["trainable_parameters"]) == trainable
        assert float(row["inference_p50_ms"]) > 0.0
        assert float(row["inference_p90_ms"]) >= float(row["inference_p50_ms"])
        assert float(row["training_step_p50_ms"]) > 0.0
        assert float(row["forward_estimated_flops_v1"]) > 0.0
        reconstructed = (
            2.0 * float(row["forward_dense_real_macs"])
            + 6.0 * float(row["forward_spectral_complex_macs"])
            + float(row["forward_fft_real_flops_estimate"])
        )
        assert abs(float(row["forward_estimated_flops_v1"]) - reconstructed) < 1e-6
        assert int(row["training_peak_mps_allocated_bytes_observed"]) > 0
    assert all(row["confirmatory_superiority_eligible"] == "False" for row in readiness)
    assert len(checkpoints) == 3 * 4
    assert len(targets) == 3 * 3
    assert any(row["target_reached"] == "False" for row in targets)
    assert report["protocol"]["fresh_process_per_model_repeat"] is True
    assert report["protocol"]["validation_or_test_errors_used_for_cost_ranking"] is False
    assert report["provenance"]["source_dataset_npz_public"] is False
    assert report["provenance"]["checkpoint_weights_used"] is False
    assert report["provenance"] == dashboard["provenance"]
    assert report["profiles"] == dashboard["profiles"]
    assert report["provenance"]["requirements_sha256"]
    assert len(report["provenance"]["git_source_base_commit"]) == 40
    assert report["environment"] == dashboard["environment"]
    assert report["environment"]["hardware_model"]
    assert report["environment"]["processor"]
    assert int(report["environment"]["physical_memory_bytes"]) > 0

    checksums = {}
    for line in (RESULTS / "v3e_compute_checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines():
        digest, filename = line.split("  ", 1)
        checksums[filename] = digest
    for filename, expected in checksums.items():
        actual = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
    assert set(checksums) == CHECKSUM_FILES
    print(
        "T16 v3e compute-accounting validation passed: "
        f"profiles={len(profiles)}, trials={len(trials)}, "
        f"status={dashboard['scientific_status']}"
    )


if __name__ == "__main__":
    main()
