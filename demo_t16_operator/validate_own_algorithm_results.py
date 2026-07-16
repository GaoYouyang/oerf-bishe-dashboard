#!/usr/bin/env python3
"""Validate the T16 v3b own-algorithm development benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "own_algorithm_benchmark"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    dashboard = json.loads((RESULTS / "own_algorithm_dashboard.json").read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "own_algorithm_report.json").read_text(encoding="utf-8"))
    samples = rows("own_algorithm_samples.csv")
    clusters = rows("own_algorithm_clusters.csv")
    training = rows("own_algorithm_training.csv")
    pairwise = rows("own_algorithm_pairwise.csv")
    superiority = rows("own_algorithm_superiority.csv")

    assert dashboard["development_fields_only"] is True
    assert dashboard["algorithm_name_status"] == "provisional_working_label_not_novelty_claim"
    assert dashboard["independent_test_field_count"] == 96
    assert dashboard["model_seed_count"] == 3
    assert dashboard["q_audit_used_for_training_or_selection"] is False
    assert report["protocol"]["all_neural_methods_receive_identical_ray_backprojection_channels"] is True
    assert report["protocol"]["neural_comparator_locked_by_validation_only"] is True
    assert report["protocol"]["q_audit_used_for_training_or_selection"] is False
    assert len(report["input_channels"]) == 42
    assert len(training) == 36
    assert len(samples) == 96 * 3 * 3 * 5
    assert len(clusters) == 96 * 3 * 5
    assert len(pairwise) == 96 * 3
    assert len(superiority) == 3
    assert {row["method"] for row in samples} == {
        "ridge", "ridge_unet_aug", "ridge_fno_aug", "ridge_deeponet", "ray_set_operator"
    }
    parameter_sets = {
        method: {int(row["parameters"]) for row in training if row["method"] == method}
        for method in {row["method"] for row in training}
    }
    assert parameter_sets == {
        "ridge_unet_aug": {94193},
        "ridge_fno_aug": {44203},
        "ridge_deeponet": {49313},
        "ray_set_operator": {45973},
    }
    assert all(int(row["model_seed_count"]) == 3 for row in clusters)
    assert {row["locked_neural_baseline"] for row in pairwise}.issubset(
        {"ridge_unet_aug", "ridge_fno_aug", "ridge_deeponet"}
    )

    manifest = (RESULTS / "own_algorithm_checksums.sha256").read_text(encoding="ascii").splitlines()
    for line in manifest:
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest() == digest
    print("T16 provisional own-algorithm benchmark validation passed")
    print(f"sample_rows={len(samples)}")
    print(f"cluster_rows={len(clusters)}")
    print(f"scientific_status={dashboard['scientific_status']}")


if __name__ == "__main__":
    main()
