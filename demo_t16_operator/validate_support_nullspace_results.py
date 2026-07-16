#!/usr/bin/env python3
"""Validate the T16 matched free/nullspace corrector result package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "configs" / "independent_dual_support.json"
DATASET_CONFIG = ROOT / "configs" / "smoke.json"
RESULTS = ROOT / "results" / "support_nullspace_corrector"
METHODS = {
    "support_fit_base",
    "free_correction",
    "nullspace_correction",
    "oracle_null_upper_bound",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    base_config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    dataset_config = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "support_nullspace_report.json").read_text(encoding="utf-8"))
    runs = read_csv("support_nullspace_runs.csv")
    samples = read_csv("support_nullspace_samples.csv")
    summary = read_csv("support_nullspace_summary.csv")
    splits = [name for name in dataset_config["splits"] if name.startswith("test_")]
    seeds = [int(value) for value in base_config["training_seeds"]]
    sample_count = sum(int(dataset_config["splits"][name]["count"]) for name in splits)
    expected_cells = sample_count * len(seeds)

    assert len(runs) == len(seeds) * len(splits) * len(METHODS)
    assert len(samples) == expected_cells * len(METHODS)
    assert len(summary) == len(splits) * len(METHODS)
    assert set(row["method"] for row in samples) == METHODS
    assert len(
        {
            (int(row["seed"]), row["split"], int(row["sample_index"]), row["method"])
            for row in samples
        }
    ) == len(samples)
    assert Counter(row["method"] for row in samples) == Counter(
        {method: expected_cells for method in METHODS}
    )
    assert set(int(row["seed"]) for row in samples) == set(seeds)

    by_key: dict[tuple[int, str, int], dict[str, dict[str, str]]] = {}
    for row in samples:
        key = (int(row["seed"]), row["split"], int(row["sample_index"]))
        by_key.setdefault(key, {})[row["method"]] = row
    assert len(by_key) == expected_cells
    for methods in by_key.values():
        assert set(methods) == METHODS
        base = methods["support_fit_base"]
        assert math.isclose(float(base["field_improvement_vs_base_pct"]), 0.0, abs_tol=1e-12)
        assert math.isclose(float(base["support_correction_leakage"]), 0.0, abs_tol=1e-12)
        assert float(methods["oracle_null_upper_bound"]["rel_l2"]) <= float(base["rel_l2"]) + 1e-10

    null_rows = [row for row in samples if row["method"] == "nullspace_correction"]
    free_rows = [row for row in samples if row["method"] == "free_correction"]
    oracle_rows = [row for row in samples if row["method"] == "oracle_null_upper_bound"]
    max_null_leakage = max(float(row["support_correction_leakage"]) for row in null_rows)
    assert max_null_leakage < 1e-6
    assert max(float(row["support_correction_leakage"]) for row in oracle_rows) < 1e-10
    assert sum(float(row["support_correction_leakage"]) for row in free_rows) / len(free_rows) > 1e-4
    assert all(float(row["correction_norm_ratio"]) <= 0.50001 for row in null_rows + free_rows)

    training = report["model"]["training_records"]
    assert len(training) == len(seeds)
    for record in training:
        assert set(record["modes"]) == {"free_correction", "nullspace_correction"}
        free_parameters = int(record["modes"]["free_correction"]["parameters"])
        null_parameters = int(record["modes"]["nullspace_correction"]["parameters"])
        assert free_parameters == null_parameters > 0
    assert report["model"]["free_and_null_correctors_parameter_matched"]
    assert report["model"]["free_and_null_correctors_identical_initialization_within_seed"]
    assert int(report["dataset"]["test_sample_seed_cells"]) == expected_cells
    assert math.isclose(
        float(report["key_findings"]["maximum_null_support_correction_leakage"]),
        max_null_leakage,
    )

    manifest = (RESULTS / "support_nullspace_checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines()
    assert len(manifest) == 4
    for line in manifest:
        expected_digest, filename = line.split("  ", 1)
        actual_digest = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, filename

    print("T16 support-nullspace corrector validation passed")
    print(f"sample_seed_cells={expected_cells}")
    print(f"sample_metric_rows={len(samples)}")
    print(f"matched_corrector_parameters={training[0]['modes']['free_correction']['parameters']}")
    print(f"maximum_null_support_leakage={max_null_leakage:.3e}")
    print("checksum_manifest=verified")


if __name__ == "__main__":
    main()
