#!/usr/bin/env python3
"""Validate the independent dual-operator support-fit result package."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "independent_dual_support.json"
DATASET_CONFIG = ROOT / "configs" / "smoke.json"
RESULTS = ROOT / "results" / "independent_dual_support"
METHODS = {
    "residual_head",
    "absolute_head",
    "uniform_dual",
    "support_fit_mix",
    "query_router",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    dataset_config = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "dual_report.json").read_text(encoding="utf-8"))
    runs = read_csv("dual_runs.csv")
    summary = read_csv("dual_summary.csv")
    samples = read_csv("dual_sample_metrics.csv")
    regret = read_csv("dual_oracle_regret.csv")
    selection = read_csv("dual_selection_audit.csv")
    features = read_csv("dual_feature_alignment.csv")
    seeds = [int(value) for value in config["training_seeds"]]
    splits = [name for name in dataset_config["splits"] if name.startswith("test_")]
    test_samples = sum(int(dataset_config["splits"][name]["count"]) for name in splits)
    expected_cells = test_samples * len(seeds)

    assert config["expert_sharing"] == "independent"
    assert report["model"]["expert_sharing"] == "independent"
    assert len(runs) == len(seeds) * len(splits) * len(METHODS)
    assert len(summary) == len(splits) * len(METHODS)
    assert len(samples) == expected_cells * len(METHODS)
    assert len(regret) == 6
    assert len(selection) == 12
    assert len(features) == 72
    assert set(row["method"] for row in samples) == METHODS
    assert Counter(row["method"] for row in samples) == Counter(
        {method: expected_cells for method in METHODS}
    )
    assert len(
        {
            (int(row["seed"]), row["split"], int(row["sample_index"]), row["method"])
            for row in samples
        }
    ) == len(samples)
    assert all(0.0 <= float(row["router_weight"]) <= 1.0 for row in samples)
    assert all(float(row["rel_l2"]) >= 0.0 for row in samples)
    assert all(float(row["heldout_reprojection_rel_l2"]) >= 0.0 for row in samples)

    seed_runs = report["model"]["seed_runs"]
    assert len(seed_runs) == len(seeds)
    parameter_counts = {int(row["parameters"]) for row in seed_runs}
    assert len(parameter_counts) == 1
    parameters = next(iter(parameter_counts))
    assert parameters > 2 * 40_000
    assert "independent" in report["model"]["architecture"].lower()
    assert any("twice" in claim.lower() for claim in report["claims_boundary"])
    assert float(report["key_findings"]["support_fit_field_oracle_or_better_fraction"]) > 0.0

    manifest = (RESULTS / "dual_checksums.sha256").read_text(encoding="ascii").splitlines()
    assert len(manifest) == 7
    for line in manifest:
        expected_digest, filename = line.split("  ", 1)
        actual_digest = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, filename

    print("T16 independent dual support-fit validation passed")
    print(f"sample_seed_cells={expected_cells}")
    print(f"sample_metric_rows={len(samples)}")
    print(f"independent_parameters={parameters}")
    print(f"support_fit_endpoint_regret={report['key_findings']['support_fit_field_regret']:.6f}")
    print("checksum_manifest=verified")


if __name__ == "__main__":
    main()
