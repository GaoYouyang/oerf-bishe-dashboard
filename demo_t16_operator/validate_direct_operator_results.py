#!/usr/bin/env python3
"""Integrity checks for the training-matched direct inverse-operator pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "direct_operator_pilot"
METHODS = {"physics_lift", "ridge", "unet", "fno", "ridge_unet", "ridge_fno"}
NEURAL = {"unet", "fno", "ridge_unet", "ridge_fno"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    dashboard = json.loads((RESULTS / "direct_operator_dashboard.json").read_text())
    report = json.loads((RESULTS / "direct_operator_report.json").read_text())
    tuning = read_csv("direct_operator_baseline_tuning.csv")
    training = read_csv("direct_operator_training.csv")
    samples = read_csv("direct_operator_samples.csv")
    clusters = read_csv("direct_operator_clusters.csv")
    summary = read_csv("direct_operator_summary.csv")
    domains = read_csv("direct_operator_domains.csv")
    verdicts = read_csv("direct_operator_verdicts.csv")

    budgets = {4, 6, 8}
    assert dashboard["training_mask_matches_evaluation"] is True
    assert dashboard["development_fields_only"] is True
    assert dashboard["independent_test_field_count"] == 96
    assert dashboard["model_seed_count"] == 3
    assert set(dashboard["reconstruction_budgets"]) == budgets
    assert dashboard["audit_query_index"] == 3
    assert dashboard["fixed_query_index"] == 4
    assert report["protocol"]["q_audit_used_for_training_or_selection"] is False
    assert len(report["dataset"]["input_channels"]) == 15
    assert len(report["dataset"]["ridge_residual_input_channels"]) == 15

    assert len(tuning) == 3 * 7
    assert len(training) == 3 * 3 * 4
    assert len(samples) == 3 * 96 * 3 * 6
    assert len(clusters) == 96 * 3 * 6
    assert len(summary) == 3 * 6
    assert len(domains) == 3 * 6 * 4
    assert len(verdicts) == 3 * 4
    assert {row["method"] for row in summary} == METHODS
    assert {row["method"] for row in verdicts} == NEURAL
    assert {int(row["total_budget"]) for row in summary} == budgets

    sample_counts = Counter(
        (int(row["source_index"]), int(row["total_budget"]), row["method"])
        for row in samples
    )
    assert set(sample_counts.values()) == {3}
    cluster_counts = Counter(
        (int(row["source_index"]), int(row["total_budget"])) for row in clusters
    )
    assert set(cluster_counts.values()) == {len(METHODS)}
    assert {int(row["model_seed_count"]) for row in clusters} == {3}

    for budget in budgets:
        selected = [
            row for row in tuning if int(row["total_budget"]) == budget and row["selected"] == "True"
        ]
        assert len(selected) == 1
        champion = dashboard["classical_champions"][str(budget)]
        assert selected[0]["method"] == champion
        champion_rows = [
            row
            for row in clusters
            if int(row["total_budget"]) == budget and row["method"] == champion
        ]
        assert champion_rows
        assert max(abs(float(row["improvement_vs_classical_pct"])) for row in champion_rows) < 1e-10

    for rows in (training, samples, clusters, summary, domains, verdicts):
        for row in rows:
            for value in row.values():
                if value in {"", "True", "False"}:
                    continue
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                assert math.isfinite(numeric)

    manifest = (RESULTS / "direct_operator_checksums.sha256").read_text().splitlines()
    assert len(manifest) == 9
    for line in manifest:
        expected, filename = line.split("  ", 1)
        actual = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual == expected

    assert dashboard["scientific_status"].startswith("DEVELOPMENT_PILOT_")
    assert report["scientific_status"] == dashboard["scientific_status"]
    print("T16 training-matched direct operator integrity validation passed")
    print(f"test_fields={dashboard['independent_test_field_count']}")
    print(f"model_seeds={dashboard['model_seed_count']}")
    print(f"sample_rows={len(samples)}")
    print(f"cluster_rows={len(clusters)}")
    print(f"scientific_status={dashboard['scientific_status']}")


if __name__ == "__main__":
    main()
