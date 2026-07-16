#!/usr/bin/env python3
"""Validate committed equal-camera-budget T16 evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "fair_camera_budget"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = json.loads((RESULTS / "fair_camera_budget_report.json").read_text(encoding="utf-8"))
    dashboard = json.loads(
        (RESULTS / "fair_camera_budget_dashboard.json").read_text(encoding="utf-8")
    )
    samples = read_csv("fair_camera_budget_samples.csv")
    clusters = read_csv("fair_camera_budget_clusters.csv")
    summary = read_csv("fair_camera_budget_summary.csv")
    verdicts = read_csv("fair_camera_budget_verdicts.csv")
    fields = int(report["design"]["independent_test_fields"])
    seeds = int(report["design"]["model_seed_count"])
    budgets = len(report["design"]["total_budgets"])
    strategies = int(report["design"]["query_strategy_count"])
    methods = len(report["design"]["methods"])
    assert fields == 88
    assert len(samples) == fields * seeds * budgets * strategies * methods
    assert len(clusters) == fields * budgets * strategies * methods
    assert len(summary) == budgets * strategies * methods
    assert len(verdicts) == budgets
    assert len(dashboard["summary"]) == len(summary)
    assert dashboard["independent_test_fields"] == fields
    assert len(
        {
            (
                row["seed"],
                row["sample_index"],
                row["total_budget"],
                row["query_strategy"],
                row["method"],
            )
            for row in samples
        }
    ) == len(samples)
    assert all(int(row["audit_count"]) == 1 for row in samples)
    assert all(int(row["audit_index"]) == 3 for row in samples)
    assert all(int(row["query_index"]) == 4 for row in samples if row["query_strategy"] == "fixed")
    assert all(int(row["query_index"]) != int(row["audit_index"]) for row in samples)
    assert all(int(row["model_seed_count"]) == seeds for row in clusters)
    assert all(int(row["independent_field_count"]) == fields for row in summary)
    assert all(int(row["source_domain_count"]) == 5 for row in summary)
    assert max(
        abs(float(row["field_improvement_vs_support_pct"]))
        for row in samples
        if row["method"] == "support_fit_base"
    ) < 1e-8
    assert max(
        abs(float(row["field_improvement_vs_union_pct"]))
        for row in samples
        if row["method"] == "union_support_fit_direct"
    ) < 1e-8
    maximum_leakage = max(
        float(row["support_correction_leakage"])
        for row in samples
        if row["method"] in {"learned_query_correction", "numeric_query_null_update"}
    )
    assert maximum_leakage < 1e-5
    assert report["protocol"]["field_truth_in_inference"] is False
    assert "Q_audit" in report["protocol"]["audit_rule"]
    assert report["protocol"]["training_mask_matches_controlled_evaluation"] is False
    assert report["scientific_status"].startswith("PILOT_ONLY")

    manifest = (RESULTS / "fair_camera_budget_checksums.sha256").read_text(
        encoding="ascii"
    )
    for line in manifest.splitlines():
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest() == digest

    print("T16 equal-reconstruction-budget integrity validation passed")
    print(f"independent_fields={fields}")
    print(f"model_seeds={seeds}")
    print(f"method_rows={len(samples)}")
    print(f"maximum_support_leakage={maximum_leakage:.3e}")
    print(f"scientific_status={report['scientific_status']}")
    print("checksum_manifest=verified")


if __name__ == "__main__":
    main()
