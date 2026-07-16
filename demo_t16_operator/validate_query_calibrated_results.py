#!/usr/bin/env python3
"""Validate committed query-calibrated support-nullspace evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "query_calibrated_nullspace"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    report = json.loads((RESULTS / "query_calibrated_report.json").read_text(encoding="utf-8"))
    samples = read_csv("query_calibrated_samples.csv")
    runs = read_csv("query_calibrated_runs.csv")
    summary = read_csv("query_calibrated_summary.csv")
    cells = int(report["design"]["test_sample_seed_cells"])
    methods = len(report["design"]["methods"])
    seeds = int(report["design"]["seed_count"])
    assert len(samples) == cells * methods
    assert len(runs) == seeds * 5 * methods
    assert len(summary) == 5 * methods
    assert len(
        {
            (row["seed"], row["split"], row["sample_index"], row["method"])
            for row in samples
        }
    ) == len(samples)
    assert all(0.0 <= float(row["alpha"]) <= 1.0 for row in samples)
    maximum_leakage = max(
        float(row["support_correction_leakage"])
        for row in samples
        if row["method"] == "query_line_search_all"
    )
    assert maximum_leakage < 1e-6
    assert abs(
        maximum_leakage
        - float(report["key_findings"]["maximum_support_correction_leakage"])
    ) < 1e-12

    manifest = (RESULTS / "query_calibrated_checksums.sha256").read_text(
        encoding="ascii"
    )
    for line in manifest.splitlines():
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest() == digest

    print("T16 query-calibrated nullspace validation passed")
    print(f"sample_seed_cells={cells}")
    print(f"method_sample_rows={len(samples)}")
    print(f"maximum_support_leakage={maximum_leakage:.3e}")
    print("checksum_manifest=verified")


if __name__ == "__main__":
    main()
