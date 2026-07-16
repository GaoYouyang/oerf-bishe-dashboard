#!/usr/bin/env python3
"""Validate the committed T16 support-nullspace identifiability package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "smoke.json"
RESULTS = ROOT / "results" / "nullspace_identifiability"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "nullspace_report.json").read_text(encoding="utf-8"))
    rows = read_csv("nullspace_sample_metrics.csv")
    summary = read_csv("nullspace_layout_summary.csv")
    test_splits = [name for name in config["splits"] if name.startswith("test_")]
    expected_samples = sum(int(config["splits"][name]["count"]) for name in test_splits)
    expected_summary = sum(len(config["splits"][name]["views"]) for name in test_splits)

    assert len(rows) == expected_samples
    assert len(summary) == expected_summary
    assert len({int(row["sample_index"]) for row in rows}) == expected_samples
    assert Counter(row["split"] for row in rows) == Counter(
        {name: int(config["splits"][name]["count"]) for name in test_splits}
    )
    assert all(float(row["field_improvement_pct"]) > 0.0 for row in rows)
    assert all(float(row["query_reprojection_improvement_pct"]) > 0.0 for row in rows)
    assert max(float(row["support_projection_change_rel_to_clean"]) for row in rows) < 1e-10
    assert max(float(row["null_range_orthogonality"]) for row in rows) < 1e-10
    assert max(abs(float(row["decomposition_energy_closure"]) - 1.0) for row in rows) < 1e-9
    assert all(
        int(row["support_rank"]) + int(row["support_nullity"])
        == int(config["grid_size"]) ** 2
        for row in rows
    )
    assert set(int(row["view_count"]) for row in rows) == {3, 5, 7}
    rank_by_view: dict[int, set[int]] = {}
    for row in rows:
        rank_by_view.setdefault(int(row["view_count"]), set()).add(int(row["support_rank"]))
    assert all(len(values) == 1 for values in rank_by_view.values())
    assert all(
        float(row["oracle_null_support_clean_reprojection"])
        <= float(row["base_support_clean_reprojection"]) + 1e-10
        for row in rows
    )

    findings = report["key_findings"]
    mean_improvement = sum(float(row["field_improvement_pct"]) for row in rows) / len(rows)
    assert math.isclose(float(findings["mean_field_improvement_pct"]), mean_improvement)
    assert bool(findings["all_samples_field_improved"])
    assert int(report["dataset"]["sample_count"]) == expected_samples
    assert int(report["dataset"]["unique_support_masks"]) == 3

    manifest = (RESULTS / "nullspace_checksums.sha256").read_text(encoding="ascii").splitlines()
    assert len(manifest) == 3
    for line in manifest:
        expected_digest, filename = line.split("  ", 1)
        actual_digest = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, filename

    print("T16 nullspace identifiability validation passed")
    print(f"test_samples={expected_samples}")
    print(f"summary_cells={len(summary)}")
    print(f"rank_by_view={rank_by_view}")
    print(f"mean_field_improvement_pct={mean_improvement:.6f}")
    print("support_projection_preservation=verified")
    print("checksum_manifest=verified")


if __name__ == "__main__":
    main()
