#!/usr/bin/env python3
"""Validate the CG-PDNO engineering-smoke result bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "cg_pdno_smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report = json.loads((RESULT_DIR / "report.json").read_text(encoding="utf-8"))
    checksums = json.loads((RESULT_DIR / "checksums.json").read_text(encoding="utf-8"))
    assert report["evidence_label"] == "engineering_smoke_only"
    assert report["claim_status"] == "NOT_AUTHORIZED_FOR_SUPERIORITY"
    assert report["contract"]["learned_stop"] is False
    assert report["adjoint_relative_error"] < 1e-5
    assert report["geometry_overlap"]["train_validation"] == []
    assert report["geometry_overlap"]["development_test"] == []
    for name, expected in checksums.items():
        path = RESULT_DIR / name
        assert path.exists()
        assert sha256(path) == expected
    with (RESULT_DIR / "sample_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = 3 * (16 + 24)
    assert len(rows) == expected
    assert {row["split"] for row in rows} == {"validation", "test"}
    print(f"PASS: {len(rows)} sample rows; adjoint and disjoint-geometry checks passed")


if __name__ == "__main__":
    main()
