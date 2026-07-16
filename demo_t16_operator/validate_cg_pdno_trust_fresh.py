#!/usr/bin/env python3
"""Validate the development-fresh CG-PDNO trust experiment and diagnosis."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results" / "cg_pdno_trust_fresh"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report = json.loads((RESULT / "report.json").read_text(encoding="utf-8"))
    diagnostic = json.loads(
        (RESULT / "residual_gate_diagnostic.json").read_text(encoding="utf-8")
    )
    checksums = json.loads((RESULT / "checksums.json").read_text(encoding="utf-8"))
    assert report["evidence_label"] == "development_fresh_after_trust_design"
    assert report["claim_status"] == "NOT_AUTHORIZED_FOR_SUPERIORITY"
    assert report["contract"]["learned_stop"] is False
    assert report["adjoint_relative_error"] < 1e-5
    assert report["geometry_overlap"] == {"train_validation": [], "development_test": []}
    assert report["aggregate"]["test"]["mean_relative_gain_percent"] > 0.0
    assert report["aggregate"]["test"]["maximum_harm_rate_over_1_percent"] == 0.0
    assert (
        report["guarded_aggregate"]["test"]["mean_relative_gain_percent"]
        < report["aggregate"]["test"]["mean_relative_gain_percent"]
    )
    assert diagnostic["evidence_label"] == "posthoc_development_diagnostic"
    assert diagnostic["selected_threshold"] == 0.9
    assert diagnostic["gated_validation"]["harm_rate_over_1_percent"] == 0.0
    assert {row["geometry_id"] for row in diagnostic["validation_failure_fields"]} == {
        "g_0110110",
        "g_0011110",
    }
    for name, expected in checksums.items():
        assert sha256(RESULT / name) == expected
    expected, name = (RESULT / "residual_gate_diagnostic.sha256").read_text().split()
    assert sha256(RESULT / name) == expected
    with (RESULT / "sample_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * (18 + 30)
    required = {
        "model_whitened_residual_rms",
        "fallback_whitened_residual_rms",
        "correction_to_fallback_norm",
        "trust_budget",
    }
    assert required <= set(rows[0])
    print(
        "PASS: 144 rows, disjoint geometries, hashes, negative trust result and "
        "post-hoc residual diagnosis verified"
    )


if __name__ == "__main__":
    main()
