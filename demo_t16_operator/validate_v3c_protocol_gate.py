#!/usr/bin/env python3
"""Validate the v3c zero-init and dev2 commitment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "v3c_protocol_gate"


def main() -> None:
    report = json.loads((RESULTS / "v3c_protocol_gate.json").read_text(encoding="utf-8"))
    with (RESULTS / "v3c_dev2_split_commitments.csv").open(newline="", encoding="utf-8") as handle:
        commitments = list(csv.DictReader(handle))
    assert report["scientific_status"] == "V3C_ARCHITECTURE_GATE_PASS_BLIND_FINAL_NOT_SEALED"
    assert report["architecture_gate_pass"] is True
    assert report["zero_initialization"]["maximum_absolute_output_difference_vs_base_fno"] == 0.0
    assert report["zero_initialization"]["maximum_absolute_initial_correction"] == 0.0
    assert report["zero_initialization"]["maximum_absolute_initial_head_weight"] == 0.0
    assert report["zero_initialization"]["maximum_absolute_initial_head_bias"] == 0.0
    assert report["zero_initialization"]["stratified_exactness_sample_count"] == 36
    assert report["zero_initialization"]["maximum_frozen_base_parameter_drift"] == 0.0
    assert report["zero_initialization"]["head_gradient_norm_after_first_backward"] > 0.0
    assert report["zero_initialization"]["optimizer_steps_checked"] == 3
    assert report["zero_initialization"]["correction_l2_after_checked_steps"] > 0.0
    assert report["parameters"]["base_frozen"] is True
    assert report["parameters"]["adapter_trainable"] > 0
    assert report["dev2"]["field_count"] == 328
    assert report["dev2"]["sample_seed_overlap_with_v3b"] == 0
    assert report["dev2"]["field_content_hash_overlap_with_v3b"] == 0
    assert report["dev2"]["clean_observation_hash_overlap_with_v3b"] == 0
    assert len(commitments) == 6
    assert sum(int(row["field_count"]) for row in commitments) == 328
    assert report["blind_final"]["confirmatory_claim_allowed"] is False
    assert report["blind_final"]["seed_stored_in_repository"] is False
    for line in (RESULTS / "v3c_protocol_checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", 1)
        actual = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert actual == digest
    print("T16 v3c zero-init and dev2 protocol validation passed")
    print(f"dev2_fields={report['dev2']['field_count']}")
    print(f"adapter_trainable={report['parameters']['adapter_trainable']}")
    print(f"scientific_status={report['scientific_status']}")


if __name__ == "__main__":
    main()
