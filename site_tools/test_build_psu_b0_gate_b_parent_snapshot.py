from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from site_tools.build_psu_b0_gate_b_parent_snapshot import (
    FIELDS,
    REPOSITORY_ROOT,
    extract_snapshot_rows,
    file_sha256,
)


SNAPSHOT = (
    REPOSITORY_ROOT
    / "demo_t16_operator/results/psu_b0_factor_pdhg_gate_b_parent_snapshot"
)
PUBLIC_SUMMARY = (
    REPOSITORY_ROOT
    / "demo_t16_operator/results/psu_b0_pdhg_scale_smoke_v2_public/summary.json"
)


def test_tracked_parent_snapshot_has_exact_provenance_and_coverage() -> None:
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    public = json.loads(PUBLIC_SUMMARY.read_text(encoding="utf-8"))
    assert manifest["status"] == "TRACKED_MINIMAL_SYNTHETIC_PARENT_SNAPSHOT"
    assert manifest["row_count"] == 128
    assert tuple(manifest["columns"]) == FIELDS
    assert manifest["contains_experimental_flow_truth"] is False
    assert manifest["contains_credentials_or_private_paths"] is False
    assert manifest["source_private_metric_rows_sha256"] == public["integrity"][
        "private_artifact_sha256_by_opaque_role"
    ]["metric_rows"]
    assert manifest["source_public_summary_sha256"] == file_sha256(PUBLIC_SUMMARY)
    assert manifest["snapshot_metric_rows_sha256"] == file_sha256(
        SNAPSHOT / "metric_rows.csv"
    )
    with (SNAPSHOT / "metric_rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert extract_snapshot_rows(rows) == rows


def test_parent_snapshot_extractor_rejects_duplicate_rows() -> None:
    with (SNAPSHOT / "metric_rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with pytest.raises(ValueError, match="duplicate"):
        extract_snapshot_rows([*rows, rows[0]])
