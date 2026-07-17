from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from site_tools.run_psu_b0_factor_gate_b import REPOSITORY_ROOT


RESULTS = (
    REPOSITORY_ROOT
    / "demo_t16_operator/results/psu_b0_gate_b_connectivity_diagnostic"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_connectivity_diagnostic_freezes_one_shared_a_only_gauge_without_solver_calls() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    with (RESULTS / "connectivity_rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    declared = {}
    for line in (RESULTS / "checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        declared[name] = digest
    assert declared == {
        "report.json": _sha256(RESULTS / "report.json"),
        "connectivity_rows.csv": _sha256(RESULTS / "connectivity_rows.csv"),
    }
    assert report["status"] == "SETUP_ONLY_A_CONNECTIVITY_FIXED_BEFORE_FACTOR_SOLVER"
    assert report["sample_count"] == len(rows) == 16
    assert report["support_active_voxel_count"] == [2744]
    assert report["data_coupled_voxel_count"] == [2322]
    assert report["data_null_support_voxel_count"] == [422]
    assert report["active_data_row_count"] == [4608]
    assert report["unique_active_primal_mask_count"] == 1
    assert report["factor_setup_count"] == 16
    assert report["factor_solver_calls"] == 0
    assert report["factor_metric_rows_observed"] == 0
    assert report["truth_scoring_performed"] is False
    assert {int(row["absolute_data_forward_setup_calls"]) for row in rows} == {1}
    assert {int(row["absolute_data_transpose_setup_calls"]) for row in rows} == {1}
    assert {int(row["signed_data_solver_calls"]) for row in rows} == {0}
    assert {int(row["tv_setup_or_solver_calls"]) for row in rows} == {0}
    assert len({row["active_primal_indices_sha256"] for row in rows}) == 1
