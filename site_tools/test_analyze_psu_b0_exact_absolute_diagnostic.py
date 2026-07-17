from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from site_tools.analyze_psu_b0_exact_absolute_diagnostic import (
    REPOSITORY_ROOT, ValidationError, derive_public_tables, load_release, run,
)


SOURCE = REPOSITORY_ROOT / "demo_t16_operator/results/psu_b0_exact_absolute_root_cause"


def _rewrite_report_checksum(root: Path) -> None:
    report_digest = hashlib.sha256((root / "report.json").read_bytes()).hexdigest()
    manifest = root / "checksums.sha256"
    lines = []
    for line in manifest.read_text(encoding="ascii").splitlines():
        _, separator, name = line.partition("  ")
        assert separator
        digest = report_digest if name == "report.json" else line.split("  ", 1)[0]
        lines.append(f"{digest}  {name}\n")
    manifest.write_text("".join(lines), encoding="ascii")


def test_release_tables_expose_residual_field_gap_and_nonbinding_boundary() -> None:
    report, trajectory, tightness, _ = load_release(SOURCE)
    frontier, gains, _, decision = derive_public_tables(
        trajectory,
        tightness,
        report["decision"],
    )
    assert len(frontier) == 36
    assert len(gains) == 48
    assert decision["opened_field_count"] == 16
    assert decision["exact_abs_row_paired_mean_residual_gain_percent"] == pytest.approx(64.97, abs=0.1)
    assert decision["exact_abs_row_ratio_of_means_residual_gain_percent"] == pytest.approx(64.183111, abs=1e-5)
    assert decision["exact_abs_row_paired_mean_field_gain_percent"] < 6.0
    assert decision["exact_abs_row_ratio_of_means_field_gain_percent"] == pytest.approx(4.82846, abs=1e-4)
    assert decision["replicate_cluster_count"] == 2
    assert decision["iid_field_count_claimed"] is False
    assert decision["exact_abs_row_descriptive_mean_minimum_evaluated_checkpoint"] == 64
    assert decision["exact_abs_row_mean_k128_gt_k64_descriptive"] is True
    assert decision["exact_abs_row_k128_gt_k64_opened_row_count"] == 10
    graph = next(row for row in frontier if row["method"] == "graph_pcgls")
    assert graph["comparison_binding"] is False
    assert graph["support_contract"] == "full support"
    assert graph["prior_contract"] == "Sobolev graph prior"


def test_run_emits_public_bundle_and_detects_input_checksum_drift(tmp_path: Path) -> None:
    copied = tmp_path / "release"
    shutil.copytree(SOURCE, copied)
    output = tmp_path / "public"
    summary = run(copied, output)
    assert summary["claim_boundary"]["new_algorithm_claimed"] is False
    assert summary["claim_boundary"]["formal_gate_b_reopened"] is False
    assert summary["claim_boundary"]["solver_recurrence_operator"] == "SIGNED_A"
    assert summary["claim_boundary"]["causal_cancellation_mechanism_proved"] is False
    assert summary["statistical_contract"]["iid_rows_claimed"] is False
    assert summary["known_confounders"]["synthetic_view_scaling_uses_clean_truth"] is True
    assert (output / "diagnostic.png").stat().st_size > 10_000
    assert (output / "diagnostic.pdf").stat().st_size > 10_000
    with (output / "paired_k128_gains.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 48
    (copied / "trajectory_rows.csv").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="checksum mismatch"):
        load_release(copied)


def test_release_rejects_synced_claim_boundary_and_formal_arithmetic_drift(
    tmp_path: Path,
) -> None:
    boundary_copy = tmp_path / "boundary"
    shutil.copytree(SOURCE, boundary_copy)
    report = json.loads((boundary_copy / "report.json").read_text(encoding="utf-8"))
    report["claim_boundary"]["formal_gate_b_reopened"] = True
    (boundary_copy / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_report_checksum(boundary_copy)
    with pytest.raises(ValidationError, match="claim boundary drift"):
        load_release(boundary_copy)

    arithmetic_copy = tmp_path / "arithmetic"
    shutil.copytree(SOURCE, arithmetic_copy)
    report = json.loads((arithmetic_copy / "report.json").read_text(encoding="utf-8"))
    report["decision"]["mean_normalized_residual_gain_percent"]["exact_abs_row"] = 1.0
    (arithmetic_copy / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_report_checksum(arithmetic_copy)
    with pytest.raises(ValidationError, match="formal decision arithmetic drift"):
        run(arithmetic_copy, tmp_path / "arithmetic-public")


def test_public_export_rejects_undeclared_stale_files(tmp_path: Path) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "stale.txt").write_text("stale\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="unexpected stale public files"):
        run(SOURCE, output)
