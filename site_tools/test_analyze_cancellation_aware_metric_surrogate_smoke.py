from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from site_tools.analyze_cancellation_aware_metric_surrogate_smoke import (
    DEFAULT_INPUT,
    PUBLIC_FILES,
    ValidationError,
    derive_public_tables,
    load_release,
    run,
)


def _rewrite_manifest_digest(root: Path, name: str) -> None:
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    lines = []
    for line in (root / "checksums.sha256").read_text(encoding="ascii").splitlines():
        old_digest, separator, observed_name = line.partition("  ")
        assert separator
        lines.append(f"{digest if observed_name == name else old_digest}  {observed_name}\n")
    (root / "checksums.sha256").write_text("".join(lines), encoding="ascii")


def test_canonical_release_recomputes_mean_per_rig_safety_mass_and_calls() -> None:
    report, metrics, predictions, _ = load_release(DEFAULT_INPUT)
    methods, per_rig, gates, aggregate = derive_public_tables(report, metrics, predictions)
    assert len(methods) == 6
    assert len(per_rig) == 24
    assert len(gates) == 7
    assert aggregate["calibrated_vs_factor_mean_field_gain_percent"] == pytest.approx(-15342970.167630501, rel=1e-12)
    assert aggregate["calibrated_vs_scalar_mean_field_gain_percent"] == pytest.approx(-17591406.6688204, rel=1e-12)
    assert aggregate["calibrated_stable_field_win_rig_count"] == 2
    assert aggregate["calibrated_unsafe_rig_count"] == 4
    assert aggregate["calibrated_total_violation_count"] == 39
    calibrated = next(row for row in methods if row["method"] == "calibrated_envelope")
    learned = next(row for row in methods if row["method"] == "learned_oracle_free")
    factor = next(row for row in methods if row["method"] == "factor")
    assert calibrated["mean_final_field_relative_l2"] == pytest.approx(151737.30229749542)
    assert learned["mean_final_field_relative_l2"] == pytest.approx(2.180473359157691e26)
    assert factor["mass_coverage_fraction"] == pytest.approx(1.0)
    assert calibrated["mass_coverage_fraction"] < 1.0
    assert calibrated["signed_forward_solver_calls"] == 128
    assert calibrated["signed_transpose_solver_calls"] == 128
    assert calibrated["factor_mass_vector_accesses"] == 8
    assert calibrated["factor_feature_construction_calls"] == 4
    duplicate = next(row for row in methods if row["method"] == "exact_factor_interpolation_oracle")
    assert duplicate["duplicate_of"] == "exact_oracle"
    assert duplicate["counts_as_independent_evidence"] is False
    assert aggregate["calibrated_field_gain_vs_scalar_percent_by_rig"]["ood-00"] > 0
    assert aggregate["calibrated_field_gain_vs_scalar_percent_by_rig"]["ood-01"] < 0


def test_run_generates_only_declared_bundle_without_mutating_source(tmp_path: Path) -> None:
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in DEFAULT_INPUT.iterdir() if path.is_file()}
    output = tmp_path / "public"
    summary = run(DEFAULT_INPUT, output)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in DEFAULT_INPUT.iterdir() if path.is_file()}
    assert before == after
    assert {path.name for path in output.iterdir()} == PUBLIC_FILES
    assert summary["status"] == "V2_NO_AUTH"
    assert summary["claim_boundary"]["iid_sample_claimed"] is False
    assert summary["claim_boundary"]["statistical_inference_performed"] is False
    assert summary["source_decision"]["metric_substitution_authorized"] is False
    assert summary["source_decision"]["research_claim_authorized"] is False
    assert summary["source_provenance"]["source_snapshot_status"] == "COMMITTED_CLEAN_REPRODUCIBLE_FROM_COMMIT"
    assert summary["cost_contract"]["learned_factor_mass_vector_accesses"] == 8
    assert summary["cost_contract"]["calibrated_factor_feature_construction_calls"] == 4
    assert summary["cost_contract"]["end_to_end_cost_reduction_claimed"] is False
    assert (output / "diagnostic.png").stat().st_size > 30_000
    assert (output / "diagnostic.pdf").stat().st_size > 20_000
    with (output / "fresh_rig_comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert sum(row["calibrated_beats_factor_and_scalar_on_field"] == "True" for row in rows) == 2


def test_decision_and_arithmetic_tampering_fail_closed_even_with_synced_manifest(tmp_path: Path) -> None:
    decision_root = tmp_path / "decision"
    shutil.copytree(DEFAULT_INPUT, decision_root)
    report = json.loads((decision_root / "report.json").read_text(encoding="utf-8"))
    report["decision"]["metric_substitution_authorized"] = True
    (decision_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_digest(decision_root, "report.json")
    with pytest.raises(ValidationError, match="metric substitution decision drift"):
        load_release(decision_root, enforce_frozen_hashes=False)
    with pytest.raises(ValidationError, match="frozen source hash mismatch"):
        load_release(decision_root)

    arithmetic_root = tmp_path / "arithmetic"
    shutil.copytree(DEFAULT_INPUT, arithmetic_root)
    rows = list(csv.DictReader((arithmetic_root / "metric_rows.csv").open(newline="")))
    rows[0]["final_field_relative_l2"] = "0.1"
    with (arithmetic_root / "metric_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _rewrite_manifest_digest(arithmetic_root, "metric_rows.csv")
    with pytest.raises(ValidationError, match="final field|source arithmetic drift"):
        load_release(arithmetic_root, enforce_frozen_hashes=False)
    with pytest.raises(ValidationError, match="frozen source hash mismatch"):
        load_release(arithmetic_root)


def test_file_set_and_hash_tampering_fail_closed(tmp_path: Path) -> None:
    file_set_root = tmp_path / "file-set"
    shutil.copytree(DEFAULT_INPUT, file_set_root)
    (file_set_root / "unexpected.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="source file set drift"):
        load_release(file_set_root)

    hash_root = tmp_path / "hash"
    shutil.copytree(DEFAULT_INPUT, hash_root)
    manifest = (hash_root / "checksums.sha256").read_text(encoding="ascii")
    (hash_root / "checksums.sha256").write_text("0" + manifest[1:], encoding="ascii")
    with pytest.raises(ValidationError, match="source checksum mismatch"):
        load_release(hash_root)


def test_factor_feature_ledger_and_dirty_provenance_tampering_fail_closed(tmp_path: Path) -> None:
    ledger_root = tmp_path / "ledger"
    shutil.copytree(DEFAULT_INPUT, ledger_root)
    report = json.loads((ledger_root / "report.json").read_text(encoding="utf-8"))
    report["call_ledger"]["fresh_by_method"]["calibrated_envelope"]["factor_mass_vector_accesses"] = 0
    (ledger_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_digest(ledger_root, "report.json")
    with pytest.raises(ValidationError, match="factor-feature cost ledger drift"):
        load_release(ledger_root, enforce_frozen_hashes=False)

    provenance_root = tmp_path / "provenance"
    shutil.copytree(DEFAULT_INPUT, provenance_root)
    report = json.loads((provenance_root / "report.json").read_text(encoding="utf-8"))
    report["provenance"]["source_snapshot_status"] = "UNCOMMITTED_SOURCE_SNAPSHOT_NOT_REPRODUCIBLE_FROM_COMMIT_ALONE"
    (provenance_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_digest(provenance_root, "report.json")
    with pytest.raises(ValidationError, match="source provenance boundary drift"):
        load_release(provenance_root, enforce_frozen_hashes=False)


def test_public_export_rejects_stale_file(tmp_path: Path) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "stale.txt").write_text("stale\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="unexpected stale public files"):
        run(DEFAULT_INPUT, output)
