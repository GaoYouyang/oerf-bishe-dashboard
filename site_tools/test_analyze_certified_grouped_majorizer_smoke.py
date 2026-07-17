from __future__ import annotations

import hashlib
import json
import shutil
import csv
from pathlib import Path

import pytest

from site_tools.analyze_certified_grouped_majorizer_smoke import (
    DEFAULT_INPUT, PUBLIC_FILES, ValidationError, run,
)


def _sync(root: Path, name: str) -> None:
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    lines = []
    for line in (root / "checksums.sha256").read_text(encoding="ascii").splitlines():
        old, sep, observed = line.partition("  "); assert sep
        lines.append(f"{digest if observed == name else old}  {observed}\n")
    (root / "checksums.sha256").write_text("".join(lines), encoding="ascii")


def test_public_bundle_has_expected_mixed_result_and_redaction(tmp_path: Path) -> None:
    out = tmp_path / "public"
    summary = run(DEFAULT_INPUT, out)
    assert {p.name for p in out.iterdir()} == PUBLIC_FILES
    agg = summary["aggregate"]
    assert agg["all_partition_safety_violations"] == 0
    assert agg["selector_fresh_wins"] == 4
    assert agg["selector_fresh_denominator"] == 8
    assert agg["selector_worst_harm_vs_best_fixed"] == pytest.approx(0.4144015598090733)
    assert agg["selector_mean_improvement_vs_best_fixed_percent"] == pytest.approx(10.72, abs=0.01)
    assert agg["minimum_spectral_certificate_margin_to_one"] == pytest.approx(0.06047033855698314)
    certificate = agg["safety_certificate_by_partition"]
    assert set(certificate) == {"singleton_factor", "paired_local", "paired_cross", "triad_bridge", "all_in_one_exact"}
    assert certificate["all_in_one_exact"]["max_normalized_spectral_norm_squared_over_schur_bound"] == pytest.approx(0.9395296614430169)
    assert all(row["total_safety_violations"] == 0 for row in certificate.values())
    assert all(row["audit_row_count"] == 26 for row in certificate.values())
    assert summary["claim_boundary"]["research_claim_authorized"] is False
    assert summary["aggregate"]["cost_definition"] == "ANALYTIC_PROXY_NOT_WALL_TIME"
    text = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in out.iterdir() if p.suffix in {".md", ".json", ".csv"})
    assert "/Users/" not in text and "source_commit" not in text and "geometry_seed" not in text
    assert (out / "diagnostic.png").stat().st_size > 20_000
    assert (out / "diagnostic.pdf").stat().st_size > 10_000
    with (out / "safety_summary.csv").open(newline="") as stream:
        safety_rows = list(csv.DictReader(stream))
    assert len(safety_rows) == 5
    assert all(float(row["max_normalized_spectral_norm_squared_over_schur_bound"]) < 1 for row in safety_rows)
    assert all(float(row["safety_threshold"]) == 1 for row in safety_rows)


def test_provenance_and_arithmetic_tampering_fail_closed(tmp_path: Path) -> None:
    provenance = tmp_path / "provenance"; shutil.copytree(DEFAULT_INPUT, provenance)
    report = json.loads((provenance / "report.json").read_text(encoding="utf-8")); report["provenance"]["source_worktree_dirty"] = True
    (provenance / "report.json").write_text(json.dumps(report) + "\n", encoding="utf-8"); _sync(provenance, "report.json")
    with pytest.raises(ValidationError, match="provenance"):
        run(provenance, tmp_path / "out1")
    arithmetic = tmp_path / "arithmetic"; shutil.copytree(DEFAULT_INPUT, arithmetic)
    rows = list(__import__("csv").DictReader((arithmetic / "metric_rows.csv").open(newline=""))); rows[0]["total_violation_count"] = "1"
    with (arithmetic / "metric_rows.csv").open("w", newline="") as stream:
        writer = __import__("csv").DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _sync(arithmetic, "metric_rows.csv")
    with pytest.raises(ValidationError, match="violation arithmetic"):
        run(arithmetic, tmp_path / "out2")

    certificate = tmp_path / "certificate"; shutil.copytree(DEFAULT_INPUT, certificate)
    rows = list(csv.DictReader((certificate / "partition_audit_rows.csv").open(newline="")))
    rows[0]["dense_normalized_spectral_norm_squared"] = str(2 * float(rows[0]["schur_squared_upper_bound"]))
    with (certificate / "partition_audit_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _sync(certificate, "partition_audit_rows.csv")
    with pytest.raises(ValidationError, match="certificate threshold arithmetic"):
        run(certificate, tmp_path / "out3")


def test_public_stale_file_rejected(tmp_path: Path) -> None:
    out = tmp_path / "public"; out.mkdir(); (out / "secret.txt").write_text("secret")
    with pytest.raises(ValidationError, match="public output"):
        run(DEFAULT_INPUT, out)
