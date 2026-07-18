from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from site_tools.validate_n2_pvgr_n5_d4c_msra_development import (
    DEFAULT_CONFIG,
    DEFAULT_RESULT,
    validate,
)


def test_committed_d4c_development_artifact_validates() -> None:
    report = validate(DEFAULT_CONFIG, DEFAULT_RESULT)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["threshold_selected"] is None
    assert report["counts"]["synthetic_base_rows"] == 3600
    assert report["counts"]["threshold_probe_rows"] == 36000
    assert report["claim_boundary_valid"] is True
    assert report["historical_d4b_decision_retained"] is True


def test_validator_detects_recomputed_gate_drift(tmp_path: Path) -> None:
    copied = tmp_path / "result"
    shutil.copytree(DEFAULT_RESULT, copied)
    csv_path = copied / "threshold_probe_rows.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["gamma_gate"] = "False" if rows[0]["gamma_gate"] == "True" else "True"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["threshold_probe_rows.csv"]["bytes"] = csv_path.stat().st_size
    import hashlib

    manifest["artifacts"]["threshold_probe_rows.csv"]["sha256"] = hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = validate(DEFAULT_CONFIG, copied)

    assert report["valid"] is False
    assert any("recomputed gamma_gate drifted" in error for error in report["errors"])
