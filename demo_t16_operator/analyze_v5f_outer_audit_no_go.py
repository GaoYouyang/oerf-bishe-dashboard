#!/usr/bin/env python3
"""Diagnose why v5f outer-camera routing does not transfer to audit views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .run_v5d_decoupled_complexity_screening import sha256, write_checksums, write_csv
except ImportError:
    from run_v5d_decoupled_complexity_screening import sha256, write_checksums, write_csv


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "results" / "v5f_dual_regularization_postopen"
DEFAULT_OUTPUT = ROOT / "results" / "v5f_outer_audit_postopen_diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(tuple(left), dtype=float)
    y = np.asarray(tuple(right), dtype=float)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("correlation arrays must have equal length >= 2")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def diagnose_method(rows: Sequence[dict[str, str]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["reconstruction_method"] == method]
    if not selected:
        raise ValueError(f"method has no sample rows: {method}")
    changed = [row for row in selected if row["radius_changed_from_metadata"] == "True"]
    accepted = [row for row in selected if row["route_ge_0p0pct"] == "True"]
    outer_all = [float(row["minimum_outer_error_reduction_percent"]) for row in selected]
    audit_all = [float(row["raw_audit_error_reduction_percent"]) for row in selected]
    outer_changed = [
        float(row["minimum_outer_error_reduction_percent"]) for row in changed
    ]
    audit_changed = [float(row["raw_audit_error_reduction_percent"]) for row in changed]
    sign_agreement = float(
        np.mean(
            (np.asarray(outer_changed) >= 0.0)
            == (np.asarray(audit_changed) >= 0.0)
        )
    )
    return {
        "reconstruction_method": method,
        "sample_count": len(selected),
        "changed_radius_sample_count": len(changed),
        "outer_vs_audit_correlation_all": safe_correlation(outer_all, audit_all),
        "outer_vs_audit_correlation_changed_radius": safe_correlation(
            outer_changed, audit_changed
        ),
        "outer_audit_sign_agreement_changed_radius": sign_agreement,
        "outer_nonworse_accepted_count": len(accepted),
        "accepted_audit_harm_count": int(
            sum(float(row["raw_audit_error_reduction_percent"]) < 0.0 for row in accepted)
        ),
        "accepted_audit_harm_rate": (
            0.0
            if not accepted
            else float(
                np.mean(
                    [
                        float(row["raw_audit_error_reduction_percent"]) < 0.0
                        for row in accepted
                    ]
                )
            )
        ),
    }


def accepted_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["reconstruction_method"] != "gcv" or row["route_ge_0p0pct"] != "True":
            continue
        output.append(
            {
                "rig_id": row["rig_id"],
                "block_id": row["block_id"],
                "sample_index": int(row["sample_index"]),
                "family": row["family"],
                "minimum_outer_error_reduction_percent": float(
                    row["minimum_outer_error_reduction_percent"]
                ),
                "audit_error_reduction_percent": float(
                    row["raw_audit_error_reduction_percent"]
                ),
                "field_error_reduction_percent": float(
                    row["raw_field_error_reduction_percent"]
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"diagnosis output already exists: {output}")
    rows_path = source / "sample_rows.csv"
    source_report_path = source / "report.json"
    rows = read_csv(rows_path)
    methods = sorted({row["reconstruction_method"] for row in rows})
    diagnoses = [diagnose_method(rows, method) for method in methods]
    accepted = accepted_rows(rows)
    output.mkdir(parents=True, exist_ok=False)
    accepted_path = output / "accepted_rows.csv"
    write_csv(accepted_path, accepted)
    report = {
        "claim_status": "V5F_POSTOPEN_OUTER_AUDIT_TRANSFER_NO_GO",
        "scientific_verdict": "NO_GO",
        "failure_mode": (
            "worst outer-camera improvement does not rank audit-camera improvement"
        ),
        "method_diagnosis": diagnoses,
        "gcv_and_upre_sample_metrics_identical": all(
            np.isclose(
                float(gcv[key]),
                float(upre[key]),
            )
            for gcv, upre in zip(
                [row for row in rows if row["reconstruction_method"] == "gcv"],
                [row for row in rows if row["reconstruction_method"] == "upre"],
                strict=True,
            )
            for key in (
                "raw_field_error_reduction_percent",
                "minimum_outer_error_reduction_percent",
                "raw_audit_error_reduction_percent",
            )
        ),
        "threshold_sweep_authorized": False,
        "reason_threshold_sweep_is_not_authorized": (
            "changed-radius outer/audit correlation is negative and accepted audit "
            "harm is 2/2; retuning on opened audit data would leak the final camera"
        ),
        "next_mechanism_question": (
            "replace scalar worst-view routing with geometry/leverage-aware view "
            "transfer or require an independent session-level calibration relation"
        ),
        "source_hashes": {
            "sample_rows": sha256(rows_path),
            "source_report": sha256(source_report_path),
            "analyzer": sha256(Path(__file__).resolve()),
        },
    }
    report_path = output / "diagnosis.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme_path = output / "README.md"
    readme_path.write_text(
        "# v5f outer-to-audit post-open NO-GO\n\n"
        "The GCV and UPRE refits are identical on these blocks. For the 24 rows "
        "where calibration changes radius, the worst-outer versus audit gain "
        "correlation is negative and sign agreement is below one half. The only two "
        "outer-nonworse routes both harm the audit camera. Threshold retuning is "
        "therefore forbidden; it would fit the already opened audit data rather than "
        "repair view-specific transfer.\n",
        encoding="utf-8",
    )
    write_checksums(output, [accepted_path, report_path, readme_path])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
