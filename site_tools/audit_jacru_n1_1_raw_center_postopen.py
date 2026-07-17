#!/usr/bin/env python3
"""Audit N1.1 candidates against their own raw learned proposal.

This is an explicitly post-open diagnostic.  It does not change the frozen
N1.1 decision.  It asks whether a correction that looks strong against the
matched classical references quietly harms the proposal it started from.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_1_flowoff_covariance_proximal_postopen_public"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "jacru_n1_1_raw_center_postopen_audit_public"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return float(math.fsum(materialized) / len(materialized))


def _as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected CSV boolean, received {value!r}")


def _verify_packet_checksums(source: Path) -> None:
    for line in (source / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        observed = _sha256(source / name)
        if observed != expected:
            raise RuntimeError(f"source checksum mismatch for {name}: {observed} != {expected}")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["candidate_id"]), str(row["method"]), str(row["split"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        output.append(
            {
                "candidate_id": key[0],
                "method": key[1],
                "split": key[2],
                "row_count": len(group),
                "field_gain_to_raw_mean": _mean(row["field_gain_to_raw"] for row in group),
                "h1_gain_to_raw_mean": _mean(row["h1_gain_to_raw"] for row in group),
                "field_harm_rate_vs_raw": _mean(row["field_harm_vs_raw"] for row in group),
                "worst_field_gain_to_raw": min(float(row["field_gain_to_raw"]) for row in group),
                "uses_truth": bool(group[0]["uses_truth"]),
                "uses_exact_nuisance": bool(group[0]["uses_exact_nuisance"]),
            }
        )
    return output


def _postopen_diagnostic_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["candidate_id"]), str(row["method"])), {})[
            str(row["split"])
        ] = row
    decisions: list[dict[str, Any]] = []
    for (candidate_id, method), splits in sorted(groups.items()):
        checks: dict[str, bool] = {}
        for split in ("development", "ood"):
            value = splits[split]
            checks[f"{split}_mean_nonnegative"] = float(value["field_gain_to_raw_mean"]) >= 0.0
            checks[f"{split}_harm_rate_at_most_5pct"] = float(
                value["field_harm_rate_vs_raw"]
            ) <= 0.05
            checks[f"{split}_worst_at_least_minus_5pct"] = float(
                value["worst_field_gain_to_raw"]
            ) >= -0.05
        exemplar = splits["development"]
        decisions.append(
            {
                "candidate_id": candidate_id,
                "method": method,
                "uses_truth": bool(exemplar["uses_truth"]),
                "uses_exact_nuisance": bool(exemplar["uses_exact_nuisance"]),
                "checks": checks,
                "diagnostic_passed": all(checks.values()),
                "may_authorize_method": False,
            }
        )
    return decisions


def _write_checksums(output: Path) -> None:
    files = sorted(path for path in output.iterdir() if path.name != "checksums.sha256")
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    _verify_packet_checksums(source)
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("status") != "N1_1_FLOWOFF_COVARIANCE_PROXIMAL_NO_GO":
        raise RuntimeError("raw-center audit requires the frozen formal N1.1 NO-GO packet")
    metrics = _read_csv(source / "metric_rows.csv")
    references = _read_csv(source / "reference_rows.csv")
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in references:
        if row["reference_kind"] != "raw_learned":
            raise RuntimeError("unexpected reference kind")
        key = (row["case_id"], row["method"], row["model_seed"])
        if key in lookup:
            raise RuntimeError(f"duplicate raw reference: {key}")
        lookup[key] = row
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        key = (metric["case_id"], metric["method"], metric["model_seed"])
        raw = lookup[key]
        raw_field = float(raw["field_relative_l2"])
        raw_h1 = float(raw["h1_seminorm_relative_error"])
        candidate_field = float(metric["field_relative_l2"])
        candidate_h1 = float(metric["h1_seminorm_relative_error"])
        field_gain = (raw_field - candidate_field) / raw_field
        h1_gain = (raw_h1 - candidate_h1) / raw_h1
        rows.append(
            {
                "candidate_id": metric["candidate_id"],
                "method": metric["method"],
                "model_seed": metric["model_seed"],
                "case_id": metric["case_id"],
                "split": metric["split"],
                "family": metric["family"],
                "base_seed": metric["base_seed"],
                "uses_truth": _as_bool(metric["uses_truth"]),
                "uses_exact_nuisance": _as_bool(metric["uses_exact_nuisance"]),
                "raw_field_relative_l2": raw_field,
                "candidate_field_relative_l2": candidate_field,
                "field_gain_to_raw": field_gain,
                "raw_h1_seminorm_relative_error": raw_h1,
                "candidate_h1_seminorm_relative_error": candidate_h1,
                "h1_gain_to_raw": h1_gain,
                "field_harm_vs_raw": field_gain < -0.01,
            }
        )
    if len(rows) != 1260 or len(lookup) != 180:
        raise RuntimeError("formal N1.1 row contract drifted")
    aggregates = _aggregate(rows)
    decisions = _postopen_diagnostic_decisions(aggregates)
    observable_passes = [
        row
        for row in decisions
        if row["diagnostic_passed"]
        and not row["uses_truth"]
        and not row["uses_exact_nuisance"]
    ]
    registered_harm = [
        row
        for row in metrics
        if row["candidate_id"] == "paired_structured_sensor"
        and row["split"] == "development"
        and float(row["field_gain_to_best_matched"]) < -0.01
    ]
    harmed_cases = sorted({row["case_id"] for row in registered_harm})
    summary = {
        "schema_version": "jacru-n1-1-raw-center-postopen-audit-1.0",
        "status": "POSTOPEN_RAW_CENTER_SAFETY_GAP_CONFIRMED",
        "source_status": source_summary["status"],
        "source_summary_sha256": _sha256(source / "summary.json"),
        "source_checksums_sha256": _sha256(source / "checksums.sha256"),
        "row_count": len(rows),
        "aggregate_row_count": len(aggregates),
        "decision_count": len(decisions),
        "observable_diagnostic_pass_count": len(observable_passes),
        "registered_paired_structured_development_harmed_case_ids": harmed_cases,
        "registered_paired_structured_development_harmed_row_count": len(registered_harm),
        "decisions": decisions,
        "claim_boundary": {
            "postopen_unregistered_gate": True,
            "changes_frozen_n1_1_decision": False,
            "may_authorize_algorithm": False,
            "may_claim_real_or_fresh_generalization": False,
            "purpose": "design the next pre-registered raw-center safety gate",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "raw_center_rows.csv", rows)
    _write_csv(output / "aggregate_rows.csv", aggregates)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# N1.1 raw-center post-open safety audit\n\n"
        "This diagnostic compares each N1.1 correction with the raw learned proposal "
        "that produced its center. It was defined after the formal packet was opened, "
        "so it cannot modify or strengthen the frozen N1.1 decision.\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Observable diagnostic passes: `{len(observable_passes)}`\n"
        f"- Stable registered harmed cases: `{', '.join(harmed_cases)}`\n"
        "- No method, fresh, real-data, or deployment claim is authorized.\n",
        encoding="utf-8",
    )
    _write_checksums(output)
    print(json.dumps({"status": summary["status"], "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
