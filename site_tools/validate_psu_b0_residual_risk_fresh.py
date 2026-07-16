#!/usr/bin/env python3
"""Independently validate the frozen PSU residual-risk fresh report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from site_tools.run_psu_b0_residual_risk_fresh import (
    _aggregates_with_coverage,
    _candidate_gates,
    _outside_support_equivalence,
    build_public_summary,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate(
    *,
    private_report_path: Path,
    public_summary_path: Path,
    preregistration_path: Path,
) -> dict:
    private = _load(private_report_path)
    public = _load(public_summary_path)
    preregistration = _load(preregistration_path)
    if private["configuration_private"]["preregistration_sha256"] != _sha256(
        preregistration_path
    ):
        raise AssertionError("preregistration hash drift")
    rebuilt_public = build_public_summary(private)
    if rebuilt_public != public:
        raise AssertionError("public summary is not the deterministic private export")
    rows = private["dataset_private"]["per_sample_metrics"]
    split_count = len(preregistration["fresh_splits"])
    seeds = [
        int(value)
        for value in preregistration["frozen_source"]["checkpoint_seeds"]
    ]
    expected_methods_per_split = 1 + 2 * len(seeds)
    expected_rows = (
        split_count
        * expected_methods_per_split
        * next(iter(preregistration["fresh_splits"].values()))["count"]
    )
    if len(rows) != expected_rows:
        raise AssertionError("unexpected per-sample metric row count")
    keys = {
        (row["split"], row["method"], row["sample_id"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise AssertionError("duplicate per-sample metric key")
    numeric_keys = (
        "field_relative_l2",
        "gradient_relative_l2",
        "front_top10_f1",
        "combined_loss",
        "measurement_relative_l2",
    )
    if not all(
        np.isfinite(float(row[key]))
        for row in rows
        for key in numeric_keys
    ):
        raise AssertionError("non-finite fresh metric")

    rebuilt_aggregates = _aggregates_with_coverage(rows)
    if rebuilt_aggregates != private["aggregates"]:
        raise AssertionError("aggregate table drift")
    rebuilt_equivalence = []
    for split in preregistration["fresh_gates"]["outside_support"]["splits"]:
        for seed in seeds:
            rebuilt_equivalence.append(
                _outside_support_equivalence(
                    rows,
                    split=split,
                    gated_method=f"gated_seed_{seed}",
                )
            )
    if rebuilt_equivalence != private["outside_support_equivalence"]:
        raise AssertionError("outside-support equivalence drift")
    rebuilt_candidate = _candidate_gates(
        aggregates=rebuilt_aggregates,
        equivalence=rebuilt_equivalence,
        preregistration=preregistration,
        seeds=seeds,
    )
    if rebuilt_candidate != private["candidate_gates"]:
        raise AssertionError("candidate gate drift")
    if private["status"] != "RESIDUAL_RISK_FRESH_CANDIDATE_PASS_SYNTHETIC_ONLY":
        raise AssertionError("fresh candidate did not achieve its frozen synthetic gate")
    if rebuilt_candidate["passing_seed_count"] != len(seeds):
        raise AssertionError("not every declared seed passed")
    if not all(private["gates"].values()):
        raise AssertionError("one or more protocol gates failed")
    if private["claim_boundary"]["passing_authorizes_algorithm_superiority"]:
        raise AssertionError("synthetic gate must not authorize superiority")
    if private["claim_boundary"]["passing_authorizes_experimental_field_claim"]:
        raise AssertionError("synthetic gate must not authorize experiment claims")

    baseline = {
        (row["split"], row["sample_id"]): row
        for row in rows
        if row["method"] == "sobolev_selected"
    }
    accepted_harm = []
    for row in rows:
        if not str(row["method"]).startswith("gated_seed_"):
            continue
        if not bool(row["trusted"]):
            continue
        base = baseline[(row["split"], row["sample_id"])]
        gain = 100.0 * (
            float(base["field_relative_l2"])
            - float(row["field_relative_l2"])
        ) / max(float(base["field_relative_l2"]), 1e-12)
        if gain < -1.0:
            accepted_harm.append(
                {
                    "split": row["split"],
                    "method": row["method"],
                    "sample_id": row["sample_id"],
                    "family": row["family"],
                    "noise_profile": row["noise_profile"],
                    "active_view_count": int(row["active_view_count"]),
                    "gain_percent": float(gain),
                }
            )
    if len(accepted_harm) != 4:
        raise AssertionError("accepted-harm evidence changed unexpectedly")
    return {
        "status": "PASS",
        "private_report_sha256": _sha256(private_report_path),
        "public_summary_sha256": _sha256(public_summary_path),
        "preregistration_sha256": _sha256(preregistration_path),
        "metric_rows": len(rows),
        "passing_seed_count": rebuilt_candidate["passing_seed_count"],
        "accepted_harm_over_one_percent_rows": accepted_harm,
        "claim_ceiling": "SYNTHETIC_ONLY_NO_SUPERIORITY_NO_EXPERIMENTAL_FIELD_CLAIM",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-report", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(
        private_report_path=args.private_report,
        public_summary_path=args.public_summary,
        preregistration_path=args.preregistration,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
