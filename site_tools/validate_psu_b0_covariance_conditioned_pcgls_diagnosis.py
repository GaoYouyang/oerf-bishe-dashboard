#!/usr/bin/env python3
"""Independently validate the covariance-conditioned PCGLS diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from site_tools.run_psu_b0_covariance_conditioned_pcgls_diagnosis import (
    _sha256,
    aggregate_diagnosis,
    candidate_grid,
    select_candidate,
    validate_execution_plan,
    validate_partition,
)


VALIDATION_SCHEMA = (
    "psu-b0-covariance-conditioned-pcgls-diagnosis-validation-1.0"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(
    *,
    root: Path,
    config_path: Path,
    report_path: Path,
    metric_path: Path,
    replicate_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metric_rows = _read_csv(metric_path)
    replicate_rows_file = _read_csv(replicate_path)
    candidates = candidate_grid(config)
    replicate_count = len(report["replicate_ledger"])
    partition = validate_partition(
        config,
        replicate_count=replicate_count,
    )
    budget = validate_execution_plan(
        config,
        candidates=candidates,
        replicate_count=replicate_count,
    )
    baseline_id = str(report["baseline"]["candidate_id"])
    expected_methods = len(candidates) + 1
    reaction_count = len(
        json.loads(
            (root / str(config["source_smoke_config"])).read_text(
                encoding="utf-8"
            )
        )["data"]["reaction_families"]
    )
    expected_metric_rows = (
        replicate_count * expected_methods * reaction_count
    )
    expected_replicate_rows = replicate_count * expected_methods

    recomputed_replicates, summaries, _ = aggregate_diagnosis(
        metric_rows,
        baseline_id=baseline_id,
        partition=partition,
        gates=config["selection_gates"],
    )
    selected = select_candidate(
        [
            row for row in summaries if row["split"] == "selection"
        ],
        gates=config["selection_gates"],
        eligibility=config["candidate_eligibility"],
    )
    reported_selected = report["decision"]["selection_candidate"]
    finite_metrics = all(
        np.isfinite(float(row[key]))
        for row in metric_rows
        for key in (
            "field_relative_l2",
            "gradient_relative_l2",
            "front_top10_f1",
        )
    )
    checks = {
        "config_sha256_matches": report["config_sha256"]
        == _sha256(config_path),
        "source_multiseed_config_sha256_matches": (
            report["source_multiseed_config_sha256"]
            == _sha256(root / str(config["source_multiseed_config"]))
        ),
        "source_multiseed_report_sha256_matches": (
            report["source_multiseed_report_sha256"]
            == _sha256(root / str(config["source_multiseed_report"]))
        ),
        "source_smoke_config_sha256_matches": (
            report["source_smoke_config_sha256"]
            == _sha256(root / str(config["source_smoke_config"]))
        ),
        "metric_row_count_matches": len(metric_rows)
        == expected_metric_rows,
        "replicate_row_count_matches": len(replicate_rows_file)
        == expected_replicate_rows
        == len(recomputed_replicates),
        "all_metrics_finite": finite_metrics,
        "candidate_identifiers_match": (
            {row["candidate_id"] for row in candidates}
            | {baseline_id}
        )
        == {row["candidate_id"] for row in metric_rows},
        "logical_call_total_matches": int(
            report["execution_optimization"][
                "observed_logical_forward_and_adjoint_calls_total"
            ]
        )
        == int(budget["logical_calls_total"]),
        "physical_call_total_matches": int(
            report["execution_optimization"][
                "observed_physical_forward_and_adjoint_calls_total"
            ]
        )
        == int(budget["physical_calls_total"]),
        "selection_identifier_reproduces": (
            selected is not None
            and reported_selected is not None
            and selected["candidate_id"]
            == reported_selected["candidate_id"]
        ),
        "selection_mean_reproduces": (
            selected is not None
            and reported_selected is not None
            and np.isclose(
                float(selected["mean_field_gain_percent"]),
                float(reported_selected["mean_field_gain_percent"]),
                rtol=0.0,
                atol=1e-12,
            )
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": (
            "VALIDATION_PASS"
            if all(checks.values())
            else "VALIDATION_FAIL"
        ),
        "checks": checks,
        "candidate_count": len(candidates),
        "replicate_count": replicate_count,
        "metric_row_count": len(metric_rows),
        "replicate_summary_row_count": len(replicate_rows_file),
        "recomputed_selected_candidate": (
            None if selected is None else selected["candidate_id"]
        ),
        "validated_call_budget": budget,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--metric-csv", type=Path, required=True)
    parser.add_argument("--replicate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = validate(
        root=root,
        config_path=args.config.resolve(),
        report_path=args.report.resolve(),
        metric_path=args.metric_csv.resolve(),
        replicate_path=args.replicate_csv.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "VALIDATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
