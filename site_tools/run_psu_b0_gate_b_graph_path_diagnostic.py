#!/usr/bin/env python3
"""Record the non-binding batch-versus-singleton graph-PCGLS diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from site_tools.run_psu_b0_factor_gate_b import (
    CHECKPOINTS,
    REPOSITORY_ROOT as RUNNER_REPOSITORY_ROOT,
    _build_runtime,
    _run_graph_replay,
    load_parent_metric_rows,
)

if RUNNER_REPOSITORY_ROOT != REPOSITORY_ROOT:
    raise RuntimeError("Gate-B diagnostic repository root mismatch")


METRICS = ("field_relative_l2", "gradient_relative_l2", "front_top10_f1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def run_diagnostic(
    *,
    config_path: Path,
    view_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version")
        != "psu-b0-factor-pdhg-gate-b-config-1.1-infrastructure-amendment"
        or config.get("status")
        != "FROZEN_PREPERFORMANCE_INFRASTRUCTURE_AMENDMENT"
    ):
        raise ValueError("graph diagnostic requires the frozen v2 config")
    device = torch.device("mps")
    operator, contexts, direction = _build_runtime(
        root=REPOSITORY_ROOT,
        config=config,
        view_root=view_root,
        device=device,
    )
    _, parent = load_parent_metric_rows(
        REPOSITORY_ROOT / config["source_paths"]["parent_metric_rows"]
    )
    difference_rows: list[dict[str, Any]] = []
    maximums: list[float] = []
    repeat_count = 3
    for diagnostic_repeat in range(repeat_count):
        single_rows, maximum = _run_graph_replay(
            operator=operator,
            contexts=contexts,
            direction=direction,
            parent_graph=parent,
            threshold=float("inf"),
            device=device,
        )
        maximums.append(maximum)
        for row in single_rows:
            key = (row["replicate"], row["sample_index"], row["iterations"])
            expected = parent[key]
            difference_rows.append(
                {
                    "diagnostic_repeat": diagnostic_repeat,
                    "replicate": row["replicate"],
                    "sample_index": row["sample_index"],
                    "reaction_family": row["reaction_family"],
                    "iterations": row["iterations"],
                    **{
                        f"singleton_{metric}": row[metric]
                        for metric in METRICS
                    },
                    **{
                        f"parent_batch_{metric}": expected[metric]
                        for metric in METRICS
                    },
                    **{
                        f"absolute_difference_{metric}": abs(
                            row[metric] - expected[metric]
                        )
                        for metric in METRICS
                    },
                }
            )
    by_metric = {
        metric: {
            "maximum_absolute_difference": max(
                row[f"absolute_difference_{metric}"] for row in difference_rows
            ),
            "mean_absolute_difference": statistics.fmean(
                row[f"absolute_difference_{metric}"] for row in difference_rows
            ),
        }
        for metric in METRICS
    }
    by_checkpoint: dict[str, Any] = {}
    for iteration in CHECKPOINTS:
        selected = [row for row in difference_rows if row["iterations"] == iteration]
        singleton_means = [
            statistics.fmean(
                row["singleton_field_relative_l2"]
                for row in selected
                if row["diagnostic_repeat"] == diagnostic_repeat
            )
            for diagnostic_repeat in range(repeat_count)
        ]
        by_checkpoint[str(iteration)] = {
            "singleton_mean_field_relative_l2_by_repeat": singleton_means,
            "singleton_mean_field_relative_l2_range": max(singleton_means)
            - min(singleton_means),
            "parent_batch_mean_field_relative_l2": statistics.fmean(
                row["parent_batch_field_relative_l2"] for row in selected
            ),
        }
    repeat_variability: dict[str, Any] = {}
    sample_keys = {
        (row["replicate"], row["sample_index"], row["iterations"])
        for row in difference_rows
    }
    for metric in METRICS:
        ranges = []
        for key in sample_keys:
            values = [
                row[f"singleton_{metric}"]
                for row in difference_rows
                if (row["replicate"], row["sample_index"], row["iterations"])
                == key
            ]
            ranges.append(max(values) - min(values))
        repeat_variability[metric] = {
            "maximum_row_range": max(ranges),
            "mean_row_range": statistics.fmean(ranges),
        }
    report = {
        "schema_version": "psu-b0-gate-b-graph-path-diagnostic-1.0",
        "status": "POSTDIAGNOSTIC_GRAPH_PATH_NONBINDING_ONLY",
        "parent_repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "diagnostic_runner_sha256": _sha256(Path(__file__).resolve()),
        "source_config_sha256": _sha256(config_path),
        "factor_solver_calls": 0,
        "factor_metric_rows_observed": 0,
        "diagnostic_repeat_count": repeat_count,
        "graph_row_count": len(difference_rows),
        "maximum_absolute_metric_difference": max(maximums),
        "differences_by_metric": by_metric,
        "singleton_repeat_variability": repeat_variability,
        "field_means_by_checkpoint": by_checkpoint,
        "samplewise_pcgls_reductions_verified_from_source": True,
        "batch_singleton_difference_interpretation": (
            "finite_precision_execution_shape_sensitivity_on_mps_inference_not_proof"
        ),
        "parent_batch_rows_authorized_as_binding_control": False,
        "single_sample_exact_k_graph_authorized_for_future_development_protocol": True,
        "algorithm_superiority_claim_authorized": False,
    }
    return report, difference_rows


def write_release(
    output: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    rows_path = output / "metric_differences.csv"
    report_path.write_bytes(_canonical_bytes(report) + b"\n")
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    (output / "checksums.sha256").write_text(
        f"{_sha256(report_path)}  report.json\n"
        f"{_sha256(rows_path)}  metric_differences.csv\n",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/configs/psu_b0_factor_pdhg_gate_b_v2_infrastructure_amendment.json",
    )
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/results/psu_b0_gate_b_graph_path_diagnostic",
    )
    args = parser.parse_args()
    report, rows = run_diagnostic(
        config_path=args.config.resolve(),
        view_root=args.view_root.resolve(),
    )
    write_release(args.output.resolve(), report, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
