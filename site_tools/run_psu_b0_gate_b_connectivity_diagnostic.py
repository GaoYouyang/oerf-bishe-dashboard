#!/usr/bin/env python3
"""Audit production A-only support connectivity without running a solver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from demo_t16_operator.psu_b0_gate_b_data_only import (  # noqa: E402
    build_single_sample_factor_setup,
)
from site_tools.run_psu_b0_factor_gate_b import (  # noqa: E402
    _build_runtime,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(repr((tuple(array.shape), str(array.dtype))).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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
        != "psu-b0-factor-pdhg-gate-b-config-1.2-graph-comparator-amendment"
        or config.get("status")
        != "FROZEN_POST_GRAPH_DIAGNOSTIC_PRE_FACTOR_PROTOCOL"
    ):
        raise ValueError("connectivity diagnostic requires the frozen v3 config")
    device = torch.device("mps")
    operator, contexts, _ = _build_runtime(
        root=REPOSITORY_ROOT,
        config=config,
        view_root=view_root,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    for context in contexts:
        for sample_index, family in enumerate(context["families"]):
            setup = build_single_sample_factor_setup(
                voxel_operator=operator,
                source_whitening=context["graph_operator"].whitening,
                sample_index=sample_index,
                measurement_scale=float(context["measurement_scale"]),
                eta=float(config["solver"]["eta"]),
            )
            positive = setup.data_column_sums[setup.data_column_sums > 0.0]
            rows.append(
                {
                    "replicate": int(context["replicate"]),
                    "sample_index": sample_index,
                    "reaction_family": str(family),
                    "support_active_voxel_count": int(setup.pipeline.n_active),
                    "data_coupled_voxel_count": int(setup.active_primal_count),
                    "data_null_support_voxel_count": int(
                        setup.pipeline.n_active - setup.active_primal_count
                    ),
                    "active_data_row_count": int(setup.data_row_mask.sum().cpu()),
                    "active_primal_indices_sha256": _tensor_sha256(
                        setup.active_primal_indices
                    ),
                    "minimum_positive_data_column_sum": float(positive.amin().cpu()),
                    "maximum_data_column_sum": float(positive.amax().cpu()),
                    "absolute_data_forward_setup_calls": int(
                        setup.setup_call_ledger.absolute_data_forward_calls
                    ),
                    "absolute_data_transpose_setup_calls": int(
                        setup.setup_call_ledger.absolute_data_transpose_calls
                    ),
                    "signed_data_solver_calls": 0,
                    "tv_setup_or_solver_calls": 0,
                }
            )
    masks = {row["active_primal_indices_sha256"] for row in rows}
    report = {
        "schema_version": "psu-b0-gate-b-a-only-connectivity-diagnostic-1.0",
        "status": "SETUP_ONLY_A_CONNECTIVITY_FIXED_BEFORE_FACTOR_SOLVER",
        "parent_repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "diagnostic_runner_sha256": _sha256(Path(__file__).resolve()),
        "source_config_sha256": _sha256(config_path),
        "sample_count": len(rows),
        "support_active_voxel_count": sorted(
            {row["support_active_voxel_count"] for row in rows}
        ),
        "data_coupled_voxel_count": sorted(
            {row["data_coupled_voxel_count"] for row in rows}
        ),
        "data_null_support_voxel_count": sorted(
            {row["data_null_support_voxel_count"] for row in rows}
        ),
        "active_data_row_count": sorted(
            {row["active_data_row_count"] for row in rows}
        ),
        "unique_active_primal_mask_count": len(masks),
        "active_primal_indices_sha256": sorted(masks),
        "factor_setup_count": len(rows),
        "factor_solver_calls": 0,
        "factor_metric_rows_observed": 0,
        "truth_scoring_performed": False,
        "a_plus_d_count_must_not_be_reused_as_a_only_count": True,
        "graph_full_support_sobolev_extrapolation_disclosed": True,
        "algorithm_superiority_claim_authorized": False,
    }
    return report, rows


def write_release(
    output: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    rows_path = output / "connectivity_rows.csv"
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
        f"{_sha256(rows_path)}  connectivity_rows.csv\n",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/configs/psu_b0_factor_pdhg_gate_b_v3_graph_comparator_amendment.json",
    )
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/results/psu_b0_gate_b_connectivity_diagnostic",
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
