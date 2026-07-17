#!/usr/bin/env python3
"""Build a sanitized public audit of the deterministic factor Gate B result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_STATUS = "GATE_B_E2_MECHANISM_NO_GO"
ITERATIONS = (4, 8, 16, 32)
METHODS = (
    "graph_pcgls",
    "scalar_a_only_pdhg",
    "view_block_a_only_pdhg",
    "voxel_factor_a_only_pdhg",
)
METHOD_LABELS = {
    "graph_pcgls": "Graph PCGLS",
    "scalar_a_only_pdhg": "Scalar PDHG",
    "view_block_a_only_pdhg": "View-block PDHG",
    "voxel_factor_a_only_pdhg": "Voxel-factor PDHG",
}
COLORS = {
    "graph_pcgls": "#167D70",
    "scalar_a_only_pdhg": "#B44C43",
    "view_block_a_only_pdhg": "#416C9B",
    "voxel_factor_a_only_pdhg": "#9A6B19",
    "pass": "#2F855A",
    "fail": "#C53030",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _verify_checksum_file(directory: Path, filename: str) -> dict[str, str]:
    checksum_path = directory / filename
    if not checksum_path.is_file():
        raise ValueError(f"missing checksum file: {filename}")
    verified: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, member = line.split("  ", 1)
        if Path(member).name != member:
            raise ValueError("checksum member must be a plain filename")
        path = directory / member
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"checksum mismatch: {member}")
        verified[member] = digest
    return verified


def verify_input_bundle(input_dir: Path) -> dict[str, str]:
    verified = _verify_checksum_file(input_dir, "checksums.sha256")
    required = {"report.json", "metric_rows.csv", "audit.json"}
    if set(verified) != required:
        raise ValueError("formal release checksum membership is not frozen")
    validation = _verify_checksum_file(input_dir, "validation_report.sha256")
    if set(validation) != {"validation_report.json"}:
        raise ValueError("validation checksum membership is not frozen")
    verified.update(validation)
    return dict(sorted(verified.items()))


def _finite(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def _summary_lookup(report: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in report["summaries"]:
        key = (str(row["method"]), int(row["iterations"]))
        if key in lookup:
            raise ValueError(f"duplicate summary row: {key}")
        lookup[key] = dict(row)
    expected = {(method, k) for method in METHODS for k in ITERATIONS}
    if set(lookup) != expected:
        raise ValueError("summary method/checkpoint grid is incomplete")
    return lookup


def _frontier_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    lookup = _summary_lookup(report)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for iterations in ITERATIONS:
            row = lookup[(method, iterations)]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "iterations": iterations,
                    "sample_count": int(row["sample_count"]),
                    "mean_field_relative_l2": _finite(row, "mean_field_relative_l2"),
                    "p90_field_relative_l2": _finite(row, "p90_field_relative_l2"),
                    "mean_gradient_relative_l2": _finite(
                        row, "mean_gradient_relative_l2"
                    ),
                    "mean_front_top10_f1": _finite(row, "mean_front_top10_f1"),
                }
            )
    return rows


def _paired_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in metric_rows:
        if int(row["iterations"]) != 32:
            continue
        key = (int(row["replicate"]), int(row["sample_index"]), str(row["method"]))
        if key in selected:
            raise ValueError(f"duplicate K32 metric row: {key}")
        selected[key] = row
    output: list[dict[str, Any]] = []
    sample_keys = sorted({(key[0], key[1]) for key in selected})
    if len(sample_keys) != 16:
        raise ValueError("expected 16 paired K32 samples")
    for replicate, sample_index in sample_keys:
        rows = {
            method: selected[(replicate, sample_index, method)] for method in METHODS
        }
        factor = _finite(rows["voxel_factor_a_only_pdhg"], "field_relative_l2")
        scalar = _finite(rows["scalar_a_only_pdhg"], "field_relative_l2")
        view = _finite(rows["view_block_a_only_pdhg"], "field_relative_l2")
        graph = _finite(rows["graph_pcgls"], "field_relative_l2")
        output.append(
            {
                "replicate": replicate,
                "sample_index": sample_index,
                "reaction_family": str(rows["graph_pcgls"]["reaction_family"]),
                "factor_gain_vs_scalar_percent": 100.0 * (scalar - factor) / scalar,
                "factor_gain_vs_view_block_percent": 100.0 * (view - factor) / view,
                "factor_error_gap_vs_graph_percent": 100.0 * (factor - graph) / graph,
                "graph_field_relative_l2": graph,
                "scalar_field_relative_l2": scalar,
                "view_block_field_relative_l2": view,
                "voxel_factor_field_relative_l2": factor,
            }
        )
    return output


def _gate_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    decision = report["decision"]
    metrics = decision["metrics"]
    thresholds = decision["thresholds"]
    gates = decision["gates"]
    specifications = (
        (
            "Mean gain vs scalar",
            "mean_reduction_vs_scalar",
            metrics["factor_mean_reduction_vs_scalar_percent"],
            ">=",
            thresholds["factor_mean_reduction_vs_scalar_percent_min"],
        ),
        (
            "Both replicates positive",
            "both_replicates_positive",
            min(metrics["replicate_reduction_vs_scalar_percent"].values()),
            ">",
            0.0,
        ),
        (
            "Positive samples",
            "positive_sample_count",
            metrics["factor_positive_sample_count"],
            ">=",
            thresholds["factor_positive_sample_count_min"],
        ),
        (
            "Worst gain vs scalar",
            "worst_gain_vs_scalar",
            metrics["factor_worst_gain_vs_scalar_percent"],
            ">=",
            thresholds["factor_worst_gain_vs_scalar_percent_min"],
        ),
        (
            "K-monotone factor mean",
            "factor_mean_monotone",
            1.0,
            "==",
            1.0,
        ),
        (
            "Mean gap vs graph",
            "same_k_graph_gap",
            metrics["factor_graph_mean_error_gap_percent"],
            "<=",
            thresholds["factor_graph_mean_error_gap_percent_max"],
        ),
        (
            "Gain vs view block",
            "voxel_attribution_vs_view_block",
            metrics["factor_mean_reduction_vs_view_block_percent"],
            ">=",
            thresholds["factor_mean_reduction_vs_view_block_percent_min"],
        ),
        (
            "Wall-time ratio",
            "single_sample_wall_time",
            metrics["factor_median_wall_time_ratio_vs_single_sample_graph"],
            "<=",
            thresholds["factor_wall_time_ratio_vs_single_sample_graph_max"],
        ),
    )
    return [
        {
            "gate": label,
            "gate_key": key,
            "observed": float(observed),
            "operator": operator,
            "threshold": float(threshold),
            "passed": bool(gates[key]),
        }
        for label, key, observed, operator, threshold in specifications
    ]


def _plot_summary(
    output: Path,
    frontier: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "#F6F8F7",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CAD6D2",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), constrained_layout=True)

    for method in METHODS:
        rows = [row for row in frontier if row["method"] == method]
        axes[0, 0].plot(
            [row["iterations"] for row in rows],
            [row["mean_field_relative_l2"] for row in rows],
            marker="o",
            linewidth=2.1,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0, 0].set_title("A. Same-call field-error frontier")
    axes[0, 0].set_xlabel("Forward/adjoint calls K")
    axes[0, 0].set_ylabel("Mean field relative L2 (lower is better)")
    axes[0, 0].set_xticks(ITERATIONS)
    axes[0, 0].set_ylim(0.35, 1.03)
    axes[0, 0].legend(frameon=False, ncol=2)

    for method in METHODS:
        rows = [row for row in frontier if row["method"] == method]
        axes[0, 1].plot(
            [row["iterations"] for row in rows],
            [row["mean_front_top10_f1"] for row in rows],
            marker="o",
            linewidth=2.1,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0, 1].set_title("B. Front preservation under the same call budget")
    axes[0, 1].set_xlabel("Forward/adjoint calls K")
    axes[0, 1].set_ylabel("Mean front top-10% F1 (higher is better)")
    axes[0, 1].set_xticks(ITERATIONS)
    axes[0, 1].set_ylim(0.08, 0.8)

    x = np.arange(len(paired))
    scalar_gain = np.asarray(
        [float(row["factor_gain_vs_scalar_percent"]) for row in paired]
    )
    view_gain = np.asarray(
        [float(row["factor_gain_vs_view_block_percent"]) for row in paired]
    )
    axes[1, 0].axhline(0.0, color="#5F6B6D", linewidth=1)
    axes[1, 0].plot(
        x,
        scalar_gain,
        marker="o",
        linewidth=1.5,
        color=COLORS["scalar_a_only_pdhg"],
        label="Voxel factor vs scalar",
    )
    axes[1, 0].plot(
        x,
        view_gain,
        marker="s",
        linewidth=1.5,
        color=COLORS["view_block_a_only_pdhg"],
        label="Voxel factor vs view block",
    )
    axes[1, 0].axhline(25.0, color="#C53030", linestyle=":", linewidth=1.3)
    labels = [
        f"r{row['replicate']}\n{str(row['reaction_family']).replace('_', ' ')}"
        for row in paired
    ]
    axes[1, 0].set_xticks(x, labels, rotation=62, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("Paired field-error reduction (%)")
    axes[1, 0].set_title("C. Small paired gains do not meet the 25% mechanism gate")
    axes[1, 0].legend(frameon=False)

    y = np.arange(len(gates))
    colors = [COLORS["pass"] if row["passed"] else COLORS["fail"] for row in gates]
    axes[1, 1].barh(y, np.ones(len(gates)), color=colors, alpha=0.9)
    axes[1, 1].set_yticks(y, [str(row["gate"]) for row in gates])
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].set_xticks([])
    axes[1, 1].grid(False)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("D. Preregistered decision gates: 5 pass / 3 fail")
    for index, row in enumerate(gates):
        observed = float(row["observed"])
        threshold = float(row["threshold"])
        text = (
            f"{'PASS' if row['passed'] else 'FAIL'}  "
            f"{observed:.3g} {row['operator']} {threshold:.3g}"
        )
        axes[1, 1].text(0.03, index, text, va="center", color="white", weight="bold")

    fig.suptitle(
        "Deterministic factor-PDHG Gate B: valid mechanism NO-GO",
        fontsize=16,
        weight="bold",
        color="#173037",
    )
    fig.savefig(output / "factor_gate_b_no_go.png", dpi=220)
    fig.savefig(output / "factor_gate_b_no_go.pdf")
    plt.close(fig)


def build_public_bundle(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_hashes = verify_input_bundle(input_dir)
    report = _load_json(input_dir / "report.json")
    validation = _load_json(input_dir / "validation_report.json")
    metric_rows = _load_csv(input_dir / "metric_rows.csv")
    if report.get("status") != EXPECTED_STATUS:
        raise ValueError("formal report is not the frozen Gate B NO-GO result")
    if validation.get("status") != "PASS_INDEPENDENT_GATE_B_RECOMPUTATION":
        raise ValueError("independent validator did not pass")
    if validation.get("gate_b_status") != EXPECTED_STATUS:
        raise ValueError("validator and formal report disagree")
    if int(report.get("row_count", -1)) != 256 or len(metric_rows) != 256:
        raise ValueError("formal metric row count is not 256")
    if bool(report["decision"]["all_gates_passed"]):
        raise ValueError("NO-GO report unexpectedly marks all gates passed")

    frontier = _frontier_rows(report)
    paired = _paired_rows(metric_rows)
    gates = _gate_rows(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "method_frontier.csv", frontier)
    _write_csv(output_dir / "paired_k32_gains.csv", paired)
    _write_csv(output_dir / "decision_gates.csv", gates)

    metrics = report["decision"]["metrics"]
    summary = {
        "schema_version": "psu-b0-factor-pdhg-gate-b-public-summary-1.0",
        "status": EXPECTED_STATUS,
        "evidence_role": report["evidence_role"],
        "source_commit": report["source_commit"],
        "config_sha256": report["config_sha256"],
        "independent_validation": {
            "status": validation["status"],
            "check_count": int(validation["independent_check_count"]),
            "metric_row_count": int(validation["metric_row_count"]),
            "timing_pair_count": int(validation["timing_pair_count"]),
        },
        "data_scope": {
            "sample_count": int(report["data_contract"]["sample_count"]),
            "replicates": list(report["data_contract"]["replicates"]),
            "reaction_families": list(report["data_contract"]["reaction_families"]),
            "real_detector_geometry": True,
            "analytic_field_truth": True,
            "synthetic_correlated_noise": True,
            "truth_derived_development_scale": True,
        },
        "connectivity": {
            "support_active_voxels": int(report["amendment"]["support_active_voxel_count"]),
            "data_coupled_voxels": int(report["amendment"]["data_coupled_voxel_count"]),
            "data_null_support_voxels": int(
                report["amendment"]["data_null_support_voxel_count"]
            ),
        },
        "headline_metrics": {
            "scalar_k32_mean_field_relative_l2": metrics[
                "scalar_k32_mean_field_relative_l2"
            ],
            "view_block_k32_mean_field_relative_l2": metrics[
                "view_block_k32_mean_field_relative_l2"
            ],
            "voxel_factor_k32_mean_field_relative_l2": metrics[
                "voxel_factor_k32_mean_field_relative_l2"
            ],
            "graph_k32_mean_field_relative_l2": metrics[
                "graph_k32_mean_field_relative_l2"
            ],
            "factor_mean_reduction_vs_scalar_percent": metrics[
                "factor_mean_reduction_vs_scalar_percent"
            ],
            "factor_mean_reduction_vs_view_block_percent": metrics[
                "factor_mean_reduction_vs_view_block_percent"
            ],
            "factor_graph_mean_error_gap_percent": metrics[
                "factor_graph_mean_error_gap_percent"
            ],
            "factor_positive_sample_count": metrics["factor_positive_sample_count"],
            "factor_worst_gain_vs_scalar_percent": metrics[
                "factor_worst_gain_vs_scalar_percent"
            ],
            "factor_median_wall_time_ratio_vs_graph": metrics[
                "factor_median_wall_time_ratio_vs_single_sample_graph"
            ],
        },
        "gate_counts": {
            "passed": sum(bool(row["passed"]) for row in gates),
            "failed": sum(not bool(row["passed"]) for row in gates),
        },
        "claim_boundary": dict(report["claim_boundary"]),
        "neural_training_authorized": False,
        "algorithm_superiority_claim_authorized": False,
        "input_release_sha256": input_hashes,
    }
    _write_json(output_dir / "summary.json", summary)
    _plot_summary(output_dir, frontier, paired, gates)

    public_files = (
        "summary.json",
        "method_frontier.csv",
        "paired_k32_gains.csv",
        "decision_gates.csv",
        "factor_gate_b_no_go.png",
        "factor_gate_b_no_go.pdf",
    )
    checksums = "".join(
        f"{_sha256(output_dir / filename)}  {filename}\n" for filename in public_files
    )
    (output_dir / "checksums.sha256").write_text(checksums, encoding="utf-8")
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "demo_t16_operator/results/psu_b0_factor_pdhg_gate_b",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "demo_t16_operator/results/psu_b0_factor_pdhg_gate_b_public",
    )
    args = parser.parse_args()
    summary = build_public_bundle(args.input.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
