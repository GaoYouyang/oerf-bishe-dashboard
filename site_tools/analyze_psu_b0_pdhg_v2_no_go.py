#!/usr/bin/env python3
"""Build a sanitized, reproducible audit of the PDHG v2 E2 NO-GO result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


EXPECTED_STATUS = "POSTOPEN_PDHG_SCALE_NO_GO"
ITERATIONS = (4, 8, 16, 32)
FAMILIES = (
    "plume",
    "wavy_front",
    "thin_front",
    "double_front",
    "annular_kernel",
    "oblique_shock",
    "vortex_pair",
    "multi_plume",
)
PALETTE = {
    "graph": "#167D70",
    "data": "#D14B40",
    "regularized": "#3D5A80",
    "threshold": "#6B7280",
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
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _verify_private_bundle(input_dir: Path) -> dict[str, str]:
    checksum_path = input_dir / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError("private input bundle has no checksums.sha256")
    verified: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split("  ", 1)
        if Path(filename).name != filename:
            raise ValueError("private checksum entry is not a plain filename")
        path = input_dir / filename
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"private bundle checksum mismatch: {filename}")
        verified[filename] = digest
    required = {
        "report.json",
        "metric_rows.csv",
        "candidate_summaries.csv",
        "method_summaries.csv",
        "operator_call_ledger.json",
        "timing_audit.json",
    }
    if not required.issubset(verified):
        raise ValueError("private input bundle is incomplete")
    return dict(sorted(verified.items()))


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def _find_values(value: Any, key: str) -> list[Any]:
    output: list[Any] = []
    if isinstance(value, Mapping):
        if key in value:
            output.append(value[key])
        for child in value.values():
            output.extend(_find_values(child, key))
    elif isinstance(value, list):
        for child in value:
            output.extend(_find_values(child, key))
    return output


def _gate_rows(
    report: Mapping[str, Any],
    winner: Mapping[str, Any],
) -> list[dict[str, Any]]:
    thresholds = report["configuration"]["decision_gates"]
    decisions = report["decision"]["gates"]
    specifications = (
        (
            "mean field gain",
            "mean_field_gain_vs_graph_budget_frontier_percent",
            ">=",
            "minimum_mean_field_gain_percent",
            "mean_field_gain_at_least_one_percent",
        ),
        (
            "field gain p10",
            "field_gain_vs_graph_budget_frontier_p10_percent",
            ">=",
            "minimum_field_p10_percent",
            "field_p10_at_least_minus_one_percent",
        ),
        (
            "field harm rate",
            "field_harm_vs_graph_budget_frontier_over_one_percent_rate",
            "<=",
            "maximum_harm_rate",
            "field_harm_rate_at_most_one_of_sixteen",
        ),
        (
            "worst field gain",
            "worst_field_gain_vs_graph_budget_frontier_percent",
            ">=",
            "minimum_worst_field_gain_percent",
            "worst_field_gain_at_least_minus_three_percent",
        ),
        (
            "mean gradient gain",
            "mean_gradient_gain_vs_graph_budget_frontier_percent",
            ">=",
            "minimum_mean_gradient_gain_percent",
            "mean_gradient_gain_nonnegative",
        ),
        (
            "mean front gain",
            "mean_front_gain_vs_graph_budget_frontier",
            ">=",
            "minimum_mean_front_gain",
            "mean_front_gain_nonnegative",
        ),
        (
            "critical-family front gain",
            "front_critical_mean_front_gain_vs_graph_budget_frontier",
            ">=",
            "minimum_front_critical_mean_gain",
            "front_critical_mean_nonnegative",
        ),
        (
            "replicate 0 mean field gain",
            "replicate_0_mean_field_gain_percent",
            ">",
            None,
            "replicate_0_mean_positive",
        ),
        (
            "replicate 8 mean field gain",
            "replicate_8_mean_field_gain_percent",
            ">",
            None,
            "replicate_8_mean_positive",
        ),
        (
            "paired wall-time ratio",
            None,
            "<=",
            "maximum_median_wall_time_ratio",
            "median_wall_time_ratio_at_most_three",
        ),
    )
    rows: list[dict[str, Any]] = []
    for label, observed_key, operator, threshold_key, gate_key in specifications:
        observed = (
            float(report["decision"]["winner_median_wall_time_ratio_vs_graph_frontier"])
            if observed_key is None
            else _finite_float(winner, observed_key)
        )
        threshold = 0.0 if threshold_key is None else float(thresholds[threshold_key])
        rows.append(
            {
                "gate": label,
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(decisions[gate_key]),
            }
        )
    return rows


def _budget_rows(
    candidates: Sequence[Mapping[str, str]],
    methods: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for iterations in ITERATIONS:
        selected = [
            row for row in candidates if int(row["iterations"]) == iterations
        ]
        best = max(
            selected,
            key=lambda row: _finite_float(
                row, "mean_field_gain_vs_graph_budget_frontier_percent"
            ),
        )
        data_id = f"pdhg_data_only_k{iterations}"
        graph_id = f"graph_s3_k{iterations}"
        rows.append(
            {
                "iterations": iterations,
                "graph_candidate_id": graph_id,
                "graph_mean_field_relative_l2": _finite_float(
                    methods[graph_id], "mean_field_relative_l2"
                ),
                "data_only_candidate_id": data_id,
                "data_only_mean_field_relative_l2": _finite_float(
                    methods[data_id], "mean_field_relative_l2"
                ),
                "best_regularized_candidate_id": best["candidate_id"],
                "best_regularized_mean_field_relative_l2": _finite_float(
                    methods[str(best["candidate_id"])], "mean_field_relative_l2"
                ),
                "best_regularized_gain_vs_data_only_percent": _finite_float(
                    best, "mean_field_gain_vs_data_only_percent"
                ),
                "best_regularized_gain_vs_graph_percent": _finite_float(
                    best, "mean_field_gain_vs_graph_budget_frontier_percent"
                ),
            }
        )
    return rows


def _candidate_rows(
    candidates: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    keys = (
        "rank",
        "candidate_id",
        "penalty",
        "alpha_fraction",
        "alpha",
        "iterations",
        "mean_field_gain_vs_data_only_percent",
        "mean_field_gain_vs_graph_budget_frontier_percent",
        "field_gain_vs_graph_budget_frontier_p10_percent",
        "worst_field_gain_vs_graph_budget_frontier_percent",
        "mean_gradient_gain_vs_graph_budget_frontier_percent",
        "mean_front_gain_vs_graph_budget_frontier",
        "median_wall_time_ratio_vs_graph_frontier",
    )
    return [{key: row[key] for key in keys} for row in candidates]


def _morphology_rows(
    metric_rows: Sequence[Mapping[str, str]],
    *,
    winner_id: str,
    baseline_id: str,
) -> list[dict[str, Any]]:
    index = {
        (
            row["candidate_id"],
            int(row["replicate"]),
            row["reaction_family"],
        ): row
        for row in metric_rows
    }
    rows: list[dict[str, Any]] = []
    for replicate in (0, 8):
        for family in FAMILIES:
            candidate = index[(winner_id, replicate, family)]
            baseline = index[(baseline_id, replicate, family)]
            candidate_error = _finite_float(candidate, "field_relative_l2")
            baseline_error = _finite_float(baseline, "field_relative_l2")
            rows.append(
                {
                    "replicate": replicate,
                    "reaction_family": family,
                    "candidate_field_relative_l2": candidate_error,
                    "baseline_field_relative_l2": baseline_error,
                    "field_gain_percent": 100.0
                    * (baseline_error - candidate_error)
                    / baseline_error,
                }
            )
    return rows


def _plot_budget(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    x = [int(row["iterations"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    ax.plot(
        x,
        [float(row["graph_mean_field_relative_l2"]) for row in rows],
        marker="o",
        linewidth=2.4,
        color=PALETTE["graph"],
        label="graph-PCGLS",
    )
    ax.plot(
        x,
        [float(row["data_only_mean_field_relative_l2"]) for row in rows],
        marker="s",
        linewidth=2.2,
        color=PALETTE["data"],
        label="data-only PDHG",
    )
    ax.plot(
        x,
        [float(row["best_regularized_mean_field_relative_l2"]) for row in rows],
        marker="^",
        linewidth=1.8,
        linestyle="--",
        color=PALETTE["regularized"],
        label="best regularized PDHG at each K",
    )
    ax.set_xticks(x)
    ax.set_xlabel("Forward/adjoint budget K")
    ax.set_ylabel("Mean field relative L2 (lower is better)")
    ax.set_title("Scalar-step PDHG remains near the zero-field error floor")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_regularization_heatmap(
    path: Path,
    candidates: Sequence[Mapping[str, str]],
) -> None:
    row_labels = [
        f"{penalty} {alpha}"
        for penalty in ("tv", "huber")
        for alpha in ("1/256", "1/64", "1/16", "1/4")
    ]
    values = np.empty((len(row_labels), len(ITERATIONS)), dtype=np.float64)
    for row_index, label in enumerate(row_labels):
        penalty, alpha = label.split()
        for column, iterations in enumerate(ITERATIONS):
            row = next(
                candidate
                for candidate in candidates
                if candidate["penalty"] == penalty
                and candidate["alpha_fraction"] == alpha
                and int(candidate["iterations"]) == iterations
            )
            values[row_index, column] = _finite_float(
                row, "mean_field_gain_vs_data_only_percent"
            )
    minimum = float(np.min(values))
    maximum = max(0.01, float(np.max(values)))
    fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    image = ax.imshow(
        values,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum),
        aspect="auto",
    )
    ax.set_xticks(range(len(ITERATIONS)), labels=[str(value) for value in ITERATIONS])
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Iteration budget K")
    ax.set_ylabel("Penalty and alpha")
    ax.set_title("Regularization gain vs data-only PDHG (%)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = f"{value:.1e}" if abs(value) < 0.001 else f"{value:.3f}"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(image, ax=ax, label="Gain (%) | positive is better")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_morphology_heatmap(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    values = np.array(
        [
            [
                next(
                    float(row["field_gain_percent"])
                    for row in rows
                    if int(row["replicate"]) == replicate
                    and row["reaction_family"] == family
                )
                for family in FAMILIES
            ]
            for replicate in (0, 8)
        ]
    )
    minimum = float(np.min(values))
    maximum = max(1.0, float(np.max(values)))
    fig, ax = plt.subplots(figsize=(11.2, 3.6), constrained_layout=True)
    image = ax.imshow(
        values,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum),
        aspect="auto",
    )
    ax.set_xticks(range(len(FAMILIES)), labels=FAMILIES, rotation=30, ha="right")
    ax.set_yticks((0, 1), labels=("replicate 0", "replicate 8"))
    ax.set_title("Rank-1 PDHG field gain vs graph-PCGLS by morphology (%)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Gain (%) | positive is better")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_runtime(path: Path, paired_runs: Sequence[Mapping[str, Any]]) -> None:
    ratios = [float(row["paired_ratio"]) for row in paired_runs]
    median = float(np.median(ratios))
    fig, (ax, gauge) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (3.2, 1.2)},
    )
    seen_orders: set[str] = set()
    for index, row in enumerate(paired_runs, start=1):
        order = str(row["order"])
        color = (
            PALETTE["regularized"]
            if order == "candidate_then_baseline"
            else PALETTE["graph"]
        )
        label = order.replace("_", " ") if order not in seen_orders else None
        ax.scatter(index, float(row["paired_ratio"]), c=color, s=54, zorder=3, label=label)
        seen_orders.add(order)
    ax.plot(range(1, len(ratios) + 1), ratios, color="#A0AEC0", linewidth=1)
    ax.axhline(
        median,
        color=PALETTE["data"],
        linewidth=1.8,
        label=f"median: {median:.3f}",
    )
    ax.set_xticks(range(1, len(ratios) + 1))
    padding = max(0.006, 0.25 * (max(ratios) - min(ratios)))
    ax.set_ylim(min(ratios) - padding, max(ratios) + padding)
    ax.set_xlabel("Adjacent AB/BA timing pair")
    ax.set_ylabel("PDHG / graph-PCGLS wall-time ratio")
    ax.set_title("Ten adjacent AB/BA ratios")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    gauge.barh((0,), (median,), color=PALETTE["regularized"], height=0.38)
    gauge.axvline(3.0, color=PALETTE["threshold"], linestyle="--", linewidth=2)
    gauge.text(median, 0.24, f"median {median:.3f}", ha="center", va="bottom", fontsize=9)
    gauge.text(3.0, -0.24, "gate 3.0", ha="right", va="top", fontsize=9)
    gauge.set_xlim(0.0, 3.15)
    gauge.set_yticks(())
    gauge.set_xlabel("ratio")
    gauge.set_title("Frozen runtime gate")
    gauge.grid(True, axis="x", alpha=0.2)
    fig.suptitle("Runtime gate passes; accuracy gates do not")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_gate_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax.axis("off")
    cell_text = [
        [
            str(row["gate"]),
            f"{float(row['observed']):.4g}",
            f"{row['operator']} {float(row['threshold']):.4g}",
            "PASS" if row["passed"] else "FAIL",
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=("Frozen gate", "Observed", "Rule", "Result"),
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.46, 0.18, 0.18, 0.12),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.55)
    for row_index, row in enumerate(rows, start=1):
        table[(row_index, 3)].set_facecolor(
            "#C6F6D5" if row["passed"] else "#FED7D7"
        )
        table[(row_index, 3)].get_text().set_color(
            PALETTE["pass"] if row["passed"] else PALETTE["fail"]
        )
    for column in range(4):
        table[(0, column)].set_facecolor("#E2E8F0")
        table[(0, column)].get_text().set_weight("bold")
    ax.set_title("Preregistered decision audit: 1 runtime pass, 9 accuracy/structure failures", pad=18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_checksums(output_dir: Path, paths: Iterable[Path]) -> None:
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(paths)]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_public_audit(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("public output directory must not already exist")
    private_checksums = _verify_private_bundle(input_dir)
    report = _load_json(input_dir / "report.json")
    if report.get("status") != EXPECTED_STATUS:
        raise ValueError("this analyzer only accepts the frozen v2 NO-GO bundle")
    if (
        int(report.get("formal_candidate_count", -1)) != 32
        or int(report.get("control_reference_count", -1)) != 17
        or int(report.get("method_count", -1)) != 49
        or int(report.get("metric_row_count", -1)) != 784
    ):
        raise ValueError("frozen method or metric-row count is invalid")

    candidates = _load_csv(input_dir / "candidate_summaries.csv")
    metric_rows = _load_csv(input_dir / "metric_rows.csv")
    method_rows = _load_csv(input_dir / "method_summaries.csv")
    methods = {row["candidate_id"]: row for row in method_rows}
    timing = _load_json(input_dir / "timing_audit.json")
    ledger = _load_json(input_dir / "operator_call_ledger.json")
    winner = candidates[0]
    winner_id = str(report["decision"]["winner_candidate_id"])
    if winner["candidate_id"] != winner_id or winner["rank"] != "1":
        raise ValueError("candidate CSV and report disagree about rank 1")
    baseline_id = str(winner["graph_budget_frontier_id"])
    gate_rows = _gate_rows(report, winner)
    budget_rows = _budget_rows(candidates, methods)
    morphology_rows = _morphology_rows(
        metric_rows,
        winner_id=winner_id,
        baseline_id=baseline_id,
    )
    candidate_rows = _candidate_rows(candidates)
    winner_timing = timing["winner_five_repeat_audit"]

    norm_rows = [
        row["shared_norm_setup"]
        for row in ledger["replicates"]
        if isinstance(row.get("shared_norm_setup"), Mapping)
    ]
    norm_ratios = [
        float(row["gradient_norm_squared_upper"])
        / float(row["data_norm_squared_upper"])
        for row in norm_rows
    ]
    regularization_counts = sorted(
        {int(value) for value in _find_values(ledger, "regularization_site_count")}
    )
    summary = {
        "schema_version": "psu-b0-pdhg-v2-public-no-go-audit-1.0",
        "status": EXPECTED_STATUS,
        "evidence_level": report["decision"]["evidence_level"],
        "interpretation": (
            "frozen_scalar_step_pdhg_grid_no_go_not_a_general_pdhg_or_tv_no_go"
        ),
        "integrity": {
            "private_bundle_checksums_verified": True,
            "private_artifact_sha256_by_opaque_role": {
                "report": private_checksums["report.json"],
                "metric_rows": private_checksums["metric_rows.csv"],
                "candidate_summaries": private_checksums[
                    "candidate_summaries.csv"
                ],
                "method_summaries": private_checksums["method_summaries.csv"],
                "call_ledger": private_checksums["operator_call_ledger.json"],
                "timing_audit": private_checksums["timing_audit.json"],
            },
            "private_paths_published": False,
            "junit_or_environment_published": False,
        },
        "counts": {
            "formal_candidates": 32,
            "controls_and_references": 17,
            "methods": 49,
            "paired_metric_rows": 784,
            "valid_candidate_rows": 512,
            "invalid_candidates": int(report["decision"]["invalid_candidate_count"]),
        },
        "preflight": {
            "status": report["pdhg_stability_preflight"]["status"],
            "valid": bool(report["pdhg_stability_preflight"]["valid"]),
            "stress_run_count": int(
                report["pdhg_stability_preflight"]["call_ledger"]["run_count"]
            ),
            "solver_forward_calls": int(
                report["pdhg_stability_preflight"]["call_ledger"][
                    "solver_forward_calls"
                ]
            ),
            "solver_adjoint_calls": int(
                report["pdhg_stability_preflight"]["call_ledger"][
                    "solver_adjoint_calls"
                ]
            ),
            "solver_gradient_calls": int(
                report["pdhg_stability_preflight"]["call_ledger"][
                    "observed_gradient_calls"
                ]
            ),
            "solver_gradient_adjoint_calls": int(
                report["pdhg_stability_preflight"]["call_ledger"][
                    "observed_gradient_adjoint_calls"
                ]
            ),
        },
        "rank_one_candidate": {
            "candidate_id": winner_id,
            "ranking_winner_is_not_a_passing_method": True,
            "mean_field_gain_vs_graph_percent": _finite_float(
                winner, "mean_field_gain_vs_graph_budget_frontier_percent"
            ),
            "field_gain_p10_percent": _finite_float(
                winner, "field_gain_vs_graph_budget_frontier_p10_percent"
            ),
            "worst_field_gain_percent": _finite_float(
                winner, "worst_field_gain_vs_graph_budget_frontier_percent"
            ),
            "mean_gradient_gain_percent": _finite_float(
                winner, "mean_gradient_gain_vs_graph_budget_frontier_percent"
            ),
            "mean_front_gain": _finite_float(
                winner, "mean_front_gain_vs_graph_budget_frontier"
            ),
            "paired_wall_time_ratio": float(
                report["decision"]["winner_median_wall_time_ratio_vs_graph_frontier"]
            ),
        },
        "diagnosis": {
            "data_only_pdhg_mean_field_relative_l2_by_k": {
                str(row["iterations"]): row["data_only_mean_field_relative_l2"]
                for row in budget_rows
            },
            "graph_pcgls_mean_field_relative_l2_by_k": {
                str(row["iterations"]): row["graph_mean_field_relative_l2"]
                for row in budget_rows
            },
            "best_regularized_gain_vs_data_only_percent": max(
                float(row["mean_field_gain_vs_data_only_percent"])
                for row in candidates
            ),
            "gradient_to_data_norm_squared_ratio_by_replicate": norm_ratios,
            "regularization_site_counts": regularization_counts,
            "supported_conclusion": (
                "the frozen scalar-step formulation is conditioning-limited and "
                "regularization-inactive at K<=32 on this E2 oracle-scale benchmark"
            ),
            "not_supported": [
                "PDHG is generally ineffective",
                "TV or Huber regularization is generally ineffective",
                "the result generalizes to fresh seeds, cameras, or experiments",
            ],
        },
        "decision_gates": gate_rows,
        "claim_boundary": {
            "already_opened_replicates_only": True,
            "oracle_scale_synthetic_benchmark": True,
            "experimental_deployment_scale_available": False,
            "fresh_data_opened": False,
            "neural_training_authorized": False,
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{time.time_ns()}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        paths = [
            staging / "summary.json",
            staging / "budget_frontier.csv",
            staging / "candidate_grid.csv",
            staging / "morphology_gain.csv",
            staging / "decision_gates.csv",
            staging / "budget_frontier.png",
            staging / "regularization_gain_heatmap.png",
            staging / "morphology_gain_heatmap.png",
            staging / "winner_runtime_pairs.png",
            staging / "decision_gate_audit.png",
            staging / "README.md",
        ]
        _write_json(paths[0], summary)
        _write_csv(paths[1], budget_rows)
        _write_csv(paths[2], candidate_rows)
        _write_csv(paths[3], morphology_rows)
        _write_csv(paths[4], gate_rows)
        _plot_budget(paths[5], budget_rows)
        _plot_regularization_heatmap(paths[6], candidates)
        _plot_morphology_heatmap(paths[7], morphology_rows)
        _plot_runtime(paths[8], winner_timing["paired_runs"])
        _plot_gate_table(paths[9], gate_rows)
        paths[10].write_text(
            "\n".join(
                (
                    "# PSU B0 PDHG v2 public NO-GO audit",
                    "",
                    "- Status: `POSTOPEN_PDHG_SCALE_NO_GO`.",
                    "- Rank 1 is a ranking label, not a passing algorithm.",
                    "- The runtime gate passed; all nine accuracy/structure gates failed.",
                    "- This rules out only the frozen scalar-step grid on two already-opened replicates.",
                    "- Raw paths, JUnit XML, environment details, and private geometry metadata are excluded.",
                    "",
                    "All plotted values are regenerated from the checksum-verified private CSV/JSON bundle.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        _write_checksums(staging, paths)
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_public_audit(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "winner": summary["rank_one_candidate"]["candidate_id"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
