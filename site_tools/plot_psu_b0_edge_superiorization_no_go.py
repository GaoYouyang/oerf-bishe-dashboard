#!/usr/bin/env python3
"""Plot the call-budget and mechanism audit for TV/Huber SupPCG."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gain(
    candidate: dict[str, str],
    reference: dict[str, str],
) -> float:
    reference_error = float(reference["field_relative_l2"])
    return 100.0 * (
        reference_error - float(candidate["field_relative_l2"])
    ) / max(reference_error, 1e-12)


def build_figure(
    *,
    scale_report_path: Path,
    scale_rows_path: Path,
    tail_report_path: Path,
    tail_rows_path: Path,
) -> plt.Figure:
    scale_report = _load_json(scale_report_path)
    tail_report = _load_json(tail_report_path)
    scale_rows = _load_csv(scale_rows_path)
    tail_rows = _load_csv(tail_rows_path)
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    figure.patch.set_facecolor("#f4f7f6")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.grid(axis="y", color="#dce6e2", linewidth=0.8, alpha=0.9)
        axis.spines[["top", "right"]].set_visible(False)

    by_key = {
        (
            int(row["replicate"]),
            int(row["sample_index"]),
            row["candidate_id"],
        ): row
        for row in tail_rows
    }
    stages = [4, 5, 6, 7, 8, 9]
    depth_values = []
    depth_p10 = []
    depth_worst = []
    for stage in stages:
        values = []
        for row in tail_rows:
            if row["candidate_id"] != f"graph_s3_k{stage}":
                continue
            reference = by_key[
                (
                    int(row["replicate"]),
                    int(row["sample_index"]),
                    "component_s3_k4",
                )
            ]
            values.append(_gain(row, reference))
        depth_values.append(float(np.mean(values)))
        depth_p10.append(float(np.quantile(values, 0.1)))
        depth_worst.append(float(np.min(values)))
    axis = axes[0, 0]
    axis.plot(
        stages,
        depth_values,
        marker="o",
        linewidth=2.2,
        color="#177e75",
        label="mean",
    )
    axis.plot(
        stages,
        depth_p10,
        marker="s",
        linewidth=1.8,
        color="#e08b43",
        label="p10",
    )
    axis.plot(
        stages,
        depth_worst,
        marker="^",
        linewidth=1.8,
        color="#b84d5d",
        label="worst",
    )
    axis.axhline(0.0, color="#65736f", linewidth=1.0)
    axis.set_title("A  Plain graph-PCGLS depth is a strong competitor")
    axis.set_xlabel("PCGLS stages")
    axis.set_ylabel("gain vs component s3-k4 (%)")
    axis.legend(frameon=False, ncol=3)

    axis = axes[0, 1]
    colors = {"tv": "#2878b5", "huber": "#d46b3c"}
    for penalty in ("tv", "huber"):
        selected = [
            row
            for row in (
                scale_report["top_superiorized_scale_smoke"]
                + tail_report["top_superiorized_scale_smoke"]
            )
            if row["penalty"] == penalty
        ]
        axis.scatter(
            [
                float(
                    row["mean_field_gain_vs_graph_same_stage_percent"]
                )
                for row in selected
            ],
            [
                float(
                    row[
                        "mean_field_gain_vs_graph_budget_floor_percent"
                    ]
                )
                for row in selected
            ],
            s=[
                42.0 + 10.0 * int(row["stages"])
                for row in selected
            ],
            alpha=0.78,
            color=colors[penalty],
            edgecolor="white",
            linewidth=0.8,
            label=penalty.upper(),
        )
    axis.axhline(0.0, color="#65736f", linewidth=1.0)
    axis.axvline(0.0, color="#65736f", linewidth=1.0)
    axis.set_title("B  Same-stage signal does not survive call matching")
    axis.set_xlabel("gain vs graph-PCGLS at same stage (%)")
    axis.set_ylabel("gain vs graph-PCGLS call-budget floor (%)")
    axis.legend(frameon=False)

    scale_summaries = {
        row["candidate_id"]: row
        for row in _load_csv(
            scale_report_path.with_name("candidate_summaries.csv")
        )
    }
    best_same_stage_id = max(
        (
            candidate_id
            for candidate_id, row in scale_summaries.items()
            if row["method"] == "superiorized_pcgls"
        ),
        key=lambda candidate_id: float(
            scale_summaries[candidate_id][
                "mean_field_gain_vs_graph_same_stage_percent"
            ]
        ),
    )
    scale_by_key = {
        (
            int(row["replicate"]),
            int(row["sample_index"]),
            row["candidate_id"],
        ): row
        for row in scale_rows
    }
    family_values: dict[str, list[float]] = {}
    for row in scale_rows:
        if row["candidate_id"] != best_same_stage_id:
            continue
        reference = scale_by_key[
            (
                int(row["replicate"]),
                int(row["sample_index"]),
                f"graph_s3_k{int(row['stages'])}",
            )
        ]
        family_values.setdefault(row["reaction_family"], []).append(
            _gain(row, reference)
        )
    families = list(family_values)
    values = [float(np.mean(family_values[name])) for name in families]
    axis = axes[1, 0]
    axis.barh(
        np.arange(len(families)),
        values,
        color=[
            "#177e75" if value >= 0.0 else "#b84d5d"
            for value in values
        ],
    )
    axis.axvline(0.0, color="#65736f", linewidth=1.0)
    axis.set_yticks(np.arange(len(families)), families)
    axis.invert_yaxis()
    axis.set_title(
        "C  Tiny same-stage mechanism signal is morphology-specific"
    )
    axis.set_xlabel(
        f"{best_same_stage_id} gain vs same-stage graph-PCGLS (%)"
    )

    axis = axes[1, 1]
    scale_decision = scale_report["decision"]
    tail_decision = tail_report["decision"]
    labels = [
        "scale mean",
        "scale p10",
        "scale worst",
        "tail mean",
        "tail p10",
        "tail worst",
    ]
    decision_values = [
        float(
            scale_decision[
                "best_mean_field_gain_vs_graph_budget_floor_percent"
            ]
        ),
        float(
            scale_decision[
                "best_field_gain_vs_graph_budget_floor_p10_percent"
            ]
        ),
        float(
            scale_decision[
                "best_worst_field_gain_vs_graph_budget_floor_percent"
            ]
        ),
        float(
            tail_decision[
                "best_mean_field_gain_vs_graph_budget_floor_percent"
            ]
        ),
        float(
            tail_decision[
                "best_field_gain_vs_graph_budget_floor_p10_percent"
            ]
        ),
        float(
            tail_decision[
                "best_worst_field_gain_vs_graph_budget_floor_percent"
            ]
        ),
    ]
    axis.bar(
        np.arange(len(labels)),
        decision_values,
        color=["#d46b3c"] * 3 + ["#b84d5d"] * 3,
    )
    axis.axhline(0.0, color="#65736f", linewidth=1.0)
    axis.set_xticks(
        np.arange(len(labels)),
        labels,
        rotation=28,
        ha="right",
    )
    axis.set_ylabel("gain vs graph call-budget floor (%)")
    axis.set_title("D  Preregistered stop: SupPCG budget NO-GO")
    axis.text(
        0.02,
        0.04,
        "Fresh sealed. No more gamma tuning. Next: one-pair primal-dual.",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#394743",
    )

    figure.suptitle(
        "TV/Huber superiorized PCGLS: correct mechanism, poor call efficiency",
        fontsize=16,
        fontweight="bold",
        color="#1f302b",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-report", type=Path, required=True)
    parser.add_argument("--scale-rows", type=Path, required=True)
    parser.add_argument("--tail-report", type=Path, required=True)
    parser.add_argument("--tail-rows", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    figure = build_figure(
        scale_report_path=args.scale_report,
        scale_rows_path=args.scale_rows,
        tail_report_path=args.tail_report,
        tail_rows_path=args.tail_rows,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output_prefix.with_suffix(".png"),
        dpi=180,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
