#!/usr/bin/env python3
"""Plot risk-quantile and front-safe single-expert PCGLS development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "baseline": "#52636b",
    "mean": "#147d70",
    "quantile": "#3578a8",
    "risk": "#d1902f",
    "multi": "#b14f61",
    "gate": "#dfe7e4",
}


def _lookup(report: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row["candidate_method"]), str(row["split"])): row
        for row in report["paired_gain_summary"]
    }


def build_figure(rq: dict, multi: dict, output: Path) -> None:
    rq_lookup = _lookup(rq)
    multi_lookup = _lookup(multi)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6))
    fig.patch.set_facecolor("#f7f9f8")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#dfe7e4", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)

    baseline_id = rq["expert_bank"]["baseline_candidate_id"]
    action_rows = [
        row
        for row in rq["finite_action_bank"]
        if row["expert_candidate_id"] != baseline_id
    ]
    expert_ids = sorted(
        {str(row["expert_candidate_id"]) for row in action_rows}
    )
    expert_labels = {
        expert_id: f"Expert {index + 1}"
        for index, expert_id in enumerate(expert_ids)
    }
    action_colors = ("#147d70", "#3578a8", "#b14f61")
    for color, expert_id in zip(
        action_colors,
        expert_ids,
        strict=True,
    ):
        rows = sorted(
            [
                row
                for row in action_rows
                if row["expert_candidate_id"] == expert_id
            ],
            key=lambda row: row["interpolation_fraction"],
        )
        axes[0, 0].plot(
            [
                100.0 * float(row["harm_over_one_percent_rate"])
                for row in rows
            ],
            [float(row["mean_field_gain_percent"]) for row in rows],
            marker="o",
            linewidth=1.7,
            color=color,
            label=expert_labels[expert_id],
        )
        for row in rows:
            axes[0, 0].annotate(
                f"{float(row['interpolation_fraction']):.2g}",
                (
                    100.0
                    * float(row["harm_over_one_percent_rate"]),
                    float(row["mean_field_gain_percent"]),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
            )
    axes[0, 0].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[0, 0].axvline(
        5.0,
        color=COLORS["baseline"],
        linewidth=1.0,
        linestyle="--",
    )
    axes[0, 0].set_xlabel("Train samples harmed by more than 1% (%)")
    axes[0, 0].set_ylabel("Mean field gain vs baseline (%)")
    axes[0, 0].set_title(
        "A  Larger expert steps buy utility by exposing a heavy tail",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    axes[0, 0].legend(frameon=False, fontsize=8.2)

    method_specs = (
        ("rq_ogse_ablation_mean_only", "Mean", COLORS["mean"]),
        (
            "rq_ogse_ablation_quantile_only",
            "Quantile",
            COLORS["quantile"],
        ),
        (
            "rq_ogse_ablation_quantile_harm",
            "Quantile + harm",
            COLORS["risk"],
        ),
        (
            "rq_ogse_ablation_mean_quantile_harm",
            "Mean + quantile + harm",
            "#805d9f",
        ),
    )
    x = np.arange(len(method_specs), dtype=np.float64)
    width = 0.34
    for split_index, (split, label) in enumerate(
        (
            ("risk_validation", "Validation"),
            ("risk_calibration", "Calibration"),
        )
    ):
        values = [
            rq_lookup[(method, split)]["mean_field_gain_percent"]
            for method, _, _ in method_specs
        ]
        axes[0, 1].bar(
            x + (split_index - 0.5) * width,
            values,
            width=width,
            color=[
                color if split_index == 0 else "#aebbb8"
                for _, _, color in method_specs
            ],
            edgecolor=[
                color for _, _, color in method_specs
            ],
            linewidth=1.1,
            label=label,
        )
    axes[0, 1].axhline(
        2.0,
        color=COLORS["baseline"],
        linewidth=1.0,
        linestyle="--",
    )
    axes[0, 1].set_xticks(
        x,
        [label for _, label, _ in method_specs],
        rotation=12,
        ha="right",
    )
    axes[0, 1].set_ylabel("Mean field error reduction (%)")
    axes[0, 1].set_title(
        "B  Risk heads remove harm but fall below the 2% utility gate",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    axes[0, 1].legend(frameon=False, fontsize=8.5)

    trade_specs = (
        (
            rq_lookup,
            "rq_ogse_ablation_mean_only",
            "Mean-only",
            COLORS["mean"],
        ),
        (
            rq_lookup,
            "rq_ogse_ablation_mean_quantile_harm",
            "Field risk",
            COLORS["risk"],
        ),
        (
            multi_lookup,
            "mo_rq_ogse_strict",
            "Field + front risk",
            COLORS["multi"],
        ),
    )
    for lookup, method, label, color in trade_specs:
        for split, marker, suffix in (
            ("risk_validation", "o", "V"),
            ("risk_calibration", "s", "C"),
        ):
            row = lookup[(method, split)]
            field = float(row["mean_field_gain_percent"])
            front = float(
                row["secondary_metric_gain"]["front_top10_f1"][
                    "mean_gain_percent"
                ]
            )
            axes[1, 0].scatter(
                field,
                front,
                color=color,
                marker=marker,
                s=74,
            )
            axes[1, 0].annotate(
                f"{label} {suffix}",
                (field, front),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axes[1, 0].axvline(
        2.0,
        color=COLORS["baseline"],
        linewidth=1.0,
        linestyle="--",
    )
    axes[1, 0].axhline(0.0, color="#88969b", linewidth=1.0)
    axes[1, 0].axvspan(
        2.0,
        4.0,
        color="#edf4ef",
        alpha=0.55,
    )
    axes[1, 0].set_xlabel("Mean field error reduction (%)")
    axes[1, 0].set_ylabel("Mean front top-10% F1 gain (%)")
    axes[1, 0].set_title(
        "C  Field utility and reacting-front fidelity do not align",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )

    rows = (
        (
            "Mean-only RQ",
            rq["overall_decision"]["primary_strict_gate_pass"],
            rq["overall_decision"]["secondary_metric_safety_pass"],
            rq_lookup[("rq_ogse_strict", "risk_calibration")],
        ),
        (
            "Field + front RQ",
            multi["overall_decision"]["primary_strict_gate_pass"],
            multi["overall_decision"]["secondary_metric_safety_pass"],
            multi_lookup[("mo_rq_ogse_strict", "risk_calibration")],
        ),
    )
    y = np.arange(len(rows), dtype=np.float64)
    field_values = [
        float(row[3]["mean_field_gain_percent"]) for row in rows
    ]
    front_values = [
        float(
            row[3]["secondary_metric_gain"]["front_top10_f1"][
                "mean_gain_percent"
            ]
        )
        for row in rows
    ]
    axes[1, 1].barh(
        y + 0.16,
        field_values,
        height=0.28,
        color=COLORS["mean"],
        label="Calibration field gain",
    )
    axes[1, 1].barh(
        y - 0.16,
        front_values,
        height=0.28,
        color=COLORS["multi"],
        label="Calibration front gain",
    )
    axes[1, 1].axvline(0.0, color="#88969b", linewidth=1.0)
    axes[1, 1].axvline(
        2.0,
        color=COLORS["baseline"],
        linewidth=1.0,
        linestyle="--",
    )
    axes[1, 1].set_yticks(y, [row[0] for row in rows])
    axes[1, 1].set_xlabel("Paired gain (%)")
    axes[1, 1].set_title(
        "D  A primary field pass is not a multi-objective GO",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )
    for index, (_, primary, secondary, _) in enumerate(rows):
        axes[1, 1].text(
            max(field_values[index], front_values[index], 0.0) + 0.12,
            index,
            (
                f"field {'PASS' if primary else 'FAIL'} | "
                f"front {'PASS' if secondary else 'FAIL'}"
            ),
            va="center",
            fontsize=8.2,
            color="#263b40",
        )
    axes[1, 1].legend(frameon=False, fontsize=8.2, loc="lower right")

    fig.suptitle(
        "RQ-OGSE-PCGLS: field gains survive, reacting-front safety does not",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#1f3438",
    )
    fig.text(
        0.055,
        0.952,
        (
            "Fixed 4F+4AT, train-only finite actions and family-stratified "
            "OOF routing. Validation/calibration are post-open diagnostics; "
            "no independent repeat or experimental claim is authorized."
        ),
        ha="left",
        fontsize=9.7,
        color="#53666b",
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.93), h_pad=2.1, w_pad=1.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq", type=Path, required=True)
    parser.add_argument("--multi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rq = json.loads(args.rq.read_text(encoding="utf-8"))
    multi = json.loads(args.multi.read_text(encoding="utf-8"))
    build_figure(rq, multi, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
