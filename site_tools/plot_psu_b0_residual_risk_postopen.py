#!/usr/bin/env python3
"""Plot the OCRRG post-open failure and feature-contract diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value


def plot(*, summary_path: Path, output_prefix: Path) -> dict:
    summary = _load(summary_path)
    views = summary["view_strata"]
    support = summary["calibration_view_support"]
    harms = summary["accepted_harm"]
    mismatch = summary["support_order_mismatch"]
    exact_view = summary["exact_view_conformal_probe"]
    if len(views) != 3 or len(harms) != 4:
        raise ValueError("post-open summary has an unexpected evidence shape")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "#f7f8f5",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#c9d0cc",
            "axes.grid": True,
            "grid.color": "#e7ebe8",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))

    axis = axes[0, 0]
    x = np.arange(len(views))
    mean = np.asarray([row["mean_gain_percent"] for row in views])
    p10 = np.asarray([row["p10_gain_percent"] for row in views])
    minimum = np.asarray([row["minimum_gain_percent"] for row in views])
    colors = [
        "#d65f4b" if row["harm_over_one_percent_count"] else "#1f8a70"
        for row in views
    ]
    axis.bar(x, mean, color=colors, width=0.62)
    axis.scatter(x, p10, marker="D", color="#315f93", zorder=3, label="p10")
    axis.scatter(
        x,
        minimum,
        marker="v",
        color="#8a651b",
        zorder=3,
        label="minimum",
    )
    for index, row in enumerate(views):
        axis.text(
            index,
            mean[index] - 0.16,
            (
                f"n={row['accepted_row_count']}\n"
                f"harm={row['harm_over_one_percent_count']}"
            ),
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    axis.axhline(-1.0, color="#a34f43", linestyle="--", linewidth=1.2)
    axis.set_xticks(x, labels=[f"{row['active_view_count']} views" for row in views])
    axis.set_ylabel("Field gain vs Sobolev (%)")
    axis.set_title("A  Accepted harm is confined to the six-view stratum")
    axis.legend(frameon=False, loc="lower right")

    axis = axes[0, 1]
    split_order = ("risk_train", "risk_validation", "risk_calibration")
    split_labels = ("train", "validation", "calibration")
    width = 0.25
    counts = {
        (row["split"], int(row["active_view_count"])): int(
            row["distinct_field_count"]
        )
        for row in support
    }
    view_counts = np.arange(6, 10)
    for index, split in enumerate(split_order):
        axis.bar(
            view_counts + (index - 1) * width,
            [counts[(split, int(view))] for view in view_counts],
            width,
            label=split_labels[index],
            color=("#315f93", "#1f8a70", "#d28b32")[index],
        )
    axis.set_xticks(view_counts)
    axis.set_xlabel("Active views")
    axis.set_ylabel("Distinct analytic fields")
    axis.set_title("B  Calibration was strongly skewed toward six views")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    marker_by_sample = {
        "fresh_iid_support-012": "o",
        "fresh_correlated_noise_ood-011": "s",
    }
    color_by_sample = {
        "fresh_iid_support-012": "#315f93",
        "fresh_correlated_noise_ood-011": "#a34f43",
    }
    label_by_sample = {
        "fresh_iid_support-012": "low-frequency plume",
        "fresh_correlated_noise_ood-011": "correlated-noise shock",
    }
    shown = set()
    for row in harms:
        sample = row["sample_id"]
        axis.scatter(
            row["spectral_correction_stress"],
            row["correlated_camera_stress"],
            marker=marker_by_sample[sample],
            s=105,
            color=color_by_sample[sample],
            edgecolor="white",
            linewidth=1.0,
            label=label_by_sample[sample] if sample not in shown else None,
            zorder=3,
        )
        axis.annotate(
            str(row["seed"])[-2:],
            (
                row["spectral_correction_stress"],
                row["correlated_camera_stress"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
        shown.add(sample)
    axis.axhline(0.0, color="#68736f", linewidth=0.9)
    axis.axvline(0.0, color="#68736f", linewidth=0.9)
    axis.set_xlabel("Spectral / correction stress score")
    axis.set_ylabel("Correlated-camera stress score")
    axis.set_title("C  One pooled linear score misses two distinct tails")
    axis.legend(frameon=False, loc="best")

    axis = axes[1, 1]
    labels = (
        "feature-order\ndecision changes",
        "accepted harm\nrows",
        "harm caught by\nview-only probe",
    )
    values = (
        int(mismatch["decision_disagreement_count"]),
        int(exact_view["current_accepted_harm_count"]),
        int(exact_view["harmful_rows_rejected_by_exact_view_probe"]),
    )
    bars = axis.bar(
        np.arange(3),
        values,
        color=("#d28b32", "#a34f43", "#607178"),
        width=0.62,
    )
    axis.set_xticks(np.arange(3), labels=labels)
    axis.set_ylabel("Rows")
    axis.set_ylim(0.0, max(values) + 2.2)
    axis.set_title("D  Contract repair comes before a new risk gate")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            str(value),
            ha="center",
            fontweight="bold",
        )
    axis.text(
        0.02,
        0.95,
        (
            "max prediction shift from support-order mismatch: "
            f"{mismatch['maximum_absolute_prediction_shift_percent']:.3f}%\n"
            "old empirical metrics reproduce; conformal interpretation does not"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=8.5,
        color="#4d5955",
    )

    figure.suptitle(
        "OCRRG post-open diagnosis | two physical tails and one feature-contract defect",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Real PSU support geometry; analytic 32^3 morphology and synthetic "
            "camera noise only. Opened fresh audit: diagnostic evidence, not a "
            "new frozen candidate and not algorithm superiority."
        ),
        ha="center",
        fontsize=9,
        color="#4c5551",
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.95))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = output_prefix.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip()
                for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
        outputs[suffix] = {
            "filename": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    plt.close(figure)
    manifest = {
        "schema_version": (
            "psu-b0-residual-risk-postopen-diagnosis-figure-manifest-1.0"
        ),
        "source_summary": summary_path.name,
        "source_summary_sha256": _sha256(summary_path),
        "status": summary["status"],
        "outputs": outputs,
        "claim_boundary": summary["claim_boundary"],
    }
    manifest_path = output_prefix.with_name(
        f"{output_prefix.name}_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    result = plot(
        summary_path=args.summary,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
