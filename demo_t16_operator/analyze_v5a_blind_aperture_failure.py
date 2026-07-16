#!/usr/bin/env python3
"""Post-lock failure analysis for the frozen v5a aperture gate.

This script is descriptive only. It must not be used to change the v5a method,
threshold, or claim after the first-open lock has been observed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT = ROOT / "results" / "v5a_blind_aperture_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for row in raw:
        if row["split"] != "independent_lock":
            continue
        rows.append(
            {
                **row,
                "sample_index": int(row["sample_index"]),
                "true_aperture_radius": float(row["true_aperture_radius"]),
                "active_reconstruction_views": int(row["active_reconstruction_views"]),
                "accepted": row["accepted"] == "True",
                "confidence": float(row["confidence"]),
                "estimated_aperture_radius": float(row["estimated_aperture_radius"]),
                "selected_gain_percent": float(row["selected_gain_percent"]),
            }
        )
    if not rows:
        raise ValueError("independent_lock rows are missing")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for the tiny deterministic diagnostic arrays."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray, *, ranked: bool = False) -> float:
    if ranked:
        x, y = rankdata(x), rankdata(y)
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def group_summary(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    groups: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for value, group in sorted(groups.items(), key=lambda item: str(item[0])):
        gain = np.asarray([float(row["selected_gain_percent"]) for row in group])
        accepted = [row for row in group if bool(row["accepted"])]
        accepted_gain = np.asarray(
            [float(row["selected_gain_percent"]) for row in accepted]
        )
        output.append(
            {
                key: value,
                "count": len(group),
                "accepted_count": int(sum(bool(row["accepted"]) for row in group)),
                "coverage": float(np.mean([bool(row["accepted"]) for row in group])),
                "mean_gain_percent": float(np.mean(gain)),
                "p10_gain_percent": float(np.quantile(gain, 0.10)),
                "harm_rate_over_1_percent": float(np.mean(gain < -1.0)),
                "accepted_mean_gain_percent": (
                    float(np.mean(accepted_gain)) if accepted else 0.0
                ),
                "accepted_p10_gain_percent": (
                    float(np.quantile(accepted_gain, 0.10)) if accepted else 0.0
                ),
                "accepted_harm_rate_over_1_percent": (
                    float(np.mean(accepted_gain < -1.0)) if accepted else 0.0
                ),
                "mean_confidence": float(
                    np.mean([float(row["confidence"]) for row in group])
                ),
                "mean_estimated_radius": float(
                    np.mean([float(row["estimated_aperture_radius"]) for row in group])
                ),
            }
        )
    return output


def build_diagnostic(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted = [row for row in rows if bool(row["accepted"])]
    confidence = np.asarray([float(row["confidence"]) for row in rows])
    gain = np.asarray([float(row["selected_gain_percent"]) for row in rows])
    accepted_confidence = np.asarray([float(row["confidence"]) for row in accepted])
    accepted_gain = np.asarray([float(row["selected_gain_percent"]) for row in accepted])
    audit_change = np.asarray(
        [
            100.0
            * (
                float(row["selected_candidate_audit_true_operator_residual_rms"])
                - float(row["pinhole_fista_equal_calls_audit_true_operator_residual_rms"])
            )
            / max(
                float(row["pinhole_fista_equal_calls_audit_true_operator_residual_rms"]),
                1e-12,
            )
            for row in accepted
        ]
    )
    audit_base = np.asarray(
        [
            float(row["pinhole_fista_equal_calls_audit_true_operator_residual_rms"])
            for row in rows
        ]
    )
    audit_candidate = np.asarray(
        [
            float(row["selected_candidate_audit_true_operator_residual_rms"])
            for row in rows
        ]
    )
    extreme_count = sum(
        float(row["estimated_aperture_radius"]) in {0.0, 0.16} for row in rows
    )
    wrong_extreme = sum(
        bool(row["accepted"])
        and abs(
            float(row["estimated_aperture_radius"])
            - float(row["true_aperture_radius"])
        )
        >= 0.06
        for row in rows
    )
    worst = sorted(rows, key=lambda row: float(row["selected_gain_percent"]))[:10]
    return {
        "status": "POST_LOCK_DIAGNOSTIC_ONLY_V5A_REMAINS_FAILED",
        "claim_boundary": (
            "descriptive analysis of the already opened synthetic lock; it cannot "
            "change the frozen v5a method, threshold, or failed gate"
        ),
        "lock_count": len(rows),
        "accepted_count": len(accepted),
        "accepted_only_risk": {
            "mean_gain_percent": float(np.mean(accepted_gain)),
            "p10_gain_percent": float(np.quantile(accepted_gain, 0.10)),
            "harm_rate_over_1_percent": float(np.mean(accepted_gain < -1.0)),
            "audit_reprojection_increase_rate": float(np.mean(audit_change > 0.0)),
            "maximum_audit_reprojection_increase_percent": float(
                np.max(audit_change)
            ),
        },
        "audit_metric_sensitivity": {
            "mean_of_samplewise_percent_change": float(
                np.mean(100.0 * (audit_candidate - audit_base) / audit_base)
            ),
            "percent_change_of_mean_rms": float(
                100.0
                * (np.mean(audit_candidate) - np.mean(audit_base))
                / np.mean(audit_base)
            ),
            "baseline_mean_rms": float(np.mean(audit_base)),
            "candidate_mean_rms": float(np.mean(audit_candidate)),
        },
        "extreme_radius_choice_count": int(extreme_count),
        "accepted_large_radius_error_count": int(wrong_extreme),
        "confidence_gain_correlation": {
            "all_pearson": correlation(confidence, gain),
            "all_spearman": correlation(confidence, gain, ranked=True),
            "accepted_pearson": correlation(accepted_confidence, accepted_gain),
            "accepted_spearman": correlation(
                accepted_confidence, accepted_gain, ranked=True
            ),
        },
        "by_family": group_summary(rows, "family"),
        "by_true_aperture_radius": group_summary(rows, "true_aperture_radius"),
        "by_active_reconstruction_views": group_summary(
            rows, "active_reconstruction_views"
        ),
        "worst_samples": [
            {
                key: row[key]
                for key in (
                    "sample_index",
                    "geometry_id",
                    "family",
                    "true_aperture_radius",
                    "active_reconstruction_views",
                    "accepted",
                    "confidence",
                    "estimated_aperture_radius",
                    "selected_gain_percent",
                )
            }
            for row in worst
        ],
        "mechanism_inference": (
            "The full-residual aperture selector separates the two held-out field "
            "families into opposite bank extremes, while confidence does not rank "
            "reconstruction gain. This is evidence of morphology/operator confounding, "
            "not a calibrated aperture estimator."
        ),
    }


def write_figure(path: Path, rows: list[dict[str, object]], report: dict[str, object]) -> None:
    family = report["by_family"]
    views = report["by_active_reconstruction_views"]
    colors = {"helical_plume": "#a34f43", "stratified_ignition": "#146f66"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    labels = [str(item["family"]).replace("_", " ") for item in family]
    axes[0].bar(
        labels,
        [float(item["mean_gain_percent"]) for item in family],
        color=[colors.get(str(item["family"]), "#315f93") for item in family],
    )
    axes[0].axhline(0.0, color="black", lw=1)
    axes[0].set(title="Family-confounded gain", ylabel="mean selected gain (%)")
    axes[0].tick_params(axis="x", rotation=18)

    for family_name in sorted(colors):
        subset = [row for row in rows if row["family"] == family_name]
        axes[1].scatter(
            [float(row["true_aperture_radius"]) for row in subset],
            [float(row["estimated_aperture_radius"]) for row in subset],
            label=family_name.replace("_", " "),
            color=colors[family_name],
            alpha=0.72,
        )
    axes[1].plot([0.04, 0.16], [0.04, 0.16], color="black", lw=1)
    axes[1].set(
        title="Morphology beats aperture",
        xlabel="truth-only radius",
        ylabel="estimated radius",
    )
    axes[1].legend(frameon=False, fontsize=8)

    for family_name in sorted(colors):
        subset = [row for row in rows if row["family"] == family_name]
        axes[2].scatter(
            [float(row["confidence"]) for row in subset],
            [float(row["selected_gain_percent"]) for row in subset],
            color=colors[family_name],
            alpha=0.78,
        )
    axes[2].axhline(0.0, color="black", lw=1)
    axes[2].set(
        title="Confidence does not rank risk",
        xlabel="frozen confidence",
        ylabel="selected gain (%)",
    )

    x = np.arange(len(views))
    axes[3].bar(
        x - 0.18,
        [float(item["mean_gain_percent"]) for item in views],
        width=0.36,
        label="mean",
        color="#315f93",
    )
    axes[3].bar(
        x + 0.18,
        [float(item["p10_gain_percent"]) for item in views],
        width=0.36,
        label="p10",
        color="#d0933b",
    )
    axes[3].axhline(0.0, color="black", lw=1)
    axes[3].set_xticks(
        x, [str(item["active_reconstruction_views"]) for item in views]
    )
    axes[3].set(title="View-count tail", xlabel="active views", ylabel="gain (%)")
    axes[3].legend(frameon=False, fontsize=8)

    fig.suptitle("v5a post-lock diagnosis: failed gate, no rescue tuning", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.result_dir / "sample_metrics.csv")
    frozen = json.loads((args.result_dir / "report.json").read_text(encoding="utf-8"))
    if "FAILED_OR_INCOMPLETE" not in frozen["claim_status"]:
        raise ValueError("v5a report is not the expected failed first-open result")
    report = build_diagnostic(rows)
    output_json = args.result_dir / "failure_diagnosis.json"
    output_png = args.result_dir / "v5a_failure_diagnosis.png"
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_figure(output_png, rows, report)
    checksum = args.result_dir / "failure_diagnosis_checksums.sha256"
    checked = [
        (Path(__file__), f"../../{Path(__file__).name}"),
        (output_json, output_json.name),
        (output_png, output_png.name),
    ]
    checksum.write_text(
        "\n".join(f"{sha256(path)}  {label}" for path, label in checked) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
