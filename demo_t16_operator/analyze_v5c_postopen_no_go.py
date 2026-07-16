#!/usr/bin/env python3
"""Create a read-only post-open diagnosis of the frozen v5c no-go result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "v5c_nested_crossview_first_open"
DEFAULT_OUTPUT = ROOT / "results" / "v5c_nested_crossview_postopen_diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_distribution(values: Sequence[float]) -> dict[str, float | None]:
    data = np.asarray(tuple(values), dtype=float)
    if data.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)),
        "p90": float(np.percentile(data, 90)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def summarize(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads((input_dir / "report.json").read_text(encoding="utf-8"))
    blocks = read_csv(input_dir / "block_metrics.csv")
    samples = read_csv(input_dir / "sample_metrics.csv")
    candidates = read_csv(input_dir / "crossview_candidates.csv")

    main_candidates = [
        row for row in candidates if row["method"] == "radius_kappa"
    ]
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in main_candidates:
        by_block[row["block_id"]].append(row)

    kappa_profiles: dict[str, Any] = {}
    max_kappa = max(float(row["kappa"]) for row in main_candidates)
    selected_kappa_at_upper_boundary = 0
    strictly_decreasing_profiles = 0
    for block_id, rows in sorted(by_block.items()):
        profile: list[dict[str, Any]] = []
        for kappa in sorted({float(row["kappa"]) for row in rows}):
            at_kappa = [row for row in rows if float(row["kappa"]) == kappa]
            best = min(at_kappa, key=lambda row: float(row["mean_validation_mse"]))
            profile.append(
                {
                    "kappa": kappa,
                    "best_radius": float(best["candidate_aperture_radius"]),
                    "best_validation_mse": float(best["mean_validation_mse"]),
                }
            )
        selected = [row for row in rows if row["selected"].lower() == "true"]
        if len(selected) != 1:
            raise ValueError(f"expected one selected candidate for {block_id}")
        selected_kappa = float(selected[0]["kappa"])
        selected_kappa_at_upper_boundary += int(selected_kappa == max_kappa)
        decreasing = all(
            right["best_validation_mse"] < left["best_validation_mse"]
            for left, right in zip(profile, profile[1:])
        )
        strictly_decreasing_profiles += int(decreasing)
        kappa_profiles[block_id] = {
            "selected_kappa": selected_kappa,
            "selected_radius": float(selected[0]["candidate_aperture_radius"]),
            "strictly_decreasing_over_frozen_grid": decreasing,
            "profile": profile,
        }

    changed_samples = [
        row
        for row in samples
        if float(row["selected_radius"]) != float(row["metadata_nearest_radius"])
    ]
    no_action_samples = [
        row for row in samples if row["outcome_code"] == "NO_ACTION_FALLBACK"
    ]
    raw_field_gain = [float(row["raw_field_gain_percent"]) for row in changed_samples]
    raw_audit_change = [
        float(row["raw_audit_change_percent"]) for row in changed_samples
    ]
    min_outer = [
        float(row["minimum_outer_improvement_percent"]) for row in changed_samples
    ]

    changed_block_summary: dict[str, Any] = {}
    for block_id in sorted({row["block_id"] for row in changed_samples}):
        rows = [row for row in changed_samples if row["block_id"] == block_id]
        field = [float(row["raw_field_gain_percent"]) for row in rows]
        audit = [float(row["raw_audit_change_percent"]) for row in rows]
        outer = [float(row["minimum_outer_improvement_percent"]) for row in rows]
        changed_block_summary[block_id] = {
            "sample_count": len(rows),
            "raw_field_gain_percent": optional_distribution(field),
            "raw_field_harm_below_minus_1_count": int(np.sum(np.asarray(field) < -1.0)),
            "raw_audit_change_percent": optional_distribution(audit),
            "raw_audit_increase_count": int(np.sum(np.asarray(audit) > 0.0)),
            "minimum_outer_improvement_percent": optional_distribution(outer),
        }

    reason_counts = Counter(
        reason
        for row in samples
        for reason in row["block_gate_minus_sample_reasons"].split("|")
        if reason
    )
    deletion_unique_counts = []
    deletion_spans = []
    for row in blocks:
        radii = np.asarray(
            [float(value) for value in row["camera_deletion_selected_radii"].split("|")],
            dtype=float,
        )
        deletion_unique_counts.append(len(set(radii.tolist())))
        deletion_spans.append(float(np.max(radii) - np.min(radii)))

    diagnosis = {
        "status": "V5C_POSTOPEN_NO_GO_DIAGNOSIS_NOT_A_NEW_LOCK",
        "source_first_open_commit": report["preopen_git_commit"],
        "source_report_sha256": sha256(input_dir / "report.json"),
        "source_claim_status": report["claim_status"],
        "verdict": "NO_GO_JOINT_RADIUS_KAPPA_CROSSVIEW_SELECTION",
        "primary_findings": {
            "nearest_bank_match_blocks": int(
                np.sum([row["nearest_bank_match"].lower() == "true" for row in blocks])
            ),
            "block_count": len(blocks),
            "fixed_ridge_nearest_bank_match_blocks": int(
                np.sum(
                    [
                        float(row["fixed_ridge_profile_radius"])
                        == float(row["oracle_nearest_radius"])
                        for row in blocks
                    ]
                )
            ),
            "operator_matrix_oracle_match_blocks": int(
                np.sum(
                    [
                        float(row["closest_operator_matrix_radius"])
                        == float(row["oracle_nearest_radius"])
                        for row in blocks
                    ]
                )
            ),
            "clean_truth_oracle_match_blocks": int(
                np.sum(
                    [
                        float(row["clean_truth_field_radius"])
                        == float(row["oracle_nearest_radius"])
                        for row in blocks
                    ]
                )
            ),
            "noisy_truth_oracle_match_blocks": int(
                np.sum(
                    [
                        float(row["noisy_truth_field_radius"])
                        == float(row["oracle_nearest_radius"])
                        for row in blocks
                    ]
                )
            ),
            "mean_true_camera_deletion_radius_stability": float(
                np.mean(
                    [
                        float(row["camera_deletion_radius_stability_fraction"])
                        for row in blocks
                    ]
                )
            ),
            "fully_camera_deletion_stable_blocks": int(
                np.sum(
                    [
                        float(row["camera_deletion_radius_stability_fraction"]) == 1.0
                        for row in blocks
                    ]
                )
            ),
            "selected_kappa_upper_boundary_blocks": selected_kappa_at_upper_boundary,
            "strictly_decreasing_best_cv_profiles": strictly_decreasing_profiles,
            "changed_operator_blocks": len(
                {
                    row["block_id"]
                    for row in blocks
                    if float(row["selected_radius"])
                    != float(row["metadata_nearest_radius"])
                }
            ),
            "no_action_sample_rows": len(no_action_samples),
            "accepted_sample_rows": int(
                np.sum([row["accepted"].lower() == "true" for row in samples])
            ),
        },
        "camera_deletion": {
            "unique_selected_radius_count_by_block": deletion_unique_counts,
            "radius_span_by_block": deletion_spans,
            "mean_radius_span": float(np.mean(deletion_spans)),
        },
        "changed_candidate_only": {
            "nominal_sample_rows": len(changed_samples),
            "raw_field_gain_percent": optional_distribution(raw_field_gain),
            "raw_field_harm_below_minus_1_count": int(
                np.sum(np.asarray(raw_field_gain) < -1.0)
            ),
            "raw_audit_change_percent_positive_is_worse": optional_distribution(
                raw_audit_change
            ),
            "raw_audit_increase_count": int(
                np.sum(np.asarray(raw_audit_change) > 0.0)
            ),
            "minimum_outer_improvement_percent": optional_distribution(min_outer),
            "outer_safe_and_audit_safe_contingency": {
                "outer_min_at_least_2_and_audit_nonincrease": int(
                    np.sum(
                        (np.asarray(min_outer) >= 2.0)
                        & (np.asarray(raw_audit_change) <= 0.0)
                    )
                ),
                "outer_min_at_least_2_and_audit_increase": int(
                    np.sum(
                        (np.asarray(min_outer) >= 2.0)
                        & (np.asarray(raw_audit_change) > 0.0)
                    )
                ),
                "outer_min_below_2_and_audit_nonincrease": int(
                    np.sum(
                        (np.asarray(min_outer) < 2.0)
                        & (np.asarray(raw_audit_change) <= 0.0)
                    )
                ),
                "outer_min_below_2_and_audit_increase": int(
                    np.sum(
                        (np.asarray(min_outer) < 2.0)
                        & (np.asarray(raw_audit_change) > 0.0)
                    )
                ),
            },
            "by_block": changed_block_summary,
        },
        "gate_reason_counts_over_nominal_rows": dict(reason_counts),
        "kappa_profiles": kappa_profiles,
        "interpretation": {
            "operator_semantics_failed": False,
            "joint_radius_kappa_cv_failed": True,
            "zero_coverage_is_safety_evidence": False,
            "all_kappa_at_boundary_invalidates_interior_optimum_claim": True,
            "camera_deletion_instability_invalidates_shared_radius_lock": True,
            "fixed_ridge_outperforming_joint_cv_is_superiority_evidence": False,
        },
        "next_falsifiable_methods": [
            "GCV or UPRE with effective degrees of freedom, then radius selection",
            "Morozov discrepancy selection using externally calibrated noise",
            "equal-complexity or equal-effective-DOF radius comparison",
            "genuinely nested inner-camera CV that tunes kappa before scoring radius",
            "camera-consensus radius score with a predeclared instability fallback",
        ],
        "next_lock_boundary": {
            "development_data": "opened v5c only for diagnosis and method development",
            "new_generator_families": ["helical_plume", "stratified_ignition"],
            "new_seed_required": True,
            "new_rig_or_session_draw_required": True,
            "real_bos_claim_authorized": False,
            "neural_operator_superiority_claim_authorized": False,
        },
    }
    plotting = {
        "blocks": blocks,
        "changed_samples": changed_samples,
        "kappa_profiles": kappa_profiles,
    }
    return diagnosis, plotting


def write_figure(path: Path, plotting: dict[str, Any]) -> None:
    blocks = plotting["blocks"]
    changed = plotting["changed_samples"]
    profiles = plotting["kappa_profiles"]
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), constrained_layout=True)

    truth = np.asarray([float(row["true_aperture_radius"]) for row in blocks])
    selected = np.asarray([float(row["selected_radius"]) for row in blocks])
    fixed = np.asarray([float(row["fixed_ridge_profile_radius"]) for row in blocks])
    oracle = np.asarray([float(row["oracle_nearest_radius"]) for row in blocks])
    axes[0, 0].plot([0.04, 0.15], [0.04, 0.15], "k--", lw=1, label="nearest-bank ideal")
    axes[0, 0].scatter(truth, selected, marker="x", s=85, color="#d1495b", label="joint CV")
    axes[0, 0].scatter(truth, fixed, marker="s", s=48, facecolors="none", edgecolors="#00798c", label="fixed ridge")
    axes[0, 0].scatter(truth, oracle, marker=".", s=45, color="#222222", label="oracle bank")
    axes[0, 0].set(title="Calibration ranking after opening", xlabel="true radius", ylabel="selected radius")
    axes[0, 0].legend(fontsize=8)

    for block_id, record in profiles.items():
        kappa = [row["kappa"] for row in record["profile"]]
        mse = [row["best_validation_mse"] for row in record["profile"]]
        axes[0, 1].plot(kappa, mse, marker="o", ms=3, alpha=0.8, label=block_id.replace("development_", ""))
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Every best-CV curve runs into kappa max", xlabel="dimensionless kappa", ylabel="best inner CV MSE")
    axes[0, 1].legend(fontsize=6, ncol=2)

    for index, row in enumerate(blocks):
        deletion = np.asarray([float(value) for value in row["camera_deletion_selected_radii"].split("|")])
        axes[1, 0].plot(np.arange(len(deletion)), deletion, color="#888888", alpha=0.45)
        axes[1, 0].scatter(np.arange(len(deletion)), deletion, s=28, label=row["block_id"] if index == 0 else None)
        axes[1, 0].axhline(float(row["selected_radius"]), color="#d1495b", alpha=0.10)
    axes[1, 0].set_xticks(range(4), ["delete cam 0", "delete cam 1", "delete cam 2", "delete cam 3"])
    axes[1, 0].set(title="Deleting one inner camera changes selected radius", ylabel="selected radius")

    field_gain = np.asarray([float(row["raw_field_gain_percent"]) for row in changed])
    audit_change = np.asarray([float(row["raw_audit_change_percent"]) for row in changed])
    outer = np.asarray([float(row["minimum_outer_improvement_percent"]) for row in changed])
    points = axes[1, 1].scatter(field_gain, audit_change, c=outer, cmap="coolwarm", s=55, edgecolors="white", linewidths=0.4)
    axes[1, 1].axhline(0.0, color="black", lw=1)
    axes[1, 1].axvline(0.0, color="black", lw=1)
    axes[1, 1].set(title="Changed candidates: field gain vs audit change", xlabel="raw field L2 gain (%)", ylabel="raw audit RMS change (%; positive is worse)")
    figure.colorbar(points, ax=axes[1, 1], label="minimum outer improvement (%)")

    figure.suptitle("v5c post-open diagnosis: joint radius-kappa CV is a no-go", fontsize=14)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def write_readme(path: Path, diagnosis: dict[str, Any]) -> None:
    primary = diagnosis["primary_findings"]
    changed = diagnosis["changed_candidate_only"]
    field = changed["raw_field_gain_percent"]
    audit = changed["raw_audit_change_percent_positive_is_worse"]
    text = f"""# v5c post-open no-go diagnosis

Status: `{diagnosis['status']}`

This directory is a read-only diagnosis of the committed first-open result. It
does not alter the frozen protocol or create a new lock.

## What failed

- joint radius-kappa CV recovered the nearest bank radius in
  `{primary['nearest_bank_match_blocks']} / {primary['block_count']}` blocks;
- fixed-ridge profiling recovered `{primary['fixed_ridge_nearest_bank_match_blocks']} / {primary['block_count']}`;
- operator, clean-truth and noisy-truth oracles remained
  `{primary['operator_matrix_oracle_match_blocks']} / {primary['block_count']}`,
  `{primary['clean_truth_oracle_match_blocks']} / {primary['block_count']}`, and
  `{primary['noisy_truth_oracle_match_blocks']} / {primary['block_count']}`;
- all `{primary['selected_kappa_upper_boundary_blocks']} / {primary['block_count']}`
  blocks selected the largest frozen kappa, and all best-CV curves decreased
  monotonically over the complete grid;
- mean true camera-deletion radius stability was
  `{primary['mean_true_camera_deletion_radius_stability']:.3f}`; no block was
  stable under all four deletions;
- final coverage was `{primary['accepted_sample_rows']} / 48`.

The forward model still ranks radius correctly when truth is available. The
failure is therefore localized to the current joint predictive-selection rule,
not to the finite-aperture bank semantics.

## Why zero coverage is not success

Only `{primary['changed_operator_blocks']}` of six blocks changed the
metadata operator. The other `{primary['no_action_sample_rows']}` rows were
explicit fallback. Among the 16 rows where the candidate actually changed:

- mean raw field gain was `{field['mean']:.3f}%`;
- `{changed['raw_field_harm_below_minus_1_count']} / 16` lost more than 1% field accuracy;
- mean raw audit change was `{audit['mean']:.3f}%` (positive is worse);
- `{changed['raw_audit_increase_count']} / 16` worsened the audit camera.

The strict gate correctly rejected these candidates, but a method that always
falls back has no demonstrated utility.

## Next algorithm test

Do not enlarge the network yet. On the opened development data, compare GCV,
UPRE/effective degrees of freedom, Morozov discrepancy, equal-complexity radius
scores, and genuinely nested inner-camera CV. Freeze the winning rule before
opening new `helical_plume` and `stratified_ignition` families, a new seed,
and a new rig/session draw.

None of these numbers supports real BOS, OERF, arbitrary-view safety, or
DeepONet/FNO/FFNO superiority.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"post-open diagnosis output exists: {output_dir}")
    diagnosis, plotting = summarize(input_dir)
    output_dir.mkdir(parents=True)
    report_path = output_dir / "diagnosis.json"
    report_path.write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figure_path = output_dir / "v5c_postopen_diagnosis.png"
    write_figure(figure_path, plotting)
    readme_path = output_dir / "README.md"
    write_readme(readme_path, diagnosis)
    targets = [report_path, figure_path, readme_path]
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in targets),
        encoding="utf-8",
    )
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
