#!/usr/bin/env python3
"""Test separate calibration and reconstruction kappas on opened v5c."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .dual_regularization import (
        error_reduction_percent,
        outer_only_route,
        refit_fixed_radius_with_method,
    )
    from .nested_crossview import whitened_per_view_rms
    from .run_v5b_rig_shared_profile_pilot import (
        DevelopmentBlock,
        build_development_blocks,
        nearest_index,
        read_json,
        relative_l2,
        support_mask_from_config,
    )
    from .run_v5d_decoupled_complexity_screening import (
        sha256,
        write_checksums,
        write_csv,
    )
except ImportError:
    from dual_regularization import (
        error_reduction_percent,
        outer_only_route,
        refit_fixed_radius_with_method,
    )
    from nested_crossview import whitened_per_view_rms
    from run_v5b_rig_shared_profile_pilot import (
        DevelopmentBlock,
        build_development_blocks,
        nearest_index,
        read_json,
        relative_l2,
        support_mask_from_config,
    )
    from run_v5d_decoupled_complexity_screening import (
        sha256,
        write_checksums,
        write_csv,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5f_dual_regularization_postopen.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5f_dual_regularization_postopen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_radius_by_block(
    rows: Sequence[dict[str, str]], variant_id: str
) -> dict[str, dict[str, str]]:
    selected = [row for row in rows if row["variant_id"] == variant_id]
    if not selected:
        raise ValueError("calibration variant has no source decisions")
    output = {row["block_id"]: row for row in selected}
    if len(output) != len(selected):
        raise ValueError("calibration variant contains duplicate block decisions")
    return output


def refit_pair(
    block: DevelopmentBlock,
    selected_radius_index: int,
    metadata_radius_index: int,
    method: str,
    support: np.ndarray,
    kappas: Sequence[float],
):
    candidate = refit_fixed_radius_with_method(
        selected_radius_index,
        method,
        block.reconstruction_bank,
        block.observations,
        block.noise_std,
        block.inner_views,
        support,
        kappas,
    )
    baseline = refit_fixed_radius_with_method(
        metadata_radius_index,
        method,
        block.reconstruction_bank,
        block.observations,
        block.noise_std,
        block.inner_views,
        support,
        kappas,
    )
    return candidate, baseline


def sample_rows_for_refit(
    block: DevelopmentBlock,
    radii: np.ndarray,
    selected_source: dict[str, str],
    method: str,
    support: np.ndarray,
    kappas: Sequence[float],
    route_thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    selected_radius_index = nearest_index(
        radii, float(selected_source["selected_radius"])
    )
    metadata_radius_index = nearest_index(radii, block.metadata_radius)
    truth_radius_index = nearest_index(radii, block.true_radius)
    candidate, baseline = refit_pair(
        block,
        selected_radius_index,
        metadata_radius_index,
        method,
        support,
        kappas,
    )
    oracle = refit_fixed_radius_with_method(
        truth_radius_index,
        method,
        block.reconstruction_bank,
        block.observations,
        block.noise_std,
        block.inner_views,
        support,
        kappas,
    )
    rows: list[dict[str, Any]] = []
    for index, (truth, observation, sigma, family) in enumerate(
        zip(
            block.fields,
            block.observations,
            block.noise_std,
            block.families,
            strict=True,
        )
    ):
        candidate_field = candidate.refit.fits[index].field
        baseline_field = baseline.refit.fits[index].field
        oracle_field = oracle.refit.fits[index].field
        candidate_field_error = relative_l2(candidate_field, truth)
        baseline_field_error = relative_l2(baseline_field, truth)
        oracle_field_error = relative_l2(oracle_field, truth)
        candidate_outer = whitened_per_view_rms(
            block.reconstruction_bank[selected_radius_index],
            candidate_field,
            observation,
            sigma,
            block.outer_views,
        )
        baseline_outer = whitened_per_view_rms(
            block.reconstruction_bank[metadata_radius_index],
            baseline_field,
            observation,
            sigma,
            block.outer_views,
        )
        outer_reductions = tuple(
            error_reduction_percent(candidate_value, baseline_value)
            for candidate_value, baseline_value in zip(
                candidate_outer, baseline_outer, strict=True
            )
        )
        candidate_audit = whitened_per_view_rms(
            block.reconstruction_bank[selected_radius_index],
            candidate_field,
            observation,
            sigma,
            block.audit_views,
        )[0]
        baseline_audit = whitened_per_view_rms(
            block.reconstruction_bank[metadata_radius_index],
            baseline_field,
            observation,
            sigma,
            block.audit_views,
        )[0]
        row: dict[str, Any] = {
            "rig_id": block.rig_id,
            "block_id": block.block_id,
            "sample_index": index,
            "family": family,
            "reconstruction_method": method,
            "true_aperture_radius": block.true_radius,
            "nearest_truth_bank_radius": float(radii[truth_radius_index]),
            "metadata_aperture_radius": block.metadata_radius,
            "metadata_bank_radius": float(radii[metadata_radius_index]),
            "selected_calibration_radius": float(radii[selected_radius_index]),
            "radius_changed_from_metadata": bool(
                selected_radius_index != metadata_radius_index
            ),
            "calibration_radius_matches_nearest_truth": bool(
                selected_radius_index == truth_radius_index
            ),
            "calibration_fold_deletion_stability": float(
                selected_source[
                    "fold_score_deletion_radius_stability_fraction"
                ]
            ),
            "calibration_relative_radius_margin": float(
                selected_source["relative_radius_margin"]
            ),
            "candidate_reconstruction_kappa": candidate.choice.kappa,
            "baseline_reconstruction_kappa": baseline.choice.kappa,
            "oracle_reconstruction_kappa": oracle.choice.kappa,
            "candidate_field_relative_l2": candidate_field_error,
            "baseline_field_relative_l2": baseline_field_error,
            "oracle_field_relative_l2": oracle_field_error,
            "raw_field_error_reduction_percent": error_reduction_percent(
                candidate_field_error, baseline_field_error
            ),
            "candidate_outer_rms": "|".join(
                f"{value:.12g}" for value in candidate_outer
            ),
            "baseline_outer_rms": "|".join(
                f"{value:.12g}" for value in baseline_outer
            ),
            "outer_error_reductions_percent": "|".join(
                f"{value:.12g}" for value in outer_reductions
            ),
            "minimum_outer_error_reduction_percent": float(
                min(outer_reductions)
            ),
            "candidate_audit_rms": candidate_audit,
            "baseline_audit_rms": baseline_audit,
            "raw_audit_error_reduction_percent": error_reduction_percent(
                candidate_audit, baseline_audit
            ),
            "decision_uses_truth": False,
            "decision_uses_audit": False,
        }
        for threshold in route_thresholds:
            label = f"route_ge_{threshold:.1f}pct".replace("-", "m").replace(".", "p")
            row[label] = outer_only_route(
                bool(row["radius_changed_from_metadata"]),
                outer_reductions,
                minimum_per_view_reduction_percent=float(threshold),
            )
        rows.append(row)
    return rows


def summarize_rows(
    rows: Sequence[dict[str, Any]], route_thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({str(row["reconstruction_method"]) for row in rows}):
        selected = [row for row in rows if row["reconstruction_method"] == method]
        field_gain = np.asarray(
            [float(row["raw_field_error_reduction_percent"]) for row in selected]
        )
        audit_gain = np.asarray(
            [float(row["raw_audit_error_reduction_percent"]) for row in selected]
        )
        changed = np.asarray(
            [bool(row["radius_changed_from_metadata"]) for row in selected]
        )
        base: dict[str, Any] = {
            "reconstruction_method": method,
            "sample_count": len(selected),
            "changed_radius_sample_count": int(np.sum(changed)),
            "raw_mean_field_error_reduction_percent": float(np.mean(field_gain)),
            "raw_p10_field_error_reduction_percent": float(
                np.quantile(field_gain, 0.10)
            ),
            "raw_field_harm_rate_over_1_percent": float(
                np.mean(field_gain < -1.0)
            ),
            "raw_mean_audit_error_reduction_percent": float(np.mean(audit_gain)),
        }
        for threshold in route_thresholds:
            label = f"route_ge_{threshold:.1f}pct".replace("-", "m").replace(".", "p")
            accepted = np.asarray([bool(row[label]) for row in selected])
            selected_field_gain = np.where(accepted, field_gain, 0.0)
            selected_audit_gain = np.where(accepted, audit_gain, 0.0)
            prefix = label
            base[f"{prefix}_accepted_count"] = int(np.sum(accepted))
            base[f"{prefix}_coverage"] = float(np.mean(accepted))
            base[f"{prefix}_mean_selected_field_gain_percent"] = float(
                np.mean(selected_field_gain)
            )
            base[f"{prefix}_p10_selected_field_gain_percent"] = float(
                np.quantile(selected_field_gain, 0.10)
            )
            base[f"{prefix}_selected_field_harm_rate_over_1_percent"] = float(
                np.mean(selected_field_gain < -1.0)
            )
            base[f"{prefix}_mean_selected_audit_gain_percent"] = float(
                np.mean(selected_audit_gain)
            )
            base[f"{prefix}_accepted_audit_harm_rate"] = (
                0.0 if not np.any(accepted) else float(np.mean(audit_gain[accepted] < 0.0))
            )
        output.append(base)
    return output


def write_figure(
    path: Path,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    methods = [str(row["reconstruction_method"]) for row in summaries]
    gains = [
        [
            float(row["raw_field_error_reduction_percent"])
            for row in rows
            if row["reconstruction_method"] == method
            and row["radius_changed_from_metadata"]
        ]
        for method in methods
    ]
    axes[0, 0].boxplot(gains, tick_labels=methods, showmeans=True)
    axes[0, 0].axhline(0.0, color="tab:gray", linestyle="--")
    axes[0, 0].set(
        title="Field gain where calibration changes radius",
        ylabel="relative-L2 error reduction (%)",
    )

    for method, color in zip(methods, ("tab:blue", "tab:green"), strict=False):
        selected = [row for row in rows if row["reconstruction_method"] == method]
        axes[0, 1].scatter(
            [float(row["minimum_outer_error_reduction_percent"]) for row in selected],
            [float(row["raw_audit_error_reduction_percent"]) for row in selected],
            label=method,
            alpha=0.75,
            color=color,
        )
    axes[0, 1].axhline(0.0, color="tab:gray", linestyle="--")
    axes[0, 1].axvline(0.0, color="tab:gray", linestyle="--")
    axes[0, 1].set(
        title="Can outer cameras predict the audit camera?",
        xlabel="worst outer-camera error reduction (%)",
        ylabel="audit-camera error reduction (%)",
    )
    axes[0, 1].legend()

    locations = np.arange(len(methods))
    width = 0.36
    axes[1, 0].bar(
        locations - width / 2,
        [float(row["route_ge_0p0pct_coverage"]) for row in summaries],
        width,
        label="outer >= 0%",
    )
    axes[1, 0].bar(
        locations + width / 2,
        [float(row["route_ge_2p0pct_coverage"]) for row in summaries],
        width,
        label="outer >= 2%",
    )
    axes[1, 0].set(
        title="Selective coverage",
        ylabel="fraction of all samples",
        xticks=locations,
        xticklabels=methods,
        ylim=(0.0, 1.0),
    )
    axes[1, 0].legend()

    axes[1, 1].bar(
        locations - width / 2,
        [
            float(row["route_ge_0p0pct_mean_selected_field_gain_percent"])
            for row in summaries
        ],
        width,
        label="field gain",
    )
    axes[1, 1].bar(
        locations + width / 2,
        [
            float(row["route_ge_0p0pct_mean_selected_audit_gain_percent"])
            for row in summaries
        ],
        width,
        label="audit gain",
    )
    axes[1, 1].axhline(0.0, color="tab:gray", linestyle="--")
    axes[1, 1].set(
        title="Outer-nonworse routed result",
        ylabel="mean error reduction over all samples (%)",
        xticks=locations,
        xticklabels=methods,
    )
    axes[1, 1].legend()
    figure.suptitle(
        "v5f dual-regularization post-open development - not confirmatory",
        fontsize=15,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"post-open output already exists: {output}")
    config = read_json(config_path)
    source_config_path = (ROOT / str(config["source_config"])).resolve()
    source_report_path = (ROOT / str(config["source_v5e_report"])).resolve()
    source_selection_path = (
        ROOT / str(config["source_v5e_selection_rows"])
    ).resolve()
    source_config = read_json(source_config_path)
    source_rows = selected_radius_by_block(
        read_csv(source_selection_path), str(config["calibration_variant_id"])
    )
    methods = tuple(str(value) for value in config["reconstruction_methods"])
    if not methods or any(method not in {"gcv", "upre"} for method in methods):
        raise ValueError("reconstruction methods must be gcv and/or upre")
    if len(set(methods)) != len(methods):
        raise ValueError("reconstruction methods must be unique")
    thresholds = tuple(float(value) for value in config["outer_route_thresholds_percent"])
    if thresholds != tuple(sorted(set(thresholds))):
        raise ValueError("outer route thresholds must be sorted and unique")
    kappas = tuple(float(value) for value in config["kappas"])
    radii = np.asarray(source_config["candidate_aperture_radii"], dtype=float)
    support = support_mask_from_config(source_config)
    blocks, _ = build_development_blocks(source_config)
    if set(source_rows) != {block.block_id for block in blocks}:
        raise ValueError("source decisions do not match regenerated blocks")

    rows: list[dict[str, Any]] = []
    for block in blocks:
        for method in methods:
            rows.extend(
                sample_rows_for_refit(
                    block,
                    radii,
                    source_rows[block.block_id],
                    method,
                    support,
                    kappas,
                    thresholds,
                )
            )
    summaries = summarize_rows(rows, thresholds)
    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = output / "config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows_path = output / "sample_rows.csv"
    summary_path = output / "method_summary.csv"
    write_csv(rows_path, rows)
    write_csv(summary_path, summaries)
    figure_path = output / "v5f_dual_regularization_postopen.png"
    write_figure(figure_path, rows, summaries)
    report = {
        "claim_status": config["claim_status"],
        "scientific_boundary": (
            "post-open development on v5c decisions and truth; no new data, frozen "
            "session, real BOS, or superiority evidence"
        ),
        "calibration_method": "diagonal df-corrected discrepancy target 1.0",
        "field_reconstruction_methods": list(methods),
        "calibration_and_reconstruction_kappas_are_separate": True,
        "routing_uses_truth": False,
        "routing_uses_audit_camera": False,
        "method_summary": summaries,
        "independent_rig_session_count": len(source_config["rigs"]),
        "sample_rows_are_not_iid": True,
        "next_step": (
            "retain only a rule whose outer-only route predicts audit and field tails; "
            "then freeze it before any new family/seed/rig/session run"
        ),
        "source_hashes": {
            "config": sha256(config_path),
            "source_config": sha256(source_config_path),
            "source_v5e_report": sha256(source_report_path),
            "source_v5e_selection_rows": sha256(source_selection_path),
            "runner": sha256(Path(__file__).resolve()),
            "dual_regularization": sha256((ROOT / "dual_regularization.py").resolve()),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme_path = output / "README.md"
    readme_path.write_text(
        "# v5f dual-regularization post-open development\n\n"
        "The optical radius comes from the saved v5e df-corrected discrepancy "
        "decision. The final field is independently refit with GCV or UPRE. Outer "
        "cameras may route; truth and the audit camera are opened only for this "
        "post-open verdict. The same v5c fields and rigs are reused, so no result in "
        "this folder is confirmatory.\n",
        encoding="utf-8",
    )
    write_checksums(
        output,
        [config_snapshot, rows_path, summary_path, figure_path, report_path, readme_path],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
