#!/usr/bin/env python3
"""Diagnose geometry-aware and residual-field transfer on opened v5f data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .dual_regularization import error_reduction_percent
    from .nested_crossview import whitened_per_view_rms
    from .rig_shared_profile import whitened_support_system
    from .run_v5b_rig_shared_profile_pilot import (
        DevelopmentBlock,
        build_development_blocks,
        nearest_index,
        read_json,
        support_mask_from_config,
    )
    from .run_v5d_decoupled_complexity_screening import (
        sha256,
        write_checksums,
        write_csv,
    )
    from .run_v5f_dual_regularization_postopen import refit_pair
    from .view_transfer_geometry import (
        camera_support_matrix,
        gram_cosine,
        group_predictive_leverage,
        operator_change_cosine,
        projection_similarity,
        residual_field_transfer,
        similarity_weighted_gain,
    )
except ImportError:
    from dual_regularization import error_reduction_percent
    from nested_crossview import whitened_per_view_rms
    from rig_shared_profile import whitened_support_system
    from run_v5b_rig_shared_profile_pilot import (
        DevelopmentBlock,
        build_development_blocks,
        nearest_index,
        read_json,
        support_mask_from_config,
    )
    from run_v5d_decoupled_complexity_screening import (
        sha256,
        write_checksums,
        write_csv,
    )
    from run_v5f_dual_regularization_postopen import refit_pair
    from view_transfer_geometry import (
        camera_support_matrix,
        gram_cosine,
        group_predictive_leverage,
        operator_change_cosine,
        projection_similarity,
        residual_field_transfer,
        similarity_weighted_gain,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5g_geometry_residual_transfer_postopen.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5g_geometry_residual_transfer_postopen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def circular_angle_distance(first: float, second: float, period: float) -> float:
    """Shortest distance between two projection angles on a fixed period."""

    cycle = float(period)
    if not np.isfinite(cycle) or cycle <= 0.0:
        raise ValueError("period must be finite and positive")
    delta = (float(first) - float(second) + 0.5 * cycle) % cycle - 0.5 * cycle
    return float(abs(delta))


def selected_radius_by_block(
    rows: Sequence[dict[str, str]], variant_id: str
) -> dict[str, dict[str, str]]:
    selected = [row for row in rows if row["variant_id"] == variant_id]
    output = {row["block_id"]: row for row in selected}
    if not output or len(output) != len(selected):
        raise ValueError("calibration decisions are empty or duplicated")
    return output


def audit_gain_by_sample(
    rows: Sequence[dict[str, str]], method: str
) -> dict[tuple[str, int], float]:
    selected = [row for row in rows if row["reconstruction_method"] == method]
    output = {
        (row["block_id"], int(row["sample_index"])): float(
            row["raw_audit_error_reduction_percent"]
        )
        for row in selected
    }
    if not output or len(output) != len(selected):
        raise ValueError("audit outcomes are empty or duplicated")
    return output


def methods_have_identical_sample_rows(
    rows: Sequence[dict[str, str]], first: str, second: str
) -> bool:
    """Verify that choosing one duplicate v5f reconstruction method is lossless."""

    def indexed(method: str) -> dict[tuple[str, int], dict[str, str]]:
        return {
            (row["block_id"], int(row["sample_index"])): {
                key: value
                for key, value in row.items()
                if key != "reconstruction_method"
            }
            for row in rows
            if row["reconstruction_method"] == method
        }

    first_rows = indexed(first)
    second_rows = indexed(second)
    return bool(first_rows and first_rows == second_rows)


def attach_audit_outcomes(
    prediction_rows: Sequence[dict[str, Any]],
    outcomes: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    """Join opened labels only after every predictor has been constructed."""

    expected = {(row["block_id"], int(row["sample_index"])) for row in prediction_rows}
    if expected != set(outcomes):
        raise ValueError("prediction and audit sample keys disagree")
    return [
        {
            **row,
            "actual_audit_gain_percent": outcomes[
                (row["block_id"], int(row["sample_index"]))
            ],
        }
        for row in prediction_rows
    ]


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or float(np.std(first)) <= 1e-12 or float(np.std(second)) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _same_sign(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.sign(first) == np.sign(second)))


def predictor_summaries(
    rows: Sequence[dict[str, Any]], predictors: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize opened labels without treating sample rows as independent rigs."""

    changed = [row for row in rows if bool(row["radius_changed_from_metadata"])]
    if not changed:
        raise ValueError("geometry transfer diagnostic needs changed-radius rows")
    summaries: list[dict[str, Any]] = []
    rig_rows: list[dict[str, Any]] = []
    for predictor in predictors:
        predicted = np.asarray([float(row[predictor]) for row in changed])
        actual = np.asarray([float(row["actual_audit_gain_percent"]) for row in changed])
        by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_rig: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in changed:
            by_block[str(row["block_id"])].append(row)
            by_rig[str(row["rig_id"])].append(row)
        block_predicted = np.asarray(
            [np.mean([float(row[predictor]) for row in group]) for group in by_block.values()]
        )
        block_actual = np.asarray(
            [
                np.mean([float(row["actual_audit_gain_percent"]) for row in group])
                for group in by_block.values()
            ]
        )
        for rig_id, group in sorted(by_rig.items()):
            rig_rows.append(
                {
                    "predictor": predictor,
                    "rig_id": rig_id,
                    "changed_sample_count": len(group),
                    "mean_predicted_audit_gain_percent": float(
                        np.mean([float(row[predictor]) for row in group])
                    ),
                    "mean_actual_audit_gain_percent": float(
                        np.mean(
                            [
                                float(row["actual_audit_gain_percent"])
                                for row in group
                            ]
                        )
                    ),
                }
            )
        predictor_rigs = [row for row in rig_rows if row["predictor"] == predictor]
        rig_predicted = np.asarray(
            [float(row["mean_predicted_audit_gain_percent"]) for row in predictor_rigs]
        )
        rig_actual = np.asarray(
            [float(row["mean_actual_audit_gain_percent"]) for row in predictor_rigs]
        )
        summaries.append(
            {
                "predictor": predictor,
                "changed_sample_count": len(changed),
                "independent_rig_count": len(by_rig),
                "changed_block_count": len(by_block),
                "descriptive_sample_pearson_correlation": _safe_correlation(
                    predicted, actual
                ),
                "descriptive_sample_sign_agreement_fraction": _same_sign(
                    predicted, actual
                ),
                "descriptive_sample_mean_absolute_error_percent": float(
                    np.mean(np.abs(predicted - actual))
                ),
                "block_sign_agreement_fraction": _same_sign(
                    block_predicted, block_actual
                ),
                "block_mean_absolute_error_percent": float(
                    np.mean(np.abs(block_predicted - block_actual))
                ),
                "rig_sign_agreement_fraction": _same_sign(rig_predicted, rig_actual),
                "equal_weight_rig_mean_absolute_error_percent": float(
                    np.mean(np.abs(rig_predicted - rig_actual))
                ),
            }
        )
    return summaries, rig_rows


def prediction_hash(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prediction_rows(
    blocks: Sequence[DevelopmentBlock],
    source_config: dict[str, Any],
    source_v5f_config: dict[str, Any],
    selected_by_block: dict[str, dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    radii = np.asarray(source_config["candidate_aperture_radii"], dtype=np.float64)
    support = support_mask_from_config(source_config)
    kappas = tuple(float(value) for value in source_v5f_config["kappas"])
    method = str(config["reconstruction_method"])
    period = float(config["angle_period_degrees"])
    rig_angles = {
        str(rig["id"]): tuple(float(value) for value in rig["angles_degrees"])
        for rig in source_config["rigs"]
    }
    output: list[dict[str, Any]] = []
    for block in blocks:
        selected_source = selected_by_block[block.block_id]
        selected_index = nearest_index(
            radii, float(selected_source["selected_radius"])
        )
        metadata_index = nearest_index(radii, block.metadata_radius)
        candidate, baseline = refit_pair(
            block,
            selected_index,
            metadata_index,
            method,
            support,
            kappas,
        )
        angles = rig_angles[block.rig_id]
        audit_view = block.audit_views[0]
        nearest_angle_offset = int(
            np.argmin(
                [
                    circular_angle_distance(angles[view], angles[audit_view], period)
                    for view in block.outer_views
                ]
            )
        )
        for sample_index, (observation, sigma, family) in enumerate(
            zip(block.observations, block.noise_std, block.families, strict=True)
        ):
            candidate_field = candidate.refit.fits[sample_index].field
            baseline_field = baseline.refit.fits[sample_index].field
            candidate_outer = whitened_per_view_rms(
                block.reconstruction_bank[selected_index],
                candidate_field,
                observation,
                sigma,
                block.outer_views,
            )
            baseline_outer = whitened_per_view_rms(
                block.reconstruction_bank[metadata_index],
                baseline_field,
                observation,
                sigma,
                block.outer_views,
            )
            outer_gains = tuple(
                error_reduction_percent(candidate_value, baseline_value)
                for candidate_value, baseline_value in zip(
                    candidate_outer, baseline_outer, strict=True
                )
            )
            zeros = np.zeros_like(observation)
            audit_baseline_matrix = camera_support_matrix(
                block.reconstruction_bank[metadata_index],
                zeros,
                sigma,
                audit_view,
                support,
            )
            audit_candidate_matrix = camera_support_matrix(
                block.reconstruction_bank[selected_index],
                zeros,
                sigma,
                audit_view,
                support,
            )
            outer_baseline_matrices = tuple(
                camera_support_matrix(
                    block.reconstruction_bank[metadata_index],
                    zeros,
                    sigma,
                    view,
                    support,
                )
                for view in block.outer_views
            )
            outer_candidate_matrices = tuple(
                camera_support_matrix(
                    block.reconstruction_bank[selected_index],
                    zeros,
                    sigma,
                    view,
                    support,
                )
                for view in block.outer_views
            )
            projection_weights = tuple(
                projection_similarity(
                    matrix,
                    audit_baseline_matrix,
                    relative_tolerance=float(config["svd_relative_tolerance"]),
                )
                for matrix in outer_baseline_matrices
            )
            gram_weights = tuple(
                gram_cosine(matrix, audit_baseline_matrix)
                for matrix in outer_baseline_matrices
            )
            if selected_index == metadata_index:
                change_weights = tuple(1.0 for _ in block.outer_views)
            else:
                change_weights = tuple(
                    abs(
                        operator_change_cosine(
                            candidate_matrix,
                            baseline_matrix,
                            audit_candidate_matrix,
                            audit_baseline_matrix,
                        )
                    )
                    for candidate_matrix, baseline_matrix in zip(
                        outer_candidate_matrices,
                        outer_baseline_matrices,
                        strict=True,
                    )
                )
            inner_matrix, _, _ = whitened_support_system(
                block.reconstruction_bank[metadata_index],
                zeros,
                sigma,
                block.inner_views,
                support,
            )
            ridge_lambda = float(baseline.refit.effective_lambdas[sample_index])
            audit_leverage = group_predictive_leverage(
                inner_matrix, audit_baseline_matrix, ridge_lambda
            ).mean_per_measurement
            outer_leverages = tuple(
                group_predictive_leverage(
                    inner_matrix, matrix, ridge_lambda
                ).mean_per_measurement
                for matrix in outer_baseline_matrices
            )
            nearest_leverage_offset = int(
                np.argmin(
                    [
                        abs(
                            np.log(
                                max(value, 1e-15) / max(audit_leverage, 1e-15)
                            )
                        )
                        for value in outer_leverages
                    ]
                )
            )
            residual_transfer = residual_field_transfer(
                block.reconstruction_bank[metadata_index],
                block.reconstruction_bank[selected_index],
                baseline_field,
                candidate_field,
                observation,
                sigma,
                block.outer_views,
                block.audit_views,
                support,
                ridge_lambda,
            )
            output.append(
                {
                    "rig_id": block.rig_id,
                    "block_id": block.block_id,
                    "sample_index": sample_index,
                    "family": family,
                    "reconstruction_method": method,
                    "metadata_bank_radius": float(radii[metadata_index]),
                    "selected_calibration_radius": float(radii[selected_index]),
                    "radius_changed_from_metadata": bool(
                        selected_index != metadata_index
                    ),
                    "outer_camera_indices": "|".join(
                        str(value) for value in block.outer_views
                    ),
                    "target_camera_index": audit_view,
                    "outer_gains_percent": "|".join(
                        f"{value:.12g}" for value in outer_gains
                    ),
                    "projection_similarities": "|".join(
                        f"{value:.12g}" for value in projection_weights
                    ),
                    "gram_similarities": "|".join(
                        f"{value:.12g}" for value in gram_weights
                    ),
                    "operator_change_similarities": "|".join(
                        f"{value:.12g}" for value in change_weights
                    ),
                    "outer_mean_leverages": "|".join(
                        f"{value:.12g}" for value in outer_leverages
                    ),
                    "target_mean_leverage": audit_leverage,
                    "residual_source_fit_whitened_rms": (
                        residual_transfer.source_fit_whitened_rms
                    ),
                    "outer_observation_minimum_gain": float(min(outer_gains)),
                    "outer_observation_mean_gain": float(np.mean(outer_gains)),
                    "outer_observation_nearest_angle_gain": float(
                        outer_gains[nearest_angle_offset]
                    ),
                    "outer_observation_projection_weighted_gain": (
                        similarity_weighted_gain(projection_weights, outer_gains)
                    ),
                    "outer_observation_gram_weighted_gain": (
                        similarity_weighted_gain(gram_weights, outer_gains)
                    ),
                    "outer_observation_operator_change_weighted_gain": (
                        similarity_weighted_gain(change_weights, outer_gains)
                    ),
                    "outer_observation_nearest_leverage_gain": float(
                        outer_gains[nearest_leverage_offset]
                    ),
                    "source_residual_field_transfer_gain": float(
                        residual_transfer.predicted_error_reductions_percent[0]
                    ),
                    "prediction_uses_source_observation": True,
                    "prediction_uses_target_operator": True,
                    "prediction_uses_target_observation": False,
                    "prediction_uses_truth_field": False,
                    "oracle_sigma_truth_derived": True,
                }
            )
    return output


def write_figure(
    path: Path, rows: Sequence[dict[str, Any]], summaries: Sequence[dict[str, Any]]
) -> None:
    changed = [row for row in rows if bool(row["radius_changed_from_metadata"])]
    actual = np.asarray([float(row["actual_audit_gain_percent"]) for row in changed])
    names = [
        "outer_observation_minimum_gain",
        "outer_observation_projection_weighted_gain",
        "outer_observation_operator_change_weighted_gain",
        "source_residual_field_transfer_gain",
    ]
    labels = ["worst outer", "projection weighted", "change weighted", "residual field"]
    lookup = {row["predictor"]: row for row in summaries}
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, name, label in zip(axes.flat, names, labels, strict=True):
        predicted = np.asarray([float(row[name]) for row in changed])
        axis.scatter(predicted, actual, alpha=0.7, color="tab:blue")
        bounds = [float(min(np.min(predicted), np.min(actual))), float(max(np.max(predicted), np.max(actual)))]
        axis.plot(bounds, bounds, linestyle="--", color="black", linewidth=1)
        correlation = lookup[name]["descriptive_sample_pearson_correlation"]
        correlation_text = "n/a" if correlation is None else f"{correlation:.3f}"
        axis.set_title(
            f"{label}: descriptive r={correlation_text}, sign={lookup[name]['descriptive_sample_sign_agreement_fraction']:.3f}"
        )
        axis.set_xlabel("predicted audit gain (%)")
        axis.set_ylabel("opened audit gain (%)")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "v5g post-open geometry and residual transfer diagnostic - no method freeze",
        fontsize=14,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    config = read_json(config_path)
    source_config_path = (ROOT / str(config["source_config"])).resolve()
    source_v5f_config_path = (ROOT / str(config["source_v5f_config"])).resolve()
    source_selection_path = (ROOT / str(config["source_v5e_selection_rows"])).resolve()
    source_report_path = (ROOT / str(config["source_v5f_report"])).resolve()
    source_samples_path = (ROOT / str(config["source_v5f_sample_rows"])).resolve()
    source_config = read_json(source_config_path)
    source_v5f_config = read_json(source_v5f_config_path)
    source_sample_rows = read_csv(source_samples_path)
    if not methods_have_identical_sample_rows(source_sample_rows, "gcv", "upre"):
        raise ValueError("v5g expects identical v5f GCV and UPRE sample rows")
    selected = selected_radius_by_block(
        read_csv(source_selection_path), str(config["calibration_variant_id"])
    )
    blocks, _ = build_development_blocks(source_config)
    prediction_rows = build_prediction_rows(
        blocks, source_config, source_v5f_config, selected, config
    )
    pre_audit_hash = prediction_hash(prediction_rows)
    outcomes = audit_gain_by_sample(
        source_sample_rows, str(config["reconstruction_method"])
    )
    rows = attach_audit_outcomes(prediction_rows, outcomes)
    predictors = tuple(str(value) for value in config["predictors"])
    summaries, rig_rows = predictor_summaries(rows, predictors)

    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = output / "config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prediction_path = output / "prediction_rows.csv"
    summary_path = output / "predictor_summary.csv"
    rig_path = output / "rig_summary.csv"
    write_csv(prediction_path, rows)
    write_csv(summary_path, summaries)
    write_csv(rig_path, rig_rows)
    figure_path = output / "v5g_geometry_residual_transfer_postopen.png"
    write_figure(figure_path, rows, summaries)

    residual_summary = next(
        row
        for row in summaries
        if row["predictor"] == "source_residual_field_transfer_gain"
    )
    report = {
        "claim_status": config["claim_status"],
        "scientific_verdict": "NO_METHOD_FREEZE",
        "scientific_boundary": (
            "post-open predictor development on v5f; target operator geometry is used, "
            "target observations are excluded until all predictions are hashed; only two "
            "independent rigs and three changed-radius blocks exist"
        ),
        "pre_audit_prediction_sha256": pre_audit_hash,
        "prediction_row_count": len(rows),
        "changed_radius_prediction_count": sum(
            bool(row["radius_changed_from_metadata"]) for row in rows
        ),
        "independent_rig_count": len({row["rig_id"] for row in rows}),
        "changed_block_count": len(
            {
                row["block_id"]
                for row in rows
                if bool(row["radius_changed_from_metadata"])
            }
        ),
        "predictor_summary": summaries,
        "residual_field_transfer_summary": residual_summary,
        "primary_cluster_conclusion": (
            "residual-field transfer agrees in only 1/3 changed blocks and 0/2 "
            "independent rigs; descriptive 17/24 sample signs cannot override this"
        ),
        "predictor_family_designed_after_audit_open": True,
        "predictor_ranking_authorized": False,
        "threshold_sweep_authorized": False,
        "sample_rows_are_iid": False,
        "confidence_interval_or_significance_authorized": False,
        "source_outer_observation_used_for_prediction": True,
        "target_operator_used_for_prediction": True,
        "target_observation_used_for_prediction": False,
        "truth_field_used_for_prediction": False,
        "noise_std_is_synthetic_clean_inner_rms_oracle": True,
        "v5f_gcv_and_upre_sample_rows_identical": True,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "next_step": (
            "retain residual-field transfer only as a candidate architecture; test a frozen "
            "source-residual encoder and physics decoder on a new rig/session, or stop the "
            "RigCal branch if independent target-view sign consistency remains absent"
        ),
        "source_hashes": {
            "config": sha256(config_path),
            "source_config": sha256(source_config_path),
            "source_v5f_config": sha256(source_v5f_config_path),
            "source_v5e_selection_rows": sha256(source_selection_path),
            "source_v5f_report": sha256(source_report_path),
            "source_v5f_sample_rows": sha256(source_samples_path),
            "runner": sha256(Path(__file__).resolve()),
            "geometry_module": sha256((ROOT / "view_transfer_geometry.py").resolve()),
            "rig_shared_profile": sha256((ROOT / "rig_shared_profile.py").resolve()),
            "block_builder": sha256(
                (ROOT / "run_v5b_rig_shared_profile_pilot.py").resolve()
            ),
            "v5f_runner": sha256(
                (ROOT / "run_v5f_dual_regularization_postopen.py").resolve()
            ),
            "dual_regularization": sha256(
                (ROOT / "dual_regularization.py").resolve()
            ),
            "nested_crossview": sha256((ROOT / "nested_crossview.py").resolve()),
            "finite_aperture_forward": sha256(
                (ROOT / "finite_aperture_bost.py").resolve()
            ),
            "reaction_field_noise_generator": sha256(
                (ROOT / "independent_reaction_bost.py").resolve()
            ),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme_path = output / "README.md"
    readme_path.write_text(
        "# v5g geometry and residual transfer post-open diagnostic\n\n"
        "This opened-v5f development artifact compares fixed geometry summaries and "
        "a source-observation-assisted residual-field transfer. Target observations "
        "are joined only after the prediction hash is frozen. The diagonal sigma is "
        "an oracle derived from clean synthetic inner-view signal RMS. Sample rows are "
        "not IID; the primary result is 1/3 block and 0/2 rig sign agreement. No "
        "predictor is authorized for routing or confirmatory claims.\n",
        encoding="utf-8",
    )
    write_checksums(
        output,
        (
            config_snapshot,
            prediction_path,
            summary_path,
            rig_path,
            figure_path,
            report_path,
            readme_path,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
