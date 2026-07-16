#!/usr/bin/env python3
"""Development-only low-rank screening for DCO-BOST / GC-BiLOC."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo_t16_operator.finite_aperture_bost import (
        build_finite_aperture_operator_bank,
        finite_aperture_reference_scale,
    )
    from demo_t16_operator.gc_rio.data import _flatten_operator
    from demo_t16_operator.gc_rio.protocol import sha256_json
    from demo_t16_operator.independent_reaction_bost import make_reaction_field
else:
    from .finite_aperture_bost import (
        build_finite_aperture_operator_bank,
        finite_aperture_reference_scale,
    )
    from .gc_rio.data import _flatten_operator
    from .gc_rio.protocol import sha256_json
    from .independent_reaction_bost import make_reaction_field


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5s_dco_low_rank_screening.json"
OUTPUT_DIR = ROOT / "results" / "v5s_dco_low_rank_screening"
NOMINAL_FEATURE_KEYS = (
    "nominal_aperture_radius",
    "nominal_cone_u",
    "nominal_cone_z",
    "nominal_bend",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(output: Path, names: Sequence[str]) -> None:
    lines = [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    ]
    (output / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rig_manifest(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(config["seed"]))
    train_count = int(config["train_rigs"])
    total = train_count + int(config["development_rigs"])
    nominal = config["nominal_ranges"]
    hidden = config["hidden_truth_mismatch"]
    base_angles = np.linspace(4.0, 176.0, int(config["views"]))
    rows = []
    for index in range(total):
        angles = np.sort(
            np.clip(
                base_angles
                + rng.uniform(
                    -float(config["angle_jitter_degrees"]),
                    float(config["angle_jitter_degrees"]),
                    size=len(base_angles),
                ),
                0.0,
                179.5,
            )
        )
        radius = rng.uniform(*nominal["aperture_radius"])
        cone_u = rng.uniform(*nominal["cone_u"])
        cone_z = rng.uniform(*nominal["cone_z"])
        bend = rng.uniform(*nominal["bend"])
        phase = 2.0 * np.pi * (radius - nominal["aperture_radius"][0]) / (
            nominal["aperture_radius"][1] - nominal["aperture_radius"][0]
        )
        radius_scale = rng.uniform(*hidden["radius_scale"])
        radius_offset = rng.uniform(*hidden["radius_offset"])
        truth_angles = angles + float(hidden["angle_systematic_degrees"]) * np.sin(
            np.deg2rad(2.0 * angles)
        ) + rng.normal(
            0.0, float(hidden["angle_hidden_jitter_degrees"]), size=len(angles)
        )
        rows.append(
            {
                "rig_id": f"dco-rig-{index:03d}",
                "split": "train" if index < train_count else "development",
                "nominal_angles_degrees": [float(value) for value in angles],
                "nominal_aperture_radius": float(radius),
                "nominal_cone_u": float(cone_u),
                "nominal_cone_z": float(cone_z),
                "nominal_bend": float(bend),
                "truth_angles_degrees": [float(value) for value in truth_angles],
                "truth_aperture_radius": float(radius * radius_scale + radius_offset),
                "truth_cone_u": float(
                    cone_u
                    * (1.0 + float(hidden["cone_relative"]) * np.sin(phase))
                    + rng.normal(0.0, 0.0015)
                ),
                "truth_cone_z": float(
                    cone_z
                    * (1.0 + float(hidden["cone_relative"]) * np.cos(phase))
                    + rng.normal(0.0, 0.0015)
                ),
                "truth_bend": float(
                    bend
                    * (1.0 + float(hidden["bend_relative"]) * np.sin(phase + 0.7))
                    + rng.normal(0.0, 0.001)
                ),
                "truth_fields_are_feature_forbidden": True,
            }
        )
    return rows


def nominal_geometry_features(row: Mapping[str, Any]) -> np.ndarray:
    """Return known geometry only; truth-side fields are forbidden."""

    angles = np.asarray(row["nominal_angles_degrees"], dtype=np.float64)
    values = [float(row[key]) for key in NOMINAL_FEATURE_KEYS]
    return np.asarray(
        [
            *np.sin(np.deg2rad(angles)),
            *np.cos(np.deg2rad(angles)),
            *values,
            float(np.mean(np.diff(angles))),
            float(np.std(np.diff(angles))),
        ],
        dtype=np.float64,
    )


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack([nominal_geometry_features(row) for row in rows])
    if center is None:
        center = np.mean(raw, axis=0)
    if scale is None:
        scale = np.std(raw, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    standardized = (raw - center) / scale
    design = np.concatenate(
        [np.ones((len(rows), 1)), standardized, np.square(standardized)], axis=1
    )
    return design, center, scale


def fit_ridge(
    design: np.ndarray, targets: np.ndarray, relative_ridge: float
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    gram = x.T @ x
    penalty = float(relative_ridge) * max(float(np.trace(gram)) / len(gram), 1e-12)
    regularizer = penalty * np.eye(gram.shape[0])
    regularizer[0, 0] = 0.0
    return np.linalg.solve(gram + regularizer, x.T @ y)


def hosvd_bases(discrepancy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(discrepancy, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("discrepancy must be [rig,measurement,voxel]")
    measurement_unfold = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    voxel_unfold = values.transpose(2, 0, 1).reshape(values.shape[2], -1)
    u, measurement_singular, _ = np.linalg.svd(measurement_unfold, full_matrices=False)
    v, voxel_singular, _ = np.linalg.svd(voxel_unfold, full_matrices=False)
    return u, v, measurement_singular, voxel_singular


def project_cores(
    discrepancy: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    return np.einsum("mi,gmn,nj->gij", u, discrepancy, v, optimize=True)


def reconstruct_operators(
    cores: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    return np.einsum("mi,gij,nj->gmn", u, cores, v, optimize=True)


def _build_operator_pair(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n = int(config["grid_size"])
    depth = int(config["depth"])
    renderer = config["renderer"]
    nominal_angles = np.asarray(row["nominal_angles_degrees"], dtype=np.float64)
    tick = time.perf_counter()
    common_scale = finite_aperture_reference_scale(
        n,
        depth,
        nominal_angles,
        aperture_samples=int(renderer["truth_aperture_samples"]),
        path_samples=int(renderer["truth_path_samples"]),
        cone_u=float(row["nominal_cone_u"]),
        cone_z=float(row["nominal_cone_z"]),
        bend=float(row["nominal_bend"]),
    )
    nominal = build_finite_aperture_operator_bank(
        n,
        depth,
        nominal_angles,
        [float(row["nominal_aperture_radius"])],
        aperture_samples=int(renderer["nominal_aperture_samples"]),
        path_samples=int(renderer["nominal_path_samples"]),
        cone_u=float(row["nominal_cone_u"]),
        cone_z=float(row["nominal_cone_z"]),
        bend=float(row["nominal_bend"]),
        normalization_scale=common_scale,
    )[0]
    nominal_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    truth = build_finite_aperture_operator_bank(
        n,
        depth,
        np.asarray(row["truth_angles_degrees"], dtype=np.float64),
        [float(row["truth_aperture_radius"])],
        aperture_samples=int(renderer["truth_aperture_samples"]),
        path_samples=int(renderer["truth_path_samples"]),
        cone_u=float(row["truth_cone_u"]),
        cone_z=float(row["truth_cone_z"]),
        bend=float(row["truth_bend"]),
        normalization_scale=common_scale,
    )[0]
    truth_seconds = time.perf_counter() - tick
    return (
        _flatten_operator(nominal).reshape(-1, n * n * depth).astype(np.float64),
        _flatten_operator(truth).reshape(-1, n * n * depth).astype(np.float64),
        {
            "nominal_build_seconds": nominal_seconds,
            "truth_build_seconds": truth_seconds,
            "normalization_scale": common_scale,
        },
    )


def _relative_discrepancy_error(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(prediction - truth, axis=(1, 2))
    denominator = np.maximum(np.linalg.norm(truth, axis=(1, 2)), 1e-15)
    return numerator / denominator


def _nearest_predictions(
    train_design: np.ndarray,
    development_design: np.ndarray,
    train_discrepancy: np.ndarray,
) -> np.ndarray:
    train = train_design[:, 1 : 1 + (train_design.shape[1] - 1) // 2]
    development = development_design[:, 1 : 1 + (train_design.shape[1] - 1) // 2]
    output = []
    for row in development:
        index = int(np.argmin(np.linalg.norm(train - row[None], axis=1)))
        output.append(train_discrepancy[index])
    return np.stack(output)


def _gradient_cosine(operator: np.ndarray, truth: np.ndarray, field: np.ndarray) -> float:
    measurement = truth @ field
    approximate_gradient = operator.T @ measurement
    truth_gradient = truth.T @ measurement
    denominator = np.linalg.norm(approximate_gradient) * np.linalg.norm(truth_gradient)
    return float(approximate_gradient @ truth_gradient / max(denominator, 1e-15))


def run() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not bool(config["design_lock_construction_forbidden"]):
        raise ValueError("v5s must forbid design-lock construction")
    manifest = build_rig_manifest(config)
    nominal_operators = []
    truth_operators = []
    timing_rows = []
    for row in manifest:
        nominal, truth, timing = _build_operator_pair(row, config)
        nominal_operators.append(nominal)
        truth_operators.append(truth)
        timing_rows.append({"rig_id": row["rig_id"], **timing})
    nominal_all = np.stack(nominal_operators)
    truth_all = np.stack(truth_operators)
    discrepancy_all = truth_all - nominal_all
    train_mask = np.asarray([row["split"] == "train" for row in manifest])
    development_mask = ~train_mask
    train_rows = [row for row in manifest if row["split"] == "train"]
    development_rows = [row for row in manifest if row["split"] == "development"]
    train_discrepancy = discrepancy_all[train_mask]
    development_discrepancy = discrepancy_all[development_mask]
    train_design, feature_center, feature_scale = _feature_matrix(train_rows)
    development_design, _, _ = _feature_matrix(
        development_rows, center=feature_center, scale=feature_scale
    )
    u_full, v_full, measurement_singular, voxel_singular = hosvd_bases(
        train_discrepancy
    )

    sweep_rows = []
    best: dict[str, Any] | None = None
    for rank_m in config["rank_grid"]:
        for rank_v in config["rank_grid"]:
            u = u_full[:, : int(rank_m)]
            v = v_full[:, : int(rank_v)]
            train_cores = project_cores(train_discrepancy, u, v)
            for ridge in config["ridge_grid"]:
                coefficients = fit_ridge(
                    train_design,
                    train_cores.reshape(len(train_cores), -1),
                    float(ridge),
                )
                predicted_cores = (development_design @ coefficients).reshape(
                    len(development_rows), int(rank_m), int(rank_v)
                )
                prediction = reconstruct_operators(predicted_cores, u, v)
                errors = _relative_discrepancy_error(
                    prediction, development_discrepancy
                )
                record = {
                    "method": "gc_biloc_ridge",
                    "rank_measurement": int(rank_m),
                    "rank_voxel": int(rank_v),
                    "relative_ridge": float(ridge),
                    "development_mean_relative_discrepancy_error": float(
                        np.mean(errors)
                    ),
                    "development_worst_relative_discrepancy_error": float(
                        np.max(errors)
                    ),
                }
                sweep_rows.append(record)
                if best is None or (
                    record["development_mean_relative_discrepancy_error"],
                    int(rank_m) * int(rank_v),
                ) < (
                    best["development_mean_relative_discrepancy_error"],
                    best["rank_measurement"] * best["rank_voxel"],
                ):
                    best = {**record, "coefficients": coefficients}
    assert best is not None
    rank_m = int(best["rank_measurement"])
    rank_v = int(best["rank_voxel"])
    selected_u = u_full[:, :rank_m]
    selected_v = v_full[:, :rank_v]
    predicted_cores = (development_design @ best["coefficients"]).reshape(
        len(development_rows), rank_m, rank_v
    )
    biloc_prediction = reconstruct_operators(
        predicted_cores, selected_u, selected_v
    )

    mean_prediction = np.broadcast_to(
        np.mean(train_discrepancy, axis=0), development_discrepancy.shape
    ).copy()
    nearest_prediction = _nearest_predictions(
        train_design, development_design, train_discrepancy
    )
    full_ridge_best: dict[str, Any] | None = None
    for ridge in config["ridge_grid"]:
        coefficients = fit_ridge(
            train_design,
            train_discrepancy.reshape(len(train_discrepancy), -1),
            float(ridge),
        )
        prediction = (development_design @ coefficients).reshape(
            development_discrepancy.shape
        )
        error = float(
            np.mean(_relative_discrepancy_error(prediction, development_discrepancy))
        )
        if full_ridge_best is None or error < full_ridge_best["error"]:
            full_ridge_best = {
                "error": error,
                "ridge": float(ridge),
                "prediction": prediction,
            }
    assert full_ridge_best is not None
    oracle_cores = project_cores(
        development_discrepancy, selected_u, selected_v
    )
    oracle_shared_prediction = reconstruct_operators(
        oracle_cores, selected_u, selected_v
    )
    predictions = {
        "zero_nominal": np.zeros_like(development_discrepancy),
        "mean_discrepancy": mean_prediction,
        "nearest_geometry": nearest_prediction,
        "full_matrix_geometry_ridge": full_ridge_best["prediction"],
        "gc_biloc_ridge": biloc_prediction,
        "oracle_shared_subspace": oracle_shared_prediction,
    }

    development_nominal = nominal_all[development_mask]
    development_truth = truth_all[development_mask]
    probe_rng = np.random.default_rng(int(config["seed"]) + 991)
    probe_fields = []
    for family in config["probe_families"]:
        for _ in range(int(config["probes_per_family"])):
            probe_fields.append(
                make_reaction_field(
                    str(family),
                    int(config["grid_size"]),
                    int(config["depth"]),
                    probe_rng,
                ).reshape(-1)
            )
    rig_metric_rows = []
    summary = {}
    for method, correction in predictions.items():
        corrected = development_nominal + correction
        discrepancy_error = _relative_discrepancy_error(
            correction, development_discrepancy
        )
        operator_error = np.linalg.norm(
            corrected - development_truth, axis=(1, 2)
        ) / np.maximum(np.linalg.norm(development_truth, axis=(1, 2)), 1e-15)
        forward_errors = []
        gradient_cosines = []
        for rig_index, (operator, truth) in enumerate(
            zip(corrected, development_truth, strict=True)
        ):
            rig_forward = []
            rig_gradient = []
            for field in probe_fields:
                truth_measurement = truth @ field
                rig_forward.append(
                    np.linalg.norm(operator @ field - truth_measurement)
                    / max(np.linalg.norm(truth_measurement), 1e-15)
                )
                rig_gradient.append(_gradient_cosine(operator, truth, field))
            forward_errors.extend(rig_forward)
            gradient_cosines.extend(rig_gradient)
            rig_metric_rows.append(
                {
                    "rig_id": development_rows[rig_index]["rig_id"],
                    "method": method,
                    "relative_discrepancy_error": float(
                        discrepancy_error[rig_index]
                    ),
                    "relative_operator_error": float(operator_error[rig_index]),
                    "mean_probe_forward_relative_error": float(
                        np.mean(rig_forward)
                    ),
                    "mean_probe_gradient_cosine": float(
                        np.mean(rig_gradient)
                    ),
                }
            )
        summary[method] = {
            "mean_relative_discrepancy_error": float(np.mean(discrepancy_error)),
            "worst_relative_discrepancy_error": float(np.max(discrepancy_error)),
            "mean_relative_operator_error": float(np.mean(operator_error)),
            "mean_probe_forward_relative_error": float(np.mean(forward_errors)),
            "mean_probe_gradient_cosine": float(np.mean(gradient_cosines)),
            "worst_probe_gradient_cosine": float(np.min(gradient_cosines)),
        }
    best_cheap_name = min(
        ("mean_discrepancy", "nearest_geometry", "full_matrix_geometry_ridge"),
        key=lambda name: summary[name]["mean_relative_discrepancy_error"],
    )
    relative_improvement = 1.0 - summary["gc_biloc_ridge"][
        "mean_relative_discrepancy_error"
    ] / max(summary[best_cheap_name]["mean_relative_discrepancy_error"], 1e-15)
    rule = config["screening_rule"]
    passed = (
        relative_improvement
        >= float(rule["minimum_relative_improvement_over_best_cheap_baseline"])
        and summary["oracle_shared_subspace"]["mean_relative_discrepancy_error"]
        <= float(rule["maximum_oracle_shared_subspace_error"])
    )
    report = {
        "schema": config["schema"],
        "evidence_label": config["evidence_label"],
        "claim_ceiling": config["claim_ceiling"],
        "config_sha256": sha256_json(config),
        "source_provenance": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "config_file": _file_sha256(CONFIG_PATH),
            "finite_aperture_renderer": _file_sha256(
                ROOT / "finite_aperture_bost.py"
            ),
            "reaction_field_generator": _file_sha256(
                ROOT / "independent_reaction_bost.py"
            ),
            "boundary": (
                "Hashes identify the code used for this development run; they do "
                "not constitute a pre-run public preregistration."
            ),
        },
        "manifest_sha256_before_operator_build": sha256_json(manifest),
        "sample_accounting": {
            "train_rigs": len(train_rows),
            "development_rigs": len(development_rows),
            "design_lock_rigs": 0,
            "measurements_per_rig": int(development_truth.shape[1]),
            "voxels": int(development_truth.shape[2]),
            "probe_fields": len(probe_fields),
        },
        "selected_development_hyperparameters": {
            "rank_measurement": rank_m,
            "rank_voxel": rank_v,
            "relative_ridge": float(best["relative_ridge"]),
            "full_matrix_ridge": float(full_ridge_best["ridge"]),
        },
        "singular_energy": {
            "measurement_first_4": float(
                np.sum(np.square(measurement_singular[:4]))
                / np.sum(np.square(measurement_singular))
            ),
            "measurement_first_16": float(
                np.sum(np.square(measurement_singular[:16]))
                / np.sum(np.square(measurement_singular))
            ),
            "voxel_first_4": float(
                np.sum(np.square(voxel_singular[:4]))
                / np.sum(np.square(voxel_singular))
            ),
            "voxel_first_16": float(
                np.sum(np.square(voxel_singular[:16]))
                / np.sum(np.square(voxel_singular))
            ),
        },
        "development_summary": summary,
        "best_cheap_baseline": best_cheap_name,
        "gc_biloc_relative_improvement_over_best_cheap_baseline": relative_improvement,
        "screening_rule": rule,
        "decision": (
            "GC_BILOC_DEVELOPMENT_SIGNAL_NO_FRESH_GATE"
            if passed
            else "GC_BILOC_DEVELOPMENT_NO_GO"
        ),
        "build_timing": {
            "mean_nominal_operator_seconds": float(
                np.mean([row["nominal_build_seconds"] for row in timing_rows])
            ),
            "mean_truth_operator_seconds": float(
                np.mean([row["truth_build_seconds"] for row in timing_rows])
            ),
            "boundary": "Local CPU construction timing only; no runtime superiority claim.",
        },
        "limitations": [
            "The hidden mismatch generator is synthetic and hand-designed.",
            "Hyperparameters and structure are selected on development rigs; no fresh rigs are constructed.",
            "The full truth matrices are used for scoring and shared-subspace diagnosis, not for the geometry predictor at test time.",
            "No inverse reconstruction result is produced by v5s.",
        ],
    }
    manifest_rows = []
    for row in manifest:
        manifest_rows.append(
            {
                **{key: value for key, value in row.items() if not isinstance(value, list)},
                "nominal_angles_degrees": " ".join(
                    f"{value:.6f}" for value in row["nominal_angles_degrees"]
                ),
                "truth_angles_degrees": " ".join(
                    f"{value:.6f}" for value in row["truth_angles_degrees"]
                ),
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "rig_manifest.csv", manifest_rows)
    _write_csv(OUTPUT_DIR / "sweep.csv", sweep_rows)
    _write_csv(OUTPUT_DIR / "development_rig_metrics.csv", rig_metric_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(
        OUTPUT_DIR,
        ["rig_manifest.csv", "sweep.csv", "development_rig_metrics.csv", "report.json"],
    )
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "selected": report["selected_development_hyperparameters"],
                "singular_energy": report["singular_energy"],
                "best_cheap_baseline": report["best_cheap_baseline"],
                "relative_improvement": report[
                    "gc_biloc_relative_improvement_over_best_cheap_baseline"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
