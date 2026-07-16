#!/usr/bin/env python3
"""Post-v5s camera-local tangent representation diagnosis.

This script deliberately gives the structured models truth-side parameter offsets.
It is an oracle representation test, not a deployable calibration method.
"""

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
    )
    from demo_t16_operator.gc_rio.data import _flatten_operator
    from demo_t16_operator.gc_rio.protocol import sha256_json
    from demo_t16_operator.independent_reaction_bost import make_reaction_field
    from demo_t16_operator.run_v5s_dco_low_rank_screening import (
        _build_operator_pair,
        _gradient_cosine,
        build_rig_manifest,
    )
else:
    from .finite_aperture_bost import build_finite_aperture_operator_bank
    from .gc_rio.data import _flatten_operator
    from .gc_rio.protocol import sha256_json
    from .independent_reaction_bost import make_reaction_field
    from .run_v5s_dco_low_rank_screening import (
        _build_operator_pair,
        _gradient_cosine,
        build_rig_manifest,
    )


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "v5t_camera_local_tangent_diagnosis.json"
OUTPUT_DIR = ROOT / "results" / "v5t_camera_local_tangent_diagnosis"
V5S_REPORT = ROOT / "results" / "v5s_dco_low_rank_screening" / "report.json"
V5S_RIG_METRICS = (
    ROOT
    / "results"
    / "v5s_dco_low_rank_screening"
    / "development_rig_metrics.csv"
)
GLOBAL_PARAMETERS = ("aperture_radius", "cone_u", "cone_z", "bend")


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums(output: Path, names: Sequence[str]) -> None:
    lines = [f"{_file_sha256(output / name)}  {name}" for name in names]
    (output / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def nominal_parameter_vector(row: Mapping[str, Any]) -> np.ndarray:
    """Return only operator parameters available on the nominal side."""

    angles = np.asarray(row["nominal_angles_degrees"], dtype=np.float64)
    return np.asarray(
        [
            *angles,
            float(row["nominal_aperture_radius"]),
            float(row["nominal_cone_u"]),
            float(row["nominal_cone_z"]),
            float(row["nominal_bend"]),
        ],
        dtype=np.float64,
    )


def truth_parameter_vector(row: Mapping[str, Any]) -> np.ndarray:
    angles = np.asarray(row["truth_angles_degrees"], dtype=np.float64)
    return np.asarray(
        [
            *angles,
            float(row["truth_aperture_radius"]),
            float(row["truth_cone_u"]),
            float(row["truth_cone_z"]),
            float(row["truth_bend"]),
        ],
        dtype=np.float64,
    )


def parameter_names(views: int) -> list[str]:
    return [f"angle_{index}" for index in range(int(views))] + list(
        GLOBAL_PARAMETERS
    )


def _renderer_parameters(vector: np.ndarray, views: int) -> dict[str, Any]:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (int(views) + len(GLOBAL_PARAMETERS),):
        raise ValueError("parameter vector has the wrong shape")
    return {
        "angles": values[:views],
        "aperture_radius": float(values[views]),
        "cone_u": float(values[views + 1]),
        "cone_z": float(values[views + 2]),
        "bend": float(values[views + 3]),
    }


def _high_fidelity_operator(
    vector: np.ndarray,
    source_config: Mapping[str, Any],
    normalization_scale: float,
) -> np.ndarray:
    views = int(source_config["views"])
    parameters = _renderer_parameters(vector, views)
    renderer = source_config["renderer"]
    n = int(source_config["grid_size"])
    depth = int(source_config["depth"])
    operator = build_finite_aperture_operator_bank(
        n,
        depth,
        parameters["angles"],
        [parameters["aperture_radius"]],
        aperture_samples=int(renderer["truth_aperture_samples"]),
        path_samples=int(renderer["truth_path_samples"]),
        cone_u=parameters["cone_u"],
        cone_z=parameters["cone_z"],
        bend=parameters["bend"],
        normalization_scale=float(normalization_scale),
    )[0]
    return _flatten_operator(operator).reshape(-1, n * n * depth).astype(np.float64)


def _high_fidelity_single_view_block(
    vector: np.ndarray,
    view_index: int,
    source_config: Mapping[str, Any],
    normalization_scale: float,
) -> np.ndarray:
    views = int(source_config["views"])
    parameters = _renderer_parameters(vector, views)
    renderer = source_config["renderer"]
    n = int(source_config["grid_size"])
    depth = int(source_config["depth"])
    operator = build_finite_aperture_operator_bank(
        n,
        depth,
        np.asarray([parameters["angles"][view_index]], dtype=np.float64),
        [parameters["aperture_radius"]],
        aperture_samples=int(renderer["truth_aperture_samples"]),
        path_samples=int(renderer["truth_path_samples"]),
        cone_u=parameters["cone_u"],
        cone_z=parameters["cone_z"],
        bend=parameters["bend"],
        normalization_scale=float(normalization_scale),
    )[0]
    return _flatten_operator(operator).reshape(depth * n, n * n * depth).astype(
        np.float64
    )


def finite_difference_tangent(
    nominal: np.ndarray,
    truth: np.ndarray,
    source_config: Mapping[str, Any],
    diagnosis_config: Mapping[str, Any],
    normalization_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return first-order, diagonal second-order and additive-secant operators."""

    views = int(source_config["views"])
    n = int(source_config["grid_size"])
    depth = int(source_config["depth"])
    steps_config = diagnosis_config["finite_difference_steps"]
    names = parameter_names(views)
    steps = np.asarray(
        [
            *([float(steps_config["angle_degrees"])] * views),
            float(steps_config["aperture_radius"]),
            float(steps_config["cone_u"]),
            float(steps_config["cone_z"]),
            float(steps_config["bend"]),
        ],
        dtype=np.float64,
    )
    delta = np.asarray(truth - nominal, dtype=np.float64)
    base = _high_fidelity_operator(nominal, source_config, normalization_scale)
    first = base.copy()
    second = base.copy()
    secant = base.copy()
    builds = {"full_view": 1, "single_view": 0}
    for index, (name, step, offset) in enumerate(
        zip(names, steps, delta, strict=True)
    ):
        plus = nominal.copy()
        minus = nominal.copy()
        endpoint = nominal.copy()
        plus[index] += step
        minus[index] -= step
        endpoint[index] = truth[index]
        if name.startswith("angle_"):
            view_index = int(name.split("_", maxsplit=1)[1])
            plus_block = _high_fidelity_single_view_block(
                plus, view_index, source_config, normalization_scale
            )
            minus_block = _high_fidelity_single_view_block(
                minus, view_index, source_config, normalization_scale
            )
            endpoint_block = _high_fidelity_single_view_block(
                endpoint, view_index, source_config, normalization_scale
            )
            row_slice = slice(view_index * depth * n, (view_index + 1) * depth * n)
            base_block = base[row_slice]
            derivative = (plus_block - minus_block) / (2.0 * step)
            curvature = (plus_block - 2.0 * base_block + minus_block) / (step**2)
            first[row_slice] += derivative * offset
            second[row_slice] += derivative * offset + 0.5 * curvature * offset**2
            secant[row_slice] += endpoint_block - base_block
            builds["single_view"] += 3
        else:
            plus_operator = _high_fidelity_operator(
                plus, source_config, normalization_scale
            )
            minus_operator = _high_fidelity_operator(
                minus, source_config, normalization_scale
            )
            endpoint_operator = _high_fidelity_operator(
                endpoint, source_config, normalization_scale
            )
            derivative = (plus_operator - minus_operator) / (2.0 * step)
            curvature = (plus_operator - 2.0 * base + minus_operator) / (step**2)
            first += derivative * offset
            second += derivative * offset + 0.5 * curvature * offset**2
            secant += endpoint_operator - base
            builds["full_view"] += 3
    return first, second, secant, builds


def _relative_error(prediction: np.ndarray, truth: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(prediction - truth) / max(np.linalg.norm(reference), 1e-15)
    )


def _load_v5s_full_ridge_rows() -> dict[str, dict[str, float]]:
    with V5S_RIG_METRICS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        if row["method"] == "full_matrix_geometry_ridge":
            selected[row["rig_id"]] = {
                "relative_discrepancy_error": float(
                    row["relative_discrepancy_error"]
                ),
                "relative_operator_error": float(row["relative_operator_error"]),
                "mean_probe_forward_relative_error": float(
                    row["mean_probe_forward_relative_error"]
                ),
                "mean_probe_gradient_cosine": float(
                    row["mean_probe_gradient_cosine"]
                ),
            }
    return selected


def run() -> dict[str, Any]:
    diagnosis_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_config_path = ROOT / str(diagnosis_config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not bool(diagnosis_config["design_lock_construction_forbidden"]):
        raise ValueError("v5t must forbid design-lock construction")
    v5s_report = json.loads(V5S_REPORT.read_text(encoding="utf-8"))
    manifest = build_rig_manifest(source_config)
    if sha256_json(manifest) != v5s_report["manifest_sha256_before_operator_build"]:
        raise RuntimeError("v5s manifest reproduction failed")
    development_rows = [row for row in manifest if row["split"] == "development"]
    v5s_full_ridge = _load_v5s_full_ridge_rows()
    if {row["rig_id"] for row in development_rows} != set(v5s_full_ridge):
        raise RuntimeError("v5s full-ridge rig rows do not match the reproduced manifest")

    probe_rng = np.random.default_rng(int(source_config["seed"]) + 1991)
    probe_fields = []
    for family in source_config["probe_families"]:
        for _ in range(int(source_config["probes_per_family"])):
            probe_fields.append(
                make_reaction_field(
                    str(family),
                    int(source_config["grid_size"]),
                    int(source_config["depth"]),
                    probe_rng,
                ).reshape(-1)
            )

    metric_rows = []
    decomposition_rows = []
    build_counts = {"full_view": 0, "single_view": 0}
    tick = time.perf_counter()
    for row in development_rows:
        low_operator, truth_operator, timing = _build_operator_pair(row, source_config)
        nominal_parameters = nominal_parameter_vector(row)
        truth_parameters = truth_parameter_vector(row)
        high_nominal = _high_fidelity_operator(
            nominal_parameters, source_config, timing["normalization_scale"]
        )
        first, second, secant, builds = finite_difference_tangent(
            nominal_parameters,
            truth_parameters,
            source_config,
            diagnosis_config,
            timing["normalization_scale"],
        )
        for key, value in builds.items():
            build_counts[key] += value
        total_discrepancy = truth_operator - low_operator
        renderer_gap = high_nominal - low_operator
        parameter_gap = truth_operator - high_nominal
        gap_denominator = max(
            np.linalg.norm(renderer_gap) * np.linalg.norm(parameter_gap), 1e-15
        )
        decomposition_rows.append(
            {
                "rig_id": row["rig_id"],
                "renderer_gap_fraction_of_total_norm": float(
                    np.linalg.norm(renderer_gap)
                    / max(np.linalg.norm(total_discrepancy), 1e-15)
                ),
                "parameter_gap_fraction_of_total_norm": float(
                    np.linalg.norm(parameter_gap)
                    / max(np.linalg.norm(total_discrepancy), 1e-15)
                ),
                "renderer_parameter_gap_cosine": float(
                    np.vdot(renderer_gap, parameter_gap) / gap_denominator
                ),
                "maximum_absolute_parameter_offset": float(
                    np.max(np.abs(truth_parameters - nominal_parameters))
                ),
            }
        )
        methods = {
            "low_fidelity_nominal": low_operator,
            "high_fidelity_nominal": high_nominal,
            "first_order_tangent_oracle": first,
            "diagonal_second_order_oracle": second,
            "additive_secant_oracle": secant,
        }
        for method, operator in methods.items():
            forward_errors = []
            gradient_cosines = []
            for field in probe_fields:
                truth_measurement = truth_operator @ field
                forward_errors.append(
                    np.linalg.norm(operator @ field - truth_measurement)
                    / max(np.linalg.norm(truth_measurement), 1e-15)
                )
                gradient_cosines.append(
                    _gradient_cosine(operator, truth_operator, field)
                )
            metric_rows.append(
                {
                    "rig_id": row["rig_id"],
                    "method": method,
                    "total_discrepancy_relative_error": _relative_error(
                        operator, truth_operator, total_discrepancy
                    ),
                    "parameter_gap_relative_error": _relative_error(
                        operator - high_nominal, parameter_gap, parameter_gap
                    ),
                    "relative_operator_error": float(
                        np.linalg.norm(operator - truth_operator)
                        / max(np.linalg.norm(truth_operator), 1e-15)
                    ),
                    "mean_probe_forward_relative_error": float(
                        np.mean(forward_errors)
                    ),
                    "mean_probe_gradient_cosine": float(
                        np.mean(gradient_cosines)
                    ),
                    "worst_probe_gradient_cosine": float(
                        np.min(gradient_cosines)
                    ),
                }
            )

    methods = sorted({row["method"] for row in metric_rows})
    summary = {}
    for method in methods:
        rows = [row for row in metric_rows if row["method"] == method]
        summary[method] = {
            "mean_total_discrepancy_relative_error": float(
                np.mean([row["total_discrepancy_relative_error"] for row in rows])
            ),
            "worst_total_discrepancy_relative_error": float(
                np.max([row["total_discrepancy_relative_error"] for row in rows])
            ),
            "mean_parameter_gap_relative_error": float(
                np.mean([row["parameter_gap_relative_error"] for row in rows])
            ),
            "mean_relative_operator_error": float(
                np.mean([row["relative_operator_error"] for row in rows])
            ),
            "mean_probe_forward_relative_error": float(
                np.mean([row["mean_probe_forward_relative_error"] for row in rows])
            ),
            "mean_probe_gradient_cosine": float(
                np.mean([row["mean_probe_gradient_cosine"] for row in rows])
            ),
            "worst_probe_gradient_cosine": float(
                np.min([row["worst_probe_gradient_cosine"] for row in rows])
            ),
        }

    structured_methods = (
        "first_order_tangent_oracle",
        "diagonal_second_order_oracle",
        "additive_secant_oracle",
    )
    best_method = min(
        structured_methods,
        key=lambda name: summary[name]["mean_total_discrepancy_relative_error"],
    )
    best_rows = {
        row["rig_id"]: row
        for row in metric_rows
        if row["method"] == best_method
    }
    beating_rigs = np.mean(
        [
            best_rows[rig_id]["total_discrepancy_relative_error"]
            < metrics["relative_discrepancy_error"]
            for rig_id, metrics in v5s_full_ridge.items()
        ]
    )
    v5s_full_mean = float(
        v5s_report["development_summary"]["full_matrix_geometry_ridge"][
            "mean_relative_discrepancy_error"
        ]
    )
    improvement = 1.0 - summary[best_method][
        "mean_total_discrepancy_relative_error"
    ] / max(v5s_full_mean, 1e-15)
    rule = diagnosis_config["diagnostic_reference_rule"]
    reference_pass = (
        summary[best_method]["mean_parameter_gap_relative_error"]
        <= float(rule["maximum_parameter_gap_relative_error"])
        and improvement
        >= float(rule["minimum_mean_improvement_over_v5s_full_matrix_ridge"])
        and beating_rigs
        >= float(rule["minimum_rig_fraction_beating_v5s_full_matrix_ridge"])
    )
    report = {
        "schema": diagnosis_config["schema"],
        "evidence_label": diagnosis_config["evidence_label"],
        "claim_ceiling": diagnosis_config["claim_ceiling"],
        "config_sha256": sha256_json(diagnosis_config),
        "source_config_sha256": sha256_json(source_config),
        "reproduced_v5s_manifest_sha256": sha256_json(manifest),
        "source_provenance": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "diagnosis_config": _file_sha256(CONFIG_PATH),
            "source_config": _file_sha256(source_config_path),
            "finite_aperture_renderer": _file_sha256(
                ROOT / "finite_aperture_bost.py"
            ),
            "v5s_runner": _file_sha256(
                ROOT / "run_v5s_dco_low_rank_screening.py"
            ),
        },
        "oracle_firewall": {
            "truth_parameter_offsets_used": True,
            "deployable_calibration_estimator_trained": False,
            "target_fields_used_for_parameter_estimation": False,
            "interpretation": (
                "The diagnostic asks whether the declared local parameter tangent "
                "can represent mismatch when given true offsets. It does not test "
                "whether those offsets can be inferred from real calibration data."
            ),
        },
        "sample_accounting": {
            "development_rigs": len(development_rows),
            "probe_fields": len(probe_fields),
            "design_lock_rigs": 0,
            "full_view_operator_builds": build_counts["full_view"],
            "single_view_operator_builds": build_counts["single_view"],
        },
        "decomposition_summary": {
            "mean_renderer_gap_fraction_of_total_norm": float(
                np.mean(
                    [
                        row["renderer_gap_fraction_of_total_norm"]
                        for row in decomposition_rows
                    ]
                )
            ),
            "mean_parameter_gap_fraction_of_total_norm": float(
                np.mean(
                    [
                        row["parameter_gap_fraction_of_total_norm"]
                        for row in decomposition_rows
                    ]
                )
            ),
            "mean_renderer_parameter_gap_cosine": float(
                np.mean(
                    [row["renderer_parameter_gap_cosine"] for row in decomposition_rows]
                )
            ),
        },
        "development_summary": summary,
        "v5s_full_matrix_ridge_mean_relative_discrepancy_error": v5s_full_mean,
        "best_structured_oracle": best_method,
        "best_structured_improvement_over_v5s_full_matrix_ridge": improvement,
        "best_structured_rig_fraction_beating_v5s_full_matrix_ridge": float(
            beating_rigs
        ),
        "diagnostic_reference_rule": rule,
        "decision": (
            "CAMERA_LOCAL_TANGENT_REPRESENTATION_SIGNAL_POSTOPEN"
            if reference_pass
            else "CAMERA_LOCAL_TANGENT_REPRESENTATION_NO_GO_POSTOPEN"
        ),
        "wall_clock_seconds": float(time.perf_counter() - tick),
        "limitations": [
            "All rigs and hidden mismatch rules were already opened in v5s.",
            "Truth-side parameter offsets are supplied to the structured oracle.",
            "The finite-aperture renderer is a prescribed linear weak-BOST surrogate.",
            "No calibration estimator, query-budget comparison, inverse reconstruction, or real data is evaluated.",
            "The diagonal second-order model omits cross-parameter Hessian terms.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "rig_metrics.csv", metric_rows)
    _write_csv(OUTPUT_DIR / "gap_decomposition.csv", decomposition_rows)
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_checksums(
        OUTPUT_DIR, ["rig_metrics.csv", "gap_decomposition.csv", "report.json"]
    )
    return report


def main() -> None:
    report = run()
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "decomposition": report["decomposition_summary"],
                "best_structured_oracle": report["best_structured_oracle"],
                "improvement_over_v5s_full_ridge": report[
                    "best_structured_improvement_over_v5s_full_matrix_ridge"
                ],
                "rig_fraction_beating_v5s_full_ridge": report[
                    "best_structured_rig_fraction_beating_v5s_full_matrix_ridge"
                ],
                "development_summary": report["development_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
