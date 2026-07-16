#!/usr/bin/env python3
"""Evaluate the PSU BOS numeric samples against the official loader contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _variable(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in report["variables"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one sampled variable named {name!r}")
    return matches[0]


def _full_values(variable: dict[str, Any]) -> list[float]:
    if variable["sample_mode"] != "full":
        raise ValueError(f"variable {variable['name']!r} was not fully decoded")
    ordered = sorted(variable["samples"], key=lambda item: item["matlab_flat_index"])
    return [float(item["value"]) for item in ordered]


def _measurement_vectors(variable: dict[str, Any]) -> dict[int, tuple[float, ...]]:
    components = int(variable["shape"][0])
    grouped: dict[int, dict[int, float]] = {}
    for sample in variable["samples"]:
        component, measurement = sample["subscripts_zero_based"]
        grouped.setdefault(int(measurement), {})[int(component)] = float(sample["value"])
    result: dict[int, tuple[float, ...]] = {}
    for measurement, values in grouped.items():
        if sorted(values) != list(range(components)):
            raise ValueError(
                f"incomplete components for {variable['name']!r} measurement {measurement}"
            )
        result[measurement] = tuple(values[index] for index in range(components))
    return result


def _axis_contract(
    variable: dict[str, Any], axis: int, expected_extent_m: float
) -> dict[str, Any]:
    grouped: dict[int, list[float]] = {}
    for sample in variable["samples"]:
        subscripts = sample["subscripts_zero_based"]
        grouped.setdefault(int(subscripts[axis]), []).append(float(sample["value"]))
    dimension = int(variable["shape"][axis])
    if 0 not in grouped or dimension - 1 not in grouped:
        raise ValueError(f"{variable['name']} samples omit axis endpoints")
    axis_values = {index: math.fsum(values) / len(values) for index, values in grouped.items()}
    within_index_spread = max(
        max(values) - min(values) for values in grouped.values()
    )
    lower = axis_values[0]
    upper = axis_values[dimension - 1]
    spacing = (upper - lower) / (dimension - 1)
    inferred_cell_centered_extent = spacing * dimension
    return {
        "field": variable["name"],
        "axis": axis,
        "dimension": dimension,
        "lower_cell_center_m": lower,
        "upper_cell_center_m": upper,
        "spacing_m": spacing,
        "inferred_cell_centered_extent_m": inferred_cell_centered_extent,
        "expected_extent_m": expected_extent_m,
        "same_axis_index_max_spread": within_index_spread,
        "separable_on_landmarks": within_index_spread <= 1e-12,
        "centered": abs(lower + upper) <= 1e-12,
        "extent_matches_official": abs(
            inferred_cell_centered_extent - expected_extent_m
        )
        <= 1e-12,
    }


def _all_selected_streams_valid(reports: Iterable[dict[str, Any]]) -> bool:
    return all(
        report.get("status") == "SELECTED_NUMERIC_STREAMS_CONFORMANT"
        and all(
            variable.get("integrity_status") == "FULL_SELECTED_STREAM_VALIDATED"
            for variable in report.get("variables", [])
        )
        for report in reports
    )


def build_conformance_report(
    scalar_report: dict[str, Any],
    xyz_report: dict[str, Any],
    ray_report: dict[str, Any],
    v_report: dict[str, Any],
    *,
    expected_views: int = 9,
    expected_grid_shape: tuple[int, int, int] = (400, 350, 350),
    expected_domain_m: tuple[float, float, float] = (0.150, 0.130, 0.130),
) -> dict[str, Any]:
    reports = (scalar_report, xyz_report, ray_report, v_report)
    source_snapshots = {
        (
            report["source_file"],
            int(report["source_file_size_bytes"]),
            int(report["subsystem_offset"]),
        )
        for report in reports
    }
    source_consistent = len(source_snapshots) == 1

    siz = [int(value) for value in _full_values(_variable(scalar_report, "siz"))]
    if len(siz) != 3:
        raise ValueError("siz must contain image rows, image columns and view count")
    image_rows, image_columns, views = siz
    measurements = image_rows * image_columns * views

    xyz_variables = [_variable(xyz_report, name) for name in ("X", "Y", "Z")]
    axis_contracts = [
        _axis_contract(variable, axis, expected_domain_m[axis])
        for axis, variable in enumerate(xyz_variables)
    ]

    c_variable = _variable(ray_report, "c")
    ru_variable = _variable(ray_report, "Ruvecs")
    rv_variable = _variable(ray_report, "Rvvecs")
    aperture_variable = _variable(ray_report, "Rapvec")
    v_variable = _variable(v_report, "v")
    c_vectors = _measurement_vectors(c_variable)
    ru_vectors = _measurement_vectors(ru_variable)
    rv_vectors = _measurement_vectors(rv_variable)
    v_vectors = _measurement_vectors(v_variable)
    apertures = _measurement_vectors(aperture_variable)
    measurement_index_sets = {
        tuple(sorted(values))
        for values in (c_vectors, ru_vectors, rv_vectors, v_vectors, apertures)
    }
    aligned_measurement_samples = len(measurement_index_sets) == 1

    v_norms = [math.sqrt(math.fsum(value * value for value in vector)) for vector in v_vectors.values()]
    v_norm_max_error = max(abs(value - 1.0) for value in v_norms)
    camera_centers = {
        tuple(round(value, 6) for value in vector) for vector in c_vectors.values()
    }
    aperture_values = [vector[0] for vector in apertures.values()]

    shapes_match_measurements = all(
        int(variable["shape"][1]) == measurements
        for variable in (c_variable, ru_variable, rv_variable, v_variable, aperture_variable)
    )
    grid_shapes_match = all(
        tuple(int(value) for value in variable["shape"]) == expected_grid_shape
        for variable in xyz_variables
    )
    finite_ray_samples = all(
        math.isfinite(value)
        for vectors in (c_vectors, ru_vectors, rv_vectors, v_vectors, apertures)
        for vector in vectors.values()
        for value in vector
    )

    checks = {
        "source_snapshot_consistent": source_consistent,
        "selected_stream_integrity_valid": _all_selected_streams_valid(reports),
        "official_nine_view_count": views == expected_views,
        "ray_width_matches_siz_product": shapes_match_measurements,
        "grid_shapes_match_official": grid_shapes_match,
        "coordinate_landmarks_are_separable": all(
            item["separable_on_landmarks"] for item in axis_contracts
        ),
        "coordinate_grid_is_centered": all(item["centered"] for item in axis_contracts),
        "coordinate_extents_match_official_meters": all(
            item["extent_matches_official"] for item in axis_contracts
        ),
        "ray_sample_indices_aligned": aligned_measurement_samples,
        "ray_samples_are_finite": finite_ray_samples,
        "direction_samples_are_unit_vectors": v_norm_max_error <= 1e-6,
        "sampled_camera_centers_match_view_count": len(camera_centers) == views,
        "sampled_apertures_are_positive": min(aperture_values) > 0,
    }
    status = (
        "LOADER_NUMERIC_CONTRACT_CONFORMANT"
        if all(checks.values())
        else "LOADER_NUMERIC_CONTRACT_REVIEW_REQUIRED"
    )
    source_file, source_size, subsystem_offset = next(iter(source_snapshots))
    return {
        "schema_version": "psu-hsof-loader-conformance-1.0",
        "status": status,
        "evidence_scope": "SELECTED_NUMERIC_LOADER_CONTRACT_ONLY_NO_NIRT_RECONSTRUCTION",
        "source_snapshot": {
            "file": source_file,
            "file_size_bytes": source_size,
            "subsystem_offset": subsystem_offset,
        },
        "checks": checks,
        "configuration": {
            "image_rows": image_rows,
            "image_columns": image_columns,
            "views": views,
            "measurement_count": measurements,
            "grid_shape": list(expected_grid_shape),
        },
        "coordinate_contract": axis_contracts,
        "ray_contract": {
            "sampled_measurements": len(v_vectors),
            "sampled_unique_camera_centers": len(camera_centers),
            "direction_norm_min": min(v_norms),
            "direction_norm_max": max(v_norms),
            "direction_norm_max_abs_error": v_norm_max_error,
            "aperture_min": min(aperture_values),
            "aperture_max": max(aperture_values),
        },
        "interpretation": [
            "the 9-view MAT snapshot is numerically consistent with the official loader geometry",
            "coordinate values are stored in meters on a centered cell-centered rectilinear grid",
            "sampled v vectors are unit directions and sampled camera centers resolve nine view poses",
            "Ruvecs and Rvvecs are checked for alignment and finiteness only; no unsupported orthonormality assumption is imposed",
        ],
        "limitations": [
            "only deterministic geometric landmarks and measurement rows are decoded as scalar values",
            "full selected compressed streams are validated, but full-array extrema and distributions are not computed",
            "deflection arrays, masks and heavy calibration fields are not yet numerically audited",
            "this report does not execute the author TensorFlow NIRT code or establish reconstruction quality",
        ],
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--ray", type=Path, required=True)
    parser.add_argument("--v", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_conformance_report(
        _load(args.scalar), _load(args.xyz), _load(args.ray), _load(args.v)
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "LOADER_NUMERIC_CONTRACT_CONFORMANT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
