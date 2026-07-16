#!/usr/bin/env python3
"""Run a deterministic B2 finite-aperture domain audit on PSU BOST shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .psu_bost_aperture_domain import (
        CONTRACT_VERSION as APERTURE_CONTRACT_VERSION,
        deterministic_paired_uniform_aperture_samples,
        evaluate_aperture_domain,
        generate_aperture_sample_points,
    )
    from .psu_bost_forward_geometry import (
        CONTRACT_VERSION as FORWARD_CONTRACT_VERSION,
        intersect_forward_ray_box,
        intersect_forward_ray_box_cone,
    )
    from .run_psu_fixed_domain_geometry_audit import _open_view_contract
except ImportError:  # Direct script execution.
    from psu_bost_aperture_domain import (  # type: ignore[no-redef]
        CONTRACT_VERSION as APERTURE_CONTRACT_VERSION,
        deterministic_paired_uniform_aperture_samples,
        evaluate_aperture_domain,
        generate_aperture_sample_points,
    )
    from psu_bost_forward_geometry import (  # type: ignore[no-redef]
        CONTRACT_VERSION as FORWARD_CONTRACT_VERSION,
        intersect_forward_ray_box,
        intersect_forward_ray_box_cone,
    )
    from run_psu_fixed_domain_geometry_audit import (  # type: ignore[no-redef]
        _open_view_contract,
    )


REPORT_SCHEMA = "psu-bost-aperture-domain-audit-1.0"
AGGREGATE_SCHEMA = "psu-bost-aperture-domain-all-view-audit-1.0"
DOMAINS = ("B0", "B1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _empty_support_accumulator(ray_count: int, sample_count: int) -> dict[str, Any]:
    return {
        "ray_count": ray_count,
        "centerline_hit_count": 0,
        "centerline_miss_count": 0,
        "eligible_sample_count": 0,
        "in_domain_sample_count": 0,
        "out_of_domain_sample_count": 0,
        "box_out_sample_count": 0,
        "cone_only_out_sample_count": 0,
        "all_samples_in_domain_ray_count": 0,
        "any_sample_out_of_domain_ray_count": 0,
        "empty_sample_support_ray_count": 0,
        "retained_sample_count_histogram": [0] * (sample_count + 1),
    }


def _add_histogram(target: list[int], values: np.ndarray, sample_count: int) -> None:
    update = np.bincount(values, minlength=sample_count + 1)
    if update.size != sample_count + 1:
        raise ValueError("retained sample count exceeded the fixed sample count")
    for index, count in enumerate(update.tolist()):
        target[index] += int(count)


def _nearest_rank_fraction(histogram: Sequence[int], quantile: float) -> float | None:
    total = int(sum(histogram))
    if total == 0:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    rank = 1 if quantile == 0.0 else int(math.ceil(quantile * total))
    cumulative = 0
    for retained, count in enumerate(histogram):
        cumulative += int(count)
        if cumulative >= rank:
            return retained / (len(histogram) - 1)
    raise AssertionError("histogram rank was not found")


def _finalize_support_accumulator(value: Mapping[str, Any]) -> dict[str, Any]:
    ray_count = int(value["ray_count"])
    hit_count = int(value["centerline_hit_count"])
    eligible_samples = int(value["eligible_sample_count"])
    in_samples = int(value["in_domain_sample_count"])
    histogram = [int(item) for item in value["retained_sample_count_histogram"]]
    if sum(histogram) != hit_count:
        raise ValueError("retained-sample histogram does not match centerline hits")
    return {
        **value,
        "centerline_hit_fraction": hit_count / ray_count if ray_count else None,
        "fixed_denominator_sample_retained_fraction": (
            in_samples / eligible_samples if eligible_samples else None
        ),
        "any_sample_out_of_domain_ray_fraction_of_hits": (
            int(value["any_sample_out_of_domain_ray_count"]) / hit_count
            if hit_count
            else None
        ),
        "empty_sample_support_ray_fraction_of_hits": (
            int(value["empty_sample_support_ray_count"]) / hit_count
            if hit_count
            else None
        ),
        "retained_fraction_minimum": _nearest_rank_fraction(histogram, 0.0),
        "retained_fraction_p10_nearest_rank": _nearest_rank_fraction(
            histogram, 0.10
        ),
        "retained_fraction_median_nearest_rank": _nearest_rank_fraction(
            histogram, 0.50
        ),
        "retained_fraction_p90_nearest_rank": _nearest_rank_fraction(
            histogram, 0.90
        ),
    }


def _accumulate_support(
    accumulator: dict[str, Any],
    *,
    hit: np.ndarray,
    retained_count: np.ndarray,
    box_out_count: np.ndarray,
    cone_only_out_count: np.ndarray,
    sample_count: int,
    selection: np.ndarray | None = None,
) -> None:
    if selection is None:
        selected_hit = hit
        selected_retained = retained_count
        selected_box_out = box_out_count
        selected_cone_out = cone_only_out_count
    else:
        selected_hit = hit[selection]
        selected_retained = retained_count[selection]
        selected_box_out = box_out_count[selection]
        selected_cone_out = cone_only_out_count[selection]
    eligible_retained = selected_retained[selected_hit]
    eligible_box_out = selected_box_out[selected_hit]
    eligible_cone_out = selected_cone_out[selected_hit]
    hit_count = int(np.count_nonzero(selected_hit))
    accumulator["centerline_hit_count"] += hit_count
    accumulator["centerline_miss_count"] += int(selected_hit.size - hit_count)
    accumulator["eligible_sample_count"] += hit_count * sample_count
    accumulator["in_domain_sample_count"] += int(eligible_retained.sum())
    accumulator["out_of_domain_sample_count"] += int(
        hit_count * sample_count - eligible_retained.sum()
    )
    accumulator["box_out_sample_count"] += int(eligible_box_out.sum())
    accumulator["cone_only_out_sample_count"] += int(eligible_cone_out.sum())
    accumulator["all_samples_in_domain_ray_count"] += int(
        np.count_nonzero(eligible_retained == sample_count)
    )
    accumulator["any_sample_out_of_domain_ray_count"] += int(
        np.count_nonzero(eligible_retained < sample_count)
    )
    accumulator["empty_sample_support_ray_count"] += int(
        np.count_nonzero(eligible_retained == 0)
    )
    _add_histogram(
        accumulator["retained_sample_count_histogram"],
        eligible_retained,
        sample_count,
    )


def _domain_chunk_samples(
    *,
    origin: np.ndarray,
    ray: Mapping[str, Any],
    rx: np.ndarray,
    ry: np.ndarray,
    rap: np.ndarray,
    df: np.ndarray,
    design: Mapping[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    vertex: np.ndarray,
    axis: np.ndarray,
    theta: float,
    domain: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    hit = np.asarray(ray["hit"], dtype=bool)
    chunk_rows = hit.size
    retained = np.zeros(chunk_rows, dtype=np.int64)
    box_out = np.zeros(chunk_rows, dtype=np.int64)
    cone_only_out = np.zeros(chunk_rows, dtype=np.int64)
    diagnostics = {
        "negative_radius_count": 0,
        "nonfinite_radius_count": 0,
        "basis_validation_failure_count": 0,
    }
    if not np.any(hit):
        return hit, retained, box_out, cone_only_out, diagnostics

    selected = np.flatnonzero(hit)
    enter = np.asarray(ray["enter"], dtype=np.float64)[selected]
    exit_ = np.asarray(ray["exit"], dtype=np.float64)[selected]
    direction = np.asarray(ray["direction_unit"], dtype=np.float64)[selected]
    start = origin[selected] + enter[:, None] * direction
    stop = origin[selected] + exit_[:, None] * direction
    selected_rap = rap[selected]
    selected_df = df[selected]
    rin = selected_rap * (1.0 - enter / selected_df)
    rout = selected_rap * (1.0 - exit_ / selected_df)
    diagnostics["nonfinite_radius_count"] = int(
        np.count_nonzero(~np.isfinite(rin)) + np.count_nonzero(~np.isfinite(rout))
    )
    diagnostics["negative_radius_count"] = int(
        np.count_nonzero(rin < 0.0) + np.count_nonzero(rout < 0.0)
    )
    if diagnostics["nonfinite_radius_count"] or diagnostics["negative_radius_count"]:
        return hit, retained, box_out, cone_only_out, diagnostics

    points = generate_aperture_sample_points(
        start,
        stop,
        rx[selected],
        ry[selected],
        rin,
        rout,
        design["longitudinal_fractions"],
        design["unit_disk_offsets"],
    )
    membership = evaluate_aperture_domain(
        points,
        lower,
        upper,
        cone_vertex=vertex if domain == "B1" else None,
        cone_axis=axis if domain == "B1" else None,
        cone_theta=theta if domain == "B1" else None,
    )
    indicator = np.asarray(membership["indicator"], dtype=bool)
    box_indicator = np.asarray(membership["box_indicator"], dtype=bool)
    cone_indicator = np.asarray(membership["cone_indicator"], dtype=bool)
    retained[selected] = np.count_nonzero(indicator, axis=1)
    box_out[selected] = np.count_nonzero(~box_indicator, axis=1)
    if domain == "B1":
        cone_only_out[selected] = np.count_nonzero(
            box_indicator & ~cone_indicator, axis=1
        )
    return hit, retained, box_out, cone_only_out, diagnostics


def audit_aperture_domain_view(
    *,
    view_dir: Path,
    sample_count: int = 16,
    chunk_rows: int = 16_384,
    outer_minimum: tuple[float, float, float] = (-0.110, -0.110, -0.110),
    outer_maximum: tuple[float, float, float] = (0.110, 0.110, 0.110),
    cone_vertex: tuple[float, float, float] = (0.060, 0.015, 0.0),
    cone_axis: tuple[float, float, float] = (-1.0, -0.1, 0.0),
    cone_angle_degrees: float = 25.0,
) -> dict[str, Any]:
    if chunk_rows < 2:
        raise ValueError("chunk_rows must be at least two")
    design = deterministic_paired_uniform_aperture_samples(sample_count)
    lower = np.asarray(outer_minimum, dtype=np.float64)
    upper = np.asarray(outer_maximum, dtype=np.float64)
    vertex = np.asarray(cone_vertex, dtype=np.float64)
    axis = np.asarray(cone_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    theta = math.radians(cone_angle_degrees)

    base_arrays, masks, metadata = _open_view_contract(view_dir)
    bundle_dir = view_dir / "bundle"
    arrays = {
        **base_arrays,
        "rx": np.load(bundle_dir / "Rxvecs.npy", mmap_mode="r", allow_pickle=False),
        "ry": np.load(bundle_dir / "Ryvecs.npy", mmap_mode="r", allow_pickle=False),
        "rap": np.load(bundle_dir / "Rapvec.npy", mmap_mode="r", allow_pickle=False),
        "df": np.load(bundle_dir / "Dfvec.npy", mmap_mode="r", allow_pickle=False),
    }
    rows = int(metadata["rows"])
    if arrays["rx"].shape != (rows, 3) or arrays["ry"].shape != (rows, 3):
        raise ValueError("Rxvecs and Ryvecs must have shape (N, 3)")
    if arrays["rap"].shape != (rows, 1) or arrays["df"].shape != (rows, 1):
        raise ValueError("Rapvec and Dfvec must have shape (N, 1)")

    totals = {
        domain: _empty_support_accumulator(rows, sample_count) for domain in DOMAINS
    }
    mask_totals = {
        name: {
            domain: _empty_support_accumulator(int(indices.size), sample_count)
            for domain in DOMAINS
        }
        for name, indices in masks.items()
    }
    diagnostics = {
        domain: {
            "negative_radius_count": 0,
            "nonfinite_radius_count": 0,
            "basis_validation_failure_count": 0,
        }
        for domain in DOMAINS
    }
    started = time.perf_counter()

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        origin = np.asarray(arrays["origin"][start:stop], dtype=np.float64)
        direction = np.asarray(arrays["direction"][start:stop], dtype=np.float64)
        rx = np.asarray(arrays["rx"][start:stop], dtype=np.float64)
        ry = np.asarray(arrays["ry"][start:stop], dtype=np.float64)
        rap = np.asarray(arrays["rap"][start:stop, 0], dtype=np.float64)
        df = np.asarray(arrays["df"][start:stop, 0], dtype=np.float64)
        if np.any(~np.isfinite(df)) or np.any(df <= 0.0):
            raise ValueError("Dfvec must contain finite positive values")
        if np.any(~np.isfinite(rap)) or np.any(rap < 0.0):
            raise ValueError("Rapvec must contain finite non-negative values")

        rays = {
            "B0": intersect_forward_ray_box(
                origin, direction, lower, upper, layout="rows"
            ),
            "B1": intersect_forward_ray_box_cone(
                origin,
                direction,
                lower,
                upper,
                vertex,
                axis,
                theta,
                layout="rows",
            ),
        }
        chunk_results: dict[str, tuple[np.ndarray, ...]] = {}
        for domain in DOMAINS:
            result = _domain_chunk_samples(
                origin=origin,
                ray=rays[domain],
                rx=rx,
                ry=ry,
                rap=rap,
                df=df,
                design=design,
                lower=lower,
                upper=upper,
                vertex=vertex,
                axis=axis,
                theta=theta,
                domain=domain,
            )
            chunk_results[domain] = result[:4]
            for key, value in result[4].items():
                diagnostics[domain][key] += int(value)
            _accumulate_support(
                totals[domain],
                hit=result[0],
                retained_count=result[1],
                box_out_count=result[2],
                cone_only_out_count=result[3],
                sample_count=sample_count,
            )

        for name, indices in masks.items():
            left = int(np.searchsorted(indices, start, side="left"))
            right = int(np.searchsorted(indices, stop, side="left"))
            local = np.asarray(indices[left:right] - start, dtype=np.int64)
            for domain in DOMAINS:
                result = chunk_results[domain]
                _accumulate_support(
                    mask_totals[name][domain],
                    hit=result[0],
                    retained_count=result[1],
                    box_out_count=result[2],
                    cone_only_out_count=result[3],
                    sample_count=sample_count,
                    selection=local,
                )

    invalid = any(
        value
        for domain in DOMAINS
        for value in diagnostics[domain].values()
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "B2_DETERMINISTIC_APERTURE_DOMAIN_AUDIT_INVALID"
            if invalid
            else "B2_DETERMINISTIC_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED"
        ),
        "evidence_scope": "REAL_ONE_VIEW_DETERMINISTIC_PAIRED_LOW_DISCREPANCY_APERTURE_SUPPORT_AUDIT_NO_TENSORFLOW_NO_RECONSTRUCTION",
        "view_id_zero_based": metadata["view_id_zero_based"],
        "source": {
            "bundle_manifest_sha256": metadata["bundle_manifest_sha256"],
            "setup_manifest_sha256": metadata["setup_manifest_sha256"],
            "corrected_mask_manifest_sha256": metadata["mask_manifest_sha256"],
            "forward_geometry_sha256": _sha256_file(
                Path(__file__).with_name("psu_bost_forward_geometry.py")
            ),
            "aperture_geometry_sha256": _sha256_file(
                Path(__file__).with_name("psu_bost_aperture_domain.py")
            ),
            "audit_implementation_sha256": _sha256_file(Path(__file__)),
        },
        "configuration": {
            "rows": rows,
            "chunk_rows": chunk_rows,
            "sample_count_per_centerline_hit": sample_count,
            "sample_design": design["design"],
            "longitudinal_fractions": design["longitudinal_fractions"].tolist(),
            "unit_disk_offsets": design["unit_disk_offsets"].tolist(),
            "outer_minimum_m": list(outer_minimum),
            "outer_maximum_m": list(outer_maximum),
            "cone_vertex_m": list(cone_vertex),
            "cone_axis_normalized": axis.tolist(),
            "cone_angle_degrees": cone_angle_degrees,
            "forward_contract_version": FORWARD_CONTRACT_VERSION,
            "aperture_contract_version": APERTURE_CONTRACT_VERSION,
            "normalization_policy": "FIXED_ORIGINAL_SAMPLE_COUNT_NO_SURVIVOR_RENORMALIZATION",
        },
        "domains": {
            domain: _finalize_support_accumulator(totals[domain])
            for domain in DOMAINS
        },
        "mask_conditioned": {
            name: {
                domain: _finalize_support_accumulator(value)
                for domain, value in domain_values.items()
            }
            for name, domain_values in mask_totals.items()
        },
        "diagnostics": diagnostics,
        "runtime_observation": {
            "wall_seconds": time.perf_counter() - started,
            "scope": "CACHED_LOCAL_DIAGNOSTIC_NOT_A_SPEED_BENCHMARK",
        },
        "decision": {
            "discrete_aperture_support_audited": not invalid,
            "continuous_aperture_containment_proved": False,
            "fixed_denominator_indicator_implemented": True,
            "geometry_safe_mask_built": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": "QMC_SAMPLE_COUNT_SENSITIVITY_THEN_B3_GEOMETRY_SAFE_MASK_AND_HELD_OUT_REPROJECTION",
        },
        "limitations": [
            "the deterministic low-discrepancy samples approximate the released random path-and-disk marginals but do not prove continuous aperture containment",
            "sample-retained fractions are support diagnostics under a fixed design, not reconstruction errors or confidence intervals",
            "the B1 cone parameters remain an unconfirmed computational sampling-hull hypothesis",
            "no TensorFlow loss, density field, held-out camera, inverse reconstruction, or algorithm comparison is run",
        ],
        "upstream_view_contract": {
            "bundle_status": metadata["bundle_status"],
            "setup_status": metadata["setup_status"],
            "mask_status": metadata["mask_status"],
        },
    }


def _sum_support_records(
    records: Sequence[Mapping[str, Any]],
    extractor: Any,
    sample_count: int,
) -> dict[str, Any]:
    values = [extractor(record) for record in records]
    accumulator = _empty_support_accumulator(
        sum(int(value["ray_count"]) for value in values), sample_count
    )
    for value in values:
        for key in accumulator:
            if key == "ray_count":
                continue
            if key == "retained_sample_count_histogram":
                for index, count in enumerate(value[key]):
                    accumulator[key][index] += int(count)
            else:
                accumulator[key] += int(value[key])
    return _finalize_support_accumulator(accumulator)


def aggregate_aperture_domain_views(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one view record is required")
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    if view_ids != list(range(len(records))):
        raise ValueError("view ids must be the ordered contiguous range from zero")
    sample_counts = {
        int(record["configuration"]["sample_count_per_centerline_hit"])
        for record in records
    }
    if len(sample_counts) != 1:
        raise ValueError("all views must use the same fixed sample count")
    sample_count = sample_counts.pop()
    invalid = [
        view_id
        for view_id, record in zip(view_ids, records)
        if record["status"]
        != "B2_DETERMINISTIC_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED"
    ]
    domains = {
        domain: _sum_support_records(
            records, lambda record, d=domain: record["domains"][d], sample_count
        )
        for domain in DOMAINS
    }
    masks = {
        mask: {
            domain: _sum_support_records(
                records,
                lambda record, m=mask, d=domain: record["mask_conditioned"][m][d],
                sample_count,
            )
            for domain in DOMAINS
        }
        for mask in ("amask_all", "imask_all")
    }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "execution_status": "COMPLETE",
        "scientific_verdict": (
            "INVALID"
            if invalid
            else "DISCRETE_APERTURE_SUPPORT_AUDIT_COMPLETE_B3_AND_HELD_OUT_REQUIRED"
        ),
        "status": (
            "B2_ALL_VIEW_APERTURE_DOMAIN_AUDIT_INVALID"
            if invalid
            else "B2_ALL_VIEW_APERTURE_DOMAIN_AUDIT_PASS_B3_REQUIRED"
        ),
        "view_count": len(records),
        "sample_count_per_centerline_hit": sample_count,
        "views": list(records),
        "aggregate": {
            "invalid_view_ids": invalid,
            "domains": domains,
            "mask_conditioned": masks,
        },
        "decision": {
            "discrete_aperture_support_audited": not invalid,
            "continuous_aperture_containment_proved": False,
            "fixed_denominator_indicator_implemented": True,
            "geometry_safe_mask_built": False,
            "training_ready": "NO",
            "algorithm_superiority_claim": "LOCKED",
            "next_gate": "QMC_SAMPLE_COUNT_SENSITIVITY_THEN_B3_GEOMETRY_SAFE_MASK_AND_HELD_OUT_REPROJECTION",
        },
        "limitations": [
            "B2 uses a deterministic discrete design and requires sample-count sensitivity before quantitative interpretation",
            "B1 cone physical semantics remain unconfirmed",
            "held-out camera and inverse reconstruction remain unrun",
            "the exhaustive centerline set does not turn the aperture discretization into a statistical confidence interval",
        ],
    }


def run_all_view_aperture_domain_audit(
    *,
    audit_root: Path,
    output_path: Path,
    view_count: int,
    sample_count: int = 16,
    chunk_rows: int = 16_384,
) -> dict[str, Any]:
    records = [
        audit_aperture_domain_view(
            view_dir=audit_root / f"view_{view_id:02d}",
            sample_count=sample_count,
            chunk_rows=chunk_rows,
        )
        for view_id in range(view_count)
    ]
    report = aggregate_aperture_domain_views(records)
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--chunk-rows", type=int, default=16_384)
    args = parser.parse_args()
    report = run_all_view_aperture_domain_audit(
        audit_root=args.audit_root,
        output_path=args.output,
        view_count=args.view_count,
        sample_count=args.sample_count,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["scientific_verdict"] != "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
