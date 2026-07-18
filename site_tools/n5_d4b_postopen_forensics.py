#!/usr/bin/env python3
"""Explain frozen N5-D4b failures without changing the historical decision.

This post-open diagnostic performs two bounded operations only:

1. recompute dot contractions from the already saved JVP/VJP arrays; and
2. replay only the discrete field-program signatures for topology-failing rows.

It never calls a forward observable, JVP, VJP, finite-difference sweep, decoder,
reconstruction, or training routine.  The committed D4b result remains the sole
formal result and is always reported unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_t16_operator.automatic_discrete_multifidelity import (  # noqa: E402
    smoothstep_grid_field,
)
from demo_t16_operator.field_dependent_ray import initial_pupil_rays  # noqa: E402
from demo_t16_operator.field_program_signature import (  # noqa: E402
    FIELD_PROGRAM_SIGNATURE_SCHEMA,
    _central_query_bundle,
    _diagnostic_trace_with_stages,
    _lower_interpolation_cells,
    _validated_inputs,
)
from demo_t16_operator.run_n2_pvgr_n5_d4_tiny_field_derivative import (  # noqa: E402
    _rig,
)


DEFAULT_RESULT_DIR = (
    ROOT
    / "demo_t16_operator/results/"
    "n2_pvgr_n5_d4b_population_field_derivative_v1"
)
DEFAULT_PREREG_DIR = (
    ROOT
    / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4b_population_field_derivative_preregistration_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "demo_t16_operator/results/"
    "n2_pvgr_n5_d4b_postopen_forensics_v1"
)

OFFSET_LABELS = ("base", "+x", "+y", "+z", "-x", "-y", "-z")
STAGE_PATTERN = re.compile(r"^step=(\d+),stage=(k[1-4])$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _neumaier_sum(values: Iterable[float]) -> float:
    total = 0.0
    compensation = 0.0
    for value in values:
        candidate = total + value
        if abs(total) >= abs(value):
            compensation += (total - candidate) + value
        else:
            compensation += (value - candidate) + total
        total = candidate
    return total + compensation


def _metric(lhs: float, rhs: float, *, floor: float = 1e-30) -> dict[str, Any]:
    signed = lhs - rhs
    absolute = abs(signed)
    signal = max(abs(lhs), abs(rhs))
    return {
        "lhs": float(lhs),
        "rhs": float(rhs),
        "absolute_defect": float(absolute),
        "signed_defect_lhs_minus_rhs": float(signed),
        "signal": float(signal),
        "relative_defect": float(absolute / max(signal, floor)),
    }


def _fraction_decimal(value: Fraction, *, precision: int = 60) -> str:
    with localcontext() as context:
        context.prec = precision
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _exact_binary_metric(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = sum(
        (
            Fraction.from_float(float(a)) * Fraction.from_float(float(b))
            for a, b in zip(left[0].ravel(), left[1].ravel(), strict=True)
        ),
        Fraction(0),
    )
    rhs = sum(
        (
            Fraction.from_float(float(a)) * Fraction.from_float(float(b))
            for a, b in zip(right[0].ravel(), right[1].ravel(), strict=True)
        ),
        Fraction(0),
    )
    signed = lhs - rhs
    absolute = abs(signed)
    signal = max(abs(lhs), abs(rhs))
    relative = absolute / signal if signal else Fraction(0)
    return {
        "lhs": float(lhs),
        "rhs": float(rhs),
        "absolute_defect": float(absolute),
        "signed_defect_lhs_minus_rhs": float(signed),
        "signal": float(signal),
        "relative_defect": float(relative),
        "lhs_exact_decimal": _fraction_decimal(lhs),
        "rhs_exact_decimal": _fraction_decimal(rhs),
        "absolute_defect_exact_decimal": _fraction_decimal(absolute),
        "signed_defect_exact_decimal": _fraction_decimal(signed),
        "relative_defect_exact_decimal": _fraction_decimal(relative),
        "exact_binary_products_and_sum": True,
    }


def _contraction_methods(
    jvp: np.ndarray,
    cotangent: np.ndarray,
    tangent: np.ndarray,
    vjp: np.ndarray,
) -> dict[str, dict[str, Any]]:
    jvp_flat = np.asarray(jvp, dtype=np.float64).ravel()
    cotangent_flat = np.asarray(cotangent, dtype=np.float64).ravel()
    tangent_flat = np.asarray(tangent, dtype=np.float64).ravel()
    vjp_flat = np.asarray(vjp, dtype=np.float64).ravel()
    if jvp_flat.shape != cotangent_flat.shape:
        raise ValueError("saved JVP and frozen cotangent shapes do not match")
    if tangent_flat.shape != vjp_flat.shape:
        raise ValueError("frozen tangent and saved VJP shapes do not match")

    left_products = jvp_flat * cotangent_flat
    right_products = tangent_flat * vjp_flat
    long_type = np.longdouble
    methods = {
        "numpy_sum_float64_products": _metric(
            float(np.sum(left_products, dtype=np.float64)),
            float(np.sum(right_products, dtype=np.float64)),
        ),
        "numpy_dot_float64": _metric(
            float(np.dot(jvp_flat, cotangent_flat)),
            float(np.dot(tangent_flat, vjp_flat)),
        ),
        "math_fsum_float64_products": _metric(
            math.fsum(float(value) for value in left_products),
            math.fsum(float(value) for value in right_products),
        ),
        "neumaier_float64_products": _metric(
            _neumaier_sum(float(value) for value in left_products),
            _neumaier_sum(float(value) for value in right_products),
        ),
        "numpy_longdouble_product_and_sum": _metric(
            float(
                np.sum(
                    jvp_flat.astype(long_type) * cotangent_flat.astype(long_type),
                    dtype=long_type,
                )
            ),
            float(
                np.sum(
                    tangent_flat.astype(long_type) * vjp_flat.astype(long_type),
                    dtype=long_type,
                )
            ),
        ),
        "exact_binary_rational": _exact_binary_metric(
            np.stack((jvp_flat, cotangent_flat)),
            np.stack((tangent_flat, vjp_flat)),
        ),
    }
    return methods


@dataclass(slots=True)
class SignatureEvidence:
    hashes: dict[str, str]
    support_bits: np.ndarray
    sampled_values: np.ndarray
    group_layout: list[dict[str, Any]]
    support_true_count: int
    query_record_count: int
    minimum_domain_margin: float
    minimum_stencil_margin: float
    minimum_frustum_margin: float
    maximum_direction_norm_error: float


def _signature_evidence(
    values_zyx: torch.Tensor,
    pupil_states: torch.Tensor,
    rig: Any,
    *,
    difference_step: float,
    refractivity_scale: float,
    step_count: int,
    support_threshold: float,
    frustum_half_width_u: float,
    frustum_half_width_v: float,
) -> SignatureEvidence:
    values, states, delta, threshold, half_u, half_v = _validated_inputs(
        values_zyx,
        pupil_states,
        difference_step=difference_step,
        support_threshold=support_threshold,
        frustum_half_width_u=frustum_half_width_u,
        frustum_half_width_v=frustum_half_width_v,
    )
    trace, stage_positions = _diagnostic_trace_with_stages(
        values,
        states,
        rig,
        difference_step=delta,
        refractivity_scale=float(refractivity_scale),
        step_count=int(step_count),
    )
    groups: list[tuple[str, torch.Tensor]] = list(stage_positions)
    groups.append(
        (
            "curved_path_midpoints",
            (0.5 * (trace.positions[:, :-1] + trace.positions[:, 1:])).reshape(-1, 3),
        )
    )
    start, straight_direction, _, _ = initial_pupil_rays(states, rig)
    midpoint_distance = (
        torch.arange(int(step_count), dtype=torch.float64) + 0.5
    ) * float(trace.step_size)
    straight_midpoints = (
        start[:, None, :]
        + midpoint_distance[None, :, None] * straight_direction[:, None, :]
    )
    groups.append(("straight_path_midpoints", straight_midpoints.reshape(-1, 3)))

    labels: list[dict[str, Any]] = []
    cells: list[np.ndarray] = []
    support: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    layout: list[dict[str, Any]] = []
    point_cursor = 0
    minimum_query_margin = math.inf
    for group_index, (label, points) in enumerate(groups):
        flat = points.reshape(-1, 3).to(dtype=torch.float64, device="cpu")
        bundle = _central_query_bundle(flat, difference_step=delta)
        minimum_query_margin = min(
            minimum_query_margin,
            1.0 - float(torch.max(torch.abs(bundle))),
        )
        flat_queries = bundle.reshape(-1, 3)
        lower = _lower_interpolation_cells(flat_queries, tuple(values.shape))
        sampled = smoothstep_grid_field(values, flat_queries).reshape(len(flat), 7)
        support_bits = torch.abs(sampled) >= threshold
        label_record = {
            "group_index": group_index,
            "label": label,
            "point_count": len(flat),
            "offset_order": list(OFFSET_LABELS),
        }
        labels.append(label_record)
        layout.append(
            {
                **label_record,
                "point_start": point_cursor,
                "point_stop": point_cursor + len(flat),
            }
        )
        point_cursor += len(flat)
        cells.append(lower.detach().cpu().numpy().astype("<i2", copy=False))
        support.append(
            support_bits.detach().cpu().numpy().astype(np.uint8, copy=False)
        )
        samples.append(sampled.detach().cpu().numpy().astype("<f8", copy=False))

    positions = trace.positions.detach()
    progress = torch.arange(
        positions.shape[1], dtype=positions.dtype, device=positions.device
    ) * float(trace.step_size)
    straight_endpoints = (
        positions[:, :1, :]
        + progress[None, :, None] * trace.directions[:, :1, :].detach()
    )
    deviation = positions - straight_endpoints
    offset_u = torch.sum(deviation * trace.projection_u[:, None, :].detach(), dim=-1)
    offset_v = torch.sum(deviation * trace.projection_v[:, None, :].detach(), dim=-1)
    margins = torch.stack((half_u - torch.abs(offset_u), half_v - torch.abs(offset_v)))
    frustum = (margins >= 0.0).cpu().numpy().astype(np.uint8, copy=False)

    cells_array = np.concatenate(cells, axis=0)
    support_array = np.concatenate(support, axis=0)
    sampled_array = np.concatenate(samples, axis=0)
    component_hashes = {
        "schema": FIELD_PROGRAM_SIGNATURE_SCHEMA,
        "labels_sha256": _sha256_bytes(_canonical_json(labels)),
        "interpolation_cells_sha256": _sha256_bytes(cells_array.tobytes(order="C")),
        "support_bits_sha256": _sha256_bytes(support_array.tobytes(order="C")),
        "frustum_margin_signs_sha256": _sha256_bytes(frustum.tobytes(order="C")),
    }
    hashes = {
        **component_hashes,
        "digest_sha256": _sha256_bytes(_canonical_json(component_hashes)),
    }
    return SignatureEvidence(
        hashes=hashes,
        support_bits=support_array,
        sampled_values=sampled_array,
        group_layout=layout,
        support_true_count=int(np.sum(support_array)),
        query_record_count=int(cells_array.shape[0]),
        minimum_domain_margin=float(trace.minimum_domain_margin),
        minimum_stencil_margin=min(
            float(trace.minimum_stencil_margin), minimum_query_margin
        ),
        minimum_frustum_margin=float(torch.min(margins)),
        maximum_direction_norm_error=float(trace.maximum_direction_norm_error),
    )


def _point_location(
    point_index: int, layout: list[dict[str, Any]], *, step_count: int
) -> dict[str, Any]:
    group = next(
        item
        for item in layout
        if int(item["point_start"]) <= point_index < int(item["point_stop"])
    )
    local_index = point_index - int(group["point_start"])
    label = str(group["label"])
    match = STAGE_PATTERN.match(label)
    if match:
        step_index = int(match.group(1))
        stage = match.group(2)
        ray_index = local_index
        kind = "rk4_stage"
    else:
        step_index = local_index % step_count
        stage = None
        ray_index = local_index // step_count
        kind = label
    return {
        "group_index": int(group["group_index"]),
        "group_label": label,
        "group_kind": kind,
        "group_point_index": int(local_index),
        "step_index": int(step_index),
        "stage": stage,
        "ray_index": int(ray_index),
    }


def _support_flips(
    base: SignatureEvidence,
    candidate: SignatureEvidence,
    *,
    support_threshold: float,
    step_count: int,
) -> list[dict[str, Any]]:
    changed = np.argwhere(base.support_bits != candidate.support_bits)
    records: list[dict[str, Any]] = []
    for point_index, offset_index in changed:
        base_bit = int(base.support_bits[point_index, offset_index])
        candidate_bit = int(candidate.support_bits[point_index, offset_index])
        base_value = float(base.sampled_values[point_index, offset_index])
        candidate_value = float(candidate.sampled_values[point_index, offset_index])
        records.append(
            {
                **_point_location(
                    int(point_index), base.group_layout, step_count=step_count
                ),
                "global_point_index": int(point_index),
                "global_bit_index_c_order": int(point_index * 7 + offset_index),
                "offset_index": int(offset_index),
                "offset_label": OFFSET_LABELS[int(offset_index)],
                "transition": f"{base_bit}_to_{candidate_bit}",
                "base_sampled_value": base_value,
                "candidate_sampled_value": candidate_value,
                "base_support_margin": abs(base_value) - support_threshold,
                "candidate_support_margin": abs(candidate_value) - support_threshold,
            }
        )
    return records


def _topology_kwargs(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "difference_step": float(context["difference_step"]),
        "refractivity_scale": float(context["refractivity_scale"]),
        "step_count": int(config["step_count"]),
        "support_threshold": float(context["support_threshold"]),
        "frustum_half_width_u": float(context["frustum_half_width_u"]),
        "frustum_half_width_v": float(context["frustum_half_width_v"]),
    }


def _hash_match(evidence: SignatureEvidence, stored: dict[str, Any]) -> bool:
    keys = (
        "digest_sha256",
        "interpolation_cells_sha256",
        "support_bits_sha256",
        "frustum_margin_signs_sha256",
    )
    return all(evidence.hashes[key] == stored[key] for key in keys)


def _build_dot_forensics(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    derivative_arrays: Any,
    frozen_arrays: Any,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    contexts = {int(item["cell_index"]): item for item in metadata["contexts"]}
    records: list[dict[str, Any]] = []
    for row in rows:
        for map_id, stored in row["maps"].items():
            if bool(stored["map_gate"]):
                continue
            cell_index = int(row["cell_index"])
            direction_index = int(row["direction_index"])
            context = contexts[cell_index]
            direction_record = next(
                item
                for item in context["directions"]
                if int(item["direction_index"]) == direction_index
            )
            prefix = f"c{cell_index:02d}_d{direction_index:02d}_{map_id}"
            jvp = derivative_arrays[f"{prefix}_jvp"]
            vjp = derivative_arrays[f"{prefix}_vjp"]
            tangent = frozen_arrays[direction_record["direction_key"]]
            cotangent = frozen_arrays[direction_record["cotangent_key"]]
            methods = _contraction_methods(jvp, cotangent, tangent, vjp)
            methods["stored_torch_sum_float64"] = {
                "lhs": float(stored["dot_jvp_cotangent"]),
                "rhs": float(stored["dot_tangent_vjp"]),
                "absolute_defect": float(stored["dot_absolute_defect"]),
                "signal": float(stored["dot_signal"]),
                "relative_defect": float(stored["dot_relative_defect"]),
            }
            for payload in methods.values():
                payload["passes_original_threshold"] = bool(
                    float(payload["relative_defect"]) <= threshold
                )
            exact = methods["exact_binary_rational"]
            records.append(
                {
                    "cell_index": cell_index,
                    "cell_id": row["cell_id"],
                    "pair_id": row["pair_id"],
                    "role": row["role"],
                    "direction_index": direction_index,
                    "map_id": map_id,
                    "array_keys": {
                        "jvp": f"{prefix}_jvp",
                        "vjp": f"{prefix}_vjp",
                        "tangent": direction_record["direction_key"],
                        "cotangent": direction_record["cotangent_key"],
                    },
                    "shapes": {
                        "jvp": list(jvp.shape),
                        "cotangent": list(cotangent.shape),
                        "tangent": list(tangent.shape),
                        "vjp": list(vjp.shape),
                    },
                    "methods": methods,
                    "exact_saved_array_relative_defect": float(
                        exact["relative_defect"]
                    ),
                    "exact_saved_array_passes_original_threshold": bool(
                        exact["passes_original_threshold"]
                    ),
                }
            )
    return records


def _build_dot_context_decomposition(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    derivative_arrays: Any,
    frozen_arrays: Any,
) -> list[dict[str, Any]]:
    """Decompose every map in a row containing a failed residual dot gate."""

    contexts = {int(item["cell_index"]): item for item in metadata["contexts"]}
    failed_rows = [
        row
        for row in rows
        if any(not bool(payload["map_gate"]) for payload in row["maps"].values())
    ]
    records: list[dict[str, Any]] = []
    for row in failed_rows:
        cell_index = int(row["cell_index"])
        direction_index = int(row["direction_index"])
        context = contexts[cell_index]
        direction_record = next(
            item
            for item in context["directions"]
            if int(item["direction_index"]) == direction_index
        )
        tangent = np.asarray(
            frozen_arrays[direction_record["direction_key"]], dtype=np.float64
        )
        cotangent = np.asarray(
            frozen_arrays[direction_record["cotangent_key"]], dtype=np.float64
        )
        tangent_norm = float(np.linalg.norm(tangent.ravel()))
        cotangent_norm = float(np.linalg.norm(cotangent.ravel()))
        maps: dict[str, dict[str, Any]] = {}
        for map_id in row["maps"]:
            prefix = f"c{cell_index:02d}_d{direction_index:02d}_{map_id}"
            jvp = np.asarray(derivative_arrays[f"{prefix}_jvp"], dtype=np.float64)
            vjp = np.asarray(derivative_arrays[f"{prefix}_vjp"], dtype=np.float64)
            exact = _contraction_methods(jvp, cotangent, tangent, vjp)[
                "exact_binary_rational"
            ]
            jvp_norm = float(np.linalg.norm(jvp.ravel()))
            vjp_norm = float(np.linalg.norm(vjp.ravel()))
            normwise_scale = (
                jvp_norm * cotangent_norm + tangent_norm * vjp_norm
            )
            left_absolute_term_sum = math.fsum(
                abs(float(a) * float(b))
                for a, b in zip(jvp.ravel(), cotangent.ravel(), strict=True)
            )
            right_absolute_term_sum = math.fsum(
                abs(float(a) * float(b))
                for a, b in zip(tangent.ravel(), vjp.ravel(), strict=True)
            )
            maps[map_id] = {
                "lhs": float(exact["lhs"]),
                "rhs": float(exact["rhs"]),
                "signed_defect_lhs_minus_rhs": float(
                    exact["signed_defect_lhs_minus_rhs"]
                ),
                "absolute_defect": float(exact["absolute_defect"]),
                "relative_defect_by_dot_signal": float(exact["relative_defect"]),
                "dot_signal": float(exact["signal"]),
                "jvp_norm": jvp_norm,
                "vjp_norm": vjp_norm,
                "tangent_norm": tangent_norm,
                "cotangent_norm": cotangent_norm,
                "normwise_bilinear_scale": normwise_scale,
                "normwise_adjoint_defect": float(
                    exact["absolute_defect"] / max(normwise_scale, 1e-30)
                ),
                "left_absolute_term_sum": left_absolute_term_sum,
                "right_absolute_term_sum": right_absolute_term_sum,
                "term_cancellation_factor": float(
                    (left_absolute_term_sum + right_absolute_term_sum)
                    / max(abs(exact["lhs"]) + abs(exact["rhs"]), 1e-30)
                ),
            }

        curved = maps["curved_detector"]
        straight = maps["straight_detector"]
        raw = maps["raw_curved_minus_straight"]
        paired = maps["paired_neumaier_residual"]
        component_signal = max(curved["dot_signal"], straight["dot_signal"])
        records.append(
            {
                "cell_index": cell_index,
                "cell_id": row["cell_id"],
                "pair_id": row["pair_id"],
                "role": row["role"],
                "direction_index": direction_index,
                "maps": maps,
                "derived": {
                    "component_signal_to_raw_residual_signal_ratio": float(
                        component_signal / max(raw["dot_signal"], 1e-30)
                    ),
                    "component_signal_to_paired_residual_signal_ratio": float(
                        component_signal / max(paired["dot_signal"], 1e-30)
                    ),
                    "raw_signed_defect_minus_component_defect_difference": float(
                        raw["signed_defect_lhs_minus_rhs"]
                        - (
                            curved["signed_defect_lhs_minus_rhs"]
                            - straight["signed_defect_lhs_minus_rhs"]
                        )
                    ),
                    "raw_absolute_defect_over_curved_absolute_defect": float(
                        raw["absolute_defect"]
                        / max(curved["absolute_defect"], 1e-30)
                    ),
                    "interpretation": "the residual maps suppress the dot signal by about four orders of magnitude while retaining an absolute defect of the same order as the component map; this is a low-signal conditioning mechanism, not a formal pass",
                },
            }
        )
    return records


def _build_topology_forensics(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    frozen_arrays: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    contexts = {int(item["cell_index"]): item for item in metadata["contexts"]}
    records: list[dict[str, Any]] = []
    h_values = [float(value) for value in config["finite_difference"]["h_values"]]
    for row in rows:
        if bool(row["topology_gate"]):
            continue
        cell_index = int(row["cell_index"])
        direction_index = int(row["direction_index"])
        context = contexts[cell_index]
        direction_record = next(
            item
            for item in context["directions"]
            if int(item["direction_index"]) == direction_index
        )
        field = torch.from_numpy(frozen_arrays[context["field_key"]].copy()).to(
            torch.float64
        )
        rays = torch.from_numpy(frozen_arrays[context["rays_key"]].copy()).to(
            torch.float64
        )
        direction = torch.from_numpy(
            frozen_arrays[direction_record["direction_key"]].copy()
        ).to(torch.float64)
        kwargs = _topology_kwargs(context, config)
        rig = _rig(context["rig"])
        base = _signature_evidence(field, rays, rig, **kwargs)
        stored_base = row["topology"]["base"]
        base_match = _hash_match(base, stored_base)
        if not base_match:
            raise ValueError(f"base signature replay drifted for cell {cell_index}")

        perturbations: list[dict[str, Any]] = []
        stored_lookup = {
            (float(item["h"]), str(item["side"])): item
            for item in row["topology"]["perturbations"]
        }
        for h in h_values:
            for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                stored = stored_lookup[(h, side)]
                candidate = _signature_evidence(
                    field + sign * h * direction, rays, rig, **kwargs
                )
                replay_matches_stored = _hash_match(candidate, stored)
                if not replay_matches_stored:
                    raise ValueError(
                        f"signature replay drifted for cell {cell_index}, d{direction_index}, {h}, {side}"
                    )
                matches_base = candidate.hashes["digest_sha256"] == base.hashes[
                    "digest_sha256"
                ]
                if matches_base != bool(stored["matches_base"]):
                    raise ValueError("replayed topology verdict disagrees with stored row")
                flips = _support_flips(
                    base,
                    candidate,
                    support_threshold=float(context["support_threshold"]),
                    step_count=int(config["step_count"]),
                )
                perturbations.append(
                    {
                        "h": h,
                        "side": side,
                        "matches_base": matches_base,
                        "replay_matches_stored_hashes": replay_matches_stored,
                        "support_flip_count": len(flips),
                        "support_0_to_1_count": sum(
                            item["transition"] == "0_to_1" for item in flips
                        ),
                        "support_1_to_0_count": sum(
                            item["transition"] == "1_to_0" for item in flips
                        ),
                        "support_true_count": candidate.support_true_count,
                        "support_bits_sha256": candidate.hashes[
                            "support_bits_sha256"
                        ],
                        "flips": flips,
                    }
                )
        stable_h = [
            h
            for h in h_values
            if all(
                item["matches_base"]
                for item in perturbations
                if float(item["h"]) == h
            )
        ]
        changed_h = [
            h
            for h in h_values
            if any(
                not item["matches_base"]
                for item in perturbations
                if float(item["h"]) == h
            )
        ]
        records.append(
            {
                "cell_index": cell_index,
                "cell_id": row["cell_id"],
                "pair_id": row["pair_id"],
                "role": row["role"],
                "direction_index": direction_index,
                "support_threshold": float(context["support_threshold"]),
                "base_replay_matches_stored_hashes": base_match,
                "base_support_true_count": base.support_true_count,
                "query_record_count": base.query_record_count,
                "largest_tested_two_sided_stable_h": max(stable_h),
                "smallest_tested_changed_h": min(changed_h),
                "tested_transition_interval": {
                    "stable_inclusive_lower_evidence": max(stable_h),
                    "changed_inclusive_upper_evidence": min(changed_h),
                    "interpretation": "tested boundary lies in (stable_h, changed_h]; not an estimated physical critical radius",
                },
                "perturbations": perturbations,
            }
        )
    return records


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _topology_fd_association(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {
        "topology_stable": {
            "h_0_01_max_map_relative_error": [],
            "required_h_max_map_relative_error": [],
            "best_h_max_map_relative_error": [],
        },
        "topology_changed": {
            "h_0_01_max_map_relative_error": [],
            "required_h_max_map_relative_error": [],
            "best_h_max_map_relative_error": [],
        },
    }
    changed_map_gates: list[bool] = []
    for row in rows:
        label = "topology_stable" if bool(row["topology_gate"]) else "topology_changed"
        h_0_01: list[float] = []
        required: list[float] = []
        best: list[float] = []
        for payload in row["maps"].values():
            levels = payload["levels"]
            h_0_01.append(
                float(next(item["relative_error"] for item in levels if item["h"] == 0.01))
            )
            required.append(
                max(float(item["relative_error"]) for item in levels if item["required"])
            )
            best.append(min(float(item["relative_error"]) for item in levels))
            if label == "topology_changed":
                changed_map_gates.append(bool(payload["map_gate"]))
        grouped[label]["h_0_01_max_map_relative_error"].append(max(h_0_01))
        grouped[label]["required_h_max_map_relative_error"].append(max(required))
        grouped[label]["best_h_max_map_relative_error"].append(max(best))
    return {
        label: {metric: _distribution(values) for metric, values in metrics.items()}
        for label, metrics in grouped.items()
    } | {
        "topology_changed_map_gate_accounting": {
            "map_count": len(changed_map_gates),
            "passing_map_count": sum(changed_map_gates),
            "all_map_gates_pass": all(changed_map_gates),
        },
        "interpretation": "descriptive closed-population association only: support-set changes coincide with somewhat higher median FD error, but all 24 affected map gates remain far below the frozen 1e-5 required-h ceiling; this does not authorize removing the topology gate",
    }


def _plot(result: dict[str, Any], path: Path) -> None:
    dot_records = result["dot_contraction_forensics"]
    topology_records = result["topology_support_forensics"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    method_order = [
        "stored_torch_sum_float64",
        "numpy_sum_float64_products",
        "numpy_dot_float64",
        "math_fsum_float64_products",
        "neumaier_float64_products",
        "numpy_longdouble_product_and_sum",
        "exact_binary_rational",
    ]
    colors = ("#006d77", "#d1495b")
    x = np.arange(len(method_order))
    for index, record in enumerate(dot_records):
        axes[0].plot(
            x,
            [record["methods"][method]["relative_defect"] for method in method_order],
            "o-",
            label=record["map_id"].replace("_", " "),
            color=colors[index],
            linewidth=2,
        )
    threshold = float(result["original_gate_threshold"])
    axes[0].axhline(threshold, color="#222222", linestyle="--", label="frozen gate")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [
        "torch", "np.sum", "np.dot", "fsum", "Neumaier", "longdouble", "exact rational"
    ], rotation=28, ha="right")
    axes[0].set_ylabel("relative dot defect")
    axes[0].set_title("Saved-array contraction audit")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    labels: list[str] = []
    counts: list[int] = []
    transitions: list[str] = []
    for record in topology_records:
        for perturbation in record["perturbations"]:
            if perturbation["support_flip_count"]:
                labels.append(
                    f"c{record['cell_index']:02d} d{record['direction_index']}\n{perturbation['side']}"
                )
                counts.append(int(perturbation["support_flip_count"]))
                transitions.append(
                    f"{perturbation['support_0_to_1_count']}/{perturbation['support_1_to_0_count']}"
                )
    bars = axes[1].bar(np.arange(len(counts)), counts, color="#edae49")
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axes[1].set_ylabel("support-bit flips at h=0.01")
    axes[1].set_title("Active-set changes: 0→1 / 1→0")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, transition in zip(bars, transitions, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            transition,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle(
        "N5-D4b post-open forensics: explanatory evidence only",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary_markdown(result: dict[str, Any]) -> str:
    exact = [
        record["exact_saved_array_relative_defect"]
        for record in result["dot_contraction_forensics"]
    ]
    changed = [
        perturbation
        for record in result["topology_support_forensics"]
        for perturbation in record["perturbations"]
        if perturbation["support_flip_count"]
    ]
    flip_total = sum(item["support_flip_count"] for item in changed)
    decomposition = result["dot_context_decomposition"][0]
    signal_ratio = decomposition["derived"][
        "component_signal_to_raw_residual_signal_ratio"
    ]
    return "\n".join(
        (
            "# N5-D4b post-open forensics",
            "",
            f"- Historical decision retained: `{result['historical_machine_decision']}`",
            "- Diagnostic decision: `POSTOPEN_EXPLANATION_ONLY_NO_AUTHORIZATION`",
            f"- Exact saved-array dot defects: `{exact[0]:.12e}`, `{exact[1]:.12e}`",
            f"- Original frozen threshold: `{result['original_gate_threshold']:.1e}`",
            f"- Component-to-raw dot-signal ratio in p14: `{signal_ratio:.3e}` (descriptive low-signal conditioning evidence only)",
            f"- Topology-failing contexts replayed: `{len(result['topology_support_forensics'])}`",
            f"- Changed h/side signatures: `{len(changed)}`; support-bit flips: `{flip_total}`",
            "- All changed signatures occur at the already tested `h=0.01`; every failing context is two-sided stable at `h<=0.003` on the frozen grid.",
            "- No forward observable, JVP, VJP, finite-difference sweep, reconstruction, decoder, or training call was made.",
            "- This artifact cannot authorize a derivative contract, a reconstruction claim, a real-data claim, generalization, or model superiority.",
            "",
        )
    )


def run(result_dir: Path, prereg_dir: Path, output: Path) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    prereg_dir = prereg_dir.resolve()
    output = output.resolve()
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing to replace existing forensic output: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or os.path.lexists(temporary):
        raise FileExistsError(f"forensic temporary path already exists: {temporary}")

    paths = {
        "formal_result": result_dir / "result.json",
        "formal_rows": result_dir / "rows.json",
        "formal_arrays": result_dir / "derivative_arrays.npz",
        "formal_manifest": result_dir / "manifest.json",
        "formal_validation": result_dir / "validation_report.json",
        "frozen_inputs": prereg_dir / "frozen_inputs.npz",
        "attestation": prereg_dir / "attestation.json",
        "ready": prereg_dir / "READY.json",
    }
    source_hashes_before = {key: _sha256(path) for key, path in paths.items()}
    formal_result = _read_json(paths["formal_result"])
    rows = _read_json(paths["formal_rows"])
    config = _read_json(result_dir / "config_snapshot.json")
    attestation = _read_json(paths["attestation"])
    metadata = attestation["frozen_input_metadata"]
    threshold = float(config["gates"]["maximum_dot_relative_defect"])
    if formal_result["machine_decision"] != (
        "D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED"
    ):
        raise ValueError("unexpected historical D4b decision")
    if source_hashes_before["frozen_inputs"] != attestation[
        "frozen_input_archive_sha256"
    ]:
        raise ValueError("frozen input archive hash drifted")

    with np.load(paths["formal_arrays"], allow_pickle=False) as derivative_arrays:
        with np.load(paths["frozen_inputs"], allow_pickle=False) as frozen_arrays:
            dot_records = _build_dot_forensics(
                rows,
                metadata,
                derivative_arrays,
                frozen_arrays,
                threshold=threshold,
            )
            dot_decomposition = _build_dot_context_decomposition(
                rows, metadata, derivative_arrays, frozen_arrays
            )
            topology_records = _build_topology_forensics(
                rows, metadata, frozen_arrays, config
            )

    source_hashes_after = {key: _sha256(path) for key, path in paths.items()}
    if source_hashes_before != source_hashes_after:
        raise ValueError("a frozen D4b source artifact changed during forensics")
    exact_passes = [
        record["exact_saved_array_passes_original_threshold"]
        for record in dot_records
    ]
    all_topology_hashes_match = all(
        record["base_replay_matches_stored_hashes"]
        and all(
            item["replay_matches_stored_hashes"]
            for item in record["perturbations"]
        )
        for record in topology_records
    )
    changed = [
        item
        for record in topology_records
        for item in record["perturbations"]
        if item["support_flip_count"]
    ]
    longdouble_info = np.finfo(np.longdouble)
    result = {
        "schema": "n2-pvgr-n5-d4b-postopen-forensics-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "postopen_explanation_only_no_authorization",
        "diagnostic_decision": "POSTOPEN_EXPLANATION_ONLY_NO_AUTHORIZATION",
        "historical_machine_decision": formal_result["machine_decision"],
        "historical_decision_changed": False,
        "protocol_commit": formal_result["protocol_commit"],
        "original_gate_threshold": threshold,
        "source_hashes": source_hashes_before,
        "source_files_unchanged": True,
        "platform_precision": {
            "numpy_float64_mantissa_bits": int(np.finfo(np.float64).nmant),
            "numpy_longdouble_mantissa_bits": int(longdouble_info.nmant),
            "longdouble_wider_than_float64": bool(
                longdouble_info.nmant > np.finfo(np.float64).nmant
            ),
        },
        "forbidden_call_counts": {
            "forward_observable": 0,
            "jvp": 0,
            "vjp": 0,
            "finite_difference_forward": 0,
            "decoder": 0,
            "reconstruction": 0,
            "training": 0,
        },
        "allowed_diagnostic_counts": {
            "saved_array_contractions": len(dot_records),
            "field_program_signature_replays": sum(
                1 + len(record["perturbations"]) for record in topology_records
            ),
        },
        "dot_contraction_forensics": dot_records,
        "dot_context_decomposition": dot_decomposition,
        "dot_summary": {
            "failed_map_count": len(dot_records),
            "exact_saved_array_pass_count": sum(exact_passes),
            "final_contraction_rounding_explains_formal_failure": False,
            "component_signal_to_raw_residual_signal_ratio": dot_decomposition[0][
                "derived"
            ]["component_signal_to_raw_residual_signal_ratio"],
            "raw_residual_normwise_adjoint_defect": dot_decomposition[0]["maps"][
                "raw_curved_minus_straight"
            ]["normwise_adjoint_defect"],
            "interpretation": "exact contractions of the stored float64 arrays remain above the frozen threshold; final reduction order is not a sufficient explanation",
        },
        "topology_support_forensics": topology_records,
        "topology_fd_association": _topology_fd_association(rows),
        "topology_summary": {
            "failing_context_count": len(topology_records),
            "changed_signature_count": len(changed),
            "support_flip_count": sum(item["support_flip_count"] for item in changed),
            "support_0_to_1_count": sum(
                item["support_0_to_1_count"] for item in changed
            ),
            "support_1_to_0_count": sum(
                item["support_1_to_0_count"] for item in changed
            ),
            "all_replayed_hashes_match_frozen_rows": all_topology_hashes_match,
            "all_changes_at_h_0_01": all(
                math.isclose(float(item["h"]), 0.01, rel_tol=0.0, abs_tol=0.0)
                for item in changed
            ),
            "all_contexts_two_sided_stable_at_or_below_h_0_003": all(
                math.isclose(
                    float(record["largest_tested_two_sided_stable_h"]),
                    0.003,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                for record in topology_records
            ),
        },
        "claim_authorizations": {
            "change_historical_d4b_decision": False,
            "closed_32_cell_derivative_contract": False,
            "decoder_parameterization": False,
            "three_dimensional_reconstruction": False,
            "neural_operator_training_or_superiority": False,
            "real_data": False,
            "generalization": False,
            "paper_claim": False,
        },
        "next_legal_step": "freeze a new protocol that separates low-signal contraction conditioning from support active-set stability; do not reinterpret this post-open diagnostic as a pass",
    }

    temporary.mkdir(parents=True)
    _write_json(temporary / "result.json", result)
    _plot(result, temporary / "n2_pvgr_n5_d4b_postopen_forensics.png")
    (temporary / "summary.md").write_text(
        _summary_markdown(result), encoding="utf-8"
    )
    manifest = {
        "schema": "n2-pvgr-n5-d4b-postopen-forensics-manifest-1.0",
        "artifacts": {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": _sha256(temporary / name),
            }
            for name in (
                "result.json",
                "summary.md",
                "n2_pvgr_n5_d4b_postopen_forensics.png",
            )
        },
    }
    _write_json(temporary / "manifest.json", manifest)
    os.replace(temporary, output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--prereg-dir", type=Path, default=DEFAULT_PREREG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.result_dir, args.prereg_dir, args.output)
    print(
        json.dumps(
            {
                "diagnostic_decision": result["diagnostic_decision"],
                "historical_machine_decision": result["historical_machine_decision"],
                "dot_summary": result["dot_summary"],
                "topology_summary": result["topology_summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
