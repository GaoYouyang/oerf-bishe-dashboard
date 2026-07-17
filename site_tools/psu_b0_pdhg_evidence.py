"""Stable numeric evidence anchors for the preregistered PSU-B0 PDHG screen."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import numpy as np
import torch


GEOMETRY_FIELDS = (
    "sample_points",
    "projection_u",
    "projection_v",
    "detector_xy",
    "line_length",
    "system_constant",
)
WHITENER_BUFFERS = (
    "matrix",
    "calibration_mean",
    "scale_by_view",
    "predictive_scale_by_view",
)


def _canonical_json(value: Any, *, label: str) -> tuple[Any, bytes]:
    """Return a JSON-normalized value and its compact canonical encoding."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        normalized = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable: {exc}") from exc
    return normalized, payload


def _numeric_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"{name} must be a strided numeric tensor")
        tensor = value.detach().resolve_conj().resolve_neg().cpu()
        try:
            array = tensor.numpy()
        except (RuntimeError, TypeError) as exc:
            raise TypeError(
                f"{name} has a torch dtype that cannot be represented canonically"
            ) from exc
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor")

    if array.dtype.kind not in "iufc":
        raise TypeError(f"{name} must have a numeric dtype, got {array.dtype}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")

    little_endian_dtype = array.dtype.newbyteorder("<")
    little_endian = array.astype(little_endian_dtype, copy=False)
    return np.ascontiguousarray(little_endian)


def canonical_numeric_fingerprint(name: str, value: Any) -> dict[str, Any]:
    """Fingerprint a numeric array after canonical CPU/C/little-endian conversion."""

    if not isinstance(name, str) or not name:
        raise ValueError("fingerprint name must be a non-empty string")
    array = _numeric_array(value, name=name)
    metadata = {
        "name": name,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    _, header = _canonical_json(metadata, label=f"{name} fingerprint metadata")
    raw = array.tobytes(order="C")

    digest = hashlib.sha256()
    digest.update(b"psu-b0-canonical-numeric-v1\0")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return {**metadata, "sha256": digest.hexdigest()}


def _composite_anchor(payload: dict[str, Any]) -> dict[str, Any]:
    normalized, encoded = _canonical_json(payload, label="anchor payload")
    return {**normalized, "sha256": hashlib.sha256(encoded).hexdigest()}


def geometry_operator_anchor(
    geometry: Mapping[str, Any],
    provenance: Any,
    support: Any,
    grid_metadata: Any,
) -> dict[str, Any]:
    """Anchor the exact geometry arrays, support, provenance, and grid contract."""

    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be a mapping")
    actual = set(geometry)
    required = set(GEOMETRY_FIELDS)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required, key=str)
        raise ValueError(
            "geometry must contain exactly "
            f"{list(GEOMETRY_FIELDS)!r}; missing={missing!r}, extra={extra!r}"
        )

    normalized_provenance, _ = _canonical_json(provenance, label="provenance")
    normalized_grid, _ = _canonical_json(grid_metadata, label="grid_metadata")
    fingerprints = {
        name: canonical_numeric_fingerprint(name, geometry[name])
        for name in GEOMETRY_FIELDS
    }
    return _composite_anchor(
        {
            "schema": "psu-b0-geometry-operator-anchor-1.0",
            "geometry": fingerprints,
            "support": canonical_numeric_fingerprint("support", support),
            "provenance": normalized_provenance,
            "grid_metadata": normalized_grid,
        }
    )


def _buffer(owner: Any, name: str, *, label: str) -> Any:
    if isinstance(owner, Mapping):
        if name not in owner:
            raise ValueError(f"{label} is missing operational buffer {name!r}")
        return owner[name]
    if not hasattr(owner, name):
        raise ValueError(f"{label} is missing operational buffer {name!r}")
    return getattr(owner, name)


def _whitener_fingerprints(owner: Any, *, label: str) -> dict[str, Any]:
    return {
        name: canonical_numeric_fingerprint(
            f"{label}.{name}",
            _buffer(owner, name, label=label),
        )
        for name in WHITENER_BUFFERS
    }


def whitener_anchor(
    scale_by_view: Any,
    component_whitening: Any,
    graph_whitening: Any,
    gate_rows: Any,
    seed_metadata: Any,
) -> dict[str, Any]:
    """Anchor both operational whiteners and their calibration decisions."""

    normalized_gates, _ = _canonical_json(gate_rows, label="gate_rows")
    normalized_seeds, _ = _canonical_json(seed_metadata, label="seed_metadata")
    return _composite_anchor(
        {
            "schema": "psu-b0-whitener-anchor-1.0",
            "scale_by_view": canonical_numeric_fingerprint(
                "scale_by_view", scale_by_view
            ),
            "component_whitening": _whitener_fingerprints(
                component_whitening,
                label="component_whitening",
            ),
            "graph_whitening": _whitener_fingerprints(
                graph_whitening,
                label="graph_whitening",
            ),
            "gate_rows": normalized_gates,
            "seed_metadata": normalized_seeds,
        }
    )


def _anchor_differences(expected: Any, observed: Any, path: str) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        differences = []
        expected_keys = set(expected)
        observed_keys = set(observed)
        for key in sorted(expected_keys - observed_keys, key=str):
            differences.append(f"{path}.{key}: missing from observed")
        for key in sorted(observed_keys - expected_keys, key=str):
            differences.append(f"{path}.{key}: unexpected in observed")
        for key in sorted(expected_keys & observed_keys, key=str):
            differences.extend(
                _anchor_differences(expected[key], observed[key], f"{path}.{key}")
            )
        return differences

    sequence_types = (list, tuple)
    if isinstance(expected, sequence_types) and isinstance(observed, sequence_types):
        differences = []
        if len(expected) != len(observed):
            differences.append(
                f"{path}: length expected {len(expected)}, observed {len(observed)}"
            )
        for index, (left, right) in enumerate(zip(expected, observed)):
            differences.extend(_anchor_differences(left, right, f"{path}[{index}]"))
        return differences

    if type(expected) is not type(observed) or expected != observed:
        return [f"{path}: expected {expected!r}, observed {observed!r}"]
    return []


def verify_exact_anchor(expected: Any, observed: Any, label: str) -> None:
    """Require exact anchor equality and report field-level differences."""

    if not isinstance(label, str) or not label:
        raise ValueError("anchor label must be a non-empty string")
    differences = _anchor_differences(expected, observed, "anchor")
    if differences:
        preview = differences[:12]
        remainder = len(differences) - len(preview)
        details = "\n".join(f"  - {difference}" for difference in preview)
        if remainder:
            details += f"\n  - ... and {remainder} more difference(s)"
        raise ValueError(f"{label} exact anchor mismatch:\n{details}")


__all__ = [
    "canonical_numeric_fingerprint",
    "geometry_operator_anchor",
    "verify_exact_anchor",
    "whitener_anchor",
]
