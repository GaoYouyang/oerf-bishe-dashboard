"""Shared JSONL wire helpers for the N5-D5 callable-interface bridge.

The independent validator intentionally does not import this module.  It
reimplements canonicalization, vector generation, and metric checks so that a
bug in these helpers cannot certify its own output.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUEST_SCHEMA = "n5-d5-adapter-request-1.0"
RESPONSE_SCHEMA = "n5-d5-adapter-response-1.0"
DESCRIPTOR_SCHEMA = "n5-d5-adapter-descriptor-1.0"
PATH_ROLES = ("curved", "straight", "direct_residual")
OPERATIONS = ("describe", "forward", "jvp", "vjp")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray | Iterable[float]) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    return sha256_bytes(array.tobytes(order="C"))


def finite_vector(
    name: str,
    value: object,
    *,
    expected_size: int,
) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    if array.size != expected_size:
        raise ValueError(
            f"{name} has size {array.size}, expected {expected_size}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} contains non-finite values")
    return array


def normalize(value: np.ndarray) -> np.ndarray:
    vector = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize an empty or non-finite direction")
    return vector / norm


def generated_vectors(
    *,
    input_dimension: int,
    output_dimension: int,
    base_seed: int,
    tangent_seed: int,
    cotangent_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_rng = np.random.default_rng(int(base_seed))
    tangent_rng = np.random.default_rng(int(tangent_seed))
    cotangent_rng = np.random.default_rng(int(cotangent_seed))
    base = base_rng.normal(scale=0.2, size=input_dimension).astype(np.float64)
    tangents = np.stack(
        [normalize(tangent_rng.normal(size=input_dimension)) for _ in range(2)],
        axis=0,
    )
    cotangents = np.stack(
        [normalize(cotangent_rng.normal(size=output_dimension))],
        axis=0,
    )
    return base, tangents, cotangents


def state_digest(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def response_line(value: dict[str, Any]) -> str:
    return canonical_json(value) + "\n"
