"""Teaching skeleton for a private N5-D5 laboratory renderer adapter.

Copy this file into ``private_library/`` and replace every method below with
calls into the laboratory renderer.  This public file intentionally contains
no JSONL transport, geometry, data, credentials, or working fallback.  Codex
will connect the implemented backend to the frozen transport only after the
private source has been reviewed locally.

Every method must use the same field parameterization, ray order, units,
precision, sampling rules, and path semantics.  Never return a precomputed
answer for the public probes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


NOT_IMPLEMENTED = "N5_D5_PRIVATE_ADAPTER_NOT_IMPLEMENTED"


def describe_renderer() -> dict[str, Any]:
    """Return dimensions, units, path identities, states, and cost semantics."""
    raise NotImplementedError(NOT_IMPLEMENTED)


def forward_renderer(
    path_role: str,
    field_vector: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, int]]:
    """Return output, actual branch state, diagnostic state, and call cost."""
    raise NotImplementedError(NOT_IMPLEMENTED)


def jvp_renderer(
    path_role: str,
    field_vector: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, int]]:
    """Return Jv, linearization branch state, and call cost."""
    raise NotImplementedError(NOT_IMPLEMENTED)


def vjp_renderer(
    path_role: str,
    field_vector: np.ndarray,
    cotangent: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, int]]:
    """Return J^T q, linearization branch state, and call cost."""
    raise NotImplementedError(NOT_IMPLEMENTED)


def canonical_field_vector() -> np.ndarray:
    """Return the exact flattened field/decoder vector described by config.json."""
    raise NotImplementedError(NOT_IMPLEMENTED)


def source_review_notes() -> dict[str, str]:
    """Record who reviewed branch provenance, path semantics, units, and cost."""
    raise NotImplementedError(NOT_IMPLEMENTED)


if __name__ == "__main__":
    raise SystemExit(
        "This is a non-executable teaching skeleton. Copy it into "
        "private_library/ and implement the six backend functions first."
    )
