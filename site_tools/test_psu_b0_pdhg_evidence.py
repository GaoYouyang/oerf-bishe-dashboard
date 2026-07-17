"""Tests for stable PSU-B0 PDHG evidence fingerprints."""

from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from site_tools.psu_b0_pdhg_evidence import (
    canonical_numeric_fingerprint,
    geometry_operator_anchor,
    verify_exact_anchor,
    whitener_anchor,
)


def _geometry() -> dict[str, np.ndarray]:
    return {
        "sample_points": np.arange(24, dtype=np.float64).reshape(2, 4, 3),
        "projection_u": np.arange(6, dtype=np.float32).reshape(2, 3),
        "projection_v": np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        "detector_xy": np.arange(4, dtype=np.float64).reshape(2, 2),
        "line_length": np.asarray([0.4, 0.5], dtype=np.float64),
        "system_constant": np.asarray([1.2, 1.3], dtype=np.float64),
    }


def _geometry_anchor(**overrides: object) -> dict:
    arguments = {
        "geometry": _geometry(),
        "provenance": [{"view": 0, "source": "fixture"}],
        "support": np.ones((2, 2, 2), dtype=np.float32),
        "grid_metadata": {"shape": [2, 2, 2], "minimum": [-0.1] * 3},
    }
    arguments.update(overrides)
    return geometry_operator_anchor(**arguments)


def _whitener(offset: float) -> SimpleNamespace:
    return SimpleNamespace(
        matrix=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) + offset,
        calibration_mean=torch.arange(4, dtype=torch.float32).reshape(1, 2, 2),
        scale_by_view=torch.tensor([[0.1 + offset]], dtype=torch.float32),
        predictive_scale_by_view=torch.tensor([1.05], dtype=torch.float32),
    )


def _whitener_anchor(**overrides: object) -> dict:
    arguments = {
        "scale_by_view": torch.tensor([[0.1]], dtype=torch.float32),
        "component_whitening": _whitener(0.0),
        "graph_whitening": _whitener(1.0),
        "gate_rows": [{"view": 0, "graph_activated": True}],
        "seed_metadata": {"calibration_seed": 17, "noise_seed": 23},
    }
    arguments.update(overrides)
    return whitener_anchor(**arguments)


def test_numeric_fingerprint_is_copy_layout_backend_and_process_stable() -> None:
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    non_contiguous = base[:, ::2]
    copied = np.ascontiguousarray(non_contiguous)
    expected = canonical_numeric_fingerprint("values", non_contiguous)
    assert expected == canonical_numeric_fingerprint("values", copied)
    assert expected == canonical_numeric_fingerprint(
        "values", torch.from_numpy(copied).clone()
    )

    code = (
        "import json, numpy as np; "
        "from site_tools.psu_b0_pdhg_evidence import canonical_numeric_fingerprint; "
        "x=np.arange(24,dtype=np.float64).reshape(4,6)[:,::2]; "
        "print(json.dumps(canonical_numeric_fingerprint('values',x),sort_keys=True))"
    )
    child = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    assert child == expected


def test_numeric_fingerprint_includes_name_dtype_shape_and_values() -> None:
    base = np.asarray([1.0, 2.0], dtype=np.float32)
    expected = canonical_numeric_fingerprint("values", base)
    mutated = base.copy()
    mutated[1] += 1.0
    assert canonical_numeric_fingerprint("other", base) != expected
    assert canonical_numeric_fingerprint("values", base.astype(np.float64)) != expected
    assert canonical_numeric_fingerprint("values", base.reshape(1, 2)) != expected
    assert canonical_numeric_fingerprint("values", mutated) != expected


def test_numeric_fingerprint_normalizes_explicit_little_endian() -> None:
    native = np.asarray([1.5, -2.25], dtype=np.float64)
    big_endian = native.astype(">f8")
    assert canonical_numeric_fingerprint("values", native) == (
        canonical_numeric_fingerprint("values", big_endian)
    )


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1.0, np.nan]),
        torch.tensor([1.0, float("inf")]),
    ],
)
def test_numeric_fingerprint_rejects_nonfinite(value: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_numeric_fingerprint("values", value)


def test_geometry_anchor_changes_for_element_shape_and_provenance_mutations() -> None:
    expected = _geometry_anchor()

    changed_element = _geometry()
    changed_element["projection_u"][0, 0] += 1.0
    changed_shape = _geometry()
    changed_shape["line_length"] = changed_shape["line_length"].reshape(2, 1)
    assert _geometry_anchor(geometry=changed_element)["sha256"] != expected["sha256"]
    assert _geometry_anchor(geometry=changed_shape)["sha256"] != expected["sha256"]
    assert _geometry_anchor(
        provenance=[{"view": 1, "source": "fixture"}]
    )["sha256"] != expected["sha256"]


def test_geometry_anchor_requires_exact_keys_and_serializable_provenance() -> None:
    missing = _geometry()
    missing.pop("detector_xy")
    with pytest.raises(ValueError, match="missing=.*detector_xy"):
        _geometry_anchor(geometry=missing)
    with pytest.raises(ValueError, match="provenance must be JSON serializable"):
        _geometry_anchor(provenance={"bad": object()})


def test_whitener_anchor_changes_for_operational_gate_and_seed_mutations() -> None:
    expected = _whitener_anchor()

    graph = _whitener(1.0)
    graph.matrix[0, 0, 0] += 1.0
    assert _whitener_anchor(graph_whitening=graph)["sha256"] != expected["sha256"]
    assert _whitener_anchor(
        scale_by_view=torch.tensor([[0.2]], dtype=torch.float32)
    )["sha256"] != expected["sha256"]
    assert _whitener_anchor(
        gate_rows=[{"view": 0, "graph_activated": False}]
    )["sha256"] != expected["sha256"]
    assert _whitener_anchor(
        seed_metadata={"calibration_seed": 18, "noise_seed": 23}
    )["sha256"] != expected["sha256"]


def test_whitener_anchor_requires_every_operational_buffer() -> None:
    component = vars(_whitener(0.0)).copy()
    component.pop("calibration_mean")
    with pytest.raises(ValueError, match="calibration_mean"):
        _whitener_anchor(component_whitening=component)


def test_verify_exact_anchor_passes_exact_and_fails_closed_with_diff() -> None:
    expected = _geometry_anchor()
    verify_exact_anchor(expected, deepcopy(expected), "geometry")

    observed = deepcopy(expected)
    observed["provenance"][0]["view"] = 9
    observed["sha256"] = "0" * 64
    with pytest.raises(ValueError) as error:
        verify_exact_anchor(expected, observed, "geometry")
    message = str(error.value)
    assert "geometry exact anchor mismatch" in message
    assert "anchor.provenance[0].view" in message
    assert "anchor.sha256" in message
