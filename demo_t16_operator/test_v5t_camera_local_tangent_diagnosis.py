from __future__ import annotations

from typing import Any

import numpy as np

from demo_t16_operator.run_v5t_camera_local_tangent_diagnosis import (
    _renderer_parameters,
    nominal_parameter_vector,
    parameter_names,
    truth_parameter_vector,
)


class NominalGuard(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key.startswith("truth_"):
            raise AssertionError("nominal parameter extraction touched truth metadata")
        return super().__getitem__(key)


def _row() -> dict[str, Any]:
    return {
        "nominal_angles_degrees": [5, 33, 61, 89, 117, 145, 173],
        "nominal_aperture_radius": 0.06,
        "nominal_cone_u": 0.07,
        "nominal_cone_z": 0.05,
        "nominal_bend": 0.03,
        "truth_angles_degrees": [6, 34, 62, 90, 118, 146, 174],
        "truth_aperture_radius": 0.09,
        "truth_cone_u": 0.08,
        "truth_cone_z": 0.06,
        "truth_bend": 0.04,
    }


def test_nominal_parameter_vector_has_truth_firewall() -> None:
    row = NominalGuard(_row())
    vector = nominal_parameter_vector(row)
    assert vector.shape == (11,)
    np.testing.assert_allclose(vector[:7], [5, 33, 61, 89, 117, 145, 173])


def test_parameter_vector_round_trip_and_truth_delta() -> None:
    row = _row()
    nominal = nominal_parameter_vector(row)
    truth = truth_parameter_vector(row)
    assert parameter_names(7)[-4:] == [
        "aperture_radius",
        "cone_u",
        "cone_z",
        "bend",
    ]
    parameters = _renderer_parameters(nominal, 7)
    np.testing.assert_allclose(parameters["angles"], nominal[:7])
    assert parameters["aperture_radius"] == nominal[7]
    np.testing.assert_allclose(truth - nominal, [1] * 7 + [0.03, 0.01, 0.01, 0.01])
