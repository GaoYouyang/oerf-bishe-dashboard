from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from demo_t16_operator.run_v5s_dco_low_rank_screening import (
    _file_sha256,
    _feature_matrix,
    fit_ridge,
    hosvd_bases,
    nominal_geometry_features,
    project_cores,
    reconstruct_operators,
)


class TruthFeatureGuard(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key.startswith("truth_"):
            raise AssertionError("geometry feature extraction touched truth metadata")
        return super().__getitem__(key)


def _rig(offset: float = 0.0) -> TruthFeatureGuard:
    return TruthFeatureGuard(
        {
            "nominal_angles_degrees": [5 + offset, 33, 61, 89, 117, 145, 173],
            "nominal_aperture_radius": 0.06 + 0.001 * offset,
            "nominal_cone_u": 0.07,
            "nominal_cone_z": 0.05,
            "nominal_bend": 0.03,
            "truth_aperture_radius": 9.9,
        }
    )


def test_nominal_geometry_features_have_truth_firewall() -> None:
    features = nominal_geometry_features(_rig())
    assert features.shape == (20,)
    design, center, scale = _feature_matrix([_rig(0.0), _rig(1.0), _rig(2.0)])
    assert design.shape == (3, 41)
    assert center.shape == scale.shape == (20,)
    assert np.all(np.isfinite(design))


def test_hosvd_projection_and_ridge_shapes() -> None:
    rng = np.random.default_rng(4)
    discrepancy = rng.normal(size=(5, 7, 6))
    u, v, _, _ = hosvd_bases(discrepancy)
    cores = project_cores(discrepancy, u[:, :3], v[:, :2])
    reconstruction = reconstruct_operators(cores, u[:, :3], v[:, :2])
    assert cores.shape == (5, 3, 2)
    assert reconstruction.shape == discrepancy.shape
    design = np.column_stack([np.ones(5), np.linspace(-1, 1, 5)])
    coefficients = fit_ridge(design, cores.reshape(5, -1), 1e-3)
    assert coefficients.shape == (2, 6)


def test_file_hash_is_content_addressed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("operator-audit\n", encoding="utf-8")
    assert _file_sha256(artifact) == hashlib.sha256(artifact.read_bytes()).hexdigest()
