from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from site_tools.build_psu_rotation40_geometry_binding import (
    PUBLIC_STATUS,
    build_rotation40_geometry_binding,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rx(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=float)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rows = 6
    support = tmp_path / "support"
    observations = tmp_path / "observations"
    base_vectors: dict[int, np.ndarray] = {}
    for camera_id, zero_view in ((2, 0), (3, 1), (4, 2)):
        values = np.column_stack(
            (
                np.linspace(0.7, 0.9, rows),
                np.linspace(0.1 * camera_id, 0.2 * camera_id, rows),
                np.linspace(-0.2, 0.2, rows),
            )
        )
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        base_vectors[camera_id] = values
        for angle, view_id in ((0, zero_view), (50, zero_view + 3), (90, zero_view + 6)):
            directory = support / f"view_{view_id:02d}" / "bundle"
            directory.mkdir(parents=True)
            rotation = _rx(angle)
            vector_payload = {
                "c": np.tile([1.0, 0.2 * camera_id, -0.1], (rows, 1)) @ rotation,
                "v": values @ rotation,
                "Ruvecs": np.tile([0.0, 1.0, 0.0], (rows, 1)) @ rotation,
                "Rvvecs": np.tile([0.0, 0.0, 1.0], (rows, 1)) @ rotation,
                "Rxvecs": np.tile([0.0, 1.0, 0.0], (rows, 1)) @ rotation,
                "Ryvecs": np.tile([0.0, 0.0, 1.0], (rows, 1)) @ rotation,
            }
            scalar_payload = {
                "Rapvec": np.full((rows, 1), 0.01 * camera_id),
                "Dfvec": np.full((rows, 1), 1.0 + camera_id),
                "Csys_all": np.full((rows, 1), 0.5 + camera_id),
            }
            for name, array in {**vector_payload, **scalar_payload}.items():
                np.save(directory / f"{name}.npy", array.astype(np.float32))
            (directory / "view_bundle_manifest.json").write_text(
                json.dumps({"status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"}),
                encoding="utf-8",
            )
        camera_dir = observations / f"camera_{camera_id:02d}"
        camera_dir.mkdir(parents=True)
        np.save(camera_dir / "measured_uv_px.npy", np.arange(rows * 2, dtype=np.float32).reshape(rows, 2))
        np.save(camera_dir / "active_mask.npy", np.array([1, 0, 1, 1, 0, 0], dtype=bool))
        (camera_dir / "shard_manifest.json").write_text(
            json.dumps(
                {
                    "status": "ROTATION40_CAMERA_DISPLACEMENT_AND_MASK_SHARD_VERIFIED",
                    "row_order": "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON",
                }
            ),
            encoding="utf-8",
        )

    ray_cells = np.empty((7, 1), dtype=object)
    for camera_id in range(1, 8):
        base = base_vectors.get(camera_id, base_vectors[2])
        ray_cells[camera_id - 1, 0] = (base @ _rx(40)).T
    mat = tmp_path / "geometry.mat"
    homogeneous = np.eye(4)
    homogeneous[:3, :3] = _rx(40)
    savemat(mat, {"modelAngle": 40, "Arotcam": homogeneous, "ray_dir": ray_cells})
    config = {
        "schema_version": "psu-rotation40-geometry-binding-config-1.0",
        "status": "FROZEN_BEFORE_ROTATION40_GEOMETRY_BINDING",
        "dataset": {
            "doi": "10.26208/1VE2-5C19",
            "rotation_degrees": 40,
            "camera_ids": [2, 3, 4],
            "row_order": "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON",
        },
        "geometry_source": {
            "bytes": mat.stat().st_size,
            "sha256": _sha(mat),
            "required_variables": ["modelAngle", "Arotcam", "ray_dir"],
        },
        "camera_mapping": {
            "2": {"support_zero_view_id": 0, "support_50_view_id": 3, "support_90_view_id": 6},
            "3": {"support_zero_view_id": 1, "support_50_view_id": 4, "support_90_view_id": 7},
            "4": {"support_zero_view_id": 2, "support_50_view_id": 5, "support_90_view_id": 8},
        },
        "tolerances": {
            "rotation_matrix_orthogonality_max_abs": 1e-12,
            "known_rotation_row_max_abs": 1e-6,
            "official_rotation40_ray_max_abs": 1e-6,
            "scalar_invariance_max_abs": 0.0,
        },
        "claim_firewall": {
            "development_only": True,
            "final_rotations_opened": False,
            "experimental_field_truth_available": False,
            "reprojection_scored_by_this_stage": False,
            "algorithm_superiority": False,
            "publish_raw_geometry_or_measurements": False,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, mat, support, observations


def _run(tmp_path: Path, config: Path, mat: Path, support: Path, observations: Path):
    return build_rotation40_geometry_binding(
        config_path=config,
        geometry_mat_path=mat,
        support_view_root=support,
        observation_root=observations,
        output_root=tmp_path / "output",
        public_summary_path=tmp_path / "public.json",
        chunk_rows=2,
    )


def test_binds_only_active_rows_and_keeps_claims_closed(tmp_path: Path) -> None:
    config, mat, support, observations = _fixture(tmp_path)
    private, public = _run(tmp_path, config, mat, support, observations)
    assert public["status"] == PUBLIC_STATUS
    assert public["claim_boundary"]["camera_geometry_available"] is True
    assert public["claim_boundary"]["reprojection_scored"] is False
    assert public["claim_boundary"]["algorithm_superiority"] is False
    assert len(private["camera_rows"]) == 3
    indices = np.load(tmp_path / "output/camera_02/active_indices.npy")
    np.testing.assert_array_equal(indices, [0, 2, 3])
    rays = np.load(tmp_path / "output/camera_02/v.npy")
    assert rays.shape == (3, 3)
    assert private["camera_rows"][0]["official_ray_binding_max_abs"] < 1e-6


def test_rejects_observation_row_order_drift(tmp_path: Path) -> None:
    config, mat, support, observations = _fixture(tmp_path)
    manifest = observations / "camera_02/shard_manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["row_order"] = "C_ORDER"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="row order"):
        _run(tmp_path, config, mat, support, observations)


def test_rejects_official_ray_mismatch(tmp_path: Path) -> None:
    config, mat, support, observations = _fixture(tmp_path)
    payload = loadmat_for_test(mat)
    payload["ray_dir"][1, 0][0, 0] += 0.1
    savemat(mat, payload)
    contract = json.loads(config.read_text(encoding="utf-8"))
    contract["geometry_source"]["bytes"] = mat.stat().st_size
    contract["geometry_source"]["sha256"] = _sha(mat)
    config.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="ray rows do not bind"):
        _run(tmp_path, config, mat, support, observations)


def loadmat_for_test(path: Path) -> dict[str, np.ndarray]:
    from scipy.io import loadmat

    payload = loadmat(path, variable_names=["modelAngle", "Arotcam", "ray_dir"])
    return {name: payload[name] for name in ("modelAngle", "Arotcam", "ray_dir")}
