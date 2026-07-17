from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from site_tools.audit_psu_rotation40_cell_payload import audit_rotation40_payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, nonbinary: bool = False) -> tuple[Path, Path]:
    shape = (2, 3)
    cells: dict[str, np.ndarray] = {}
    for name in ("typevector_free", "typevector_new", "u_new", "v_new"):
        cell = np.empty((1, 7), dtype=object)
        for index in range(7):
            if name == "typevector_free":
                values = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
            elif name == "typevector_new":
                values = np.array([[1, 1, 1], [0, 1, 0]], dtype=np.float32)
                if nonbinary and index == 0:
                    values[0, 0] = 0.5
            elif name == "u_new":
                values = np.arange(6, dtype=np.float64).reshape(shape) + 10 * index
            else:
                values = -(np.arange(6, dtype=np.float64).reshape(shape) + 10 * index + 0.5)
            cell[0, index] = values
        cells[name] = cell
    mat = tmp_path / "HSOF_DEF_ROT_040.mat"
    savemat(mat, cells, do_compression=True)
    access = {
        "schema_version": "psu-rotation40-development-access-private-1.0",
        "status": "ROTATION40_DEVELOPMENT_MEMBER_EXTRACTED_AND_VERIFIED",
        "dataset": {
            "rotation_degrees": 40,
            "extracted_sha256": _sha(mat),
            "member_uncompressed_bytes": mat.stat().st_size,
        },
    }
    access_path = tmp_path / "access.json"
    access_path.write_text(json.dumps(access), encoding="utf-8")
    return mat, access_path


def test_audits_cells_applies_author_sign_and_writes_private_shards(tmp_path: Path) -> None:
    mat, access = _fixture(tmp_path)
    private, public = audit_rotation40_payload(
        mat_path=mat,
        access_report_path=access,
        private_shard_root=tmp_path / "shards",
        public_summary_path=tmp_path / "public.json",
        expected_shape=(2, 3),
        shard_camera_ids=(2, 3, 4),
    )
    assert private["camera_count"] == 7
    assert public["claim_boundary"]["reprojection_scored"] is False
    assert public["claim_boundary"]["final_rotations_opened"] is False
    measured = np.load(tmp_path / "shards/camera_02/measured_uv_px.npy")
    assert measured.shape == (6, 2)
    np.testing.assert_array_equal(measured[:, 0], [10, 13, 11, 14, 12, 15])
    np.testing.assert_array_equal(measured[:, 1], [10.5, 13.5, 11.5, 14.5, 12.5, 15.5])
    manifest = json.loads(
        (tmp_path / "shards/camera_02/shard_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["row_order"] == "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON"
    active = np.load(tmp_path / "shards/camera_02/active_mask.npy")
    ambient = np.load(tmp_path / "shards/camera_02/ambient_mask.npy")
    excluded = np.load(tmp_path / "shards/camera_02/excluded_mask.npy")
    assert np.all((active.astype(int) + ambient.astype(int) + excluded.astype(int)) == 1)
    np.testing.assert_array_equal(active, [True, False, True, True, False, False])


def test_rejects_checksum_tamper(tmp_path: Path) -> None:
    mat, access = _fixture(tmp_path)
    with mat.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        audit_rotation40_payload(
            mat_path=mat,
            access_report_path=access,
            private_shard_root=tmp_path / "shards",
            expected_shape=(2, 3),
        )


def test_rejects_nonbinary_mask(tmp_path: Path) -> None:
    mat, access = _fixture(tmp_path, nonbinary=True)
    with pytest.raises(ValueError, match="not binary"):
        audit_rotation40_payload(
            mat_path=mat,
            access_report_path=access,
            private_shard_root=tmp_path / "shards",
            expected_shape=(2, 3),
        )


def test_rejects_wrong_expected_shape(tmp_path: Path) -> None:
    mat, access = _fixture(tmp_path)
    with pytest.raises(ValueError, match="unsupported shape"):
        audit_rotation40_payload(
            mat_path=mat,
            access_report_path=access,
            private_shard_root=tmp_path / "shards",
            expected_shape=(3, 2),
        )
