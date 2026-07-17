from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from site_tools.psu_rotation40_active_store import PSURotation40ActiveRayStore


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "binding"
    rows = 6
    origins = np.tile([-1.0, 0.0, 0.0], (rows, 1)).astype(np.float32)
    directions = np.tile([1.0, 0.0, 0.0], (rows, 1)).astype(np.float32)
    for camera_id in (2, 3, 4):
        directory = root / f"camera_{camera_id:02d}"
        directory.mkdir(parents=True)
        arrays = {
            "active_indices": np.arange(rows, dtype=np.int64) + 10 * camera_id,
            "measured_uv_px": np.column_stack(
                (np.arange(rows), np.arange(rows) + camera_id)
            ).astype(np.float32),
            "c": origins,
            "v": directions,
            "Ruvecs": np.tile([0.0, 1.0, 0.0], (rows, 1)).astype(np.float32),
            "Rvvecs": np.tile([0.0, 0.0, 1.0], (rows, 1)).astype(np.float32),
            "Rxvecs": np.tile([0.0, 1.0, 0.0], (rows, 1)).astype(np.float32),
            "Ryvecs": np.tile([0.0, 0.0, 1.0], (rows, 1)).astype(np.float32),
            "Rapvec": np.full((rows, 1), 0.001, dtype=np.float32),
            "Dfvec": np.full((rows, 1), 2.0, dtype=np.float32),
            "Csys_all": np.full((rows, 1), 0.5, dtype=np.float32),
        }
        for name, values in arrays.items():
            np.save(directory / f"{name}.npy", values)
        (directory / "geometry_manifest.json").write_text(
            json.dumps(
                {
                    "status": "ROTATION40_ACTIVE_ROW_GEOMETRY_AND_OBSERVATIONS_BOUND_PRIVATE",
                    "camera_id": camera_id,
                    "row_order": "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON",
                    "active_row_count": rows,
                }
            ),
            encoding="utf-8",
        )
    return root


def test_store_streams_contiguous_camera_chunks_and_observations(tmp_path: Path) -> None:
    store = PSURotation40ActiveRayStore(
        _fixture(tmp_path), rays_per_camera=3, sample_count=4, chunk_rays=2
    )
    assert store.ray_count == 9
    observations = store.load_observations(dtype=torch.float64, device=torch.device("cpu"))
    assert observations.shape == (1, 9, 2)
    chunks = list(store.iter_chunks())
    assert chunks[0].start_index == 0
    assert chunks[-1].stop_index == 9
    assert [chunk.view_id for chunk in chunks] == [2, 2, 3, 3, 4, 4]
    assert all(chunk.sample_points_xyz.shape[1] == 4 for chunk in chunks)
    assert sum(chunk.b0_hit_count for chunk in chunks) == 9


def test_store_all_rows_mode_is_deterministic(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    first = PSURotation40ActiveRayStore(root, sample_count=4, chunk_rays=4)
    second = PSURotation40ActiveRayStore(root, sample_count=4, chunk_rays=4)
    assert first.ray_count == 18
    np.testing.assert_array_equal(
        first.load_observations(dtype=torch.float32, device=torch.device("cpu")).numpy(),
        second.load_observations(dtype=torch.float32, device=torch.device("cpu")).numpy(),
    )
