"""Deterministic active-ray store for the PSU rotation-40 development run."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from demo_t16_operator.psu_b0_streaming_operator import StreamingRayChunk
from site_tools.psu_b0_real_support_store import deterministic_quantile_indices
from site_tools.psu_bost_aperture_domain import (
    deterministic_paired_uniform_aperture_samples,
    generate_aperture_sample_points,
)
from site_tools.psu_bost_forward_geometry import intersect_forward_ray_box


CAMERA_STATUS = "ROTATION40_ACTIVE_ROW_GEOMETRY_AND_OBSERVATIONS_BOUND_PRIVATE"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


@dataclass(frozen=True)
class _CameraSource:
    camera_id: int
    selected_indices: np.ndarray
    arrays: dict[str, np.ndarray]

    @property
    def ray_count(self) -> int:
        return int(self.selected_indices.size)


class PSURotation40ActiveRayStore:
    """Stream verified rotation-40 cameras 2/3/4 without loading all rays."""

    def __init__(
        self,
        root: Path,
        *,
        rays_per_camera: int | None = None,
        sample_count: int = 16,
        chunk_rays: int = 32768,
        grid_minimum_xyz: tuple[float, float, float] = (-0.11, -0.11, -0.11),
        grid_maximum_xyz: tuple[float, float, float] = (0.11, 0.11, 0.11),
    ) -> None:
        self.root = Path(root)
        self.sample_count = int(sample_count)
        self.chunk_rays = int(chunk_rays)
        if self.sample_count < 2 or self.chunk_rays < 1:
            raise ValueError("sample_count must be >=2 and chunk_rays must be positive")
        self.grid_minimum_xyz = tuple(float(value) for value in grid_minimum_xyz)
        self.grid_maximum_xyz = tuple(float(value) for value in grid_maximum_xyz)
        if any(
            upper <= lower
            for lower, upper in zip(
                self.grid_minimum_xyz, self.grid_maximum_xyz, strict=True
            )
        ):
            raise ValueError("grid maximum must exceed grid minimum")
        self.aperture_design = deterministic_paired_uniform_aperture_samples(
            self.sample_count
        )
        self.rays_per_camera = None if rays_per_camera is None else int(rays_per_camera)
        self.cameras = tuple(self._load_camera(camera_id) for camera_id in (2, 3, 4))
        self.ray_count = int(sum(camera.ray_count for camera in self.cameras))

    def _load_camera(self, camera_id: int) -> _CameraSource:
        directory = self.root / f"camera_{camera_id:02d}"
        manifest = _read_json(directory / "geometry_manifest.json")
        if manifest.get("status") != CAMERA_STATUS or int(manifest.get("camera_id", -1)) != camera_id:
            raise ValueError(f"camera {camera_id} geometry binding is unverified")
        if manifest.get("row_order") != "MATLAB_COLUMN_MAJOR_MATCHING_AUTHOR_EPSU_COLON":
            raise ValueError(f"camera {camera_id} row-order contract changed")
        names = (
            "active_indices",
            "measured_uv_px",
            "c",
            "v",
            "Ruvecs",
            "Rvvecs",
            "Rxvecs",
            "Ryvecs",
            "Rapvec",
            "Dfvec",
            "Csys_all",
        )
        arrays = {
            name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in names
        }
        active_count = int(manifest["active_row_count"])
        if any(array.shape[0] != active_count for array in arrays.values()):
            raise ValueError(f"camera {camera_id} active-array row counts differ")
        all_rows = np.arange(active_count, dtype=np.int64)
        if self.rays_per_camera is None:
            selected = all_rows
        else:
            selected = deterministic_quantile_indices(all_rows, self.rays_per_camera)
        return _CameraSource(camera_id=camera_id, selected_indices=selected, arrays=arrays)

    @property
    def selection_mode(self) -> str:
        return (
            "all_rotation40_active_rows"
            if self.rays_per_camera is None
            else "ordered_quantiles_within_rotation40_active_rows_without_measurement_magnitude"
        )

    def selection_summary(self) -> dict[str, Any]:
        return {
            "selection_mode": self.selection_mode,
            "sample_count_per_ray": self.sample_count,
            "chunk_rays": self.chunk_rays,
            "total_ray_count": self.ray_count,
            "camera_rows": [
                {
                    "camera_id": camera.camera_id,
                    "available_active_ray_count": int(camera.arrays["v"].shape[0]),
                    "selected_ray_count": camera.ray_count,
                }
                for camera in self.cameras
            ],
            "contains_measurement_values": False,
            "contains_local_paths": False,
        }

    def load_observations(
        self, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        values = np.empty((self.ray_count, 2), dtype=numpy_dtype)
        start = 0
        for camera in self.cameras:
            stop = start + camera.ray_count
            values[start:stop] = camera.arrays["measured_uv_px"][camera.selected_indices]
            start = stop
        return torch.from_numpy(values)[None].to(device=device, dtype=dtype)

    def _chunk(
        self,
        camera: _CameraSource,
        local_start: int,
        local_stop: int,
        global_start: int,
    ) -> StreamingRayChunk:
        indices = np.asarray(
            camera.selected_indices[local_start:local_stop], dtype=np.int64
        )
        arrays = camera.arrays
        origins = np.asarray(arrays["c"][indices], dtype=np.float64)
        directions = np.asarray(arrays["v"][indices], dtype=np.float64)
        box = intersect_forward_ray_box(
            origins,
            directions,
            self.grid_minimum_xyz,
            self.grid_maximum_xyz,
            layout="rows",
        )
        start = origins + box["enter"][:, None] * box["direction_unit"]
        stop = origins + box["exit"][:, None] * box["direction_unit"]
        original_enter = box["enter"] / box["direction_norm"]
        original_exit = box["exit"] / box["direction_norm"]
        aperture = np.asarray(arrays["Rapvec"][indices, 0], dtype=np.float64)
        focal_distance = np.asarray(arrays["Dfvec"][indices, 0], dtype=np.float64)
        inner_radius = aperture * (1.0 - original_enter / focal_distance)
        outer_radius = aperture * (1.0 - original_exit / focal_distance)
        sample_points = generate_aperture_sample_points(
            start,
            stop,
            np.asarray(arrays["Rxvecs"][indices], dtype=np.float64),
            np.asarray(arrays["Ryvecs"][indices], dtype=np.float64),
            inner_radius,
            outer_radius,
            self.aperture_design["longitudinal_fractions"],
            self.aperture_design["unit_disk_offsets"],
        )
        count = int(indices.size)
        return StreamingRayChunk(
            start_index=int(global_start),
            stop_index=int(global_start + count),
            sample_points_xyz=sample_points,
            projection_u_xyz=np.asarray(arrays["Ruvecs"][indices], dtype=np.float64),
            projection_v_xyz=np.asarray(arrays["Rvvecs"][indices], dtype=np.float64),
            line_length=np.asarray(box["length"], dtype=np.float64),
            system_constant=np.asarray(arrays["Csys_all"][indices, 0], dtype=np.float64),
            observation_uv=np.asarray(arrays["measured_uv_px"][indices], dtype=np.float64),
            view_id=int(camera.camera_id),
            b0_hit_count=int(np.count_nonzero(box["hit"])),
        )

    def iter_chunks(self) -> Iterator[StreamingRayChunk]:
        global_start = 0
        for camera in self.cameras:
            for local_start in range(0, camera.ray_count, self.chunk_rays):
                local_stop = min(local_start + self.chunk_rays, camera.ray_count)
                chunk = self._chunk(camera, local_start, local_stop, global_start)
                yield chunk
                global_start = chunk.stop_index
        if global_start != self.ray_count:
            raise RuntimeError("rotation-40 store emitted the wrong number of rays")
