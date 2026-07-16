"""Deterministic streamed PSU support-ray store for the B0 inverse baseline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from demo_t16_operator.psu_b0_streaming_operator import StreamingRayChunk
from site_tools.psu_bost_aperture_domain import (
    deterministic_paired_uniform_aperture_samples,
    generate_aperture_sample_points,
)
from site_tools.psu_bost_forward_geometry import intersect_forward_ray_box


STORE_SCHEMA = "psu-b0-real-support-streaming-store-1.0"
VIEW_BUNDLE_STATUS = "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED"
MASK_STATUS = "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS"


def deterministic_quantile_indices(indices: np.ndarray, count: int) -> np.ndarray:
    """Choose ordered active rows without inspecting displacement magnitude."""

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("indices must be a nonempty one-dimensional array")
    if values.size > 1 and np.any(values[1:] <= values[:-1]):
        raise ValueError("indices must be strictly increasing")
    requested = int(count)
    if requested < 1 or requested > values.size:
        raise ValueError("count must lie between one and the number of indices")
    positions = np.floor(
        (np.arange(requested, dtype=np.float64) + 0.5)
        * values.size
        / requested
    ).astype(np.int64)
    selected = values[positions]
    if np.unique(selected).size != requested:
        raise RuntimeError("quantile selection produced duplicate rows")
    return np.ascontiguousarray(selected, dtype=np.int64)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


@dataclass(frozen=True)
class _ViewSource:
    view_id: int
    active_indices: np.ndarray
    selected_indices: np.ndarray
    arrays: dict[str, np.ndarray]

    @property
    def ray_count(self) -> int:
        return int(self.selected_indices.size)


class PSURealSupportRayStore:
    """Read nine verified PSU support views in bounded deterministic chunks."""

    def __init__(
        self,
        view_root: Path,
        *,
        rays_per_view: int | None,
        sample_count: int = 16,
        chunk_rays: int = 32768,
        grid_minimum_xyz: tuple[float, float, float] = (-0.11, -0.11, -0.11),
        grid_maximum_xyz: tuple[float, float, float] = (0.11, 0.11, 0.11),
    ) -> None:
        self.view_root = Path(view_root)
        self.sample_count = int(sample_count)
        self.chunk_rays = int(chunk_rays)
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least two")
        if self.chunk_rays < 1:
            raise ValueError("chunk_rays must be positive")
        self.grid_minimum_xyz = tuple(float(value) for value in grid_minimum_xyz)
        self.grid_maximum_xyz = tuple(float(value) for value in grid_maximum_xyz)
        if len(self.grid_minimum_xyz) != 3 or len(self.grid_maximum_xyz) != 3:
            raise ValueError("grid bounds must contain three coordinates")
        if any(
            upper <= lower
            for lower, upper in zip(
                self.grid_minimum_xyz,
                self.grid_maximum_xyz,
                strict=True,
            )
        ):
            raise ValueError("grid maximum must exceed grid minimum")
        self.aperture_design = deterministic_paired_uniform_aperture_samples(
            self.sample_count
        )

        view_dirs = sorted(
            path for path in self.view_root.glob("view_*") if path.is_dir()
        )
        if len(view_dirs) != 9:
            raise ValueError(
                f"expected exactly nine support view directories, got {len(view_dirs)}"
            )
        self.views = tuple(
            self._load_view(path, rays_per_view=rays_per_view)
            for path in view_dirs
        )
        view_ids = [view.view_id for view in self.views]
        if view_ids != list(range(9)):
            raise ValueError(f"expected ordered support view ids 0..8, got {view_ids}")
        self.ray_count = int(sum(view.ray_count for view in self.views))
        self.rays_per_view = (
            None if rays_per_view is None else int(rays_per_view)
        )

    def _load_view(
        self,
        view_dir: Path,
        *,
        rays_per_view: int | None,
    ) -> _ViewSource:
        bundle_dir = view_dir / "bundle"
        mask_dir = view_dir / "corrected_masks"
        bundle_manifest = _load_json(bundle_dir / "view_bundle_manifest.json")
        mask_manifest = _load_json(
            mask_dir / "corrected_view_masks_manifest.json"
        )
        if bundle_manifest.get("status") != VIEW_BUNDLE_STATUS:
            raise ValueError(f"unverified view bundle: {view_dir.name}")
        if mask_manifest.get("status") != MASK_STATUS:
            raise ValueError(f"unverified corrected masks: {view_dir.name}")
        view_id = int(bundle_manifest["view"]["view_id_zero_based"])
        active = _load_array(mask_dir / "amask_all_zero_based.npy")
        if active.ndim != 1 or active.size == 0:
            raise ValueError(f"invalid active mask: {view_dir.name}")
        if active.size > 1 and np.any(active[1:] <= active[:-1]):
            raise ValueError(f"active mask is not strictly increasing: {view_dir.name}")
        selected = (
            active
            if rays_per_view is None
            else deterministic_quantile_indices(active, int(rays_per_view))
        )
        names = (
            "c",
            "v",
            "Ruvecs",
            "Rvvecs",
            "Rxvecs",
            "Ryvecs",
            "Rapvec",
            "Dfvec",
            "Csys_all",
            "epsu_all",
            "epsv_all",
        )
        arrays = {
            name: _load_array(bundle_dir / f"{name}.npy")
            for name in names
        }
        row_count = int(bundle_manifest["view"]["measurement_count"])
        for name, array in arrays.items():
            if array.shape[0] != row_count:
                raise ValueError(
                    f"{view_dir.name}/{name} row count does not match manifest"
                )
        return _ViewSource(
            view_id=view_id,
            active_indices=active,
            selected_indices=selected,
            arrays=arrays,
        )

    @property
    def selection_mode(self) -> str:
        return (
            "all_active_rows"
            if self.rays_per_view is None
            else "ordered_active_mask_quantiles_without_measurement_magnitude"
        )

    def selection_summary(self) -> dict[str, Any]:
        return {
            "schema_version": STORE_SCHEMA,
            "selection_mode": self.selection_mode,
            "sample_count_per_ray": int(self.sample_count),
            "chunk_rays": int(self.chunk_rays),
            "total_ray_count": int(self.ray_count),
            "view_rows": [
                {
                    "view_id_zero_based": int(view.view_id),
                    "active_ray_count": int(view.active_indices.size),
                    "selected_ray_count": int(view.selected_indices.size),
                }
                for view in self.views
            ],
            "contains_measurement_values": False,
            "contains_local_paths": False,
        }

    def load_observations(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        values = np.empty((self.ray_count, 2), dtype=numpy_dtype)
        start = 0
        for view in self.views:
            stop = start + view.ray_count
            indices = view.selected_indices
            values[start:stop, 0] = view.arrays["epsu_all"][indices, 0]
            values[start:stop, 1] = view.arrays["epsv_all"][indices, 0]
            start = stop
        tensor = torch.from_numpy(values)[None]
        return tensor.to(device=device, dtype=dtype)

    def _chunk(
        self,
        view: _ViewSource,
        local_start: int,
        local_stop: int,
        global_start: int,
    ) -> StreamingRayChunk:
        indices = np.asarray(
            view.selected_indices[local_start:local_stop],
            dtype=np.int64,
        )
        arrays = view.arrays
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
        focal_distance = np.asarray(
            arrays["Dfvec"][indices, 0],
            dtype=np.float64,
        )
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
        ray_count = int(indices.size)
        observation = np.column_stack(
            (
                np.asarray(arrays["epsu_all"][indices, 0]),
                np.asarray(arrays["epsv_all"][indices, 0]),
            )
        )
        return StreamingRayChunk(
            start_index=int(global_start),
            stop_index=int(global_start + ray_count),
            sample_points_xyz=sample_points,
            projection_u_xyz=np.asarray(
                arrays["Ruvecs"][indices],
                dtype=np.float64,
            ),
            projection_v_xyz=np.asarray(
                arrays["Rvvecs"][indices],
                dtype=np.float64,
            ),
            line_length=np.asarray(box["length"], dtype=np.float64),
            system_constant=np.asarray(
                arrays["Csys_all"][indices, 0],
                dtype=np.float64,
            ),
            observation_uv=observation,
            view_id=int(view.view_id),
            b0_hit_count=int(np.count_nonzero(box["hit"])),
        )

    def iter_chunks(self) -> Iterator[StreamingRayChunk]:
        global_start = 0
        for view in self.views:
            for local_start in range(0, view.ray_count, self.chunk_rays):
                local_stop = min(local_start + self.chunk_rays, view.ray_count)
                chunk = self._chunk(
                    view,
                    local_start,
                    local_stop,
                    global_start,
                )
                yield chunk
                global_start = chunk.stop_index
        if global_start != self.ray_count:
            raise RuntimeError("streaming store emitted the wrong number of rays")
