#!/usr/bin/env python3
"""Build and read a private compact stencil cache for streamed PSU B0 rays."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np
import torch

from demo_t16_operator.psu_b0_reconstruction_interface import (
    build_compact_trilinear_coordinates,
)
from demo_t16_operator.psu_b0_streaming_operator import StreamingRayChunk
from site_tools.psu_b0_real_support_store import PSURealSupportRayStore


CACHE_SCHEMA = "psu-b0-compact-stencil-cache-1.0"
CACHE_STATUS = "COMPLETE_PRIVATE_COMPACT_STENCIL_CACHE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _base_dtype(grid_shape: tuple[int, int, int]) -> np.dtype[Any]:
    voxel_count = int(np.prod(grid_shape, dtype=np.int64))
    return np.dtype(np.uint16 if voxel_count <= np.iinfo(np.uint16).max + 1 else np.int32)


def _array_record(path: Path) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "filename": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "nbytes": int(values.nbytes),
        "sha256": _sha256(path),
    }


def _safe_array_path(root: Path, record: dict[str, Any]) -> Path:
    filename = str(record.get("filename", ""))
    if not filename or Path(filename).name != filename:
        raise ValueError("cache array filename must be a plain basename")
    return root / filename


def _open_array(
    root: Path,
    record: dict[str, Any],
    *,
    verify_hash: bool,
) -> np.ndarray:
    path = _safe_array_path(root, record)
    if not path.is_file():
        raise ValueError(f"cache array is missing: {path.name}")
    values = np.load(path, mmap_mode="c", allow_pickle=False)
    if list(values.shape) != list(record.get("shape", [])):
        raise ValueError(f"cache array shape mismatch: {path.name}")
    if str(values.dtype) != str(record.get("dtype")):
        raise ValueError(f"cache array dtype mismatch: {path.name}")
    if int(values.nbytes) != int(record.get("nbytes", -1)):
        raise ValueError(f"cache array byte count mismatch: {path.name}")
    if verify_hash and _sha256(path) != str(record.get("sha256")):
        raise ValueError(f"cache array checksum mismatch: {path.name}")
    return values


def _create_memmap(
    root: Path,
    filename: str,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int, ...],
) -> np.memmap:
    return np.lib.format.open_memmap(
        root / filename,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def build_compact_cache(
    source_store: PSURealSupportRayStore,
    output_root: Path,
    *,
    grid_shape: tuple[int, int, int],
    grid_minimum_xyz: tuple[float, float, float],
    grid_maximum_xyz: tuple[float, float, float],
    fraction_dtype: str = "float64",
) -> dict[str, Any]:
    """Build one immutable private cache without opening development data."""

    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("grid_shape must contain three dimensions of at least two")
    minimum = tuple(float(value) for value in grid_minimum_xyz)
    maximum = tuple(float(value) for value in grid_maximum_xyz)
    if len(minimum) != 3 or len(maximum) != 3:
        raise ValueError("grid bounds must contain x, y, and z")
    if any(upper <= lower for lower, upper in zip(minimum, maximum, strict=True)):
        raise ValueError("grid maximum must exceed grid minimum")
    fraction_types = {
        "float32": np.dtype(np.float32),
        "float64": np.dtype(np.float64),
    }
    try:
        stored_fraction_dtype = fraction_types[str(fraction_dtype)]
    except KeyError as exc:
        raise ValueError("fraction_dtype must be float32 or float64") from exc

    target = Path(output_root)
    working = target.with_name(f"{target.name}.building")
    if target.exists():
        raise FileExistsError(f"cache target already exists: {target}")
    if working.exists():
        raise FileExistsError(f"incomplete cache build already exists: {working}")
    working.mkdir(parents=True)

    ray_count = int(source_store.ray_count)
    sample_count = int(source_store.sample_count)
    base = _create_memmap(
        working,
        "base_indices.npy",
        dtype=_base_dtype(shape),
        shape=(ray_count, sample_count),
    )
    fractions = _create_memmap(
        working,
        "fractions_xyz.npy",
        dtype=stored_fraction_dtype,
        shape=(ray_count, sample_count, 3),
    )
    valid = _create_memmap(
        working,
        "valid.npy",
        dtype=np.bool_,
        shape=(ray_count, sample_count),
    )
    projection = _create_memmap(
        working,
        "projection_uv_xyz.npy",
        dtype=np.float32,
        shape=(ray_count, 2, 3),
    )
    ray_scale = _create_memmap(
        working,
        "ray_scale.npy",
        dtype=np.float64,
        shape=(ray_count,),
    )
    observations = _create_memmap(
        working,
        "observations_uv.npy",
        dtype=np.float32,
        shape=(ray_count, 2),
    )

    chunk_records: list[dict[str, int]] = []
    expected_start = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(source_store.iter_chunks()):
        if chunk.start_index != expected_start:
            raise ValueError("source chunks must cover contiguous output slices")
        start = int(chunk.start_index)
        stop = int(chunk.stop_index)
        compact = build_compact_trilinear_coordinates(
            chunk.sample_points_xyz,
            grid_shape=shape,
            grid_minimum_xyz=minimum,
            grid_maximum_xyz=maximum,
            dtype=torch.float64,
        )
        base[start:stop] = compact.base_indices.numpy().astype(
            base.dtype,
            copy=False,
        )
        fractions[start:stop] = compact.fractions_xyz.numpy().astype(
            stored_fraction_dtype,
            copy=False,
        )
        valid[start:stop] = compact.valid.numpy()
        projection[start:stop, 0] = np.asarray(
            chunk.projection_u_xyz,
            dtype=np.float32,
        )
        projection[start:stop, 1] = np.asarray(
            chunk.projection_v_xyz,
            dtype=np.float32,
        )
        length = np.asarray(chunk.line_length, dtype=np.float64).reshape(-1)
        constant = np.asarray(chunk.system_constant, dtype=np.float64).reshape(-1)
        ray_scale[start:stop] = length * constant / float(sample_count)
        observations[start:stop] = np.asarray(
            chunk.observation_uv,
            dtype=np.float32,
        )
        chunk_records.append(
            {
                "chunk_index": int(chunk_index),
                "start_index": start,
                "stop_index": stop,
                "view_id": int(chunk.view_id),
                "b0_hit_count": int(chunk.b0_hit_count),
            }
        )
        expected_start = stop
    if expected_start != ray_count:
        raise ValueError("source store emitted the wrong number of rays")

    for values in (base, fractions, valid, projection, ray_scale, observations):
        values.flush()
    del base, fractions, valid, projection, ray_scale, observations

    array_filenames = (
        "base_indices.npy",
        "fractions_xyz.npy",
        "valid.npy",
        "projection_uv_xyz.npy",
        "ray_scale.npy",
        "observations_uv.npy",
    )
    arrays = {
        Path(filename).stem: _array_record(working / filename)
        for filename in array_filenames
    }
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "status": CACHE_STATUS,
        "evidence_scope": (
            "PRIVATE_SUPPORT_ONLY_FIXED_B0_STENCIL_CACHE_NO_DEVELOPMENT_NO_FINAL_AUDIT"
        ),
        "grid_shape_zyx": list(shape),
        "grid_minimum_xyz": list(minimum),
        "grid_maximum_xyz": list(maximum),
        "ray_count": ray_count,
        "sample_count": sample_count,
        "chunk_rays": int(source_store.chunk_rays),
        "fraction_dtype": str(stored_fraction_dtype),
        "build_wall_seconds": float(time.perf_counter() - started),
        "source_selection": source_store.selection_summary(),
        "chunks": chunk_records,
        "arrays": arrays,
        "claim_boundary": {
            "contains_support_measurements": True,
            "contains_reconstruction_volume": False,
            "development_rotation_40_opened": False,
            "final_audit_opened": False,
            "public_upload_allowed": False,
        },
    }
    (working / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    working.replace(target)
    return manifest


@dataclass(frozen=True)
class _CachedView:
    view_id: int
    ray_count: int


class PSUCompactCachedRayStore:
    """Read a complete private compact cache in deterministic source chunks."""

    def __init__(self, cache_root: Path, *, verify_hashes: bool = False) -> None:
        self.cache_root = Path(cache_root)
        manifest = _json_object(self.cache_root / "manifest.json")
        if manifest.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("unsupported compact cache schema")
        if manifest.get("status") != CACHE_STATUS:
            raise ValueError("compact cache is not complete")
        self.manifest = manifest
        self.grid_shape = tuple(int(value) for value in manifest["grid_shape_zyx"])
        self.grid_minimum_xyz = tuple(
            float(value) for value in manifest["grid_minimum_xyz"]
        )
        self.grid_maximum_xyz = tuple(
            float(value) for value in manifest["grid_maximum_xyz"]
        )
        self.ray_count = int(manifest["ray_count"])
        self.sample_count = int(manifest["sample_count"])
        self.chunk_rays = int(manifest["chunk_rays"])
        if self.ray_count < 1 or self.sample_count < 1 or self.chunk_rays < 1:
            raise ValueError("compact cache dimensions must be positive")
        source_selection = manifest.get("source_selection")
        if not isinstance(source_selection, dict):
            raise ValueError("compact cache source selection is missing")
        self._selection = source_selection
        view_rows = source_selection.get("view_rows")
        if not isinstance(view_rows, list) or len(view_rows) != 9:
            raise ValueError("compact cache must declare nine support views")
        self.views = tuple(
            _CachedView(
                view_id=int(row["view_id_zero_based"]),
                ray_count=int(row["selected_ray_count"]),
            )
            for row in view_rows
        )
        if [view.view_id for view in self.views] != list(range(9)):
            raise ValueError("compact cache support views must be ordered 0..8")
        if sum(view.ray_count for view in self.views) != self.ray_count:
            raise ValueError("compact cache view counts do not sum to ray_count")

        records = manifest.get("arrays")
        if not isinstance(records, dict):
            raise ValueError("compact cache array records are missing")
        required = {
            "base_indices",
            "fractions_xyz",
            "valid",
            "projection_uv_xyz",
            "ray_scale",
            "observations_uv",
        }
        if set(records) != required:
            raise ValueError("compact cache array set is incomplete")
        self.base_indices = _open_array(
            self.cache_root,
            records["base_indices"],
            verify_hash=verify_hashes,
        )
        self.fractions_xyz = _open_array(
            self.cache_root,
            records["fractions_xyz"],
            verify_hash=verify_hashes,
        )
        self.valid = _open_array(
            self.cache_root,
            records["valid"],
            verify_hash=verify_hashes,
        )
        self.projection_uv_xyz = _open_array(
            self.cache_root,
            records["projection_uv_xyz"],
            verify_hash=verify_hashes,
        )
        self.ray_scale = _open_array(
            self.cache_root,
            records["ray_scale"],
            verify_hash=verify_hashes,
        )
        self.observations_uv = _open_array(
            self.cache_root,
            records["observations_uv"],
            verify_hash=verify_hashes,
        )
        expected_shapes = {
            "base_indices": (self.ray_count, self.sample_count),
            "fractions_xyz": (self.ray_count, self.sample_count, 3),
            "valid": (self.ray_count, self.sample_count),
            "projection_uv_xyz": (self.ray_count, 2, 3),
            "ray_scale": (self.ray_count,),
            "observations_uv": (self.ray_count, 2),
        }
        for name, expected in expected_shapes.items():
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"compact cache array has wrong shape: {name}")

        raw_chunks = manifest.get("chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ValueError("compact cache chunk records are missing")
        expected_start = 0
        chunks: list[dict[str, int]] = []
        for index, raw in enumerate(raw_chunks):
            row = {key: int(value) for key, value in raw.items()}
            if row.get("chunk_index") != index:
                raise ValueError("compact cache chunk indices are not contiguous")
            if row.get("start_index") != expected_start:
                raise ValueError("compact cache chunks do not cover contiguous rays")
            stop = int(row.get("stop_index", -1))
            if stop <= expected_start or stop > self.ray_count:
                raise ValueError("compact cache chunk bounds are invalid")
            if int(row.get("b0_hit_count", -1)) not in range(
                0,
                stop - expected_start + 1,
            ):
                raise ValueError("compact cache hit count is invalid")
            chunks.append(row)
            expected_start = stop
        if expected_start != self.ray_count:
            raise ValueError("compact cache chunks do not cover ray_count")
        self._chunks = tuple(chunks)

    @property
    def selection_mode(self) -> str:
        return str(self._selection["selection_mode"])

    def selection_summary(self) -> dict[str, Any]:
        summary = copy.deepcopy(self._selection)
        summary["cache_schema_version"] = CACHE_SCHEMA
        summary["cache_mode"] = "private_compact_lower_corner_and_fraction_stream"
        return summary

    def load_observations(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.as_tensor(
            self.observations_uv,
            dtype=dtype,
            device=device,
        )[None]

    def iter_chunks(self) -> Iterator[StreamingRayChunk]:
        for row in self._chunks:
            start = row["start_index"]
            stop = row["stop_index"]
            yield StreamingRayChunk(
                start_index=start,
                stop_index=stop,
                sample_points_xyz=None,
                projection_u_xyz=self.projection_uv_xyz[start:stop, 0],
                projection_v_xyz=self.projection_uv_xyz[start:stop, 1],
                line_length=None,
                system_constant=None,
                observation_uv=self.observations_uv[start:stop],
                view_id=row["view_id"],
                b0_hit_count=row["b0_hit_count"],
                compact_base_indices=self.base_indices[start:stop],
                compact_fractions_xyz=self.fractions_xyz[start:stop],
                compact_valid=self.valid[start:stop],
                ray_scale=self.ray_scale[start:stop],
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--chunk-rays", type=int, default=32768)
    parser.add_argument(
        "--rays-per-view",
        type=int,
        default=0,
        help="0 means every corrected active row",
    )
    parser.add_argument(
        "--fraction-dtype",
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(int(args.torch_threads))
    store = PSURealSupportRayStore(
        args.view_root,
        rays_per_view=(
            None if int(args.rays_per_view) == 0 else int(args.rays_per_view)
        ),
        sample_count=int(args.sample_count),
        chunk_rays=int(args.chunk_rays),
    )
    manifest = build_compact_cache(
        store,
        args.output_root,
        grid_shape=(int(args.grid_size),) * 3,
        grid_minimum_xyz=(-0.11, -0.11, -0.11),
        grid_maximum_xyz=(0.11, 0.11, 0.11),
        fraction_dtype=args.fraction_dtype,
    )
    public_console = {
        "schema_version": manifest["schema_version"],
        "status": manifest["status"],
        "grid_shape_zyx": manifest["grid_shape_zyx"],
        "ray_count": manifest["ray_count"],
        "sample_count": manifest["sample_count"],
        "chunk_count": len(manifest["chunks"]),
        "fraction_dtype": manifest["fraction_dtype"],
        "build_wall_seconds": manifest["build_wall_seconds"],
        "total_cache_bytes": sum(
            int(record["nbytes"]) for record in manifest["arrays"].values()
        ),
        "development_rotation_40_opened": False,
        "final_audit_opened": False,
    }
    print(json.dumps(public_console, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
