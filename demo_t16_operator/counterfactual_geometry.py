"""Lazy same-field/multi-geometry inputs for the T16 v3k mechanism audit."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .direct_operator_data import ridge_reconstruction_matrix
except ImportError:
    from direct_operator_data import ridge_reconstruction_matrix


def _stable_hash(*values: object) -> str:
    payload = ":".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def geometry_catalog(
    data: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    """Recover the frozen geometry catalog from the v3i private dataset."""
    catalog: dict[str, dict[str, object]] = {}
    for identifier, partition, mask in zip(
        data["geometry_id"], data["geometry_partition"], data["view_mask"]
    ):
        key = str(identifier)
        value = {
            "geometry_id": key,
            "geometry_partition": str(partition),
            "mask": np.asarray(mask, dtype=np.float32),
        }
        if key in catalog:
            if catalog[key]["geometry_partition"] != value["geometry_partition"]:
                raise ValueError("one geometry appears in multiple partitions")
            if not np.array_equal(catalog[key]["mask"], value["mask"]):
                raise ValueError("one geometry id maps to multiple masks")
        else:
            catalog[key] = value
    if len(catalog) != 28:
        raise ValueError("v3k expects the frozen 28-layout v3i catalog")
    return catalog


def partition_geometry_ids(
    catalog: dict[str, dict[str, object]], partition: str
) -> list[str]:
    identifiers = sorted(
        key
        for key, value in catalog.items()
        if str(value["geometry_partition"]) == str(partition)
    )
    if not identifiers:
        raise ValueError(f"geometry partition {partition!r} is empty")
    return identifiers


def build_pair_schedule(
    data: dict[str, np.ndarray],
    source_split: str,
    geometry_partition: str,
    arm: str,
    repeats_per_source: int,
    assignment_seed: int,
    counterfactual_stride: int,
) -> list[dict[str, object]]:
    """Build an exposure-matched M1/M4 schedule or an all-layout audit schedule."""
    valid_arms = {"m1_repeat", "m4_counterfactual", "evaluation"}
    if arm not in valid_arms:
        raise ValueError(f"unknown counterfactual arm: {arm}")
    catalog = geometry_catalog(data)
    identifiers = partition_geometry_ids(catalog, geometry_partition)
    split_names = [str(value) for value in data["split_names"].tolist()]
    if source_split not in split_names:
        raise ValueError(f"unknown source split: {source_split}")
    split_id = split_names.index(source_split)
    sources = np.flatnonzero(np.asarray(data["split_id"]) == split_id).tolist()
    sources.sort(
        key=lambda index: (
            _stable_hash(assignment_seed, source_split, int(data["sample_seed"][index])),
            int(index),
        )
    )
    if not sources:
        raise ValueError(f"source split {source_split!r} is empty")

    if arm == "evaluation":
        repeats = len(identifiers)
    else:
        repeats = int(repeats_per_source)
        if repeats < 2:
            raise ValueError("counterfactual audit requires at least two exposures per field")
    stride = int(counterfactual_stride)
    if arm == "m4_counterfactual":
        if math.gcd(stride, len(identifiers)) != 1:
            raise ValueError("counterfactual stride must be coprime with partition size")
        if repeats > len(identifiers):
            raise ValueError("counterfactual repeats exceed available geometries")

    offset = int(_stable_hash(assignment_seed, source_split)[:8], 16) % len(
        identifiers
    )
    rows: list[dict[str, object]] = []
    for position, source_index in enumerate(sources):
        first = (position + offset) % len(identifiers)
        for slot in range(repeats):
            if arm == "m1_repeat":
                geometry_position = first
            elif arm == "m4_counterfactual":
                geometry_position = (first + slot * stride) % len(identifiers)
            else:
                geometry_position = slot
            identifier = identifiers[geometry_position]
            rows.append(
                {
                    "pair_index": len(rows),
                    "training_arm": arm,
                    "source_split": source_split,
                    "source_index": int(source_index),
                    "sample_seed": int(data["sample_seed"][source_index]),
                    "geometry_slot": int(slot),
                    "geometry_id": identifier,
                    "geometry_partition": geometry_partition,
                    "mask_bits": identifier.removeprefix("g_"),
                    "assignment_seed": int(assignment_seed),
                    "assignment_rule": (
                        "one balanced layout repeated at matched exposure"
                        if arm == "m1_repeat"
                        else "balanced cyclic distinct layouts"
                        if arm == "m4_counterfactual"
                        else "all frozen layouts in partition"
                    ),
                }
            )
    return rows


def schedule_balance(rows: list[dict[str, object]]) -> dict[str, object]:
    by_geometry = Counter(str(row["geometry_id"]) for row in rows)
    by_source: dict[int, set[str]] = {}
    exposure: Counter[int] = Counter()
    for row in rows:
        source = int(row["source_index"])
        by_source.setdefault(source, set()).add(str(row["geometry_id"]))
        exposure[source] += 1
    return {
        "row_count": len(rows),
        "source_count": len(by_source),
        "geometry_count": len(by_geometry),
        "minimum_rows_per_geometry": min(by_geometry.values()),
        "maximum_rows_per_geometry": max(by_geometry.values()),
        "minimum_exposures_per_source": min(exposure.values()),
        "maximum_exposures_per_source": max(exposure.values()),
        "minimum_unique_geometries_per_source": min(map(len, by_source.values())),
        "maximum_unique_geometries_per_source": max(map(len, by_source.values())),
    }


def geometry_derangement_map(
    catalog: dict[str, dict[str, object]],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    mapping: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    partitions = sorted(
        {str(value["geometry_partition"]) for value in catalog.values()}
    )
    for partition in partitions:
        identifiers = partition_geometry_ids(catalog, partition)
        if len(identifiers) < 2:
            raise ValueError("derangement requires at least two layouts per partition")
        for index, identifier in enumerate(identifiers):
            wrong = identifiers[(index + 1) % len(identifiers)]
            mapping[identifier] = wrong
            rows.append(
                {
                    "geometry_partition": partition,
                    "correct_geometry_id": identifier,
                    "wrong_geometry_id": wrong,
                    "correct_mask_bits": identifier.removeprefix("g_"),
                    "wrong_mask_bits": wrong.removeprefix("g_"),
                    "fixed_point": identifier == wrong,
                    "mapping_rule": "lexical cyclic derangement within partition",
                }
            )
    if any(key == value for key, value in mapping.items()):
        raise RuntimeError("geometry derangement contains a fixed point")
    return mapping, rows


class CounterfactualInputFactory:
    """Generate one geometry-consistent 42-channel input without Cartesian storage."""

    def __init__(self, data: dict[str, np.ndarray], ridge_relative: float):
        self.data = data
        self.catalog = geometry_catalog(data)
        self.ridge_relative = float(ridge_relative)
        self.names = [str(value) for value in data["input_channel_names"].tolist()]
        self.observation = np.asarray(data["observation"], dtype=np.float32)
        self.operator = np.asarray(data["forward_matrix"], dtype=np.float32)
        self.support = np.asarray(data["support"], dtype=np.float32)
        self.angles = np.asarray(data["angles"], dtype=np.float32)
        self.audit_query_index = int(data["audit_query_index"])
        self.total_budget = int(np.asarray(data["total_budget"])[0])
        self.view_count = len(self.angles)
        self.depth, self.height, self.width = data["field"].shape[1:]
        self.mask_start = self.names.index("camera_0_active")
        self.coordinate_channels = tuple(self.names.index(axis) for axis in ("z", "y", "x"))
        self.ray_start = int(data["ray_view_channel_start"])
        self.sin_start = int(data["ray_angle_sin_channel_start"])
        self.cos_start = int(data["ray_angle_cos_channel_start"])
        self.ray_scales = np.asarray(data["ray_view_scales"], dtype=np.float32)
        if np.any(self.ray_scales <= 0):
            raise ValueError("frozen v3i ray scales must be positive")
        self.coordinates = np.asarray(
            data["inputs"][0, list(self.coordinate_channels)], dtype=np.float32
        )
        self.inverse_cache = {
            identifier: ridge_reconstruction_matrix(
                self.operator,
                np.asarray(value["mask"], dtype=np.float32),
                self.ridge_relative,
            )
            for identifier, value in self.catalog.items()
        }
        self.raw_backprojections = np.zeros(
            (
                len(data["field"]),
                self.view_count,
                self.depth,
                self.height,
                self.width,
            ),
            dtype=np.float32,
        )
        for view_index in range(self.view_count):
            self.raw_backprojections[:, view_index] = np.einsum(
                "bdn,np->bdp",
                self.observation[:, :, view_index],
                self.operator[view_index],
                optimize=True,
            ).reshape(len(data["field"]), self.depth, self.height, self.width)
        self._assert_channel_contract()

    def _assert_channel_contract(self) -> None:
        expected = {
            0,
            1,
            2,
            *range(self.mask_start, self.mask_start + self.view_count),
            *self.coordinate_channels,
            *range(self.ray_start, self.ray_start + self.view_count),
            *range(self.sin_start, self.sin_start + self.view_count),
            *range(self.cos_start, self.cos_start + self.view_count),
        }
        if expected != set(range(len(self.names))):
            raise ValueError("counterfactual input factory does not cover the channel schema")

    def mask(self, geometry_identifier: str) -> np.ndarray:
        mask = np.asarray(self.catalog[str(geometry_identifier)]["mask"], dtype=np.float32)
        if mask[self.audit_query_index] > 0.5:
            raise RuntimeError("audit camera leaked into a counterfactual layout")
        if int(mask.sum()) != self.total_budget:
            raise RuntimeError("counterfactual layout violates the fixed camera budget")
        return mask

    def ridge(self, source_index: int, geometry_identifier: str) -> np.ndarray:
        mask = self.mask(geometry_identifier)
        selected = np.flatnonzero(mask > 0.5)
        projected = self.observation[int(source_index)][:, selected].reshape(
            self.depth, -1
        )
        volume = (
            projected @ self.inverse_cache[str(geometry_identifier)].T
        ).reshape(self.depth, self.height, self.width)
        return (np.clip(volume, 0.0, None) * self.support).astype(np.float32)

    def input_tensor(
        self, source_index: int, geometry_identifier: str, ridge: np.ndarray
    ) -> np.ndarray:
        mask = self.mask(geometry_identifier)
        x = np.zeros(
            (len(self.names), self.depth, self.height, self.width), dtype=np.float32
        )
        x[0] = ridge
        x[1] = self.support
        x[2] = float(self.total_budget) / self.view_count
        x[self.mask_start : self.mask_start + self.view_count] = mask[
            :, None, None, None
        ]
        x[list(self.coordinate_channels)] = self.coordinates
        rays = self.raw_backprojections[int(source_index)] / self.ray_scales[
            :, None, None, None
        ]
        x[self.ray_start : self.ray_start + self.view_count] = rays * mask[
            :, None, None, None
        ]
        radians = np.deg2rad(self.angles)
        x[self.sin_start : self.sin_start + self.view_count] = (
            np.sin(radians) * mask
        )[:, None, None, None]
        x[self.cos_start : self.cos_start + self.view_count] = (
            np.cos(radians) * mask
        )[:, None, None, None]
        return x


class CounterfactualGeometryDataset(Dataset):
    """A compact pair index with cached ridge volumes and lazy full inputs."""

    def __init__(
        self,
        factory: CounterfactualInputFactory,
        pairs: list[dict[str, object]],
    ):
        self.factory = factory
        self.data = factory.data
        self.pairs = [dict(row) for row in pairs]
        for index, row in enumerate(self.pairs):
            row["pair_index"] = index
        self.ridges = np.stack(
            [
                factory.ridge(int(row["source_index"]), str(row["geometry_id"]))
                for row in self.pairs
            ]
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(item)
        row = self.pairs[index]
        source = int(row["source_index"])
        geometry_identifier = str(row["geometry_id"])
        mask = self.factory.mask(geometry_identifier)
        ridge = self.ridges[index]
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "source_index": torch.tensor(source, dtype=torch.long),
            "x": torch.from_numpy(
                self.factory.input_tensor(source, geometry_identifier, ridge)
            ),
            "field": torch.from_numpy(self.data["field"][source][None]),
            "lift": torch.from_numpy(ridge[None]),
            "observation": torch.from_numpy(self.data["observation"][source]),
            "clean_observation": torch.from_numpy(
                self.data["clean_observation"][source]
            ),
            "view_mask": torch.from_numpy(mask),
        }


def descriptor_components_for_pairs(
    factory: CounterfactualInputFactory,
    pairs: list[dict[str, object]],
    geometry_mapping: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = []
    for row in pairs:
        identifier = str(row["geometry_id"])
        if geometry_mapping is not None:
            identifier = str(geometry_mapping[identifier])
        masks.append(factory.mask(identifier))
    mask_array = np.asarray(masks, dtype=np.float32)
    radians = np.deg2rad(factory.angles)
    sin = mask_array * np.sin(radians)[None]
    cos = mask_array * np.cos(radians)[None]
    return mask_array, sin.astype(np.float32), cos.astype(np.float32)


def ray_set_components_for_pairs(
    factory: CounterfactualInputFactory,
    pairs: list[dict[str, object]],
    geometry_mapping: dict[str, str] | None = None,
    shuffle_angle_pairing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build geometry-consistent local camera tokens for a pair schedule."""
    masks = []
    rays = []
    for row in pairs:
        identifier = str(row["geometry_id"])
        if geometry_mapping is not None:
            identifier = str(geometry_mapping[identifier])
        mask = factory.mask(identifier).astype(np.float32)
        source = int(row["source_index"])
        normalized = factory.raw_backprojections[source] / factory.ray_scales[
            :, None, None, None
        ]
        masks.append(mask)
        rays.append(normalized * mask[:, None, None, None])
    mask_array = np.asarray(masks, dtype=np.float32)
    ray_array = np.asarray(rays, dtype=np.float32)
    radians = np.deg2rad(factory.angles)
    angle_sin = mask_array * np.sin(radians)[None]
    angle_cos = mask_array * np.cos(radians)[None]
    if shuffle_angle_pairing:
        shuffled_sin = np.zeros_like(angle_sin)
        shuffled_cos = np.zeros_like(angle_cos)
        for row_index, mask in enumerate(mask_array):
            active = np.flatnonzero(mask > 0.5)
            if len(active) < 2:
                raise ValueError("angle-pairing derangement requires two active cameras")
            donors = np.roll(active, -1)
            shuffled_sin[row_index, active] = np.sin(radians[donors])
            shuffled_cos[row_index, active] = np.cos(radians[donors])
        angle_sin = shuffled_sin
        angle_cos = shuffled_cos
    if np.any(mask_array[:, factory.audit_query_index] != 0.0):
        raise RuntimeError("audit camera leaked into ray-set components")
    if np.any(ray_array[:, factory.audit_query_index] != 0.0):
        raise RuntimeError("audit camera signal leaked into ray-set components")
    return (
        mask_array,
        angle_sin.astype(np.float32),
        angle_cos.astype(np.float32),
        ray_array,
    )


def ray_angle_pairing_derangement_rows(
    factory: CounterfactualInputFactory,
) -> list[dict[str, object]]:
    """Describe the fixed-point-free angle reassignment within each active set."""
    rows: list[dict[str, object]] = []
    for identifier in sorted(factory.catalog):
        mask = factory.mask(identifier)
        active = np.flatnonzero(mask > 0.5)
        donors = np.roll(active, -1)
        for ray_camera, angle_camera in zip(active, donors):
            rows.append(
                {
                    "geometry_id": identifier,
                    "geometry_partition": str(
                        factory.catalog[identifier]["geometry_partition"]
                    ),
                    "ray_camera_index": int(ray_camera),
                    "correct_angle_camera_index": int(ray_camera),
                    "shuffled_angle_camera_index": int(angle_camera),
                    "fixed_point": bool(ray_camera == angle_camera),
                    "active_camera_count": int(len(active)),
                    "mapping_rule": "cyclic angle reassignment within unchanged active ray set",
                }
            )
    if any(bool(row["fixed_point"]) for row in rows):
        raise RuntimeError("ray-angle pairing derangement contains a fixed point")
    return rows
