"""Paired synthetic data for the T16 BOST inverse-operator benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .bost_physics import (
        baseline_lift,
        build_forward_matrix,
        forward_volume,
        make_grid_3d,
        make_phantom,
        support_window,
    )
except ImportError:
    from bost_physics import (
        baseline_lift,
        build_forward_matrix,
        forward_volume,
        make_grid_3d,
        make_phantom,
        support_window,
    )


FAMILY_TO_ID = {"gaussian": 0, "flame": 1, "thin_front": 2}
DATASET_SCHEMA_VERSION = 2


def select_view_indices(max_views: int, view_count: int) -> np.ndarray:
    if view_count > max_views:
        raise ValueError("view_count cannot exceed max_views")
    return np.rint(np.linspace(0, max_views - 1, view_count)).astype(int)


def fit_global_affine(raw_lifts: np.ndarray, fields: np.ndarray) -> tuple[float, float]:
    """Fit one train-only calibration shared by all validation/test samples."""
    x = raw_lifts.astype(np.float64).reshape(-1)
    y = fields.astype(np.float64).reshape(-1)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    variance = float(np.mean((x - x_mean) ** 2))
    scale = float(np.mean((x - x_mean) * (y - y_mean)) / (variance + 1e-12))
    return scale, y_mean - scale * x_mean


def generate_dataset(config: dict, output_path: Path, force: bool = False) -> Path:
    if output_path.exists() and not force:
        with np.load(output_path, allow_pickle=False) as archive:
            stored_config = json.loads(str(archive["config_json"]))
            stored_schema = int(archive["schema_version"]) if "schema_version" in archive.files else 0
        if stored_config == config and stored_schema == DATASET_SCHEMA_VERSION:
            return output_path

    n = int(config["grid_size"])
    depth = int(config["depth"])
    max_views = int(config["max_views"])
    base_seed = int(config["seed"])
    angles = np.linspace(0.0, 180.0, max_views, endpoint=False, dtype=np.float32)
    operator = build_forward_matrix(n, angles)

    arrays: dict[str, list] = {
        "field": [],
        "lift_raw": [],
        "observation": [],
        "clean_observation": [],
        "view_mask": [],
        "view_count": [],
        "noise_level": [],
        "family_id": [],
        "split_id": [],
        "sample_seed": [],
    }
    split_names = list(config["splits"].keys())

    for split_id, split_name in enumerate(split_names):
        spec = config["splits"][split_name]
        conditions = [
            (str(family), int(view_count), float(noise_level))
            for family in spec["families"]
            for view_count in spec["views"]
            for noise_level in spec["noise"]
        ]
        for local_index in range(int(spec["count"])):
            sample_seed = base_seed + split_id * 100_000 + local_index
            rng = np.random.default_rng(sample_seed)
            family, view_count, noise_level = conditions[local_index % len(conditions)]

            field = make_phantom(family, n, depth, rng)
            clean = forward_volume(field, operator).astype(np.float32)
            selected = select_view_indices(max_views, view_count)
            mask = np.zeros(max_views, dtype=np.float32)
            mask[selected] = 1.0
            observed_rms = float(np.sqrt(np.mean(clean[:, selected] ** 2)) + 1e-8)
            noisy = clean + rng.normal(scale=noise_level * observed_rms, size=clean.shape).astype(np.float32)
            lift = baseline_lift(noisy[:, selected], angles[selected], n)

            arrays["field"].append(field)
            arrays["lift_raw"].append(lift)
            arrays["observation"].append(noisy)
            arrays["clean_observation"].append(clean)
            arrays["view_mask"].append(mask)
            arrays["view_count"].append(view_count)
            arrays["noise_level"].append(noise_level)
            arrays["family_id"].append(FAMILY_TO_ID[family])
            arrays["split_id"].append(split_id)
            arrays["sample_seed"].append(sample_seed)

    packed = {key: np.asarray(value) for key, value in arrays.items()}
    train_id = split_names.index("train")
    train_mask = packed["split_id"] == train_id
    scale, offset = fit_global_affine(packed["lift_raw"][train_mask], packed["field"][train_mask])
    packed["lift"] = (scale * packed["lift_raw"] + offset).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **packed,
        angles=angles,
        forward_matrix=operator,
        support=support_window(n, depth).astype(np.float32),
        split_names=np.asarray(split_names),
        calibration=np.asarray([scale, offset], dtype=np.float64),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
        schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
    )
    return output_path


def build_input_channels(data: dict[str, np.ndarray]) -> np.ndarray:
    lift = data["lift"].astype(np.float32)
    sample_count, depth, n, _ = lift.shape
    max_views = data["view_mask"].shape[1]
    max_noise = max(float(data["noise_level"].max()), 1e-6)
    xx, yy, zz = make_grid_3d(n, depth)
    support = data["support"].astype(np.float32)

    channels = [lift[:, None]]
    channels.append(np.broadcast_to(support, (sample_count, depth, n, n))[:, None])
    view_fraction = (data["view_count"].astype(np.float32) / max_views)[:, None, None, None, None]
    channels.append(np.broadcast_to(view_fraction, (sample_count, 1, depth, n, n)))
    noise = (data["noise_level"].astype(np.float32) / max_noise)[:, None, None, None, None]
    channels.append(np.broadcast_to(noise, (sample_count, 1, depth, n, n)))
    for grid in (zz, yy, xx):
        channels.append(np.broadcast_to(grid.astype(np.float32), (sample_count, depth, n, n))[:, None])
    return np.concatenate(channels, axis=1).astype(np.float32)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data["inputs"] = build_input_channels(data)
    return data


class BOSTDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray], indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "x": torch.from_numpy(self.data["inputs"][index]),
            "field": torch.from_numpy(self.data["field"][index][None]),
            "lift": torch.from_numpy(self.data["lift"][index][None]),
            "observation": torch.from_numpy(self.data["observation"][index]),
            "clean_observation": torch.from_numpy(self.data["clean_observation"][index]),
            "view_mask": torch.from_numpy(self.data["view_mask"][index]),
        }


def split_indices(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = [str(value) for value in data["split_names"].tolist()]
    return {name: np.flatnonzero(data["split_id"] == split_id) for split_id, name in enumerate(names)}
