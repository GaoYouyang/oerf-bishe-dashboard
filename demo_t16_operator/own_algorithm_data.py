"""Per-view ray backprojections for the T16 v3b algorithm benchmark."""

from __future__ import annotations

import numpy as np


RAY_VIEW_SCHEMA_VERSION = 1


def append_ray_view_channels(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Append one train-normalized backprojection channel per camera.

    Every camera channel is formed independently before masked set aggregation.
    Inactive cameras, including Q_audit, are exactly zero. Normalization uses
    active training variants only and is frozen for validation and testing.
    """
    output = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    observation = np.asarray(data["observation"], dtype=np.float32)
    operator = np.asarray(data["forward_matrix"], dtype=np.float32)
    view_mask = np.asarray(data["view_mask"], dtype=np.float32)
    support = np.asarray(data["support"], dtype=np.float32)
    sample_count, depth, view_count, detector_width = observation.shape
    height, width = support.shape[-2:]
    if operator.shape != (view_count, detector_width, height * width):
        raise ValueError("forward matrix and observation geometry disagree")

    raw = np.zeros((sample_count, view_count, depth, height, width), dtype=np.float32)
    for view_index in range(view_count):
        backprojected = np.einsum(
            "bdn,np->bdp",
            observation[:, :, view_index],
            operator[view_index],
            optimize=True,
        ).reshape(sample_count, depth, height, width)
        raw[:, view_index] = backprojected * view_mask[:, view_index, None, None, None]

    split_names = [str(value) for value in data["split_names"].tolist()]
    train_id = split_names.index("train")
    training = np.asarray(data["split_id"]) == train_id
    scales = np.ones(view_count, dtype=np.float32)
    for view_index in range(view_count):
        active = training & (view_mask[:, view_index] > 0.5)
        if np.any(active):
            scales[view_index] = max(
                float(np.sqrt(np.mean(raw[active, view_index].astype(np.float64) ** 2))),
                1e-6,
            )
    normalized = raw / scales[None, :, None, None, None]
    normalized *= view_mask[:, :, None, None, None]

    view_channel_start = int(output["inputs"].shape[1])
    angle_radians = np.deg2rad(np.asarray(data["angles"], dtype=np.float32))
    angle_sin = np.broadcast_to(
        np.sin(angle_radians)[None, :, None, None, None], normalized.shape
    ) * view_mask[:, :, None, None, None]
    angle_cos = np.broadcast_to(
        np.cos(angle_radians)[None, :, None, None, None], normalized.shape
    ) * view_mask[:, :, None, None, None]
    angle_sin_start = view_channel_start + view_count
    angle_cos_start = angle_sin_start + view_count
    output["inputs"] = np.concatenate(
        [output["inputs"], normalized, angle_sin.astype(np.float32), angle_cos.astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    names = [str(value) for value in output["input_channel_names"].tolist()]
    names.extend(f"ray_backprojection_camera_{index}" for index in range(view_count))
    names.extend(f"camera_{index}_sin_active" for index in range(view_count))
    names.extend(f"camera_{index}_cos_active" for index in range(view_count))
    output["input_channel_names"] = np.asarray(names)
    output["ray_view_channel_start"] = np.asarray(view_channel_start, dtype=np.int64)
    output["ray_view_channel_count"] = np.asarray(view_count, dtype=np.int64)
    output["ray_angle_sin_channel_start"] = np.asarray(angle_sin_start, dtype=np.int64)
    output["ray_angle_cos_channel_start"] = np.asarray(angle_cos_start, dtype=np.int64)
    output["ray_view_scales"] = scales
    output["ray_view_schema_version"] = np.asarray(RAY_VIEW_SCHEMA_VERSION, dtype=np.int64)
    if output["inputs"].shape[1] != len(names):
        raise RuntimeError("ray-view channel manifest and tensor disagree")
    return output
