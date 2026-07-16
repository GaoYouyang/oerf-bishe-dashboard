"""Controlled camera-budget inputs for the T16 direct inverse-operator pilot."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

try:
    from .bost_physics import baseline_lift, make_grid_3d
    from .data import fit_global_affine
    from .run_fair_camera_budget import controlled_support_mask
except ImportError:
    from bost_physics import baseline_lift, make_grid_3d
    from data import fit_global_affine
    from run_fair_camera_budget import controlled_support_mask


INPUT_SCHEMA_VERSION = 1


def reconstruction_mask(
    max_views: int,
    total_budget: int,
    fixed_query_index: int,
    audit_query_index: int,
) -> np.ndarray:
    """Return the K-view reconstruction mask while reserving Q_audit."""
    if total_budget < 2:
        raise ValueError("total_budget must include support and Q_fit")
    support = controlled_support_mask(
        max_views,
        total_budget - 1,
        fixed_query_index,
        audit_query_index,
    )
    mask = support.astype(np.float32)
    mask[int(fixed_query_index)] = 1.0
    if int(mask.sum()) != int(total_budget):
        raise RuntimeError("reconstruction mask does not match the declared budget")
    if mask[int(audit_query_index)] > 0.5:
        raise RuntimeError("Q_audit leaked into the reconstruction mask")
    return mask


def reconstruction_masks(
    max_views: int,
    budgets: Iterable[int],
    fixed_query_index: int,
    audit_query_index: int,
) -> dict[int, np.ndarray]:
    return {
        int(budget): reconstruction_mask(
            max_views,
            int(budget),
            fixed_query_index,
            audit_query_index,
        )
        for budget in budgets
    }


def _split_names(data: dict[str, np.ndarray]) -> list[str]:
    return [str(value) for value in data["split_names"].tolist()]


def prepare_direct_operator_data(
    base_data: dict[str, np.ndarray],
    budgets: Iterable[int],
    fixed_query_index: int,
    audit_query_index: int,
) -> dict[str, np.ndarray]:
    """Repeat each field over budgets and build train-calibrated inputs.

    The original dataset view mask and lift are deliberately ignored. All train,
    validation, and test samples are rebuilt from the same predeclared K-view
    masks, while Q_audit remains unavailable to the reconstruction.
    """
    budget_values = [int(value) for value in budgets]
    if len(set(budget_values)) != len(budget_values):
        raise ValueError("reconstruction budgets must be unique")
    masks = reconstruction_masks(
        int(len(base_data["angles"])),
        budget_values,
        int(fixed_query_index),
        int(audit_query_index),
    )
    n = int(base_data["field"].shape[-1])
    depth = int(base_data["field"].shape[-3])
    sample_count = int(len(base_data["field"]))
    variant_source = np.repeat(np.arange(sample_count, dtype=np.int64), len(budget_values))
    variant_budget = np.tile(np.asarray(budget_values, dtype=np.int64), sample_count)
    view_masks = np.stack([masks[int(value)] for value in variant_budget]).astype(np.float32)

    raw_lifts = []
    observations = []
    for source_index, budget, mask in zip(variant_source, variant_budget, view_masks):
        selected = np.flatnonzero(mask > 0.5)
        clean = np.asarray(base_data["clean_observation"][int(source_index)], dtype=np.float32)
        noise_level = float(base_data["noise_level"][int(source_index)])
        observed_rms = float(np.sqrt(np.mean(clean[:, selected] ** 2)) + 1e-8)
        rng = np.random.default_rng(
            int(base_data["sample_seed"][int(source_index)]) + int(budget) * 10_007 + 3_109
        )
        observation = clean + rng.normal(
            scale=noise_level * observed_rms,
            size=clean.shape,
        ).astype(np.float32)
        observations.append(observation)
        raw_lifts.append(
            baseline_lift(
                observation[:, selected],
                base_data["angles"][selected],
                n,
            ).astype(np.float32)
        )
    raw_lifts = np.stack(raw_lifts)
    observations = np.stack(observations)
    fields = base_data["field"][variant_source].astype(np.float32)
    split_ids = base_data["split_id"][variant_source].astype(np.int64)
    names = _split_names(base_data)
    train_id = names.index("train")
    train_mask = split_ids == train_id
    scale, offset = fit_global_affine(raw_lifts[train_mask], fields[train_mask])
    lifts = (scale * raw_lifts + offset).astype(np.float32)

    support = base_data["support"].astype(np.float32)
    zz, yy, xx = make_grid_3d(n, depth)
    variants = len(variant_source)
    channels = [lifts[:, None]]
    channels.append(np.broadcast_to(support, (variants, depth, n, n))[:, None])
    view_fraction = (variant_budget.astype(np.float32) / len(base_data["angles"]))[:, None, None, None, None]
    channels.append(np.broadcast_to(view_fraction, (variants, 1, depth, n, n)))
    for view_index in range(view_masks.shape[1]):
        active = view_masks[:, view_index, None, None, None, None]
        channels.append(np.broadcast_to(active, (variants, 1, depth, n, n)))
    for grid in (zz, yy, xx):
        channels.append(
            np.broadcast_to(grid.astype(np.float32), (variants, depth, n, n))[:, None]
        )

    input_names = [
        "calibrated_k_view_lift",
        "support",
        "view_fraction",
        *[f"camera_{index}_active" for index in range(view_masks.shape[1])],
        "z",
        "y",
        "x",
    ]
    packed = {
        "inputs": np.concatenate(channels, axis=1).astype(np.float32),
        "field": fields,
        "lift": lifts,
        "lift_raw": raw_lifts,
        "observation": observations,
        "clean_observation": base_data["clean_observation"][variant_source].astype(np.float32),
        "view_mask": view_masks,
        "view_count": variant_budget.copy(),
        "total_budget": variant_budget,
        "noise_level": base_data["noise_level"][variant_source].astype(np.float64),
        "family_id": base_data["family_id"][variant_source].astype(np.int64),
        "split_id": split_ids,
        "source_index": variant_source,
        "sample_seed": base_data["sample_seed"][variant_source].astype(np.int64),
        "angles": base_data["angles"].astype(np.float32),
        "forward_matrix": base_data["forward_matrix"].astype(np.float32),
        "support": support,
        "split_names": base_data["split_names"].copy(),
        "calibration": np.asarray([scale, offset], dtype=np.float64),
        "input_channel_names": np.asarray(input_names),
        "input_schema_version": np.asarray(INPUT_SCHEMA_VERSION, dtype=np.int64),
        "fixed_query_index": np.asarray(fixed_query_index, dtype=np.int64),
        "audit_query_index": np.asarray(audit_query_index, dtype=np.int64),
    }
    if packed["inputs"].shape[1] != len(input_names):
        raise RuntimeError("input channel manifest and tensor disagree")
    return packed


def ridge_reconstruction_matrix(
    operator: np.ndarray,
    view_mask: np.ndarray,
    ridge_relative: float,
) -> np.ndarray:
    """Build A^T(AA^T + lambda I)^-1 for one fixed camera mask."""
    selected = np.flatnonzero(np.asarray(view_mask) > 0.5)
    matrix = np.asarray(operator[selected], dtype=np.float64).reshape(-1, operator.shape[-1])
    gram = matrix @ matrix.T
    scale = float(np.trace(gram) / max(len(gram), 1))
    ridge = max(float(ridge_relative) * scale, 1e-12)
    return matrix.T @ np.linalg.solve(
        gram + ridge * np.eye(len(gram), dtype=np.float64),
        np.eye(len(gram), dtype=np.float64),
    )


def ridge_reconstruct(
    observation: np.ndarray,
    operator: np.ndarray,
    view_mask: np.ndarray,
    ridge_relative: float,
    support: np.ndarray,
) -> np.ndarray:
    selected = np.flatnonzero(np.asarray(view_mask) > 0.5)
    inverse = ridge_reconstruction_matrix(operator, view_mask, ridge_relative)
    projected = np.asarray(observation[:, selected], dtype=np.float64).reshape(
        observation.shape[0], -1
    )
    volume = (projected @ inverse.T).reshape(observation.shape[0], support.shape[-2], support.shape[-1])
    return (np.clip(volume, 0.0, None) * support).astype(np.float32)


def replace_lift_with_ridge(
    data: dict[str, np.ndarray],
    ridge_by_budget: dict[int, float],
) -> dict[str, np.ndarray]:
    """Create a copy whose residual channel is the validation-locked ridge field."""
    output = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    ridge_fields = []
    for index in range(len(data["field"])):
        budget = int(data["total_budget"][index])
        ridge_fields.append(
            ridge_reconstruct(
                data["observation"][index],
                data["forward_matrix"],
                data["view_mask"][index],
                float(ridge_by_budget[budget]),
                data["support"],
            )
        )
    ridge_fields = np.stack(ridge_fields).astype(np.float32)
    output["lift"] = ridge_fields
    output["lift_raw"] = ridge_fields.copy()
    output["inputs"][:, 0] = ridge_fields
    names = [str(value) for value in output["input_channel_names"].tolist()]
    names[0] = "validation_tuned_ridge_lift"
    output["input_channel_names"] = np.asarray(names)
    return output
