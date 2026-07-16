"""Variable-camera manifests and geometry diagnostics for the T16 GC-SRO gate."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter, defaultdict

import numpy as np

try:
    from .bost_physics import make_grid_3d
    from .direct_operator_data import ridge_reconstruct, ridge_reconstruction_matrix
    from .own_algorithm_data import append_ray_view_channels
    from .run_direct_operator_pilot import relative_field_error
except ImportError:
    from bost_physics import make_grid_3d
    from direct_operator_data import ridge_reconstruct, ridge_reconstruction_matrix
    from own_algorithm_data import append_ray_view_channels
    from run_direct_operator_pilot import relative_field_error


GEOMETRY_SCHEMA_VERSION = 1


def geometry_id(mask: np.ndarray) -> str:
    bits = "".join("1" if value > 0.5 else "0" for value in np.asarray(mask))
    return f"g_{bits}"


def enumerate_budget_masks(
    view_count: int,
    total_budget: int,
    audit_query_index: int,
) -> list[np.ndarray]:
    allowed = [index for index in range(int(view_count)) if index != int(audit_query_index)]
    if int(total_budget) > len(allowed):
        raise ValueError("camera budget exceeds non-audit cameras")
    masks = []
    for active in itertools.combinations(allowed, int(total_budget)):
        mask = np.zeros(int(view_count), dtype=np.float32)
        mask[list(active)] = 1.0
        if mask[int(audit_query_index)] > 0.5:
            raise RuntimeError("audit camera leaked into a geometry mask")
        masks.append(mask)
    return masks


def circular_geometry_descriptors(
    angles_degrees: np.ndarray,
    mask: np.ndarray,
    period_degrees: float = 180.0,
) -> dict[str, float]:
    active = np.sort(np.asarray(angles_degrees, dtype=np.float64)[np.asarray(mask) > 0.5])
    if len(active) < 2:
        raise ValueError("geometry descriptors require at least two cameras")
    period = float(period_degrees)
    gaps = np.diff(np.concatenate([active, [active[0] + period]]))
    phase = 2.0 * np.pi * active / period
    first = np.mean(np.exp(1j * phase))
    second = np.mean(np.exp(2j * phase))
    return {
        "active_view_count": int(len(active)),
        "minimum_gap_degrees": float(np.min(gaps)),
        "maximum_gap_degrees": float(np.max(gaps)),
        "mean_gap_degrees": float(np.mean(gaps)),
        "gap_cv": float(np.std(gaps) / (np.mean(gaps) + 1e-12)),
        "first_angular_resultant": float(np.abs(first)),
        "second_angular_resultant": float(np.abs(second)),
        "first_angular_mean_cos": float(np.real(first)),
        "first_angular_mean_sin": float(np.imag(first)),
    }


def operator_geometry_descriptors(
    forward_matrix: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    operator = np.asarray(forward_matrix, dtype=np.float64)[np.asarray(mask) > 0.5]
    matrix = operator.reshape(-1, operator.shape[-1])
    singular = np.linalg.svd(matrix, compute_uv=False)
    nonzero = singular[singular > singular[0] * 1e-10]
    probabilities = nonzero / np.sum(nonzero)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-15))))
    gram = matrix @ matrix.T
    off_diagonal = gram - np.diag(np.diag(gram))
    coherence = float(
        np.max(np.abs(off_diagonal))
        / (np.max(np.diag(gram)) + 1e-12)
    )
    return {
        "operator_row_count": int(matrix.shape[0]),
        "operator_nonzero_rank": int(len(nonzero)),
        "operator_effective_rank": effective_rank,
        "operator_condition_nonzero": float(nonzero[0] / nonzero[-1]),
        "operator_smallest_nonzero_singular": float(nonzero[-1]),
        "operator_largest_singular": float(nonzero[0]),
        "operator_row_coherence": coherence,
    }


def build_geometry_manifest(
    angles_degrees: np.ndarray,
    forward_matrix: np.ndarray,
    total_budget: int,
    audit_query_index: int,
    period_degrees: float,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    masks = enumerate_budget_masks(
        len(angles_degrees), total_budget, audit_query_index
    )
    rows = []
    lookup = {}
    for mask in masks:
        identifier = geometry_id(mask)
        lookup[identifier] = mask
        active_indices = np.flatnonzero(mask > 0.5)
        row = {
            "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
            "geometry_id": identifier,
            "mask_bits": identifier.removeprefix("g_"),
            "active_camera_indices": ",".join(str(value) for value in active_indices),
            "active_angles_degrees": ",".join(
                f"{float(angles_degrees[value]):g}" for value in active_indices
            ),
            "audit_query_index": int(audit_query_index),
            "audit_query_angle_degrees": float(angles_degrees[int(audit_query_index)]),
        }
        row.update(circular_geometry_descriptors(angles_degrees, mask, period_degrees))
        row.update(operator_geometry_descriptors(forward_matrix, mask))
        rows.append(row)
    return rows, lookup


def assign_geometry_partitions(
    rows: list[dict[str, object]],
    reference_geometry_id: str,
    partition_seed: int,
    counts: dict[str, int],
) -> list[dict[str, object]]:
    expected = {"train", "validation", "geometry_ood", "stress"}
    if set(counts) != expected:
        raise ValueError("geometry partition counts use an unexpected schema")
    if sum(int(value) for value in counts.values()) != len(rows):
        raise ValueError("geometry partition counts do not cover the manifest")
    identifiers = {str(row["geometry_id"]) for row in rows}
    if reference_geometry_id not in identifiers:
        raise ValueError("reference geometry is absent from the manifest")

    stress_candidates = [
        row for row in rows if str(row["geometry_id"]) != reference_geometry_id
    ]
    stress_candidates.sort(
        key=lambda row: (
            float(row["maximum_gap_degrees"]),
            float(row["operator_condition_nonzero"]),
            str(row["geometry_id"]),
        ),
        reverse=True,
    )
    stress_ids = {
        str(row["geometry_id"])
        for row in stress_candidates[: int(counts["stress"])]
    }
    remaining = [row for row in rows if str(row["geometry_id"]) not in stress_ids]

    def stable_hash(row: dict[str, object]) -> str:
        payload = f"{int(partition_seed)}:{row['geometry_id']}".encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    remaining.sort(key=lambda row: (stable_hash(row), str(row["geometry_id"])))
    train_count = int(counts["train"])
    reference_position = next(
        index
        for index, row in enumerate(remaining)
        if str(row["geometry_id"]) == reference_geometry_id
    )
    if reference_position >= train_count:
        remaining[train_count - 1], remaining[reference_position] = (
            remaining[reference_position],
            remaining[train_count - 1],
        )
    train_rows = remaining[:train_count]
    train_ids = {str(row["geometry_id"]) for row in train_rows}
    tail = [row for row in remaining if str(row["geometry_id"]) not in train_ids]
    validation_count = int(counts["validation"])
    validation_ids = {
        str(row["geometry_id"]) for row in tail[:validation_count]
    }
    geometry_ood_ids = {
        str(row["geometry_id"])
        for row in tail[
            validation_count : validation_count + int(counts["geometry_ood"])
        ]
    }
    if len(train_ids | validation_ids | geometry_ood_ids | stress_ids) != len(rows):
        raise RuntimeError("geometry partitions overlap or leave gaps")

    output = []
    for row in rows:
        identifier = str(row["geometry_id"])
        if identifier in train_ids:
            partition = "train"
            selection_rule = "seeded_sha256_partition; reference forced into train"
        elif identifier in validation_ids:
            partition = "validation"
            selection_rule = "seeded_sha256_partition"
        elif identifier in geometry_ood_ids:
            partition = "geometry_ood"
            selection_rule = "seeded_sha256_partition"
        else:
            partition = "stress"
            selection_rule = "largest max-gap then nonzero condition; field errors unused"
        output.append(
            {
                **row,
                "partition": partition,
                "partition_seed": int(partition_seed),
                "partition_selection_rule": selection_rule,
                "reference_fixed_k6_geometry": identifier == reference_geometry_id,
            }
        )
    return sorted(output, key=lambda row: str(row["geometry_id"]))


def geometry_entropy_bits(view_masks: np.ndarray) -> float:
    identifiers = [geometry_id(mask) for mask in np.asarray(view_masks)]
    counts = Counter(identifiers)
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(identifiers)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def mean_pairwise_jaccard_distance(masks: list[np.ndarray]) -> float:
    distances = []
    for left, right in itertools.combinations(masks, 2):
        left_active = np.asarray(left) > 0.5
        right_active = np.asarray(right) > 0.5
        union = np.sum(left_active | right_active)
        distances.append(1.0 - np.sum(left_active & right_active) / max(union, 1))
    return float(np.mean(distances)) if distances else 0.0


def deterministic_noisy_observations(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    total_budget: int,
) -> np.ndarray:
    observations = []
    for index in indices:
        clean = np.asarray(data["clean_observation"][int(index)], dtype=np.float32)
        rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)) + 1e-8)
        rng = np.random.default_rng(
            int(data["sample_seed"][int(index)]) + int(total_budget) * 10_007 + 3_109
        )
        noisy = clean + rng.normal(
            scale=float(data["noise_level"][int(index)]) * rms,
            size=clean.shape,
        ).astype(np.float32)
        observations.append(noisy)
    return np.stack(observations)


def _projection_relative(
    prediction: np.ndarray,
    clean_observation: np.ndarray,
    forward_matrix: np.ndarray,
    mask: np.ndarray,
) -> float:
    projected = np.einsum(
        "dp,vnp->dvn",
        np.asarray(prediction, dtype=np.float64).reshape(prediction.shape[0], -1),
        np.asarray(forward_matrix, dtype=np.float64),
        optimize=True,
    )
    selected = np.asarray(mask) > 0.5
    error = projected[:, selected] - np.asarray(clean_observation, dtype=np.float64)[:, selected]
    reference = np.asarray(clean_observation, dtype=np.float64)[:, selected]
    return float(np.linalg.norm(error) / (np.linalg.norm(reference) + 1e-12))


def evaluate_geometry_ridge(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    manifest_rows: list[dict[str, object]],
    masks: dict[str, np.ndarray],
    ridge_relative: float,
    total_budget: int,
    audit_query_index: int,
) -> list[dict[str, object]]:
    noisy = deterministic_noisy_observations(data, indices, total_budget)
    split_names = [str(value) for value in data["split_names"].tolist()]
    audit_mask = np.zeros(len(data["angles"]), dtype=np.float32)
    audit_mask[int(audit_query_index)] = 1.0
    rows = []
    for manifest in manifest_rows:
        identifier = str(manifest["geometry_id"])
        mask = masks[identifier]
        for local_index, source_index in enumerate(indices):
            prediction = ridge_reconstruct(
                noisy[local_index],
                data["forward_matrix"],
                mask,
                float(ridge_relative),
                data["support"],
            )
            target = data["field"][int(source_index)]
            rows.append(
                {
                    "geometry_id": identifier,
                    "partition": str(manifest["partition"]),
                    "source_index": int(source_index),
                    "sample_seed": int(data["sample_seed"][int(source_index)]),
                    "source_split": split_names[int(data["split_id"][int(source_index)])],
                    "family_id": int(data["family_id"][int(source_index)]),
                    "noise_level": float(data["noise_level"][int(source_index)]),
                    "field_rel_l2": relative_field_error(prediction, target),
                    "observed_reprojection_rel_l2": _projection_relative(
                        prediction,
                        data["clean_observation"][int(source_index)],
                        data["forward_matrix"],
                        mask,
                    ),
                    "audit_reprojection_rel_l2": _projection_relative(
                        prediction,
                        data["clean_observation"][int(source_index)],
                        data["forward_matrix"],
                        audit_mask,
                    ),
                }
            )
    return rows


def summarize_geometry_errors(
    sample_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["geometry_id"])].append(row)
    manifest = {str(row["geometry_id"]): row for row in manifest_rows}
    output = []
    for identifier, rows in sorted(grouped.items()):
        field = np.asarray([float(row["field_rel_l2"]) for row in rows])
        audit = np.asarray([float(row["audit_reprojection_rel_l2"]) for row in rows])
        output.append(
            {
                "geometry_id": identifier,
                "partition": str(manifest[identifier]["partition"]),
                "reference_fixed_k6_geometry": bool(
                    manifest[identifier]["reference_fixed_k6_geometry"]
                ),
                "field_count": len(rows),
                "mean_field_rel_l2": float(np.mean(field)),
                "median_field_rel_l2": float(np.median(field)),
                "p90_field_rel_l2": float(np.quantile(field, 0.90)),
                "mean_audit_reprojection_rel_l2": float(np.mean(audit)),
                "maximum_gap_degrees": float(
                    manifest[identifier]["maximum_gap_degrees"]
                ),
                "gap_cv": float(manifest[identifier]["gap_cv"]),
                "first_angular_resultant": float(
                    manifest[identifier]["first_angular_resultant"]
                ),
                "operator_condition_nonzero": float(
                    manifest[identifier]["operator_condition_nonzero"]
                ),
                "operator_effective_rank": float(
                    manifest[identifier]["operator_effective_rank"]
                ),
            }
        )
    return output


def summarize_field_geometry_spread(
    sample_rows: list[dict[str, object]],
    reference_geometry_id: str,
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[int(row["source_index"])].append(row)
    output = []
    for source_index, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["field_rel_l2"]))
        best = ordered[0]
        worst = ordered[-1]
        reference = next(
            row for row in rows if str(row["geometry_id"]) == reference_geometry_id
        )
        best_error = float(best["field_rel_l2"])
        worst_error = float(worst["field_rel_l2"])
        reference_error = float(reference["field_rel_l2"])
        reference_rank = 1 + sum(
            float(row["field_rel_l2"]) < reference_error for row in rows
        )
        output.append(
            {
                "source_index": source_index,
                "sample_seed": int(best["sample_seed"]),
                "family_id": int(best["family_id"]),
                "noise_level": float(best["noise_level"]),
                "best_geometry_id": str(best["geometry_id"]),
                "best_field_rel_l2": best_error,
                "worst_geometry_id": str(worst["geometry_id"]),
                "worst_field_rel_l2": worst_error,
                "reference_geometry_id": reference_geometry_id,
                "reference_field_rel_l2": reference_error,
                "reference_geometry_rank_of_28": int(reference_rank),
                "best_to_worst_spread_pct": 100.0
                * (worst_error - best_error)
                / (best_error + 1e-12),
                "reference_regret_vs_best_pct": 100.0
                * (reference_error - best_error)
                / (best_error + 1e-12),
            }
        )
    return output


def partition_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["partition"]) for row in rows)
    return {key: int(counts[key]) for key in sorted(counts)}


def reference_mask_id(reference_mask: list[int] | np.ndarray) -> str:
    return geometry_id(np.asarray(reference_mask, dtype=np.float32))


def expected_mask_count(view_count: int, budget: int, audit_reserved: int = 1) -> int:
    return math.comb(int(view_count) - int(audit_reserved), int(budget))


def _stable_assignment_key(seed: int, sample_seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}:{int(sample_seed)}".encode("ascii")).hexdigest()


def balanced_source_geometry_assignment(
    base_data: dict[str, np.ndarray],
    manifest_rows: list[dict[str, object]],
    split_partition_map: dict[str, str],
    assignment_seed: int,
) -> list[dict[str, object]]:
    """Assign one geometry per source field, balanced within every source split."""
    split_names = [str(value) for value in base_data["split_names"].tolist()]
    if set(split_partition_map) != set(split_names):
        raise ValueError("source split to geometry partition map is incomplete")
    by_partition: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        by_partition[str(row["partition"])].append(str(row["geometry_id"]))
    for identifiers in by_partition.values():
        identifiers.sort()

    output = []
    for split_id, split_name in enumerate(split_names):
        partition = str(split_partition_map[split_name])
        identifiers = by_partition.get(partition, [])
        if not identifiers:
            raise ValueError(f"geometry partition {partition!r} is empty")
        source_indices = np.flatnonzero(base_data["split_id"] == split_id).tolist()
        source_indices.sort(
            key=lambda index: (
                _stable_assignment_key(
                    int(assignment_seed), int(base_data["sample_seed"][index])
                ),
                int(index),
            )
        )
        offset = int(
            hashlib.sha256(
                f"{int(assignment_seed)}:{split_name}".encode("utf-8")
            ).hexdigest()[:8],
            16,
        ) % len(identifiers)
        for position, source_index in enumerate(source_indices):
            geometry_identifier = identifiers[(position + offset) % len(identifiers)]
            output.append(
                {
                    "source_index": int(source_index),
                    "sample_seed": int(base_data["sample_seed"][source_index]),
                    "source_split": split_name,
                    "source_split_id": int(split_id),
                    "geometry_id": geometry_identifier,
                    "geometry_partition": partition,
                    "assignment_seed": int(assignment_seed),
                    "assignment_rule": "sha256 source ordering + balanced cyclic geometry",
                }
            )
    return sorted(output, key=lambda row: int(row["source_index"]))


def build_variable_geometry_operator_data(
    base_data: dict[str, np.ndarray],
    manifest_rows: list[dict[str, object]],
    masks: dict[str, np.ndarray],
    split_partition_map: dict[str, str],
    assignment_seed: int,
    ridge_relative: float,
    total_budget: int,
    audit_query_index: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Build a one-field/one-geometry GC-SRO dataset with shared field noise."""
    assignments = balanced_source_geometry_assignment(
        base_data, manifest_rows, split_partition_map, assignment_seed
    )
    source_indices = np.asarray(
        [int(row["source_index"]) for row in assignments], dtype=np.int64
    )
    if not np.array_equal(source_indices, np.arange(len(base_data["field"]))):
        raise ValueError("variable geometry builder requires exactly one row per source")
    view_masks = np.stack(
        [masks[str(row["geometry_id"])] for row in assignments]
    ).astype(np.float32)
    if np.any(view_masks[:, int(audit_query_index)] > 0.5):
        raise RuntimeError("audit camera leaked into variable geometry inputs")
    if not np.all(view_masks.sum(axis=1) == int(total_budget)):
        raise RuntimeError("a variable geometry row violates the camera budget")

    noisy = deterministic_noisy_observations(
        base_data, source_indices, int(total_budget)
    ).astype(np.float32)
    support = np.asarray(base_data["support"], dtype=np.float32)
    depth, height, width = base_data["field"].shape[1:]
    inverse_cache = {
        identifier: ridge_reconstruction_matrix(
            base_data["forward_matrix"], mask, float(ridge_relative)
        )
        for identifier, mask in masks.items()
    }
    ridge_fields = []
    for index, row in enumerate(assignments):
        identifier = str(row["geometry_id"])
        mask = masks[identifier]
        selected = np.flatnonzero(mask > 0.5)
        projected = noisy[index][:, selected].reshape(depth, -1)
        volume = (projected @ inverse_cache[identifier].T).reshape(
            depth, height, width
        )
        ridge_fields.append(np.clip(volume, 0.0, None) * support)
    ridge_fields = np.asarray(ridge_fields, dtype=np.float32)

    zz, yy, xx = make_grid_3d(width, depth)
    count = len(assignments)
    channels = [ridge_fields[:, None]]
    channels.append(np.broadcast_to(support, (count, depth, height, width))[:, None])
    view_fraction = np.full(
        (count, 1, depth, height, width),
        float(total_budget) / len(base_data["angles"]),
        dtype=np.float32,
    )
    channels.append(view_fraction)
    for camera_index in range(view_masks.shape[1]):
        active = view_masks[:, camera_index, None, None, None, None]
        channels.append(np.broadcast_to(active, (count, 1, depth, height, width)))
    for grid in (zz, yy, xx):
        channels.append(
            np.broadcast_to(grid.astype(np.float32), (count, depth, height, width))[
                :, None
            ]
        )
    input_names = [
        "validation_tuned_ridge_lift",
        "support",
        "view_fraction",
        *[f"camera_{index}_active" for index in range(view_masks.shape[1])],
        "z",
        "y",
        "x",
    ]
    packed = {
        "inputs": np.concatenate(channels, axis=1).astype(np.float32),
        "field": np.asarray(base_data["field"], dtype=np.float32),
        "lift": ridge_fields,
        "lift_raw": ridge_fields.copy(),
        "observation": noisy,
        "clean_observation": np.asarray(
            base_data["clean_observation"], dtype=np.float32
        ),
        "view_mask": view_masks,
        "view_count": np.full(count, int(total_budget), dtype=np.int64),
        "total_budget": np.full(count, int(total_budget), dtype=np.int64),
        "noise_level": np.asarray(base_data["noise_level"], dtype=np.float64),
        "family_id": np.asarray(base_data["family_id"], dtype=np.int64),
        "split_id": np.asarray(base_data["split_id"], dtype=np.int64),
        "source_index": source_indices,
        "sample_seed": np.asarray(base_data["sample_seed"], dtype=np.int64),
        "angles": np.asarray(base_data["angles"], dtype=np.float32),
        "forward_matrix": np.asarray(base_data["forward_matrix"], dtype=np.float32),
        "support": support,
        "split_names": np.asarray(base_data["split_names"]),
        "input_channel_names": np.asarray(input_names),
        "input_schema_version": np.asarray(2, dtype=np.int64),
        "audit_query_index": np.asarray(audit_query_index, dtype=np.int64),
        "geometry_id": np.asarray([row["geometry_id"] for row in assignments]),
        "geometry_partition": np.asarray(
            [row["geometry_partition"] for row in assignments]
        ),
        "geometry_assignment_seed": np.asarray(assignment_seed, dtype=np.int64),
        "shared_full_view_noise": np.asarray(True),
    }
    packed = append_ray_view_channels(packed)
    return packed, assignments
