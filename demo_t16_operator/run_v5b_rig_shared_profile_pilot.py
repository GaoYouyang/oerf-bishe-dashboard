#!/usr/bin/env python3
"""Run the development-only v5b rig-shared profile-identifiability pilot.

This is not a confirmatory lock.  It uses an explicit small linear ridge model
to test whether sharing one finite-aperture parameter across multiple reaction
fields reduces the morphology/operator confounding observed in v5a.  The truth
and reconstruction operators use different quadrature densities, but still
share one prescribed weak-deflection model family; this remains an inverse-
crime-adjacent mechanism test rather than real BOST evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .finite_aperture_bost import build_finite_aperture_operator_bank
    from .independent_reaction_bost import (
        correlated_camera_noise,
        make_reaction_field,
        reaction_support,
    )
    from .rig_shared_profile import (
        ProfileSelection,
        RidgeProfileFit,
        apply_metadata_prior,
        fit_support_ridge,
        operator_radius_derivative,
        profile_fisher_scalar,
        profile_shared_radius,
        whitened_normal_mean_diagonal,
        whitened_view_rms,
    )
except ImportError:
    from finite_aperture_bost import build_finite_aperture_operator_bank
    from independent_reaction_bost import (
        correlated_camera_noise,
        make_reaction_field,
        reaction_support,
    )
    from rig_shared_profile import (
        ProfileSelection,
        RidgeProfileFit,
        apply_metadata_prior,
        fit_support_ridge,
        operator_radius_derivative,
        profile_fisher_scalar,
        profile_shared_radius,
        whitened_normal_mean_diagonal,
        whitened_view_rms,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5b_rig_shared_profile_pilot.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5b_rig_shared_profile_pilot"


@dataclass(frozen=True)
class DevelopmentBlock:
    rig_id: str
    block_id: str
    true_radius: float
    metadata_radius: float
    families: tuple[str, ...]
    fields: tuple[np.ndarray, ...]
    clean_observations: tuple[np.ndarray, ...]
    observations: tuple[np.ndarray, ...]
    noise_std: tuple[np.ndarray, ...]
    reconstruction_bank: np.ndarray
    truth_operator: np.ndarray
    inner_views: tuple[int, ...]
    outer_views: tuple[int, ...]
    audit_views: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def signal_rms_from_fit_views(clean: np.ndarray, fit_views: tuple[int, ...]) -> float:
    visible = np.zeros(clean.shape[1], dtype=bool)
    visible[list(fit_views)] = True
    if not np.any(visible):
        raise ValueError("noise scaling needs at least one fit camera")
    return float(np.sqrt(np.mean(clean[:, visible] ** 2)) + 1e-8)


def relative_l2(field: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(field) - np.asarray(truth))
        / max(np.linalg.norm(np.asarray(truth)), 1e-12)
    )


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def family_eta_squared(rows: list[dict[str, Any]], value_key: str) -> float | None:
    values = np.asarray([float(row[value_key]) for row in rows], dtype=float)
    labels = np.asarray([str(row["family"]) for row in rows])
    total = float(np.sum((values - np.mean(values)) ** 2))
    if total <= 1e-15:
        return None
    between = 0.0
    for label in sorted(set(labels)):
        group = values[labels == label]
        between += len(group) * float((np.mean(group) - np.mean(values)) ** 2)
    return float(np.clip(between / total, 0.0, 1.0))


def safe_percent_change(candidate: float, baseline: float) -> float:
    return float(100.0 * (candidate - baseline) / max(abs(baseline), 1e-12))


def _balanced_field_specs(config: dict[str, Any]) -> list[tuple[str, int, float]]:
    families = [str(value) for value in config["families"]]
    levels = [float(value) for value in config["relative_noise"]]
    specs: list[tuple[str, int, float]] = []
    for family_index, family in enumerate(families):
        for replicate in range(int(config["fields_per_family"])):
            level = levels[(family_index + replicate) % len(levels)]
            specs.append((family, replicate, level))
    return specs


def support_mask_from_config(config: dict[str, Any]) -> np.ndarray:
    """Threshold the soft reaction support and reject empty/full masks."""

    values = reaction_support(int(config["grid_size"]), int(config["depth"]))
    threshold = float(config["support_threshold"])
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("support_threshold must lie strictly between zero and one")
    mask = np.asarray(values >= threshold, dtype=bool)
    active = int(np.sum(mask))
    if active == 0 or active == mask.size:
        raise ValueError("support_threshold must produce a nonempty, non-full mask")
    return mask


def validate_forward_model_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Require a source-matched deterministic renderer-convergence gate."""

    relative = Path(str(config["forward_model_gate_report"]))
    report_path = relative if relative.is_absolute() else ROOT / relative
    if not report_path.is_file():
        raise FileNotFoundError(f"forward-model gate report is missing: {report_path}")
    report = read_json(report_path)
    operator_hash = sha256((ROOT / "finite_aperture_bost.py").resolve())
    if report.get("source_hashes", {}).get("operator") != operator_hash:
        raise ValueError("forward-model gate operator hash does not match current source")
    safe = {
        (
            int(row["reconstruction_path_samples"]),
            int(row["reconstruction_aperture_samples"]),
        )
        for row in report.get("safe_settings", [])
    }
    requested = {
        (
            int(rig["reconstruction_path_samples"]),
            int(rig["reconstruction_aperture_samples"]),
        )
        for rig in config["rigs"]
    }
    if not requested or not requested.issubset(safe):
        raise ValueError("pilot renderer setting did not pass the forward-model gate")
    return {
        "report": str(relative),
        "report_sha256": sha256(report_path),
        "operator_sha256": operator_hash,
        "safe_settings_used": [
            {"path_samples": path, "aperture_samples": aperture}
            for path, aperture in sorted(requested)
        ],
        "gate_claim_status": report.get("claim_status"),
        "gate_is_deterministic_numerical_only": True,
    }


def camera_partition(rig: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    views = len(rig["angles_degrees"])
    inner = tuple(int(value) for value in rig["inner_camera_indices"])
    outer = tuple(int(value) for value in rig["outer_camera_indices"])
    audit = (int(rig["audit_camera_index"]),)
    groups = [set(inner), set(outer), set(audit)]
    if any(not group for group in groups):
        raise ValueError("inner, outer and audit camera groups must be non-empty")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("inner, outer and audit camera groups must be disjoint")
    if set.union(*groups) != set(range(views)):
        raise ValueError("camera partition must cover the complete rig exactly once")
    return inner, outer, audit


def build_development_blocks(config: dict[str, Any]) -> tuple[list[DevelopmentBlock], list[dict[str, Any]]]:
    n, depth = int(config["grid_size"]), int(config["depth"])
    candidates = np.asarray(config["candidate_aperture_radii"], dtype=float)
    truth_radii = np.asarray(config["true_aperture_radii"], dtype=float)
    camera_factors = np.asarray(config["camera_noise_factors"], dtype=np.float64)
    field_specs = _balanced_field_specs(config)
    blocks: list[DevelopmentBlock] = []
    manifest_rows: list[dict[str, Any]] = []

    for rig_index, rig in enumerate(config["rigs"]):
        rig_id = str(rig["id"])
        angles = np.asarray(rig["angles_degrees"], dtype=np.float64)
        inner_views, outer_views, audit_views = camera_partition(rig)
        if len(camera_factors) != len(angles):
            raise ValueError("camera noise factors and rig views disagree")
        common = {
            "cone_u": float(rig["cone_u"]),
            "cone_z": float(rig["cone_z"]),
            "bend": float(rig["bend"]),
        }
        reconstruction_bank = build_finite_aperture_operator_bank(
            n,
            depth,
            angles,
            candidates,
            aperture_samples=int(rig["reconstruction_aperture_samples"]),
            path_samples=int(rig["reconstruction_path_samples"]),
            **common,
        )
        truth_bank = build_finite_aperture_operator_bank(
            n,
            depth,
            angles,
            truth_radii,
            aperture_samples=int(rig["truth_aperture_samples"]),
            path_samples=int(rig["truth_path_samples"]),
            **common,
        )
        field_rng = np.random.default_rng(int(config["data_seed"]) + rig_index * 100_000)
        fields = tuple(
            make_reaction_field(family, n, depth, field_rng).astype(np.float64)
            for family, _, _ in field_specs
        )
        families = tuple(value[0] for value in field_specs)
        metadata_bias = float(config["metadata_bias_by_rig"][rig_index])

        for radius_index, true_radius in enumerate(truth_radii):
            clean_observations: list[np.ndarray] = []
            observations: list[np.ndarray] = []
            sigmas: list[np.ndarray] = []
            for field_index, (field, spec) in enumerate(zip(fields, field_specs, strict=True)):
                clean = np.einsum(
                    "dvnp,p->dvn", truth_bank[radius_index], field.reshape(-1), optimize=True
                )
                clean_observations.append(clean.astype(np.float64))
                signal_rms = signal_rms_from_fit_views(clean, inner_views)
                sigma = float(spec[2]) * signal_rms * camera_factors
                paired_noise_seed = (
                    int(config["data_seed"])
                    + rig_index * 1_000_000
                    + field_index * 10_000
                    + 733
                )
                noise = correlated_camera_noise(
                    clean,
                    sigma,
                    np.random.default_rng(paired_noise_seed),
                    correlation_fraction=float(config["correlation_fraction"]),
                    signal_fraction=float(config["signal_fraction"]),
                )
                observations.append((clean + noise).astype(np.float64))
                sigmas.append(sigma.astype(np.float64))
                manifest_rows.append(
                    {
                        "rig_id": rig_id,
                        "block_id": f"{rig_id}:radius={true_radius:.5f}",
                        "true_aperture_radius": float(true_radius),
                        "field_index": field_index,
                        "family": spec[0],
                        "family_replicate": int(spec[1]),
                        "relative_noise": float(spec[2]),
                        "paired_noise_seed": paired_noise_seed,
                        "noise_scale_uses_only_primary_inner_cameras": True,
                    }
                )
            metadata_radius = float(
                np.clip(true_radius + metadata_bias, candidates[0], candidates[-1])
            )
            blocks.append(
                DevelopmentBlock(
                    rig_id=rig_id,
                    block_id=f"{rig_id}:radius={true_radius:.5f}",
                    true_radius=float(true_radius),
                    metadata_radius=metadata_radius,
                    families=families,
                    fields=fields,
                    clean_observations=tuple(clean_observations),
                    observations=tuple(observations),
                    noise_std=tuple(sigmas),
                    reconstruction_bank=reconstruction_bank,
                    truth_operator=truth_bank[radius_index],
                    inner_views=inner_views,
                    outer_views=outer_views,
                    audit_views=audit_views,
                )
            )
    return blocks, manifest_rows


def oracle_field_scores(
    block: DevelopmentBlock,
    *,
    clean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Truth-only headroom: can fixed fields rank the candidate operator bank?"""

    active = np.zeros(block.reconstruction_bank.shape[2], dtype=bool)
    active[list(block.inner_views)] = True
    measurement_mask = np.broadcast_to(
        active[None, :, None], block.observations[0].shape
    )
    scores: list[float] = []
    observations = block.clean_observations if clean else block.observations
    for operator in block.reconstruction_bank:
        score = 0.0
        for field, observation, sigma in zip(
            block.fields, observations, block.noise_std, strict=True
        ):
            prediction = np.einsum(
                "dvnp,p->dvn", operator, field.reshape(-1), optimize=True
            )
            expanded_sigma = np.broadcast_to(
                np.asarray(sigma)[None, :, None], observation.shape
            )
            residual = (observation - prediction) / expanded_sigma
            score += 0.5 * float(np.sum(residual[measurement_mask] ** 2))
        scores.append(score)
    distances = []
    truth_rows = block.truth_operator[:, active].reshape(-1, block.truth_operator.shape[-1])
    for operator in block.reconstruction_bank:
        candidate_rows = operator[:, active].reshape(-1, operator.shape[-1])
        distances.append(
            float(
                np.linalg.norm(candidate_rows - truth_rows)
                / max(np.linalg.norm(truth_rows), 1e-12)
            )
        )
    return np.asarray(scores), np.asarray(distances)


def _method_fit(
    selection: ProfileSelection, method_indices: int | list[int], sample: int
) -> RidgeProfileFit:
    index = method_indices[sample] if isinstance(method_indices, list) else method_indices
    return selection.candidates[int(index)].fits[sample]


def evaluate_blocks(
    config: dict[str, Any], blocks: list[DevelopmentBlock]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    radii = np.asarray(config["candidate_aperture_radii"], dtype=float)
    support = support_mask_from_config(config)
    ridge_lambda = float(config["ridge_lambda"])
    metadata_sigma = float(config["metadata_sigma"])
    metadata_weight = float(config["metadata_weight"])
    minimum_fraction = float(
        config["acceptance"]["minimum_median_profile_fraction"]
    )
    minimum_outer = float(
        config["acceptance"]["minimum_outer_improvement_percent"]
    )
    sample_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []

    for block in blocks:
        clean_truth_scores, operator_distances = oracle_field_scores(block, clean=True)
        noisy_truth_scores, _ = oracle_field_scores(block, clean=False)
        oracle_field_index = int(np.argmin(clean_truth_scores))
        noisy_oracle_field_index = int(np.argmin(noisy_truth_scores))
        closest_matrix_index = int(np.argmin(operator_distances))
        shared = profile_shared_radius(
            block.reconstruction_bank,
            radii,
            block.observations,
            block.noise_std,
            block.inner_views,
            support,
            ridge_lambda,
        )
        rigcal = apply_metadata_prior(
            shared,
            block.metadata_radius,
            metadata_sigma,
            metadata_weight,
        )
        per_scene_indices = [
            int(
                np.argmin(
                    [candidate.fits[sample].reduced_objective for candidate in shared.candidates]
                )
            )
            for sample in range(len(block.fields))
        ]
        sse_only_shared_index = int(
            np.argmin(
                [
                    np.sum([0.5 * fit.whitened_sse for fit in candidate.fits])
                    for candidate in shared.candidates
                ]
            )
        )
        metadata_index = nearest_index(radii, block.metadata_radius)
        oracle_nearest_index = nearest_index(radii, block.true_radius)
        selected_index = rigcal.selected_index
        derivative = operator_radius_derivative(
            block.reconstruction_bank, radii, selected_index
        )
        fisher = [
            profile_fisher_scalar(
                block.reconstruction_bank[selected_index],
                derivative,
                fit,
                observation,
                sigma,
                block.inner_views,
                support,
                ridge_lambda,
            )
            for fit, observation, sigma in zip(
                rigcal.selected.fits,
                block.observations,
                block.noise_std,
                strict=True,
            )
        ]
        median_fraction = float(np.median([value.retained_fraction for value in fisher]))
        boundary = selected_index in (0, len(radii) - 1)
        metadata_conflict = (
            abs(float(radii[selected_index]) - block.metadata_radius) > 2.0 * metadata_sigma
        )
        block_identifiable = bool(
            median_fraction >= minimum_fraction and not boundary and not metadata_conflict
        )

        for candidate in rigcal.candidates:
            whitened_sse_score = float(
                np.sum([0.5 * fit.whitened_sse for fit in candidate.fits])
            )
            ridge_penalty_score = float(
                np.sum([0.5 * fit.regularization for fit in candidate.fits])
            )
            profile_rows.append(
                {
                    "rig_id": block.rig_id,
                    "block_id": block.block_id,
                    "true_aperture_radius": block.true_radius,
                    "metadata_aperture_radius": block.metadata_radius,
                    "candidate_aperture_radius": candidate.radius,
                    "whitened_sse_score": whitened_sse_score,
                    "ridge_penalty_score": ridge_penalty_score,
                    "data_score": candidate.data_score,
                    "metadata_penalty": candidate.metadata_penalty,
                    "total_score": candidate.total_score,
                    "selected_shared_without_metadata": candidate.index
                    == shared.selected_index,
                    "selected_shared_sse_only": candidate.index
                    == sse_only_shared_index,
                    "selected_rigcal": candidate.index == selected_index,
                }
            )

        block_methods: dict[str, int | list[int]] = {
            "pinhole": 0,
            "metadata_nearest": metadata_index,
            "per_scene_profile": per_scene_indices,
            "rig_shared_sse_only": sse_only_shared_index,
            "rig_shared_profile": shared.selected_index,
            "rigcal_profile": selected_index,
            "oracle_nearest_bank": oracle_nearest_index,
        }
        for method, indices in block_methods.items():
            estimates = (
                [float(radii[index]) for index in indices]
                if isinstance(indices, list)
                else [float(radii[int(indices)])] * len(block.fields)
            )
            calibration_rows.append(
                {
                    "rig_id": block.rig_id,
                    "block_id": block.block_id,
                    "method": method,
                    "true_aperture_radius": block.true_radius,
                    "estimated_aperture_radius_mean": float(np.mean(estimates)),
                    "estimated_aperture_radius_mae": float(
                        np.mean(np.abs(np.asarray(estimates) - block.true_radius))
                    ),
                    "nearest_bank_match_rate": float(
                        np.mean(np.asarray(estimates) == radii[oracle_nearest_index])
                    ),
                    "boundary_hit_rate": float(
                        np.mean(np.isin(estimates, [radii[0], radii[-1]]))
                    ),
                }
            )
        for method, diagnostic_index in (
            ("oracle_field_residual", oracle_field_index),
            ("noisy_oracle_field_residual", noisy_oracle_field_index),
            ("closest_truth_operator_matrix", closest_matrix_index),
        ):
            calibration_rows.append(
                {
                    "rig_id": block.rig_id,
                    "block_id": block.block_id,
                    "method": method,
                    "true_aperture_radius": block.true_radius,
                    "estimated_aperture_radius_mean": float(radii[diagnostic_index]),
                    "estimated_aperture_radius_mae": float(
                        abs(radii[diagnostic_index] - block.true_radius)
                    ),
                    "nearest_bank_match_rate": float(
                        diagnostic_index == oracle_nearest_index
                    ),
                    "boundary_hit_rate": float(
                        diagnostic_index in (0, len(radii) - 1)
                    ),
                }
            )

        for sample, (family, truth, observation, sigma) in enumerate(
            zip(
                block.families,
                block.fields,
                block.observations,
                block.noise_std,
                strict=True,
            )
        ):
            method_values: dict[str, dict[str, float | RidgeProfileFit]] = {}
            for method, indices in block_methods.items():
                fit = _method_fit(shared, indices, sample)
                index = indices[sample] if isinstance(indices, list) else int(indices)
                operator = block.reconstruction_bank[int(index)]
                method_values[method] = {
                    "fit": fit,
                    "estimated_radius": float(radii[int(index)]),
                    "relative_l2": relative_l2(fit.field, truth),
                    "outer_rms": whitened_view_rms(
                        operator, fit.field, observation, sigma, block.outer_views
                    ),
                    "audit_rms": whitened_view_rms(
                        operator, fit.field, observation, sigma, block.audit_views
                    ),
                }

            true_fit = fit_support_ridge(
                block.truth_operator,
                observation,
                sigma,
                block.inner_views,
                support,
                ridge_lambda,
            )
            method_values["oracle_true_operator"] = {
                "fit": true_fit,
                "estimated_radius": block.true_radius,
                "relative_l2": relative_l2(true_fit.field, truth),
                "outer_rms": whitened_view_rms(
                    block.truth_operator,
                    true_fit.field,
                    observation,
                    sigma,
                    block.outer_views,
                ),
                "audit_rms": whitened_view_rms(
                    block.truth_operator,
                    true_fit.field,
                    observation,
                    sigma,
                    block.audit_views,
                ),
            }
            candidate = method_values["rigcal_profile"]
            fallback = method_values["metadata_nearest"]
            outer_improvement = float(
                100.0
                * (float(fallback["outer_rms"]) - float(candidate["outer_rms"]))
                / max(float(fallback["outer_rms"]), 1e-12)
            )
            accepted = bool(block_identifiable and outer_improvement > minimum_outer)
            deployed = candidate if accepted else fallback
            selected_gain = float(
                100.0
                * (float(fallback["relative_l2"]) - float(deployed["relative_l2"]))
                / max(float(fallback["relative_l2"]), 1e-12)
            )
            raw_gain = float(
                100.0
                * (float(fallback["relative_l2"]) - float(candidate["relative_l2"]))
                / max(float(fallback["relative_l2"]), 1e-12)
            )
            row: dict[str, Any] = {
                "rig_id": block.rig_id,
                "block_id": block.block_id,
                "sample_index_in_block": sample,
                "family": family,
                "true_aperture_radius": block.true_radius,
                "metadata_aperture_radius": block.metadata_radius,
                "per_scene_estimated_radius": method_values["per_scene_profile"][
                    "estimated_radius"
                ],
                "shared_estimated_radius": method_values["rig_shared_profile"][
                    "estimated_radius"
                ],
                "rigcal_estimated_radius": candidate["estimated_radius"],
                "profile_information": fisher[sample].profile_information,
                "profile_retained_fraction": fisher[sample].retained_fraction,
                "profile_approximate_standard_error": fisher[
                    sample
                ].approximate_standard_error,
                "block_median_profile_fraction": median_fraction,
                "block_identifiable": block_identifiable,
                "outer_improvement_percent": outer_improvement,
                "accepted": accepted,
                "raw_rigcal_gain_percent": raw_gain,
                "selected_gain_percent": selected_gain,
                "fallback_relative_l2": fallback["relative_l2"],
                "candidate_relative_l2": candidate["relative_l2"],
                "selected_relative_l2": deployed["relative_l2"],
                "fallback_outer_rms": fallback["outer_rms"],
                "candidate_outer_rms": candidate["outer_rms"],
                "fallback_audit_rms": fallback["audit_rms"],
                "candidate_audit_rms": candidate["audit_rms"],
                "selected_audit_rms": deployed["audit_rms"],
                "selected_audit_change_percent": safe_percent_change(
                    float(deployed["audit_rms"]), float(fallback["audit_rms"])
                ),
            }
            for method, values in method_values.items():
                row[f"{method}_relative_l2"] = values["relative_l2"]
                row[f"{method}_outer_rms"] = values["outer_rms"]
                row[f"{method}_audit_rms"] = values["audit_rms"]
            sample_rows.append(row)

        block_rows.append(
            {
                "rig_id": block.rig_id,
                "block_id": block.block_id,
                "true_aperture_radius": block.true_radius,
                "metadata_aperture_radius": block.metadata_radius,
                "metadata_nearest_radius": float(radii[metadata_index]),
                "per_scene_radius_mean": float(np.mean(radii[per_scene_indices])),
                "rig_shared_sse_only_radius": float(radii[sse_only_shared_index]),
                "shared_profile_radius": float(radii[shared.selected_index]),
                "rigcal_profile_radius": float(radii[selected_index]),
                "oracle_nearest_radius": float(radii[oracle_nearest_index]),
                "oracle_field_residual_radius": float(radii[oracle_field_index]),
                "noisy_oracle_field_residual_radius": float(
                    radii[noisy_oracle_field_index]
                ),
                "closest_truth_operator_matrix_radius": float(
                    radii[closest_matrix_index]
                ),
                "oracle_field_score_nearest_over_best": float(
                    clean_truth_scores[oracle_nearest_index]
                    / max(clean_truth_scores[oracle_field_index], 1e-15)
                ),
                "operator_distance_nearest_over_best": float(
                    operator_distances[oracle_nearest_index]
                    / max(operator_distances[closest_matrix_index], 1e-15)
                ),
                "median_profile_fraction": median_fraction,
                "boundary": boundary,
                "metadata_conflict": metadata_conflict,
                "block_identifiable": block_identifiable,
            }
        )

    method_names = [
        "pinhole",
        "metadata_nearest",
        "per_scene_profile",
        "rig_shared_sse_only",
        "rig_shared_profile",
        "rigcal_profile",
        "oracle_nearest_bank",
        "oracle_true_operator",
    ]
    method_summary: dict[str, dict[str, float]] = {}
    for method in method_names:
        errors = np.asarray([row[f"{method}_relative_l2"] for row in sample_rows], dtype=float)
        outer = np.asarray([row[f"{method}_outer_rms"] for row in sample_rows], dtype=float)
        audit = np.asarray([row[f"{method}_audit_rms"] for row in sample_rows], dtype=float)
        method_summary[method] = {
            "mean_relative_l2": float(np.mean(errors)),
            "p90_relative_l2": float(np.quantile(errors, 0.90)),
            "mean_outer_rms": float(np.mean(outer)),
            "mean_audit_rms": float(np.mean(audit)),
        }

    calibration_summary: dict[str, dict[str, float]] = {}
    for method in [
        "pinhole",
        "metadata_nearest",
        "per_scene_profile",
        "rig_shared_sse_only",
        "rig_shared_profile",
        "rigcal_profile",
        "oracle_nearest_bank",
        "oracle_field_residual",
        "noisy_oracle_field_residual",
        "closest_truth_operator_matrix",
    ]:
        rows = [row for row in calibration_rows if row["method"] == method]
        calibration_summary[method] = {
            "mean_aperture_mae": float(
                np.mean([row["estimated_aperture_radius_mae"] for row in rows])
            ),
            "mean_nearest_bank_match_rate": float(
                np.mean([row["nearest_bank_match_rate"] for row in rows])
            ),
            "mean_boundary_hit_rate": float(
                np.mean([row["boundary_hit_rate"] for row in rows])
            ),
        }

    gains = np.asarray([row["selected_gain_percent"] for row in sample_rows], dtype=float)
    raw_gains = np.asarray([row["raw_rigcal_gain_percent"] for row in sample_rows], dtype=float)
    accepted = np.asarray([row["accepted"] for row in sample_rows], dtype=bool)
    audit_changes = np.asarray(
        [row["selected_audit_change_percent"] for row in sample_rows], dtype=float
    )
    conditional = gains[accepted]
    safety = {
        "coverage": float(np.mean(accepted)),
        "overall_mean_gain_percent": float(np.mean(gains)),
        "overall_p10_gain_percent": float(np.quantile(gains, 0.10)),
        "overall_harm_rate_over_1_percent": float(np.mean(gains < -1.0)),
        "raw_rigcal_mean_gain_percent": float(np.mean(raw_gains)),
        "accepted_count": int(np.sum(accepted)),
        "accepted_mean_gain_percent": None
        if not np.any(accepted)
        else float(np.mean(conditional)),
        "accepted_p10_gain_percent": None
        if not np.any(accepted)
        else float(np.quantile(conditional, 0.10)),
        "accepted_harm_rate_over_1_percent": None
        if not np.any(accepted)
        else float(np.mean(conditional < -1.0)),
        "mean_of_samplewise_audit_change_percent": float(np.mean(audit_changes)),
        "percent_change_of_mean_audit_rms": safe_percent_change(
            float(np.mean([row["selected_audit_rms"] for row in sample_rows])),
            float(np.mean([row["fallback_audit_rms"] for row in sample_rows])),
        ),
        "accepted_audit_increase_rate": None
        if not np.any(accepted)
        else float(np.mean(audit_changes[accepted] > 0.0)),
        "maximum_accepted_audit_increase_percent": None
        if not np.any(accepted)
        else float(np.max(audit_changes[accepted])),
    }
    leakage = {
        "per_scene_family_eta_squared": family_eta_squared(
            sample_rows, "per_scene_estimated_radius"
        ),
        "shared_family_eta_squared": family_eta_squared(
            sample_rows, "shared_estimated_radius"
        ),
        "rigcal_family_eta_squared": family_eta_squared(
            sample_rows, "rigcal_estimated_radius"
        ),
    }
    per_scene_eta = leakage["per_scene_family_eta_squared"]
    shared_eta = leakage["shared_family_eta_squared"]
    summary = {
        "sample_count": len(sample_rows),
        "support_audit": {
            "threshold": float(config["support_threshold"]),
            "active_voxels": int(np.sum(support)),
            "total_voxels": int(support.size),
            "full_soft_support_would_have_activated_every_voxel": True,
        },
        "independent_field_rig_bundles": len(
            {(row["rig_id"], row["sample_index_in_block"]) for row in sample_rows}
        ),
        "block_count": len(block_rows),
        "method_summary": method_summary,
        "calibration_summary": calibration_summary,
        "family_leakage_diagnostic": leakage,
        "selective_safety": safety,
        "zero_acceptance_interpretation": (
            "INCONCLUSIVE_FOR_ACCEPTED_CONDITIONAL_RISK"
            if not np.any(accepted)
            else "ACCEPTED_CONDITIONAL_RISK_ESTIMATED_DESCRIPTIVELY"
        ),
        "block_rows": block_rows,
        "gate_a_descriptive_signals": {
            "shared_mae_below_per_scene": calibration_summary["rig_shared_profile"][
                "mean_aperture_mae"
            ]
            < calibration_summary["per_scene_profile"]["mean_aperture_mae"],
            "rigcal_mae_not_worse_than_metadata": calibration_summary[
                "rigcal_profile"
            ]["mean_aperture_mae"]
            <= calibration_summary["metadata_nearest"]["mean_aperture_mae"],
            "sharing_reduces_family_eta_squared": None
            if per_scene_eta is None or shared_eta is None
            else shared_eta < per_scene_eta,
            "oracle_field_recovers_every_nearest_bank_radius": all(
                row["oracle_field_residual_radius"] == row["oracle_nearest_radius"]
                for row in block_rows
            ),
            "noisy_oracle_field_recovers_every_nearest_bank_radius": all(
                row["noisy_oracle_field_residual_radius"]
                == row["oracle_nearest_radius"]
                for row in block_rows
            ),
            "closest_operator_matrix_recovers_every_nearest_bank_radius": all(
                row["closest_truth_operator_matrix_radius"]
                == row["oracle_nearest_radius"]
                for row in block_rows
            ),
        },
    }
    return sample_rows, calibration_rows, profile_rows, summary


def _diagnostic_masks(
    block: DevelopmentBlock, count: int, requested: int
) -> list[tuple[int, ...]]:
    audit = set(block.audit_views)
    available = sorted(
        set(range(block.reconstruction_bank.shape[2])) - audit
    )
    if not 2 <= int(count) <= len(available):
        raise ValueError("diagnostic inner-view count is outside the non-audit rig")
    target = 1 if int(count) == len(available) else int(requested)
    masks: list[tuple[int, ...]] = []
    for shift in range(len(available)):
        positions = np.floor(
            (np.arange(int(count), dtype=float) + 0.5) * len(available) / int(count)
        ).astype(int)
        values = tuple(sorted(available[int((position + shift) % len(available))] for position in positions))
        if values not in masks:
            masks.append(values)
        if len(masks) >= target:
            break
    if len(masks) != target:
        raise RuntimeError("could not construct enough unique diagnostic view masks")
    return masks


def run_identifiability_sweep(
    config: dict[str, Any], blocks: list[DevelopmentBlock]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Post-design development sweep; never used to define a lock threshold."""

    settings = config["identifiability_sweep"]
    radii = np.asarray(config["candidate_aperture_radii"], dtype=float)
    support = support_mask_from_config(config)
    rows: list[dict[str, Any]] = []
    for block in blocks:
        oracle_index = nearest_index(radii, block.true_radius)
        for count in settings["inner_view_counts"]:
            masks = _diagnostic_masks(block, int(count), int(settings["masks_per_count"]))
            for mask_index, views in enumerate(masks):
                normal_scales = [
                    whitened_normal_mean_diagonal(
                        block.reconstruction_bank[0],
                        observation,
                        sigma,
                        views,
                        support,
                    )
                    for observation, sigma in zip(
                        block.observations, block.noise_std, strict=True
                    )
                ]
                normal_scale = float(np.median(normal_scales))
                for ridge_lambda in settings["ridge_lambdas"]:
                    value = float(ridge_lambda)
                    selection = profile_shared_radius(
                        block.reconstruction_bank,
                        radii,
                        block.observations,
                        block.noise_std,
                        views,
                        support,
                        value,
                    )
                    per_scene_indices = [
                        int(
                            np.argmin(
                                [
                                    candidate.fits[sample].reduced_objective
                                    for candidate in selection.candidates
                                ]
                            )
                        )
                        for sample in range(len(block.fields))
                    ]
                    derivative = operator_radius_derivative(
                        block.reconstruction_bank, radii, selection.selected_index
                    )
                    fisher = [
                        profile_fisher_scalar(
                            block.reconstruction_bank[selection.selected_index],
                            derivative,
                            fit,
                            observation,
                            sigma,
                            views,
                            support,
                            value,
                        )
                        for fit, observation, sigma in zip(
                            selection.selected.fits,
                            block.observations,
                            block.noise_std,
                            strict=True,
                        )
                    ]
                    ordered_scores = np.sort(
                        np.asarray(
                            [candidate.total_score for candidate in selection.candidates],
                            dtype=float,
                        )
                    )
                    rows.append(
                        {
                            "rig_id": block.rig_id,
                            "block_id": block.block_id,
                            "true_aperture_radius": block.true_radius,
                            "inner_view_count": int(count),
                            "mask_index": mask_index,
                            "inner_camera_indices": ",".join(str(item) for item in views),
                            "ridge_lambda": value,
                            "median_normal_matrix_diagonal": normal_scale,
                            "ridge_over_normal_scale": value / max(normal_scale, 1e-15),
                            "shared_estimated_radius": float(
                                radii[selection.selected_index]
                            ),
                            "shared_aperture_absolute_error": float(
                                abs(radii[selection.selected_index] - block.true_radius)
                            ),
                            "shared_nearest_bank_match": selection.selected_index
                            == oracle_index,
                            "shared_boundary_hit": selection.selected_index
                            in (0, len(radii) - 1),
                            "per_scene_boundary_hit_rate": float(
                                np.mean(
                                    np.isin(
                                        radii[per_scene_indices], [radii[0], radii[-1]]
                                    )
                                )
                            ),
                            "per_scene_nearest_bank_match_rate": float(
                                np.mean(np.asarray(per_scene_indices) == oracle_index)
                            ),
                            "median_profile_information": float(
                                np.median([item.profile_information for item in fisher])
                            ),
                            "median_profile_retained_fraction": float(
                                np.median([item.retained_fraction for item in fisher])
                            ),
                            "relative_profile_score_gap": float(
                                (ordered_scores[1] - ordered_scores[0])
                                / max(abs(ordered_scores[0]), 1e-15)
                            ),
                        }
                    )

    aggregate_rows: list[dict[str, Any]] = []
    for count in sorted({int(row["inner_view_count"]) for row in rows}):
        for ridge_lambda in sorted({float(row["ridge_lambda"]) for row in rows}):
            group = [
                row
                for row in rows
                if int(row["inner_view_count"]) == count
                and float(row["ridge_lambda"]) == ridge_lambda
            ]
            aggregate_rows.append(
                {
                    "inner_view_count": count,
                    "ridge_lambda": ridge_lambda,
                    "setting_rows": len(group),
                    "shared_nearest_bank_match_rate": float(
                        np.mean([row["shared_nearest_bank_match"] for row in group])
                    ),
                    "mean_shared_aperture_absolute_error": float(
                        np.mean([row["shared_aperture_absolute_error"] for row in group])
                    ),
                    "shared_boundary_hit_rate": float(
                        np.mean([row["shared_boundary_hit"] for row in group])
                    ),
                    "median_profile_retained_fraction": float(
                        np.median(
                            [row["median_profile_retained_fraction"] for row in group]
                        )
                    ),
                    "median_ridge_over_normal_scale": float(
                        np.median([row["ridge_over_normal_scale"] for row in group])
                    ),
                }
            )
    best = max(
        aggregate_rows,
        key=lambda row: (
            row["shared_nearest_bank_match_rate"],
            -row["mean_shared_aperture_absolute_error"],
            -row["shared_boundary_hit_rate"],
        ),
    )
    summary = {
        "status": "POSTHOC_DEVELOPMENT_DIAGNOSTIC_NOT_THRESHOLD_SELECTION",
        "row_count": len(rows),
        "aggregate": aggregate_rows,
        "descriptive_best_setting": best,
        "maximum_median_profile_retained_fraction": float(
            max(row["median_profile_retained_fraction"] for row in rows)
        ),
        "all_settings_recover_every_block": any(
            row["shared_nearest_bank_match_rate"] == 1.0
            and row["shared_boundary_hit_rate"] == 0.0
            for row in aggregate_rows
        ),
    }
    return rows, summary


def write_figure(
    path: Path,
    sample_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    methods = [
        "metadata_nearest",
        "per_scene_profile",
        "rig_shared_profile",
        "rigcal_profile",
    ]
    colors = ["tab:gray", "tab:orange", "tab:blue", "tab:green"]
    for method, color in zip(methods, colors, strict=True):
        rows = [row for row in calibration_rows if row["method"] == method]
        axes[0, 0].scatter(
            [row["true_aperture_radius"] for row in rows],
            [row["estimated_aperture_radius_mean"] for row in rows],
            label=method,
            alpha=0.75,
            color=color,
        )
    axes[0, 0].plot([0.04, 0.15], [0.04, 0.15], "k--", lw=1)
    axes[0, 0].set(
        title="Block aperture estimate",
        xlabel="true radius (truth-only)",
        ylabel="estimated radius",
    )
    axes[0, 0].legend(fontsize=8)

    for block_id in sorted({str(row["block_id"]) for row in profile_rows}):
        rows = [row for row in profile_rows if row["block_id"] == block_id]
        scores = np.asarray([row["total_score"] for row in rows], dtype=float)
        scores -= np.min(scores)
        axes[0, 1].plot(
            [row["candidate_aperture_radius"] for row in rows],
            scores,
            marker="o",
            alpha=0.7,
            label=block_id,
        )
    axes[0, 1].set(
        title="Rig-shared profile + metadata score",
        xlabel="candidate radius",
        ylabel="score - block minimum",
    )
    axes[0, 1].legend(fontsize=6, ncol=2)

    family_colors = {
        family: color
        for family, color in zip(
            sorted({str(row["family"]) for row in sample_rows}),
            ["tab:blue", "tab:orange", "tab:green", "tab:red"],
            strict=True,
        )
    }
    for family, color in family_colors.items():
        rows = [row for row in sample_rows if row["family"] == family]
        axes[1, 0].scatter(
            [row["profile_retained_fraction"] for row in rows],
            [row["raw_rigcal_gain_percent"] for row in rows],
            label=family,
            alpha=0.75,
            color=color,
        )
    axes[1, 0].axhline(0.0, color="black", lw=1)
    axes[1, 0].set(
        title="Does profile information rank field gain?",
        xlabel="retained regularized profile-Schur fraction",
        ylabel="raw RigCal field gain vs metadata (%)",
    )
    axes[1, 0].legend(fontsize=7)

    gains = np.asarray([row["selected_gain_percent"] for row in sample_rows], dtype=float)
    audit = np.asarray(
        [row["selected_audit_change_percent"] for row in sample_rows], dtype=float
    )
    accepted = np.asarray([row["accepted"] for row in sample_rows], dtype=bool)
    axes[1, 1].scatter(
        gains[~accepted], audit[~accepted], label="fallback", alpha=0.55, color="tab:gray"
    )
    axes[1, 1].scatter(
        gains[accepted], audit[accepted], label="accepted", alpha=0.8, color="tab:purple"
    )
    axes[1, 1].axhline(0.0, color="black", lw=1)
    axes[1, 1].axvline(0.0, color="black", lw=1)
    axes[1, 1].set(
        title="Development-only selective outcome",
        xlabel="deployed field gain vs metadata (%)",
        ylabel="final-audit residual change (%)",
    )
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(
        "v5b RigCal-BOST explicit profile pilot — development only, no lock claim",
        fontsize=14,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def write_sweep_figure(path: Path, sweep_summary: dict[str, Any]) -> None:
    rows = sweep_summary["aggregate"]
    counts = sorted({int(row["inner_view_count"]) for row in rows})
    lambdas = sorted({float(row["ridge_lambda"]) for row in rows})
    match = np.zeros((len(counts), len(lambdas)), dtype=float)
    fraction = np.zeros_like(match)
    for row in rows:
        y = counts.index(int(row["inner_view_count"]))
        x = lambdas.index(float(row["ridge_lambda"]))
        match[y, x] = float(row["shared_nearest_bank_match_rate"])
        fraction[y, x] = float(row["median_profile_retained_fraction"])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    first = axes[0].imshow(match, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    second = axes[1].imshow(
        np.log10(np.maximum(fraction, 1e-12)), cmap="magma", aspect="auto"
    )
    for axis, title in zip(
        axes,
        ["Nearest-bank recovery rate", "log10 median retained profile-Schur fraction"],
        strict=True,
    ):
        axis.set_xticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
        axis.set_yticks(range(len(counts)), [str(value) for value in counts])
        axis.set(xlabel="absolute ridge lambda", ylabel="inner camera count", title=title)
    figure.colorbar(first, ax=axes[0], shrink=0.85)
    figure.colorbar(second, ax=axes[1], shrink=0.85)
    figure.suptitle("v5b identifiability sweep — posthoc development diagnostic")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    forward_model_gate = validate_forward_model_gate(config)
    blocks, manifest_rows = build_development_blocks(config)
    sample_rows, calibration_rows, profile_rows, summary = evaluate_blocks(config, blocks)
    sweep_rows, sweep_summary = run_identifiability_sweep(config, blocks)

    config_snapshot = output / "config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = output / "paired_factorial_manifest.csv"
    sample_path = output / "sample_metrics.csv"
    calibration_path = output / "calibration_metrics.csv"
    profile_path = output / "profile_curves.csv"
    sweep_path = output / "identifiability_sweep.csv"
    write_csv(manifest_path, manifest_rows)
    write_csv(sample_path, sample_rows)
    write_csv(calibration_path, calibration_rows)
    write_csv(profile_path, profile_rows)
    write_csv(sweep_path, sweep_rows)
    figure_path = output / "v5b_rig_shared_profile_pilot.png"
    write_figure(figure_path, sample_rows, calibration_rows, profile_rows)
    sweep_figure_path = output / "v5b_identifiability_sweep.png"
    write_sweep_figure(sweep_figure_path, sweep_summary)

    report = {
        "claim_status": config["claim_status"],
        "claim_boundary": (
            "development-only 8x8x5 explicit linear ridge mechanism test; truth and "
            "reconstruction use different finite-aperture quadrature densities but the "
            "same prescribed weak-deflection model family; no preregistered lock, nonlinear "
            "ray tracing, CFD, real BOS, NeRIF/TDBOST, FNO/FFNO or DeepONet comparison"
        ),
        "scientific_question": (
            "Does sharing one low-dimensional aperture parameter across fully balanced "
            "reaction fields reduce v5a morphology/operator confounding, and does the "
        "explicit regularized profile-Schur fraction rank when calibration is safe?"
        ),
        "data_separation": {
            "inner_views_fit_field_and_radius": True,
            "outer_views_gate_acceptance_only": True,
            "final_audit_views_used_for_report_only": True,
            "final_audit_escrowed_in_a_separate_process": False,
            "audit_camera_excluded_from_noise_scale": True,
            "outer_camera_excluded_from_noise_scale": True,
            "paired_noise_across_counterfactual_apertures": True,
            "complete_family_by_radius_by_rig_factorial": True,
            "confirmatory_lock_constructed": False,
        },
        "forward_model_gate": forward_model_gate,
        "summary": summary,
        "identifiability_sweep": sweep_summary,
        "source_hashes": {
            "runner": sha256(Path(__file__).resolve()),
            "profile_module": sha256((ROOT / "rig_shared_profile.py").resolve()),
            "config": sha256(config_path),
            "forward_model_gate_report": forward_model_gate["report_sha256"],
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksum_targets = [
        config_snapshot,
        manifest_path,
        sample_path,
        calibration_path,
        profile_path,
        sweep_path,
        figure_path,
        sweep_figure_path,
        report_path,
    ]
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_targets),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
