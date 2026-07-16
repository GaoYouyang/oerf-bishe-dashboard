#!/usr/bin/env python3
"""Develop a truth-free residual-risk gate without opening a fresh audit."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_reaction_phantoms import reaction_morphology_batch
from demo_t16_operator.psu_b0_residual_risk import (
    RESIDUAL_RISK_SCHEMA,
    RISK_FEATURE_NAMES,
    CalibratedResidualRiskDirection,
    RidgeRiskFit,
    fit_ridge_risk_model,
    observable_risk_features,
    one_sided_conformal_quantile,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import (
    FixedSobolevDirection,
    PositiveSpectralDirection,
    exact_line_search_reconstruction,
)
from demo_t16_operator.psu_b0_streaming_operator import zero_outer_boundary_support
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    SyntheticSplit,
    _batched_forward,
    _evaluate,
    _load_json,
    _state_sha256,
    _unique_masks,
)
from site_tools.run_psu_b0_support_envelope_diagnosis import (
    _load_checkpoint_model,
)


PRIVATE_SCHEMA = "psu-b0-residual-risk-development-private-report-1.0"
PUBLIC_SCHEMA = "psu-b0-residual-risk-development-public-summary-1.0"
STATUS = "DEVELOPMENT_COMPLETE_FRESH_AUDIT_UNOPENED"


@dataclass(frozen=True)
class DevelopmentSplit:
    data: SyntheticSplit
    noise_profiles: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) < 1 or not 0.0 < float(quantile) <= 1.0:
        raise ValueError("quantile input is invalid")
    index = int(np.ceil(float(quantile) * len(array))) - 1
    return float(np.sort(array)[min(max(index, 0), len(array) - 1)])


def _camera_noise(
    clean: torch.Tensor,
    sigma_by_view: torch.Tensor,
    *,
    profiles: tuple[str, ...],
    profile_config: dict[str, Any],
    seed: int,
) -> torch.Tensor:
    """Create declared-RMS IID or row/column-correlated camera noise."""

    count, ray_count, components = clean.shape
    view_count = sigma_by_view.shape[1]
    rays_per_view = ray_count // view_count
    detector_side = int(round(np.sqrt(rays_per_view)))
    if components != 2 or detector_side * detector_side != rays_per_view:
        raise ValueError("camera-noise proxy requires square rays per view")
    generator = torch.Generator().manual_seed(int(seed))
    clean_image = clean.reshape(
        count,
        view_count,
        detector_side,
        detector_side,
        2,
    )
    output = torch.empty_like(clean_image)
    for index, profile_name in enumerate(profiles):
        profile = profile_config[str(profile_name)]
        correlation = float(profile["correlation_fraction"])
        view_bias = float(profile["view_bias_fraction"])
        signal_fraction = float(profile["signal_fraction"])
        squared = correlation**2 + view_bias**2 + signal_fraction**2
        if squared > 1.0 + 1e-8:
            raise ValueError("camera-noise fractions exceed unit variance")
        iid_fraction = np.sqrt(max(1.0 - squared, 0.0))
        iid = torch.randn(
            (view_count, detector_side, detector_side, 2),
            generator=generator,
        )
        row = torch.randn(
            (view_count, detector_side, 1, 2),
            generator=generator,
        )
        column = torch.randn(
            (view_count, 1, detector_side, 2),
            generator=generator,
        )
        correlated = (row + column) / np.sqrt(2.0)
        bias = torch.randn(
            (view_count, 1, 1, 2),
            generator=generator,
        )
        signal_scale = torch.abs(clean_image[index])
        signal_scale = signal_scale / torch.sqrt(
            torch.mean(signal_scale.square(), dim=(1, 2), keepdim=True)
        ).clamp_min(1e-8)
        heteroscedastic = torch.randn(
            (view_count, detector_side, detector_side, 2),
            generator=generator,
        ) * signal_scale
        noise = (
            iid_fraction * iid
            + correlation * correlated
            + view_bias * bias
            + signal_fraction * heteroscedastic
        )
        noise = noise / torch.sqrt(
            torch.mean(noise.square(), dim=(1, 2), keepdim=True)
        ).clamp_min(1e-8)
        output[index] = (
            noise
            * sigma_by_view[index, :, None, None, None]
        )
    return output.reshape_as(clean)


def _build_development_split(
    *,
    name: str,
    spec: dict[str, Any],
    config: dict[str, Any],
    source_config: dict[str, Any],
    true_operator: Any,
    nominal_operator: Any,
    device: torch.device,
    forbidden_masks: set[str],
) -> tuple[DevelopmentSplit, set[str]]:
    count = int(spec["count"])
    grid_size = int(config["geometry"]["grid_size"])
    view_count = int(config["geometry"]["view_count"])
    rays_per_view = int(config["geometry"]["rays_per_view"])
    families = tuple(
        str(spec["families"][index % len(spec["families"])])
        for index in range(count)
    )
    profiles = tuple(
        str(spec["noise_profiles"][index % len(spec["noise_profiles"])])
        for index in range(count)
    )
    truth = reaction_morphology_batch(
        grid_size=grid_size,
        families=families,
        seeds=tuple(int(spec["field_seed_start"]) + index for index in range(count)),
        dtype=torch.float32,
        device="cpu",
    )
    active_range = tuple(int(value) for value in config["geometry"]["active_view_range"])
    mask, used_masks = _unique_masks(
        count=count,
        view_count=view_count,
        minimum_active=active_range[0],
        maximum_active=active_range[1],
        seed=int(spec["mask_seed"]),
        forbidden=forbidden_masks,
    )
    nominal_signal = _batched_forward(
        nominal_operator,
        truth,
        batch_size=12,
        device=device,
    )
    if str(spec["truth_operator"]) == "qmc32":
        true_signal = _batched_forward(
            true_operator,
            truth,
            batch_size=12,
            device=device,
        )
    elif str(spec["truth_operator"]) == "qmc8":
        true_signal = nominal_signal.clone()
    else:
        raise ValueError("truth_operator must be qmc32 or qmc8")
    mismatch = torch.linalg.vector_norm(
        (true_signal - nominal_signal).flatten(1),
        dim=1,
    ) / torch.linalg.vector_norm(true_signal.flatten(1), dim=1).clamp_min(1e-12)
    view_signal = true_signal.reshape(count, view_count, rays_per_view, 2)
    view_rms = torch.sqrt(
        torch.mean(view_signal.square(), dim=(2, 3)).clamp_min(1e-20)
    )
    global_floor = 0.10 * torch.mean(view_rms, dim=1, keepdim=True)
    levels = tuple(float(value) for value in spec["relative_noise_levels"])
    relative_noise = torch.as_tensor(
        [levels[index % len(levels)] for index in range(count)],
        dtype=torch.float32,
    )
    factors = torch.as_tensor(
        source_config["data"]["view_noise_factors"],
        dtype=torch.float32,
    )
    sigma = (
        relative_noise[:, None]
        * torch.maximum(view_rms, global_floor)
        * factors[None]
    ).clamp_min(1e-8)
    noise = _camera_noise(
        true_signal,
        sigma,
        profiles=profiles,
        profile_config=config["camera_noise_profiles"],
        seed=int(spec["noise_seed"]),
    )
    split = SyntheticSplit(
        name=name,
        sample_ids=tuple(f"{name}-{index:03d}" for index in range(count)),
        families=families,
        truth=truth,
        observation_uv=true_signal + noise,
        sigma_by_view=sigma,
        view_mask=mask,
        relative_noise=relative_noise,
        truth_operator=str(spec["truth_operator"]),
        operator_mismatch_relative_l2=mismatch,
    )
    return DevelopmentSplit(split, profiles), used_masks


def _initial_features(
    *,
    split: SyntheticSplit,
    operator: Any,
    candidate: PositiveSpectralDirection,
    fallback: FixedSobolevDirection,
    rays_per_view: int,
    device: torch.device,
    batch_size: int = 12,
) -> np.ndarray:
    rows = []
    support = operator.support[None, None]
    operator.reset_call_counts()
    with torch.no_grad():
        for start in range(0, len(split.truth), int(batch_size)):
            observation = split.observation_uv[start : start + batch_size].to(device)
            sigma = split.sigma_by_view[start : start + batch_size].to(device)
            mask = split.view_mask[start : start + batch_size].to(device)
            active = mask.repeat_interleave(int(rays_per_view), dim=1)[:, :, None]
            expanded_sigma = sigma.repeat_interleave(
                int(rays_per_view),
                dim=1,
            )[:, :, None]
            residual = active * observation
            gradient = operator.adjoint(residual / expanded_sigma.square())
            candidate_direction, diagnostics = candidate(
                gradient,
                residual_uv=residual,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
                stage_fraction=0.25,
            )
            fallback_direction, _ = fallback(
                gradient,
                residual_uv=residual,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
                stage_fraction=0.25,
            )
            features = observable_risk_features(
                gradient,
                residual_uv=residual,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
                candidate_direction=candidate_direction * support,
                fallback_direction=fallback_direction * support,
                candidate_diagnostics=diagnostics,
            )
            rows.append(features.detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _actual_gain_lookup(
    rows: list[dict[str, Any]],
    *,
    baseline_method: str,
    candidate_method: str,
) -> dict[str, float]:
    baseline = {
        row["sample_id"]: float(row["field_relative_l2"])
        for row in rows
        if row["method"] == baseline_method
    }
    candidate = {
        row["sample_id"]: float(row["field_relative_l2"])
        for row in rows
        if row["method"] == candidate_method
    }
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("candidate and baseline sample sets do not match")
    return {
        sample_id: 100.0
        * (baseline[sample_id] - candidate[sample_id])
        / max(baseline[sample_id], 1e-12)
        for sample_id in baseline
    }


def _selection_metrics(
    gain: np.ndarray,
    trust: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(gain, dtype=np.float64)
    accepted = np.asarray(trust, dtype=bool)
    selected = np.where(accepted, values, 0.0)
    accepted_values = values[accepted]
    return {
        "row_count": int(len(values)),
        "accepted_row_count": int(np.sum(accepted)),
        "coverage": float(np.mean(accepted)),
        "mean_selected_gain_percent": float(np.mean(selected)),
        "p10_selected_gain_percent": float(np.quantile(selected, 0.10)),
        "harm_over_one_percent_rate": float(np.mean(selected < -1.0)),
        "accepted_mean_raw_gain_percent": (
            None if len(accepted_values) == 0 else float(np.mean(accepted_values))
        ),
        "accepted_minimum_raw_gain_percent": (
            None if len(accepted_values) == 0 else float(np.min(accepted_values))
        ),
    }


def _score_gate(
    fit: RidgeRiskFit,
    features: np.ndarray,
    *,
    overprediction_quantile: float | np.ndarray,
    distance_threshold: float,
    minimum_lower_gain_percent: float,
) -> dict[str, np.ndarray]:
    prediction = fit.predict(features)
    quantile = np.asarray(overprediction_quantile, dtype=np.float64)
    lower = prediction - quantile
    distance = fit.distance(features)
    trust = (
        (lower >= float(minimum_lower_gain_percent))
        & (distance <= float(distance_threshold))
    )
    return {
        "prediction": prediction,
        "lower": lower,
        "distance": distance,
        "trust": trust,
    }


def _equivalence_probe(
    *,
    split: SyntheticSplit,
    operator: Any,
    models: dict[int, PositiveSpectralDirection],
    fallback: FixedSobolevDirection,
    fit: RidgeRiskFit,
    conformal_quantile_by_seed: dict[int, float],
    distance_threshold: float,
    selected_margin: float,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    subset = slice(0, min(6, len(split.truth)))
    observation = split.observation_uv[subset].to(device)
    sigma = split.sigma_by_view[subset].to(device)
    mask = split.view_mask[subset].to(device)
    rays_per_view = int(config["geometry"]["rays_per_view"])
    stages = int(config["solver"]["stages"])
    records = []
    for seed, model in sorted(models.items()):
        features = _initial_features(
            split=SyntheticSplit(
                name=split.name,
                sample_ids=split.sample_ids[subset],
                families=split.families[subset],
                truth=split.truth[subset],
                observation_uv=split.observation_uv[subset],
                sigma_by_view=split.sigma_by_view[subset],
                view_mask=split.view_mask[subset],
                relative_noise=split.relative_noise[subset],
                truth_operator=split.truth_operator,
                operator_mismatch_relative_l2=split.operator_mismatch_relative_l2[
                    subset
                ],
            ),
            operator=operator,
            candidate=model,
            fallback=fallback,
            rays_per_view=rays_per_view,
            device=device,
        )
        expected_trust = _score_gate(
            fit,
            features,
            overprediction_quantile=conformal_quantile_by_seed[seed],
            distance_threshold=distance_threshold,
            minimum_lower_gain_percent=selected_margin,
        )["trust"]
        gate = CalibratedResidualRiskDirection(
            candidate=model,
            fallback=fallback,
            stages=stages,
            feature_mean=fit.feature_mean,
            feature_scale=fit.feature_scale,
            coefficients=fit.coefficients,
            intercept=fit.intercept,
            overprediction_quantile=conformal_quantile_by_seed[seed],
            distance_threshold=distance_threshold,
            minimum_lower_gain_percent=selected_margin,
            minimum_active_views=int(config["geometry"]["active_view_range"][0]),
            maximum_active_views=int(config["geometry"]["active_view_range"][1]),
        ).to(device)
        operator.reset_call_counts()
        gated = exact_line_search_reconstruction(
            operator,
            observation,
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=rays_per_view,
            stages=stages,
            direction=gate,
        )
        gated_calls = operator.call_report()
        operator.reset_call_counts()
        raw = exact_line_search_reconstruction(
            operator,
            observation,
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=rays_per_view,
            stages=stages,
            direction=model,
        )
        operator.reset_call_counts()
        base = exact_line_search_reconstruction(
            operator,
            observation,
            sigma_by_view=sigma,
            view_mask=mask,
            rays_per_view=rays_per_view,
            stages=stages,
            direction=fallback,
        )
        expected = torch.where(
            torch.as_tensor(
                expected_trust,
                device=device,
            )[:, None, None, None, None],
            raw.volume,
            base.volume,
        )
        reported_trust = (
            gated.history[0]["residual_risk_trust"].detach().cpu().numpy() > 0.5
        )
        records.append(
            {
                "seed": int(seed),
                "sample_count": len(observation),
                "maximum_absolute_volume_difference": float(
                    torch.max(torch.abs(gated.volume - expected))
                ),
                "decision_match": bool(np.array_equal(expected_trust, reported_trust)),
                "gated_logical_calls": {
                    "forward": int(gated_calls["forward_calls"]),
                    "adjoint": int(gated_calls["adjoint_calls"]),
                },
            }
        )
    return records


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    """Export development aggregates without features, weights, or local paths."""

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "source_candidate": copy.deepcopy(private["source_candidate_public"]),
        "dataset": copy.deepcopy(private["dataset_public"]),
        "risk_model": copy.deepcopy(private["risk_model_public"]),
        "development_results": copy.deepcopy(private["development_results"]),
        "equivalence_probe": copy.deepcopy(private["equivalence_probe"]),
        "gates": copy.deepcopy(private["gates"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_checkpoints_or_hashes": False,
            "contains_feature_rows_or_model_weights": False,
            "contains_observations_or_volumes": False,
            "fresh_audit_values": False,
        },
    }


def run_development(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
    checkpoint_dir: Path,
    source_private_report_path: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    source_config_path = root / str(config["source_pilot"]["config"])
    source_config = _load_json(source_config_path)
    source_private = _load_json(source_private_report_path)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    started = time.perf_counter()
    grid_size = int(config["geometry"]["grid_size"])
    rays_per_view = int(config["geometry"]["rays_per_view"])
    support = zero_outer_boundary_support((grid_size,) * 3).to(device)
    true_geometry, true_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(config["geometry"]["true_finite_aperture_sample_count"]),
    )
    nominal_geometry, nominal_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            config["geometry"]["nominal_finite_aperture_sample_count"]
        ),
    )
    true_operator = _make_operator(
        true_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    nominal_operator = _make_operator(
        nominal_geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    true_operator.support.copy_(support)
    nominal_operator.support.copy_(support)

    selected_strength = float(config["source_pilot"]["selected_sobolev_strength"])
    fallback = FixedSobolevDirection(
        (grid_size,) * 3,
        strength=selected_strength,
    ).to(device)
    models: dict[int, PositiveSpectralDirection] = {}
    checkpoint_records = []
    for checkpoint in sorted(checkpoint_dir.glob("learned_seed_*.pt")):
        seed, model, record = _load_checkpoint_model(
            checkpoint=checkpoint,
            config=source_config,
            selected_strength=selected_strength,
            device=device,
        )
        models[seed] = model
        checkpoint_records.append(
            {
                "seed": seed,
                **record,
                "state_sha256": _state_sha256(
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    }
                ),
            }
        )
    expected_seeds = sorted(int(value) for value in config["source_pilot"]["checkpoint_seeds"])
    if sorted(models) != expected_seeds:
        raise ValueError("checkpoint seed set does not match development config")
    source_hashes = {
        int(row["seed"]): str(row["checkpoint_sha256"])
        for row in source_private["training"]
    }
    if any(
        source_hashes[int(row["seed"])] != str(row["state_sha256"])
        for row in checkpoint_records
    ):
        raise ValueError("source checkpoint hash mismatch")

    splits: dict[str, DevelopmentSplit] = {}
    used_masks: set[str] = set()
    for name, spec in config["development_splits"].items():
        split, used_masks = _build_development_split(
            name=name,
            spec=spec,
            config=config,
            source_config=source_config,
            true_operator=true_operator,
            nominal_operator=nominal_operator,
            device=device,
            forbidden_masks=used_masks,
        )
        splits[name] = split

    baseline_rows: dict[str, list[dict[str, Any]]] = {}
    all_metric_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for split_name, wrapped in splits.items():
        baseline, _ = _evaluate(
            method="sobolev_selected",
            split=wrapped.data,
            operator=nominal_operator,
            config=source_config,
            device=device,
            direction=fallback,
        )
        baseline_rows[split_name] = baseline
        all_metric_rows.extend(baseline)
        for seed, model in sorted(models.items()):
            method = f"raw_seed_{seed}"
            candidate_rows, _ = _evaluate(
                method=method,
                split=wrapped.data,
                operator=nominal_operator,
                config=source_config,
                device=device,
                direction=model,
            )
            all_metric_rows.extend(candidate_rows)
            gains = _actual_gain_lookup(
                baseline + candidate_rows,
                baseline_method="sobolev_selected",
                candidate_method=method,
            )
            features = _initial_features(
                split=wrapped.data,
                operator=nominal_operator,
                candidate=model,
                fallback=fallback,
                rays_per_view=rays_per_view,
                device=device,
            )
            for index, sample_id in enumerate(wrapped.data.sample_ids):
                feature_rows.append(
                    {
                        "split": split_name,
                        "sample_id": sample_id,
                        "seed": int(seed),
                        "family": wrapped.data.families[index],
                        "noise_profile": wrapped.noise_profiles[index],
                        "relative_noise": float(wrapped.data.relative_noise[index]),
                        "active_view_count": int(
                            torch.sum(wrapped.data.view_mask[index] > 0.5)
                        ),
                        "actual_gain_percent": float(gains[sample_id]),
                        "features": features[index].tolist(),
                    }
                )

    by_split = {
        name: [row for row in feature_rows if row["split"] == name]
        for name in splits
    }

    def arrays(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = by_split[name]
        return (
            np.asarray([row["features"] for row in rows], dtype=np.float64),
            np.asarray(
                [row["actual_gain_percent"] for row in rows],
                dtype=np.float64,
            ),
            np.asarray([row["seed"] for row in rows], dtype=np.int64),
        )

    train_x, train_y, train_seeds = arrays("risk_train")
    validation_x, validation_y, validation_seeds = arrays("risk_validation")
    calibration_x, calibration_y, calibration_seeds = arrays("risk_calibration")
    fit = fit_ridge_risk_model(
        train_x,
        train_y,
        validation_x,
        validation_y,
        ridge_grid=tuple(float(value) for value in config["risk_model"]["ridge_grid"]),
    )
    distance_threshold = _higher_quantile(
        fit.distance(train_x),
        float(config["risk_model"]["distance_quantile"]),
    )
    validation_prediction = fit.predict(validation_x)
    validation_quantile_by_seed = {
        seed: one_sided_conformal_quantile(
            validation_prediction[validation_seeds == seed],
            validation_y[validation_seeds == seed],
            alpha=float(config["risk_model"]["conformal_alpha"]),
        )
        for seed in expected_seeds
    }
    validation_quantile_rows = np.asarray(
        [validation_quantile_by_seed[int(seed)] for seed in validation_seeds],
        dtype=np.float64,
    )
    margin_screen = []
    for margin in config["risk_model"]["minimum_lower_gain_percent_grid"]:
        score = _score_gate(
            fit,
            validation_x,
            overprediction_quantile=validation_quantile_rows,
            distance_threshold=distance_threshold,
            minimum_lower_gain_percent=float(margin),
        )
        metrics = _selection_metrics(validation_y, score["trust"])
        admissible = (
            float(metrics["harm_over_one_percent_rate"]) <= 0.05
            and float(metrics["coverage"]) >= 0.20
        )
        margin_screen.append(
            {
                "minimum_lower_gain_percent": float(margin),
                "provisional_validation_overprediction_quantile_by_seed": {
                    str(seed): float(value)
                    for seed, value in validation_quantile_by_seed.items()
                },
                "admissible": bool(admissible),
                **metrics,
            }
        )
    admissible_rows = [row for row in margin_screen if row["admissible"]]
    if admissible_rows:
        selected_margin_row = max(
            admissible_rows,
            key=lambda row: (
                float(row["mean_selected_gain_percent"]),
                float(row["coverage"]),
                float(row["minimum_lower_gain_percent"]),
            ),
        )
        margin_selection_status = "ADMISSIBLE_MARGIN_SELECTED"
    else:
        selected_margin_row = max(
            margin_screen,
            key=lambda row: (
                -float(row["harm_over_one_percent_rate"]),
                float(row["coverage"]),
                float(row["mean_selected_gain_percent"]),
            ),
        )
        margin_selection_status = "NO_ADMISSIBLE_MARGIN_DEVELOPMENT_NO_GO"
    selected_margin = float(selected_margin_row["minimum_lower_gain_percent"])
    calibration_prediction = fit.predict(calibration_x)
    conformal_quantile_by_seed = {
        seed: one_sided_conformal_quantile(
            calibration_prediction[calibration_seeds == seed],
            calibration_y[calibration_seeds == seed],
            alpha=float(config["risk_model"]["conformal_alpha"]),
        )
        for seed in expected_seeds
    }
    final_results = {}
    for split_name, (features, gain, seeds) in {
        "risk_train": (train_x, train_y, train_seeds),
        "risk_validation": (validation_x, validation_y, validation_seeds),
        "risk_calibration": (calibration_x, calibration_y, calibration_seeds),
    }.items():
        quantile_rows = np.asarray(
            [conformal_quantile_by_seed[int(seed)] for seed in seeds],
            dtype=np.float64,
        )
        score = _score_gate(
            fit,
            features,
            overprediction_quantile=quantile_rows,
            distance_threshold=distance_threshold,
            minimum_lower_gain_percent=selected_margin,
        )
        final_results[split_name] = {
            "raw": _selection_metrics(gain, np.ones(len(gain), dtype=bool)),
            "gated": _selection_metrics(gain, score["trust"]),
            "prediction_gain_correlation": float(
                np.corrcoef(score["prediction"], gain)[0, 1]
            ),
            "feature_distance_mean": float(np.mean(score["distance"])),
            "feature_distance_maximum": float(np.max(score["distance"])),
        }

    equivalence = _equivalence_probe(
        split=splits["risk_validation"].data,
        operator=nominal_operator,
        models=models,
        fallback=fallback,
        fit=fit,
        conformal_quantile_by_seed=conformal_quantile_by_seed,
        distance_threshold=distance_threshold,
        selected_margin=selected_margin,
        config=config,
        device=device,
    )
    _synchronize(device)
    validation_gate = final_results["risk_validation"]["gated"]
    calibration_gate = final_results["risk_calibration"]["gated"]
    ready_for_fresh = bool(
        margin_selection_status == "ADMISSIBLE_MARGIN_SELECTED"
        and float(validation_gate["coverage"]) >= 0.20
        and float(validation_gate["harm_over_one_percent_rate"]) <= 0.05
        and float(validation_gate["mean_selected_gain_percent"]) > 0.0
        and float(calibration_gate["coverage"]) >= 0.20
        and float(calibration_gate["harm_over_one_percent_rate"]) <= 0.05
        and float(calibration_gate["mean_selected_gain_percent"]) > 0.0
        and all(
            row["decision_match"]
            and row["maximum_absolute_volume_difference"] <= 1e-6
            and row["gated_logical_calls"] == {"forward": 4, "adjoint": 4}
            for row in equivalence
        )
    )
    split_public = []
    for name, wrapped in splits.items():
        split_public.append(
            {
                "name": name,
                "sample_count": len(wrapped.data.truth),
                "pooled_candidate_rows": len(by_split[name]),
                "families": sorted(set(wrapped.data.families)),
                "noise_profiles": sorted(set(wrapped.noise_profiles)),
                "relative_noise_minimum": float(wrapped.data.relative_noise.min()),
                "relative_noise_maximum": float(wrapped.data.relative_noise.max()),
                "active_view_count_minimum": int(
                    torch.sum(wrapped.data.view_mask, dim=1).min()
                ),
                "active_view_count_maximum": int(
                    torch.sum(wrapped.data.view_mask, dim=1).max()
                ),
            }
        )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": STATUS,
        "evidence_scope": config["evidence_scope"],
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "view_root": str(view_root.resolve()),
            "source_private_report_path": str(source_private_report_path.resolve()),
            "source_private_report_sha256": _sha256(source_private_report_path),
            "device": device_name,
        },
        "source_candidate_private": {
            "source_config_sha256": _sha256(source_config_path),
            "checkpoint_records": checkpoint_records,
        },
        "source_candidate_public": {
            "source_status": source_private["status"],
            "checkpoint_seeds": expected_seeds,
            "selected_sobolev_strength": selected_strength,
            "candidate_parameter_count": int(
                checkpoint_records[0]["parameter_count"]
            ),
            "source_candidate_passed_original_gate": False,
        },
        "dataset_private": {
            "true_geometry_provenance": true_provenance,
            "nominal_geometry_provenance": nominal_provenance,
            "feature_rows": feature_rows,
            "metric_rows": all_metric_rows,
        },
        "dataset_public": {
            "source_dataset_doi": "10.26208/1VE2-5C19",
            "real_psu_measurement_values_used": False,
            "analytic_truth_is_cfd": False,
            "camera_noise_is_measured_psu_noise": False,
            "development_splits": split_public,
            "feature_count": len(RISK_FEATURE_NAMES),
            "feature_names": list(RISK_FEATURE_NAMES),
        },
        "risk_model_private": {
            "feature_mean": fit.feature_mean.tolist(),
            "feature_scale": fit.feature_scale.tolist(),
            "coefficients": fit.coefficients.tolist(),
            "intercept": fit.intercept,
        },
        "risk_model_public": {
            "schema": RESIDUAL_RISK_SCHEMA,
            "family": config["risk_model"]["family"],
            "ridge_lambda": fit.ridge_lambda,
            "validation_rmse": fit.validation_rmse,
            "conformal_alpha": float(config["risk_model"]["conformal_alpha"]),
            "validation_provisional_overprediction_quantile": float(
                max(validation_quantile_by_seed.values())
            ),
            "validation_provisional_overprediction_quantile_by_seed": {
                str(seed): float(value)
                for seed, value in validation_quantile_by_seed.items()
            },
            "calibration_overprediction_quantile_by_seed": {
                str(seed): float(value)
                for seed, value in conformal_quantile_by_seed.items()
            },
            "distance_threshold": float(distance_threshold),
            "distance_quantile_source": "risk_train",
            "selected_minimum_lower_gain_percent": selected_margin,
            "margin_selection_status": margin_selection_status,
            "margin_screen": margin_screen,
            "decision_stage": 1,
            "decision_held_for_all_stages": True,
            "deployment_logical_calls": {"forward": 4, "adjoint": 4},
        },
        "development_results": final_results,
        "equivalence_probe": equivalence,
        "execution": {
            "wall_seconds": float(time.perf_counter() - started),
            "process_max_rss_bytes": int(_max_rss_bytes()),
            "host": {
                "machine": platform.machine(),
                "platform": platform.platform(),
                "torch_version": torch.__version__,
            },
        },
        "gates": {
            "source_checkpoint_hashes_match": True,
            "development_masks_globally_disjoint": len(used_masks)
            == sum(int(spec["count"]) for spec in config["development_splits"].values()),
            "risk_features_use_no_truth_at_deployment": True,
            "fresh_audit_not_created_or_opened": True,
            "deployment_calls_match_raw_and_sobolev": all(
                row["gated_logical_calls"] == {"forward": 4, "adjoint": 4}
                for row in equivalence
            ),
            "implementation_matches_fixed_candidate_or_fallback_choice": all(
                row["decision_match"]
                and row["maximum_absolute_volume_difference"] <= 1e-6
                for row in equivalence
            ),
            "development_ready_to_freeze_fresh_audit": ready_for_fresh,
        },
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-private-report", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_development(
        root=args.root,
        config_path=args.config,
        view_root=args.view_root,
        checkpoint_dir=args.checkpoint_dir,
        source_private_report_path=args.source_private_report,
        device_name=args.device,
    )
    if args.private_output is not None:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        args.private_output.write_text(
            json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
