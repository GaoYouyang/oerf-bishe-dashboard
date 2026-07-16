#!/usr/bin/env python3
"""Train a fixed-SPD geometry-conditioned PSU PCGLS development prototype."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
)
from demo_t16_operator.psu_b0_conditioned_pcgls import (
    CONDITIONED_PCGLS_SCHEMA,
    GeometryConditionedSPDPreconditioner,
    view_geometry_features_from_operator,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import (
    normalized_field_loss,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_classical_frontier_development import (
    _verify_split_metadata,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_residual_risk_development import (
    DevelopmentSplit,
    _build_development_split,
    _synchronize,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _expanded_measurement_values,
    _field_metrics,
    _load_json,
    _state_sha256,
)


PRIVATE_SCHEMA = "psu-b0-conditioned-pcgls-development-private-1.0"
PUBLIC_SCHEMA = "psu-b0-conditioned-pcgls-development-public-1.0"
STATUS = "CONDITIONED_PCGLS_DEVELOPMENT_COMPLETE_FRESH_NOT_USED"


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 10000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) < 2:
        raise ValueError("bootstrap requires at least two values")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(array), size=(int(draws), len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_gain_summary(
    rows: list[dict[str, Any]],
    *,
    split: str,
    candidate_method: str,
    baseline_method: str = "static_pcgls4",
    bootstrap_seed: int,
) -> dict[str, Any]:
    group = [row for row in rows if str(row["split"]) == str(split)]
    baseline = {
        str(row["sample_id"]): float(row["field_relative_l2"])
        for row in group
        if str(row["method"]) == str(baseline_method)
    }
    candidate = {
        str(row["sample_id"]): float(row["field_relative_l2"])
        for row in group
        if str(row["method"]) == str(candidate_method)
    }
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("paired methods do not share the same sample set")
    sample_ids = sorted(baseline)
    gains = np.asarray(
        [
            100.0
            * (baseline[sample_id] - candidate[sample_id])
            / max(baseline[sample_id], 1e-12)
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )
    lower, upper = _bootstrap_interval(gains, seed=int(bootstrap_seed))
    output = {
        "split": str(split),
        "candidate_method": str(candidate_method),
        "baseline_method": str(baseline_method),
        "sample_count": len(sample_ids),
        "mean_field_gain_percent": float(np.mean(gains)),
        "median_field_gain_percent": float(np.median(gains)),
        "p10_field_gain_percent": float(np.quantile(gains, 0.10)),
        "minimum_field_gain_percent": float(np.min(gains)),
        "bootstrap_mean_95_interval_percent": [lower, upper],
        "win_count": int(np.sum(gains > 0.0)),
        "win_rate": float(np.mean(gains > 0.0)),
        "harm_over_one_percent_count": int(np.sum(gains < -1.0)),
        "harm_over_one_percent_rate": float(np.mean(gains < -1.0)),
    }
    metric_rules = (
        ("gradient_relative_l2", "lower_is_better"),
        ("front_top10_f1", "higher_is_better"),
        ("measurement_relative_l2", "lower_is_better"),
    )
    baseline_rows = {
        str(row["sample_id"]): row
        for row in group
        if str(row["method"]) == str(baseline_method)
    }
    candidate_rows = {
        str(row["sample_id"]): row
        for row in group
        if str(row["method"]) == str(candidate_method)
    }
    secondary = {}
    for metric, rule in metric_rules:
        if not all(metric in baseline_rows[key] for key in sample_ids):
            continue
        baseline_metric = np.asarray(
            [float(baseline_rows[key][metric]) for key in sample_ids],
            dtype=np.float64,
        )
        candidate_metric = np.asarray(
            [float(candidate_rows[key][metric]) for key in sample_ids],
            dtype=np.float64,
        )
        denominator = np.maximum(np.abs(baseline_metric), 1e-12)
        if rule == "lower_is_better":
            metric_gain = (
                100.0
                * (baseline_metric - candidate_metric)
                / denominator
            )
        else:
            metric_gain = (
                100.0
                * (candidate_metric - baseline_metric)
                / denominator
            )
        secondary[metric] = {
            "mean_gain_percent": float(np.mean(metric_gain)),
            "p10_gain_percent": float(np.quantile(metric_gain, 0.10)),
            "minimum_gain_percent": float(np.min(metric_gain)),
        }
    output["secondary_metric_gain"] = secondary
    return output


def _evaluate(
    *,
    method: str,
    wrapped: DevelopmentSplit,
    operator: Any,
    source_config: dict[str, Any],
    device: torch.device,
    model: GeometryConditionedSPDPreconditioner | None,
    static_direction: GeneralizedSobolevDirection | None,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = wrapped.data
    rays_per_view = int(source_config["geometry"]["rays_per_view"])
    stages = 4
    rows: list[dict[str, Any]] = []
    gain_minimum = float("inf")
    gain_maximum = 0.0
    geometric_defect = 0.0
    monotone = True
    started = time.perf_counter()
    operator.reset_call_counts()
    for start in range(0, len(split.truth), int(batch_size)):
        stop = min(start + int(batch_size), len(split.truth))
        truth = split.truth[start:stop].to(device)
        observation = split.observation_uv[start:stop].to(device)
        sigma = split.sigma_by_view[start:stop].to(device)
        mask = split.view_mask[start:stop].to(device)
        with torch.no_grad():
            if model is not None:
                preconditioner = model.materialize(
                    observation,
                    sigma_by_view=sigma,
                    view_mask=mask,
                    rays_per_view=rays_per_view,
                )
            elif static_direction is not None:
                preconditioner = static_direction
            else:
                raise ValueError(
                    "evaluation needs a model or static direction"
                )
            result = preconditioned_cgls_reconstruction(
                operator,
                observation,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
                stages=stages,
                preconditioner=preconditioner,
            )
            metrics = _field_metrics(result.volume, truth)
            combined = normalized_field_loss(
                result.volume,
                truth,
                gradient_weight=float(
                    source_config["training"]["gradient_weight"]
                ),
            )
            active = _expanded_measurement_values(
                mask,
                rays_per_view=rays_per_view,
            )
            expanded_sigma = _expanded_measurement_values(
                sigma,
                rays_per_view=rays_per_view,
            )
            measurement = torch.linalg.vector_norm(
                (result.residual_uv / expanded_sigma).flatten(1),
                dim=1,
            ) / torch.linalg.vector_norm(
                (active * observation / expanded_sigma).flatten(1),
                dim=1,
            ).clamp_min(1e-12)
        for history in result.history:
            monotone &= bool(
                torch.all(
                    history["relative_objective_after"]
                    <= history["relative_objective_before"] + 2e-5
                )
            )
            gain_minimum = min(
                gain_minimum,
                float(torch.min(history["gain_minimum"])),
            )
            gain_maximum = max(
                gain_maximum,
                float(torch.max(history["gain_maximum"])),
            )
            geometric_defect = max(
                geometric_defect,
                float(
                    torch.max(
                        torch.abs(history["gain_geometric_mean"] - 1.0)
                    )
                ),
            )
        for offset, index in enumerate(range(start, stop)):
            rows.append(
                {
                    "sample_id": split.sample_ids[index],
                    "split": split.name,
                    "family": split.families[index],
                    "noise_profile": wrapped.noise_profiles[index],
                    "truth_operator": split.truth_operator,
                    "relative_noise": float(split.relative_noise[index]),
                    "active_view_count": int(
                        torch.sum(split.view_mask[index] > 0.5)
                    ),
                    "method": str(method),
                    "field_relative_l2": float(
                        metrics["field_relative_l2"][offset]
                    ),
                    "gradient_relative_l2": float(
                        metrics["gradient_relative_l2"][offset]
                    ),
                    "front_top10_f1": float(
                        metrics["front_top10_f1"][offset]
                    ),
                    "combined_loss": float(combined[offset]),
                    "measurement_relative_l2": float(measurement[offset]),
                }
            )
    _synchronize(device)
    calls = operator.call_report()
    return rows, {
        "method": str(method),
        "split": split.name,
        "sample_count": len(split.truth),
        "wall_seconds": float(time.perf_counter() - started),
        "logical_calls_per_sample": {"forward": 4, "adjoint": 4},
        "batch_invocations": {
            "forward": int(calls["forward_calls"]),
            "adjoint": int(calls["adjoint_calls"]),
        },
        "data_objective_monotone": bool(monotone),
        "gain_minimum": float(gain_minimum),
        "gain_maximum": float(gain_maximum),
        "gain_geometric_mean_maximum_defect": float(geometric_defect),
    }


def _validation_loss(
    *,
    wrapped: DevelopmentSplit,
    operator: Any,
    source_config: dict[str, Any],
    device: torch.device,
    model: GeometryConditionedSPDPreconditioner,
    batch_size: int,
) -> float:
    model.eval()
    with torch.no_grad():
        rows, _ = _evaluate(
            method="validation_probe",
            wrapped=wrapped,
            operator=operator,
            source_config=source_config,
            device=device,
            model=model,
            static_direction=None,
            batch_size=batch_size,
        )
    return float(np.mean([row["combined_loss"] for row in rows]))


def _train_seed(
    *,
    seed: int,
    train: DevelopmentSplit,
    validation: DevelopmentSplit,
    operator: Any,
    source_config: dict[str, Any],
    experiment_config: dict[str, Any],
    geometry_features: torch.Tensor,
    device: torch.device,
) -> tuple[GeometryConditionedSPDPreconditioner, dict[str, Any]]:
    torch.manual_seed(int(seed))
    model_config = experiment_config["model"]
    training = experiment_config["training"]
    model = GeometryConditionedSPDPreconditioner(
        tuple(int(value) for value in operator.grid_shape),
        view_geometry_features=geometry_features,
        hidden=int(model_config["hidden"]),
        base_sobolev_strength=float(
            model_config["base_sobolev_strength"]
        ),
        base_epsilon=float(model_config["base_epsilon"]),
        maximum_log_correction=float(
            model_config["maximum_log_correction"]
        ),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    validation_every = int(training["validation_every"])
    coefficient_weight = float(training["coefficient_l2_weight"])
    generator = torch.Generator().manual_seed(int(seed) + 9100)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    learning_curve: list[dict[str, float | int]] = []
    started = time.perf_counter()
    operator.reset_call_counts()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train.data.truth), generator=generator)
        model.train()
        train_losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            truth = train.data.truth[indices].to(device)
            observation = train.data.observation_uv[indices].to(device)
            sigma = train.data.sigma_by_view[indices].to(device)
            mask = train.data.view_mask[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            materialized = model.materialize(
                observation,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=int(
                    source_config["geometry"]["rays_per_view"]
                ),
            )
            result = preconditioned_cgls_reconstruction(
                operator,
                observation,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=int(
                    source_config["geometry"]["rays_per_view"]
                ),
                stages=4,
                preconditioner=materialized,
            )
            field_loss = normalized_field_loss(
                result.volume,
                truth,
                gradient_weight=float(training["gradient_weight"]),
            ).mean()
            coefficient_loss = (
                materialized.controller_coefficients.square().mean()
            )
            objective = field_loss + coefficient_weight * coefficient_loss
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            train_losses.append(float(field_loss.detach()))
        if epoch % validation_every == 0 or epoch == epochs:
            validation_loss = _validation_loss(
                wrapped=validation,
                operator=operator,
                source_config=source_config,
                device=device,
                model=model,
                batch_size=batch_size,
            )
            learning_curve.append(
                {
                    "epoch": int(epoch),
                    "train_combined_loss": float(np.mean(train_losses)),
                    "validation_combined_loss": float(validation_loss),
                }
            )
            if validation_loss < best_loss:
                best_loss = float(validation_loss)
                best_epoch = int(epoch)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    _synchronize(device)
    calls = operator.call_report()
    return model, {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_combined_loss": float(best_loss),
        "training_wall_seconds": float(time.perf_counter() - started),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "checkpoint_sha256_private": _state_sha256(best_state),
        "learning_curve": learning_curve,
        "last_validation_batch_invocations": {
            "forward": int(calls["forward_calls"]),
            "adjoint": int(calls["adjoint_calls"]),
        },
        "state_dict": best_state,
    }


def _seed_gates(
    summaries: list[dict[str, Any]],
    *,
    seeds: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["development_gates"]
    lookup = {
        (str(row["split"]), str(row["candidate_method"])): row
        for row in summaries
    }
    output = []
    passing = 0
    for seed in seeds:
        method = f"conditioned_pcgls_seed_{int(seed)}"
        validation = lookup[("risk_validation", method)]
        calibration = lookup[("risk_calibration", method)]
        gates = {
            "validation_mean_gain": validation["mean_field_gain_percent"]
            >= float(
                thresholds[
                    "validation_mean_field_gain_percent_minimum"
                ]
            ),
            "calibration_mean_gain": calibration["mean_field_gain_percent"]
            >= float(
                thresholds[
                    "calibration_mean_field_gain_percent_minimum"
                ]
            ),
            "validation_bootstrap_lower": validation[
                "bootstrap_mean_95_interval_percent"
            ][0]
            > float(
                thresholds[
                    "validation_bootstrap_lower_percent_minimum"
                ]
            ),
            "calibration_bootstrap_lower": calibration[
                "bootstrap_mean_95_interval_percent"
            ][0]
            > float(
                thresholds[
                    "calibration_bootstrap_lower_percent_minimum"
                ]
            ),
            "validation_harm": validation["harm_over_one_percent_rate"]
            <= float(
                thresholds[
                    "validation_harm_over_one_percent_rate_maximum"
                ]
            ),
            "calibration_harm": calibration["harm_over_one_percent_rate"]
            <= float(
                thresholds[
                    "calibration_harm_over_one_percent_rate_maximum"
                ]
            ),
        }
        passed = all(gates.values())
        passing += int(passed)
        output.append(
            {
                "seed": int(seed),
                "method": method,
                "gates": gates,
                "pass": bool(passed),
            }
        )
    required = int(thresholds["seeds_required_to_pass"])
    return {
        "per_seed": output,
        "passing_seed_count": int(passing),
        "required_passing_seed_count": required,
        "development_gate_pass": bool(passing >= required),
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    training = []
    for row in private["training"]:
        training.append(
            {
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key
                not in {
                    "checkpoint_sha256_private",
                    "state_dict",
                }
            }
        )
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "configuration": copy.deepcopy(private["configuration_public"]),
        "regeneration_checks": copy.deepcopy(
            private["regeneration_checks"]
        ),
        "training": training,
        "paired_gain_summary": copy.deepcopy(
            private["paired_gain_summary"]
        ),
        "seed_gates": copy.deepcopy(private["seed_gates"]),
        "execution": copy.deepcopy(private["execution"]),
        "runtime": copy.deepcopy(private["runtime"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
    }


def run_development(
    *,
    root: Path,
    config_path: Path,
    development_report_path: Path,
    view_root: Path,
    device_name: str,
    checkpoint_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    development_config = _load_json(
        root / str(config["source_development_config"])
    )
    source_config = _load_json(
        root / str(development_config["source_pilot"]["config"])
    )
    strong_frontier = _load_json(
        root / str(config["source_strong_frontier"])
    )
    development_report = _load_json(development_report_path)
    selected = strong_frontier["selected_candidates"]["pcgls_4"][
        "parameters"
    ]
    model_config = config["model"]
    expected = (
        float(model_config["base_sobolev_strength"]),
        float(model_config["base_epsilon"]),
        4,
    )
    actual = (
        float(selected["strength"]),
        float(selected["epsilon"]),
        int(selected["stages"]),
    )
    if actual != expected:
        raise ValueError("conditioned model base drifted from strong frontier")
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    started = time.perf_counter()
    geometry_config = development_config["geometry"]
    grid_size = int(geometry_config["grid_size"])
    rays_per_view = int(geometry_config["rays_per_view"])
    support = zero_outer_boundary_support((grid_size,) * 3).to(device)
    true_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry_config["true_finite_aperture_sample_count"]
        ),
    )
    nominal_geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry_config["nominal_finite_aperture_sample_count"]
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
    geometry_features = view_geometry_features_from_operator(
        nominal_operator,
        rays_per_view=rays_per_view,
    )
    used_masks: set[str] = set()
    splits: dict[str, DevelopmentSplit] = {}
    for split_name in (
        "risk_train",
        "risk_validation",
        "risk_calibration",
    ):
        wrapped, used_masks = _build_development_split(
            name=split_name,
            spec=development_config["development_splits"][split_name],
            config=development_config,
            source_config=source_config,
            true_operator=true_operator,
            nominal_operator=nominal_operator,
            device=device,
            forbidden_masks=used_masks,
        )
        splits[split_name] = wrapped
    source_rows = development_report["dataset_private"]["metric_rows"]
    for wrapped in splits.values():
        _verify_split_metadata(wrapped, source_rows)

    training_reports = []
    models: dict[int, GeometryConditionedSPDPreconditioner] = {}
    seeds = [int(value) for value in config["training"]["seeds"]]
    for seed in seeds:
        model, report = _train_seed(
            seed=seed,
            train=splits["risk_train"],
            validation=splits["risk_validation"],
            operator=nominal_operator,
            source_config=source_config,
            experiment_config=config,
            geometry_features=geometry_features,
            device=device,
        )
        models[seed] = model
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                report["state_dict"],
                checkpoint_dir / f"conditioned_pcgls_seed_{seed}.pt",
            )
        training_reports.append(report)

    rows: list[dict[str, Any]] = []
    execution: list[dict[str, Any]] = []
    static_direction = GeneralizedSobolevDirection(
        (grid_size,) * 3,
        strength=float(model_config["base_sobolev_strength"]),
        epsilon=float(model_config["base_epsilon"]),
    ).to(device)
    evaluation_splits = ("risk_validation", "risk_calibration")
    for split_name in evaluation_splits:
        baseline_rows, ledger = _evaluate(
            method="static_pcgls4",
            wrapped=splits[split_name],
            operator=nominal_operator,
            source_config=source_config,
            device=device,
            model=None,
            static_direction=static_direction,
            batch_size=int(config["training"]["batch_size"]),
        )
        rows.extend(baseline_rows)
        execution.append(ledger)
        for seed, model in models.items():
            candidate_rows, ledger = _evaluate(
                method=f"conditioned_pcgls_seed_{seed}",
                wrapped=splits[split_name],
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                model=model,
                static_direction=None,
                batch_size=int(config["training"]["batch_size"]),
            )
            rows.extend(candidate_rows)
            execution.append(ledger)
    summaries = []
    for seed in seeds:
        method = f"conditioned_pcgls_seed_{seed}"
        for split_index, split_name in enumerate(evaluation_splits):
            summaries.append(
                paired_gain_summary(
                    rows,
                    split=split_name,
                    candidate_method=method,
                    bootstrap_seed=seed + 100 * (split_index + 1),
                )
            )
    seed_gates = _seed_gates(summaries, seeds=seeds, config=config)
    ledgers_valid = all(
        row["logical_calls_per_sample"] == {"forward": 4, "adjoint": 4}
        and row["data_objective_monotone"]
        and row["gain_minimum"] > 0.0
        and row["gain_geometric_mean_maximum_defect"] <= 2e-5
        for row in execution
    )
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "algorithm_schema": CONDITIONED_PCGLS_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "REAL_PSU_SUPPORT_GEOMETRY_WITH_ANALYTIC_REACTION_MORPHOLOGY_"
            "AND_SYNTHETIC_CAMERA_NOISE_POSTOPEN_DEVELOPMENT_ONLY"
        ),
        "configuration_private": {
            "root": str(root.resolve()),
            "config_path": str(config_path.resolve()),
            "development_report_path": str(
                development_report_path.resolve()
            ),
            "view_root": str(view_root.resolve()),
            "checkpoint_dir": (
                None
                if checkpoint_dir is None
                else str(checkpoint_dir.resolve())
            ),
            "device": str(device),
        },
        "configuration_public": copy.deepcopy(config),
        "regeneration_checks": {
            "risk_train_metadata_matches_frozen_rows": True,
            "risk_validation_metadata_matches_frozen_rows": True,
            "risk_calibration_metadata_matches_frozen_rows": True,
            "base_matches_validation_selected_pcgls4": True,
            "checkpoint_selection_uses_only_risk_validation": True,
            "risk_calibration_not_used_for_training_or_selection": True,
            "opened_fresh_not_loaded": True,
            "fixed_spd_and_call_ledgers_pass": bool(ledgers_valid),
        },
        "training": training_reports,
        "metric_rows_private": rows,
        "paired_gain_summary": summaries,
        "seed_gates": seed_gates,
        "execution": execution,
        "runtime": {
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": int(_max_rss_bytes()),
        },
        "claim_boundary": {
            "postopen_development_only": True,
            "fresh_values_loaded": False,
            "experimental_field_truth_used": False,
            "real_psu_measurement_values_used": False,
            "analytic_morphology_is_cfd": False,
            "passing_development_is_confirmatory": False,
            "algorithm_superiority": False,
        },
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "demo_t16_operator/configs/"
            "psu_b0_conditioned_pcgls_development_v1.json"
        ),
    )
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_development(
        root=args.root,
        config_path=args.config,
        development_report_path=args.development_report,
        view_root=args.view_root,
        device_name=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    if args.private_output is not None:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        private_serializable = copy.deepcopy(private)
        for row in private_serializable["training"]:
            row.pop("state_dict", None)
        args.private_output.write_text(
            json.dumps(
                private_serializable,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.public_output is not None:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
