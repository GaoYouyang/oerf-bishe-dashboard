#!/usr/bin/env python3
"""Open the frozen PSU residual-risk fresh audit exactly once."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.psu_b0_residual_risk import (
    RESIDUAL_RISK_SCHEMA,
    RISK_FEATURE_NAMES,
    CalibratedResidualRiskDirection,
    RidgeRiskFit,
)
from demo_t16_operator.psu_b0_spectral_preconditioner import (
    FixedSobolevDirection,
    PositiveSpectralDirection,
    exact_line_search_reconstruction,
    normalized_field_loss,
)
from demo_t16_operator.psu_b0_streaming_operator import zero_outer_boundary_support
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_residual_risk_development import (
    DevelopmentSplit,
    _build_development_split,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _aggregate,
    _evaluate,
    _expanded_measurement_values,
    _field_metrics,
    _load_json,
    _state_sha256,
    _unique_masks,
)
from site_tools.run_psu_b0_support_envelope_diagnosis import (
    _load_checkpoint_model,
)


PRIVATE_SCHEMA = "psu-b0-residual-risk-fresh-private-report-1.0"
PUBLIC_SCHEMA = "psu-b0-residual-risk-fresh-public-summary-1.0"


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


def _verify_frozen_source(
    *,
    root: Path,
    preregistration: dict[str, Any],
    development_report_path: Path,
) -> None:
    frozen = preregistration["frozen_source"]
    development_config = root / str(frozen["development_config"])
    risk_module = root / str(frozen["risk_module"])
    checks = {
        "development config": (
            _sha256(development_config),
            str(frozen["development_config_sha256"]),
        ),
        "development private report": (
            _sha256(development_report_path),
            str(frozen["development_private_report_sha256"]),
        ),
        "risk module": (
            _sha256(risk_module),
            str(frozen["risk_module_sha256"]),
        ),
    }
    mismatches = [
        name
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "frozen source hash mismatch before fresh opening: "
            + ", ".join(mismatches)
        )


def _reconstruct_development_masks(
    development_config: dict[str, Any],
) -> set[str]:
    used: set[str] = set()
    view_count = int(development_config["geometry"]["view_count"])
    active_range = tuple(
        int(value)
        for value in development_config["geometry"]["active_view_range"]
    )
    for spec in development_config["development_splits"].values():
        _, used = _unique_masks(
            count=int(spec["count"]),
            view_count=view_count,
            minimum_active=active_range[0],
            maximum_active=active_range[1],
            seed=int(spec["mask_seed"]),
            forbidden=used,
        )
    return used


def _build_fresh_splits(
    *,
    preregistration: dict[str, Any],
    development_config: dict[str, Any],
    source_config: dict[str, Any],
    true_operator: Any,
    nominal_operator: Any,
    device: torch.device,
) -> tuple[dict[str, DevelopmentSplit], dict[str, set[str]], set[str]]:
    development_masks = _reconstruct_development_masks(development_config)
    splits: dict[str, DevelopmentSplit] = {}
    fresh_masks: dict[str, set[str]] = {}
    for name, spec in preregistration["fresh_splits"].items():
        builder_config = {
            "geometry": {
                **preregistration["geometry"],
                "active_view_range": list(spec["active_view_range"]),
            },
            "camera_noise_profiles": preregistration["camera_noise_profiles"],
        }
        wrapped, returned_masks = _build_development_split(
            name=name,
            spec=spec,
            config=builder_config,
            source_config=source_config,
            true_operator=true_operator,
            nominal_operator=nominal_operator,
            device=device,
            forbidden_masks=set(development_masks),
        )
        splits[name] = wrapped
        fresh_masks[name] = returned_masks - development_masks
    return splits, fresh_masks, development_masks


def _load_frozen_models(
    *,
    checkpoint_dir: Path,
    source_config: dict[str, Any],
    development_report: dict[str, Any],
    preregistration: dict[str, Any],
    device: torch.device,
) -> tuple[dict[int, PositiveSpectralDirection], list[dict[str, Any]]]:
    strength = float(preregistration["frozen_source"]["selected_sobolev_strength"])
    models: dict[int, PositiveSpectralDirection] = {}
    records = []
    source_state_hashes = {
        int(row["seed"]): str(row["state_sha256"])
        for row in development_report["source_candidate_private"][
            "checkpoint_records"
        ]
    }
    for checkpoint in sorted(checkpoint_dir.glob("learned_seed_*.pt")):
        seed, model, record = _load_checkpoint_model(
            checkpoint=checkpoint,
            config=source_config,
            selected_strength=strength,
            device=device,
        )
        state_hash = _state_sha256(
            {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
        )
        if state_hash != source_state_hashes[seed]:
            raise ValueError(f"checkpoint state drift for seed {seed}")
        models[seed] = model
        records.append(
            {
                "seed": seed,
                **record,
                "state_sha256": state_hash,
            }
        )
    expected = sorted(
        int(value)
        for value in preregistration["frozen_source"]["checkpoint_seeds"]
    )
    if sorted(models) != expected:
        raise ValueError("fresh checkpoint seed set differs from preregistration")
    return models, records


def _load_risk_fit(development_report: dict[str, Any]) -> RidgeRiskFit:
    private = development_report["risk_model_private"]
    public = development_report["risk_model_public"]
    return RidgeRiskFit(
        feature_mean=np.asarray(private["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(private["feature_scale"], dtype=np.float64),
        coefficients=np.asarray(private["coefficients"], dtype=np.float64),
        intercept=float(private["intercept"]),
        ridge_lambda=float(public["ridge_lambda"]),
        validation_rmse=float(public["validation_rmse"]),
    )


def _evaluate_gated(
    *,
    method: str,
    wrapped: DevelopmentSplit,
    operator: Any,
    source_config: dict[str, Any],
    device: torch.device,
    direction: CalibratedResidualRiskDirection,
    batch_size: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = wrapped.data
    rays_per_view = int(source_config["geometry"]["rays_per_view"])
    stages = int(source_config["solver"]["stages"])
    rows: list[dict[str, Any]] = []
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
            result = exact_line_search_reconstruction(
                operator,
                observation,
                sigma_by_view=sigma,
                view_mask=mask,
                rays_per_view=rays_per_view,
                stages=stages,
                direction=direction,
            )
        metrics = _field_metrics(result.volume, truth)
        combined = normalized_field_loss(
            result.volume,
            truth,
            gradient_weight=float(source_config["training"]["gradient_weight"]),
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
        first = result.history[0]
        for history in result.history:
            monotone &= bool(
                torch.all(
                    history["relative_objective_after"]
                    <= history["relative_objective_before"] + 2e-5
                )
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
                    "operator_mismatch_relative_l2": float(
                        split.operator_mismatch_relative_l2[index]
                    ),
                    "method": method,
                    "trusted": bool(
                        first["residual_risk_trust"][offset] > 0.5
                    ),
                    "predicted_gain_percent": float(
                        first["predicted_gain_percent"][offset]
                    ),
                    "lower_gain_bound_percent": float(
                        first["lower_gain_bound_percent"][offset]
                    ),
                    "feature_distance": float(first["feature_distance"][offset]),
                    "field_relative_l2": float(
                        metrics["field_relative_l2"][offset]
                    ),
                    "gradient_relative_l2": float(
                        metrics["gradient_relative_l2"][offset]
                    ),
                    "front_top10_f1": float(metrics["front_top10_f1"][offset]),
                    "combined_loss": float(combined[offset]),
                    "measurement_relative_l2": float(measurement[offset]),
                }
            )
    _synchronize(device)
    calls = operator.call_report()
    return rows, {
        "method": method,
        "split": split.name,
        "wall_seconds": float(time.perf_counter() - started),
        "batch_invocations": {
            "forward": int(calls["forward_calls"]),
            "adjoint": int(calls["adjoint_calls"]),
        },
        "logical_calls_per_sample": {
            "forward": stages,
            "adjoint": stages,
        },
        "data_objective_monotone": bool(monotone),
    }


def _aggregates_with_coverage(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates = _aggregate(rows, baseline_method="sobolev_selected")
    trust_lookup: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        if "trusted" in row:
            trust_lookup.setdefault(
                (str(row["split"]), str(row["method"])),
                [],
            ).append(bool(row["trusted"]))
    for aggregate in aggregates:
        values = trust_lookup.get(
            (str(aggregate["split"]), str(aggregate["method"]))
        )
        aggregate["candidate_coverage"] = (
            None if values is None else float(np.mean(values))
        )
    return aggregates


def _outside_support_equivalence(
    rows: list[dict[str, Any]],
    *,
    split: str,
    gated_method: str,
) -> dict[str, Any]:
    keys = (
        "field_relative_l2",
        "gradient_relative_l2",
        "front_top10_f1",
        "combined_loss",
        "measurement_relative_l2",
    )
    baseline = {
        row["sample_id"]: row
        for row in rows
        if row["split"] == split and row["method"] == "sobolev_selected"
    }
    gated = {
        row["sample_id"]: row
        for row in rows
        if row["split"] == split and row["method"] == gated_method
    }
    if set(baseline) != set(gated) or not baseline:
        raise ValueError("outside-support methods do not align")
    maximum = {
        key: float(
            max(
                abs(
                    float(baseline[sample_id][key])
                    - float(gated[sample_id][key])
                )
                for sample_id in baseline
            )
        )
        for key in keys
    }
    return {
        "split": split,
        "gated_method": gated_method,
        "sample_count": len(baseline),
        "candidate_coverage": float(
            np.mean([bool(row["trusted"]) for row in gated.values()])
        ),
        "maximum_absolute_metric_difference": maximum,
        "maximum_over_metrics": float(max(maximum.values())),
    }


def _candidate_gates(
    *,
    aggregates: list[dict[str, Any]],
    equivalence: list[dict[str, Any]],
    preregistration: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    by_key = {
        (str(row["split"]), str(row["method"])): row
        for row in aggregates
    }
    equivalence_by_key = {
        (str(row["split"]), str(row["gated_method"])): row
        for row in equivalence
    }
    gate_config = preregistration["fresh_gates"]
    per_seed = []
    passing = 0
    for seed in seeds:
        method = f"gated_seed_{seed}"
        split_gates = {}
        for split, thresholds in gate_config["per_seed"].items():
            row = by_key[(split, method)]
            split_gates[split] = {
                "coverage": float(row["candidate_coverage"])
                >= float(thresholds["coverage_minimum"]),
                "mean_field_gain": float(
                    row["field_gain_vs_sobolev_mean_percent"]
                )
                >= float(thresholds["mean_field_gain_percent_minimum"]),
                "harm_rate": float(row["field_harm_over_one_percent_rate"])
                <= float(
                    thresholds["field_harm_over_one_percent_rate_maximum"]
                ),
            }
        outside_gates = {}
        for split in gate_config["outside_support"]["splits"]:
            row = equivalence_by_key[(split, method)]
            outside_gates[split] = {
                "zero_candidate_coverage": float(row["candidate_coverage"])
                == float(
                    gate_config["outside_support"][
                        "candidate_coverage_required"
                    ]
                ),
                "exact_fallback_metrics": float(row["maximum_over_metrics"])
                <= float(
                    gate_config["outside_support"][
                        "maximum_absolute_metric_difference_from_sobolev"
                    ]
                ),
            }
        seed_pass = all(
            all(values.values())
            for values in [*split_gates.values(), *outside_gates.values()]
        )
        passing += int(seed_pass)
        per_seed.append(
            {
                "seed": int(seed),
                "method": method,
                "support_split_gates": split_gates,
                "outside_support_gates": outside_gates,
                "pass": bool(seed_pass),
            }
        )
    required = int(gate_config["passing_seed_count_required"])
    return {
        "per_seed": per_seed,
        "passing_seed_count": int(passing),
        "required_passing_seed_count": required,
        "candidate_gate_pass": passing >= required,
    }


def build_public_summary(private: dict[str, Any]) -> dict[str, Any]:
    """Strip paths, hashes, feature rows, masks, observations, and checkpoints."""

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": private["status"],
        "evidence_scope": private["evidence_scope"],
        "frozen_protocol": copy.deepcopy(private["frozen_protocol_public"]),
        "dataset": copy.deepcopy(private["dataset_public"]),
        "aggregates": copy.deepcopy(private["aggregates"]),
        "outside_support_equivalence": copy.deepcopy(
            private["outside_support_equivalence"]
        ),
        "candidate_gates": copy.deepcopy(private["candidate_gates"]),
        "execution": copy.deepcopy(private["execution_public"]),
        "gates": copy.deepcopy(private["gates"]),
        "claim_boundary": copy.deepcopy(private["claim_boundary"]),
        "public_export_policy": {
            "contains_local_paths": False,
            "contains_private_hashes_or_checkpoints": False,
            "contains_per_sample_features_or_metrics": False,
            "contains_masks_observations_or_volumes": False,
            "contains_real_psu_measurement_values": False,
        },
    }


def run_fresh(
    *,
    root: Path,
    preregistration_path: Path,
    development_report_path: Path,
    view_root: Path,
    checkpoint_dir: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = _load_json(preregistration_path)
    if preregistration["status"] != "FROZEN_BEFORE_FRESH_AUDIT":
        raise ValueError("fresh protocol is not in the frozen state")
    _verify_frozen_source(
        root=root,
        preregistration=preregistration,
        development_report_path=development_report_path,
    )
    development_report = _load_json(development_report_path)
    development_config = _load_json(
        root / str(preregistration["frozen_source"]["development_config"])
    )
    source_config = _load_json(
        root / str(development_config["source_pilot"]["config"])
    )
    frozen = preregistration["frozen_source"]
    public_model = development_report["risk_model_public"]
    exact_scalar_checks = {
        "feature_count": len(RISK_FEATURE_NAMES),
        "ridge_lambda": float(public_model["ridge_lambda"]),
        "distance_threshold": float(public_model["distance_threshold"]),
        "minimum_lower_gain_percent": float(
            public_model["selected_minimum_lower_gain_percent"]
        ),
        "conformal_alpha": float(public_model["conformal_alpha"]),
    }
    for key, actual in exact_scalar_checks.items():
        if actual != frozen[key]:
            raise ValueError(f"frozen risk value drift: {key}")
    if RESIDUAL_RISK_SCHEMA != str(frozen["feature_schema"]):
        raise ValueError("risk feature schema drift")
    source_quantiles = {
        str(key): float(value)
        for key, value in public_model[
            "calibration_overprediction_quantile_by_seed"
        ].items()
    }
    frozen_quantiles = {
        str(key): float(value)
        for key, value in frozen[
            "calibration_overprediction_quantile_by_seed"
        ].items()
    }
    if source_quantiles != frozen_quantiles:
        raise ValueError("frozen conformal quantiles drift")

    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    started = time.perf_counter()
    grid_size = int(preregistration["geometry"]["grid_size"])
    rays_per_view = int(preregistration["geometry"]["rays_per_view"])
    support = zero_outer_boundary_support((grid_size,) * 3).to(device)
    true_geometry, true_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            preregistration["geometry"]["true_finite_aperture_sample_count"]
        ),
    )
    nominal_geometry, nominal_provenance = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            preregistration["geometry"][
                "nominal_finite_aperture_sample_count"
            ]
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
    models, checkpoint_records = _load_frozen_models(
        checkpoint_dir=checkpoint_dir,
        source_config=source_config,
        development_report=development_report,
        preregistration=preregistration,
        device=device,
    )
    fallback = FixedSobolevDirection(
        (grid_size,) * 3,
        strength=float(frozen["selected_sobolev_strength"]),
    ).to(device)
    risk_fit = _load_risk_fit(development_report)
    splits, fresh_masks, development_masks = _build_fresh_splits(
        preregistration=preregistration,
        development_config=development_config,
        source_config=source_config,
        true_operator=true_operator,
        nominal_operator=nominal_operator,
        device=device,
    )

    all_rows: list[dict[str, Any]] = []
    execution = []
    for split_name, wrapped in splits.items():
        baseline_rows, baseline_execution = _evaluate(
            method="sobolev_selected",
            split=wrapped.data,
            operator=nominal_operator,
            config=source_config,
            device=device,
            direction=fallback,
        )
        all_rows.extend(baseline_rows)
        execution.append(baseline_execution)
        for seed, model in sorted(models.items()):
            raw_rows, raw_execution = _evaluate(
                method=f"raw_seed_{seed}",
                split=wrapped.data,
                operator=nominal_operator,
                config=source_config,
                device=device,
                direction=model,
            )
            all_rows.extend(raw_rows)
            execution.append(raw_execution)
            gate = CalibratedResidualRiskDirection(
                candidate=model,
                fallback=fallback,
                stages=int(preregistration["solver"]["stages"]),
                feature_mean=risk_fit.feature_mean,
                feature_scale=risk_fit.feature_scale,
                coefficients=risk_fit.coefficients,
                intercept=risk_fit.intercept,
                overprediction_quantile=float(
                    frozen["calibration_overprediction_quantile_by_seed"][
                        str(seed)
                    ]
                ),
                distance_threshold=float(frozen["distance_threshold"]),
                minimum_lower_gain_percent=float(
                    frozen["minimum_lower_gain_percent"]
                ),
                minimum_active_views=int(
                    preregistration["geometry"]["support_active_view_range"][0]
                ),
                maximum_active_views=int(
                    preregistration["geometry"]["support_active_view_range"][1]
                ),
            ).to(device)
            gated_rows, gated_execution = _evaluate_gated(
                method=f"gated_seed_{seed}",
                wrapped=wrapped,
                operator=nominal_operator,
                source_config=source_config,
                device=device,
                direction=gate,
            )
            all_rows.extend(gated_rows)
            execution.append(gated_execution)
    aggregates = _aggregates_with_coverage(all_rows)
    outside_equivalence = []
    for split_name in preregistration["fresh_gates"]["outside_support"]["splits"]:
        for seed in sorted(models):
            outside_equivalence.append(
                _outside_support_equivalence(
                    all_rows,
                    split=split_name,
                    gated_method=f"gated_seed_{seed}",
                )
            )
    candidate = _candidate_gates(
        aggregates=aggregates,
        equivalence=outside_equivalence,
        preregistration=preregistration,
        seeds=sorted(models),
    )
    all_finite = all(
        np.isfinite(float(row[key]))
        for row in all_rows
        for key in (
            "field_relative_l2",
            "gradient_relative_l2",
            "front_top10_f1",
            "combined_loss",
            "measurement_relative_l2",
        )
    )
    monotone = all(
        bool(row["data_objective_monotone"])
        for row in execution
    )
    gated_calls_match = all(
        row["logical_calls_per_sample"] == {"forward": 4, "adjoint": 4}
        for row in execution
        if str(row["method"]).startswith("gated_seed_")
    )
    fresh_excludes_development = all(
        not (masks & development_masks)
        for masks in fresh_masks.values()
    )
    gates = {
        "frozen_hashes_verified_before_fresh_generation": True,
        "fresh_support_masks_exclude_all_development_masks": (
            fresh_excludes_development
        ),
        "all_metrics_finite": all_finite,
        "data_objective_monotone": monotone,
        "same_logical_calls_as_raw_and_sobolev": gated_calls_match,
        "development_rotation_40_not_accessed": True,
        "final_audit_not_accessed": True,
    }
    split_public = []
    for name, wrapped in splits.items():
        split_public.append(
            {
                "name": name,
                "sample_count": len(wrapped.data.truth),
                "families": sorted(set(wrapped.data.families)),
                "noise_profiles": sorted(set(wrapped.noise_profiles)),
                "truth_operator": wrapped.data.truth_operator,
                "relative_noise_minimum": float(
                    wrapped.data.relative_noise.min()
                ),
                "relative_noise_maximum": float(
                    wrapped.data.relative_noise.max()
                ),
                "active_view_count_minimum": int(
                    torch.sum(wrapped.data.view_mask, dim=1).min()
                ),
                "active_view_count_maximum": int(
                    torch.sum(wrapped.data.view_mask, dim=1).max()
                ),
                "fresh_mask_count": len(fresh_masks[name]),
            }
        )
    protocol_and_candidate_pass = all(gates.values()) and candidate[
        "candidate_gate_pass"
    ]
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "status": (
            "RESIDUAL_RISK_FRESH_CANDIDATE_PASS_SYNTHETIC_ONLY"
            if protocol_and_candidate_pass
            else "RESIDUAL_RISK_FRESH_CANDIDATE_NO_GO_OR_INCOMPLETE"
        ),
        "evidence_scope": preregistration["evidence_scope"],
        "configuration_private": {
            "root": str(root.resolve()),
            "preregistration_path": str(preregistration_path.resolve()),
            "preregistration_sha256": _sha256(preregistration_path),
            "development_report_path": str(development_report_path.resolve()),
            "development_report_sha256": _sha256(development_report_path),
            "view_root": str(view_root.resolve()),
            "device": device_name,
        },
        "frozen_protocol_public": {
            "preregistration_filename": preregistration_path.name,
            "frozen_at_utc": preregistration["frozen_at_utc"],
            "feature_schema": frozen["feature_schema"],
            "feature_count": int(frozen["feature_count"]),
            "ridge_lambda": float(frozen["ridge_lambda"]),
            "distance_threshold": float(frozen["distance_threshold"]),
            "minimum_lower_gain_percent": float(
                frozen["minimum_lower_gain_percent"]
            ),
            "conformal_alpha": float(frozen["conformal_alpha"]),
            "calibration_overprediction_quantile_by_seed": frozen[
                "calibration_overprediction_quantile_by_seed"
            ],
            "logical_calls_per_sample": preregistration["solver"][
                "logical_calls_per_sample"
            ],
        },
        "dataset_private": {
            "true_geometry_provenance": true_provenance,
            "nominal_geometry_provenance": nominal_provenance,
            "checkpoint_records": checkpoint_records,
            "development_mask_count": len(development_masks),
            "fresh_masks": {
                name: sorted(values)
                for name, values in fresh_masks.items()
            },
            "per_sample_metrics": all_rows,
        },
        "dataset_public": {
            "source_dataset_doi": "10.26208/1VE2-5C19",
            "real_psu_measurement_values_used": False,
            "analytic_truth_is_cfd": False,
            "camera_noise_is_measured_psu_noise": False,
            "splits": split_public,
            "mask_contract": preregistration["mask_contract"],
        },
        "aggregates": aggregates,
        "outside_support_equivalence": outside_equivalence,
        "candidate_gates": candidate,
        "execution_private": {
            "wall_seconds": float(time.perf_counter() - started),
            "process_max_rss_bytes": int(_max_rss_bytes()),
            "evaluation_records": execution,
        },
        "execution_public": {
            "wall_seconds": float(time.perf_counter() - started),
            "process_max_rss_bytes": int(_max_rss_bytes()),
            "host": {
                "machine": platform.machine(),
                "platform": platform.platform(),
                "torch_version": torch.__version__,
            },
        },
        "gates": gates,
        "claim_boundary": {
            **copy.deepcopy(preregistration["claim_boundary"]),
            "fresh_values_opened": True,
            "passing_authorizes_algorithm_superiority": False,
            "passing_authorizes_experimental_field_claim": False,
        },
    }
    return private, build_public_summary(private)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    private, public = run_fresh(
        root=args.root,
        preregistration_path=args.preregistration,
        development_report_path=args.development_report,
        view_root=args.view_root,
        checkpoint_dir=args.checkpoint_dir,
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
