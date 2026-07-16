"""Shared, frozen machinery for the split v3k-F selection/audit programs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .adjoint_landweber import projected_bb_trajectory
    from .counterfactual_geometry import CounterfactualInputFactory
    from .models import make_model
    from .noise_stopping import (
        active_observation_noise_scale,
        effective_operator_calls,
        first_crossing,
        gather_path,
        generator_noise_scale,
        grouped_validation_assignment,
        residual_statistics,
        white_noise_ncp_thresholds,
    )
    from .run_v3k_a_counterfactual_supervision import (
        load_private_dataset,
        precompute_base_predictions,
    )
    from . import run_v3k_d_strong_numerical_controls as v3d
    from . import run_v3k_e_projected_bb_gate as v3e
    from .train_eval import choose_device
except ImportError:
    from adjoint_landweber import projected_bb_trajectory
    from counterfactual_geometry import CounterfactualInputFactory
    from models import make_model
    from noise_stopping import (
        active_observation_noise_scale,
        effective_operator_calls,
        first_crossing,
        gather_path,
        generator_noise_scale,
        grouped_validation_assignment,
        residual_statistics,
        white_noise_ncp_thresholds,
    )
    from run_v3k_a_counterfactual_supervision import (
        load_private_dataset,
        precompute_base_predictions,
    )
    import run_v3k_d_strong_numerical_controls as v3d
    import run_v3k_e_projected_bb_gate as v3e
    from train_eval import choose_device


ROOT = Path(__file__).resolve().parent
METHOD_LABELS = {
    "feasible_fno": "FNO + hard support",
    "fno_geometry": "FNO + fixed Landweber",
    "fno_pbb_fixed64": "FNO + fixed PBB-64",
    "fno_pbb_discrepancy": "FNO + PBB + active discrepancy",
    "fno_pbb_camera_discrepancy": "FNO + PBB + camera-balanced discrepancy",
    "fno_pbb_ncp": "FNO + PBB + NCP whiteness",
    "fno_pbb_hybrid": "FNO + PBB + discrepancy AND NCP",
    "fno_pbb_generator_sigma": "FNO + PBB + generator-sigma discrepancy (oracle)",
    "fno_pbb_truth_oracle": "FNO + PBB + truth-best stop (oracle)",
}
DEPLOYABLE_METHODS = {
    "fno_pbb_discrepancy",
    "fno_pbb_camera_discrepancy",
    "fno_pbb_ncp",
    "fno_pbb_hybrid",
}
ORACLE_METHODS = {"fno_pbb_generator_sigma", "fno_pbb_truth_oracle"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_context(config: dict, device_name: str | None = None) -> dict[str, object]:
    dataset_config = read_json(ROOT / "configs" / str(config["dataset_config"]))
    private_path = ROOT / "results" / str(config["private_dataset_npz"])
    checkpoint_path = ROOT / "results" / str(config["base_checkpoint"])
    baseline_path = ROOT / "results" / str(config["baseline_selection_commit"])
    pbb_path = ROOT / "results" / str(config["pbb_selection_commit"])
    data = load_private_dataset(private_path)
    design = config["pair_design"]
    factory = CounterfactualInputFactory(
        data, float(design["frozen_fno_input_ridge_relative"])
    )
    masks = np.asarray(
        [factory.mask(identifier) for identifier in sorted(factory.catalog)],
        dtype=np.float64,
    )
    normalization = v3d.geometry_normalization(data["forward_matrix"], masks)
    global_spectral = max(
        float(value["spectral_constant"]) for value in normalization.values()
    )
    _, regime_mapping, _ = v3d.spectral_regimes(factory, normalization)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = make_model(
        "fno",
        dataset_config["models"]["fno"],
        int(data["inputs"].shape[1]),
        residual=True,
    )
    model.load_state_dict(checkpoint, strict=True)
    pbb_commit = read_json(pbb_path)
    pbb_choice = pbb_commit["selected_projected_bb"][
        str(config["frozen_pbb"]["selection_key"])
    ]
    reference = config["white_noise_ncp_reference"]
    ncp_thresholds = white_noise_ncp_thresholds(
        int(data["field"].shape[1]),
        int(data["observation"].shape[-1]),
        int(data["view_mask"].sum(axis=1)[0]),
        samples=int(reference["samples"]),
        quantile=float(reference["quantile"]),
        seed=int(reference["seed"]),
    )
    return {
        "config": config,
        "dataset_config": dataset_config,
        "data": data,
        "factory": factory,
        "normalization": normalization,
        "global_spectral": global_spectral,
        "regime_mapping": regime_mapping,
        "model": model,
        "device": choose_device(device_name or "cpu"),
        "baseline_selection": read_json(baseline_path)["selected"],
        "pbb_choice": pbb_choice,
        "ncp_thresholds": ncp_thresholds,
        "private_path": private_path,
        "checkpoint_path": checkpoint_path,
        "baseline_path": baseline_path,
        "pbb_path": pbb_path,
    }


def build_bundle(context: dict[str, object], split: str, partition: str) -> dict[str, object]:
    config = context["config"]
    data = context["data"]
    factory = context["factory"]
    dataset, pairs = v3d.make_dataset(data, factory, split, partition, config["pair_design"])
    started = time.perf_counter()
    raw_base = precompute_base_predictions(
        context["model"], dataset, context["device"], batch_size=16
    )
    fno_seconds = time.perf_counter() - started
    baseline, baseline_runtimes = v3e.baseline_predictions(
        data,
        dataset,
        raw_base,
        factory,
        context["normalization"],
        context["global_spectral"],
        context["regime_mapping"],
        context["baseline_selection"],
    )
    sources = v3d.source_indices(dataset)
    observation = np.asarray(data["observation"][sources], dtype=np.float64)
    clean = np.asarray(data["clean_observation"][sources], dtype=np.float64)
    truth = np.asarray(data["field"][sources], dtype=np.float64)
    masks = v3d.geometry_masks(factory, dataset)
    spectral = v3d.spectral_per_sample(masks, context["normalization"])
    maximum = int(config["frozen_pbb"]["maximum_iterations"])
    choice = context["pbb_choice"]
    started = time.perf_counter()
    trajectory, diagnostics = projected_bb_trajectory(
        baseline["feasible_fno"],
        observation,
        data["forward_matrix"],
        masks,
        data["support"],
        range(1, maximum + 1),
        spectral,
        str(choice["bb_variant"]),
        float(choice["initial_step_fraction"]),
        float(choice["normalized_step_min_bound"]),
        float(choice["normalized_step_max_bound"]),
        float(config.get("curvature_floor_relative", 1e-12)),
        record_residual=True,
    )
    pbb_seconds = time.perf_counter() - started
    path = np.stack(
        [baseline["feasible_fno"]]
        + [trajectory[iteration] for iteration in range(1, maximum + 1)]
    )
    q = np.asarray(data["noise_level"][sources], dtype=np.float64)
    self_sigma = active_observation_noise_scale(observation, masks, q)
    oracle_sigma = generator_noise_scale(clean, q)
    return {
        "split": split,
        "dataset": dataset,
        "pairs": pairs,
        "sources": sources,
        "sample_seeds": np.asarray(data["sample_seed"][sources], dtype=np.int64),
        "noise_level": q,
        "truth": truth,
        "observation": observation,
        "masks": masks,
        "path": path,
        "residual_history": diagnostics["residual_before"],
        "self_stats": residual_statistics(
            diagnostics["residual_before"], masks, self_sigma
        ),
        "generator_stats": residual_statistics(
            diagnostics["residual_before"], masks, oracle_sigma
        ),
        "baseline": baseline,
        "baseline_runtimes": baseline_runtimes,
        "fno_seconds": fno_seconds,
        "pbb_seconds": pbb_seconds,
        "self_sigma": self_sigma,
        "generator_sigma": oracle_sigma,
    }


def validation_roles(bundle: dict[str, object], config: dict) -> tuple[np.ndarray, list[dict[str, object]]]:
    partition = config["validation_partition"]
    return grouped_validation_assignment(
        bundle["sources"],
        bundle["sample_seeds"],
        int(partition["tune_field_count"]),
        int(partition["assignment_seed"]),
    )


def stopping_condition(
    family: str,
    bundle: dict[str, object],
    parameters: dict[str, float],
    ncp_thresholds: dict[str, float],
) -> np.ndarray:
    self_stats = bundle["self_stats"]
    if family == "self_discrepancy":
        return self_stats["discrepancy_pooled"] <= float(parameters["tau"])
    if family == "camera_discrepancy":
        tau = float(parameters["tau"])
        return (self_stats["discrepancy_pooled"] <= tau) & (
            self_stats["discrepancy_camera_max"]
            <= tau * float(parameters["camera_max_factor"])
        )
    if family == "ncp":
        multiplier = float(parameters["ncp_multiplier"])
        return (
            self_stats["ncp_camera_mean"] <= ncp_thresholds["mean"] * multiplier
        ) & (
            self_stats["ncp_camera_max"]
            <= ncp_thresholds["maximum"] * multiplier
        )
    if family == "hybrid":
        tau = float(parameters["tau"])
        multiplier = float(parameters["ncp_multiplier"])
        return (
            (self_stats["discrepancy_pooled"] <= tau)
            & (
                self_stats["discrepancy_camera_max"]
                <= tau * float(parameters["camera_max_factor"])
            )
            & (
                self_stats["ncp_camera_mean"]
                <= ncp_thresholds["mean"] * multiplier
            )
            & (
                self_stats["ncp_camera_max"]
                <= ncp_thresholds["maximum"] * multiplier
            )
        )
    if family == "generator_discrepancy":
        return bundle["generator_stats"]["discrepancy_pooled"] <= float(
            parameters["tau"]
        )
    raise ValueError(f"unknown stopping family: {family}")


def stop_indices(
    family: str,
    bundle: dict[str, object],
    parameters: dict[str, float],
    ncp_thresholds: dict[str, float],
    maximum_iteration: int,
) -> np.ndarray:
    return first_crossing(
        stopping_condition(family, bundle, parameters, ncp_thresholds),
        maximum_iteration,
    )


def field_score(
    bundle: dict[str, object], stop: np.ndarray, rows: np.ndarray
) -> tuple[float, float, int]:
    prediction = gather_path(bundle["path"], stop)
    selected = np.asarray(rows, dtype=bool)
    return v3d.field_score(
        prediction[selected],
        bundle["truth"][selected],
        bundle["sources"][selected],
    )


def screen_row(
    family: str,
    parameters: dict[str, float],
    parameter_rank: int,
    bundle: dict[str, object],
    tune_rows: np.ndarray,
    ncp_thresholds: dict[str, float],
    maximum_iteration: int,
) -> dict[str, object]:
    stop = stop_indices(
        family, bundle, parameters, ncp_thresholds, maximum_iteration
    )
    selected = np.asarray(tune_rows, dtype=bool)
    score = field_score(bundle, stop, selected)
    a_calls, at_calls = effective_operator_calls(stop, maximum_iteration)
    return {
        "selection_role": "v_tune",
        "family": family,
        "parameter_rank": int(parameter_rank),
        "tau": None,
        "camera_max_factor": None,
        "ncp_multiplier": None,
        **parameters,
        "source_field_mean_rel_l2": float(score[0]),
        "source_field_median_rel_l2": float(score[1]),
        "independent_field_count": int(score[2]),
        "layout_row_count": int(np.sum(selected)),
        "mean_stop_iteration": float(np.mean(stop[selected])),
        "median_stop_iteration": float(np.median(stop[selected])),
        "p90_stop_iteration": float(np.quantile(stop[selected], 0.9)),
        "forced_cap_fraction": float(np.mean(stop[selected] == maximum_iteration)),
        "mean_a_calls": float(np.mean(a_calls[selected])),
        "mean_at_calls": float(np.mean(at_calls[selected])),
        "mean_total_operator_calls": float(np.mean(a_calls[selected] + at_calls[selected])),
        "maximum_total_operator_calls": int(
            np.max(a_calls[selected] + at_calls[selected])
        ),
        "per_sample_truth_used_at_runtime": False,
        "audit_camera_used": False,
        "test_domain_used": False,
    }


def choose_equivalent(
    rows: list[dict[str, object]], tolerance: float
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot choose from an empty stopping screen")
    best = min(float(row["source_field_mean_rel_l2"]) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row["source_field_mean_rel_l2"]) <= best + float(tolerance)
    ]
    return dict(
        min(
            eligible,
            key=lambda row: (
                float(row["mean_total_operator_calls"]),
                int(row["maximum_total_operator_calls"]),
                int(row["parameter_rank"]),
            ),
        )
    )


def parameter_grid(config: dict, family: str) -> list[dict[str, float]]:
    protocol = config["selection_protocol"]
    if family == "self_discrepancy":
        return [
            {"tau": float(tau)}
            for tau in protocol["self_discrepancy_tau_grid"]
        ]
    if family == "camera_discrepancy":
        return [
            {"tau": float(tau), "camera_max_factor": float(factor)}
            for tau in protocol["self_discrepancy_tau_grid"]
            for factor in protocol["camera_max_factor_grid"]
        ]
    if family == "ncp":
        return [
            {"ncp_multiplier": float(multiplier)}
            for multiplier in protocol["ncp_reference_multiplier_grid"]
        ]
    if family == "hybrid":
        return [
            {
                "tau": float(tau),
                "camera_max_factor": float(factor),
                "ncp_multiplier": float(multiplier),
            }
            for tau in protocol["self_discrepancy_tau_grid"]
            for factor in protocol["camera_max_factor_grid"]
            for multiplier in protocol["ncp_reference_multiplier_grid"]
        ]
    if family == "generator_discrepancy":
        return [
            {"tau": float(tau)}
            for tau in protocol["generator_sigma_tau_grid"]
        ]
    raise ValueError(f"unknown stopping family: {family}")


def truth_oracle_stop(bundle: dict[str, object]) -> np.ndarray:
    path = bundle["path"]
    truth = bundle["truth"]
    error = np.linalg.norm(
        (path - truth[None]).reshape(path.shape[0], path.shape[1], -1), axis=2
    ) / np.maximum(
        np.linalg.norm(truth.reshape(len(truth), -1), axis=1)[None], 1e-12
    )
    return np.argmin(error, axis=0).astype(np.int64)
