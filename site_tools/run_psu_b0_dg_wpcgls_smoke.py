#!/usr/bin/env python3
"""Test whether detector covariance whitening improves 3-D BOST reconstruction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any

import numpy as np
import torch

from demo_t16_operator.detector_covariance_whitening import (
    DetectorCovarianceWhitening,
    WhitenedMeasurementOperator,
)
from demo_t16_operator.detector_graph_covariance import (
    CovarianceFit,
    component_covariance,
    detector_graph_spectral_basis,
    fit_component_iid_covariance,
    fit_diagonal_shrinkage_covariance,
    fit_gated_graph_covariance,
    graph_heat_eigenvalues,
    simulate_flowoff_repeats,
)
from demo_t16_operator.psu_b0_classical_baselines import (
    GeneralizedSobolevDirection,
    preconditioned_cgls_reconstruction,
)
from demo_t16_operator.psu_b0_detector_graph_features import (
    build_detector_knn_graph,
)
from demo_t16_operator.psu_b0_reaction_phantoms import (
    reaction_morphology_batch,
)
from demo_t16_operator.psu_b0_streaming_operator import (
    zero_outer_boundary_support,
)
from site_tools.run_psu_b0_real_interface_audit import (
    _make_operator,
    load_real_support_geometry,
)
from site_tools.run_psu_b0_spectral_preconditioner_pilot import (
    _field_metrics,
    _load_json,
)


SCHEMA = "psu-b0-dg-wpcgls-smoke-report-1.0"
STATUS = "DEVELOPMENT_SMOKE_COMPLETE_NO_SUPERIORITY_CLAIM"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _scalar_iid_fit(calibration: np.ndarray) -> CovarianceFit:
    values = np.asarray(calibration, dtype=np.float64)
    mean = np.mean(values, axis=0)
    centered = values - mean[None]
    variance = float(np.sum(np.square(centered)) / max(centered.size - 1, 1))
    return CovarianceFit(
        kind="scalar_iid",
        mean=mean,
        sigma2=None,
        spatial_eigenvalues=None,
        component_covariance=None,
        node_amplitude=None,
        low_rank_vectors=None,
        low_rank_eigenvalues=None,
        diagonal_variance=np.full(mean.shape, max(variance, 1e-12)),
        parameters={
            "calibration_repeat_count": float(values.shape[0]),
            "scale_estimator": "GLOBAL_UNBIASED_VARIANCE",
        },
        training_nll_per_dimension=0.0,
    )


def _oracle_fit(
    family: str,
    *,
    laplacian_eigenvalues: np.ndarray,
    rays_per_view: int,
) -> CovarianceFit:
    if family == "component_iid":
        diffusion_time = 0.0
        spatial_fraction = 0.0
        correlation = 0.2
        variance_ratio = 1.25
    elif family == "graph_heat":
        diffusion_time = 1.2
        spatial_fraction = 0.75
        correlation = 0.35
        variance_ratio = 1.35
    else:
        raise ValueError(f"unsupported oracle family: {family}")
    return CovarianceFit(
        kind=f"oracle_{family}",
        mean=np.zeros((rays_per_view, 2), dtype=np.float64),
        sigma2=1.0,
        spatial_eigenvalues=graph_heat_eigenvalues(
            laplacian_eigenvalues,
            diffusion_time=diffusion_time,
            spatial_fraction=spatial_fraction,
        ),
        component_covariance=component_covariance(
            correlation=correlation,
            variance_ratio=variance_ratio,
        ),
        node_amplitude=None,
        low_rank_vectors=None,
        low_rank_eigenvalues=None,
        diagonal_variance=None,
        parameters={
            "diffusion_time": diffusion_time,
            "spatial_fraction": spatial_fraction,
            "component_correlation": correlation,
            "component_variance_ratio": variance_ratio,
            "calibration_repeat_count": 1.0e12,
            "oracle_truth_parameters": 1.0,
        },
        training_nll_per_dimension=0.0,
    )


def _calibrate(
    *,
    family: str,
    family_index: int,
    config: dict[str, Any],
    graph: dict[str, Any],
    eigenpairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, list[CovarianceFit]], list[dict[str, Any]]]:
    calibration_config = config["covariance_calibration"]
    repeat_count = int(calibration_config["repeat_count"])
    graph_grid = {
        key: tuple(float(value) for value in calibration_config["graph_grid"][key])
        for key in (
            "diffusion_times",
            "spatial_fractions",
            "correlations",
            "variance_ratios",
        )
    }
    fits: dict[str, list[CovarianceFit]] = {
        "scalar_iid": [],
        "component_iid": [],
        "diagonal_shrinkage": [],
        "dg_covgate": [],
        "oracle_covariance": [],
    }
    gate_rows = []
    for view, (laplacian_eigenvalues, eigenvectors) in enumerate(eigenpairs):
        detector_xy = graph["detector_xy"][view].detach().cpu().numpy()
        rng = np.random.default_rng(
            int(calibration_config["base_seed"])
            + 100_000 * int(family_index)
            + int(view)
        )
        calibration, _ = simulate_flowoff_repeats(
            family=family,
            repeat_count=repeat_count,
            rng=rng,
            laplacian_eigenvalues=laplacian_eigenvalues,
            eigenvectors=eigenvectors,
            detector_xy=detector_xy,
        )
        scalar = _scalar_iid_fit(calibration)
        component = fit_component_iid_covariance(
            calibration,
            laplacian_eigenvalues=laplacian_eigenvalues,
            eigenvectors=eigenvectors,
            correlations=graph_grid["correlations"],
            variance_ratios=graph_grid["variance_ratios"],
        )
        diagonal = fit_diagonal_shrinkage_covariance(
            calibration,
            prior_strength=float(
                calibration_config["diagonal_prior_strength"]
            ),
        )
        gated, gate = fit_gated_graph_covariance(
            calibration,
            laplacian_eigenvalues=laplacian_eigenvalues,
            eigenvectors=eigenvectors,
            graph_grid=graph_grid,
            validation_fraction=float(
                calibration_config["validation_fraction"]
            ),
            minimum_validation_gain_per_dimension=float(
                calibration_config[
                    "minimum_validation_gain_per_dimension"
                ]
            ),
        )
        fits["scalar_iid"].append(scalar)
        fits["component_iid"].append(component)
        fits["diagonal_shrinkage"].append(diagonal)
        fits["dg_covgate"].append(gated)
        fits["oracle_covariance"].append(
            _oracle_fit(
                family,
                laplacian_eigenvalues=laplacian_eigenvalues,
                rays_per_view=int(config["geometry"]["rays_per_view"]),
            )
        )
        gate_rows.append(
            {
                "noise_family": family,
                "view": view,
                **gate,
            }
        )
    return fits, gate_rows


def _flowon_unit_noise(
    *,
    family: str,
    family_index: int,
    sample_count: int,
    config: dict[str, Any],
    graph: dict[str, Any],
    eigenpairs: list[tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor:
    rows = []
    for view, (laplacian_eigenvalues, eigenvectors) in enumerate(eigenpairs):
        rng = np.random.default_rng(
            int(config["data"]["flowon_noise_seed"])
            + 100_000 * int(family_index)
            + int(view)
        )
        detector_xy = graph["detector_xy"][view].detach().cpu().numpy()
        samples, _ = simulate_flowoff_repeats(
            family=family,
            repeat_count=int(sample_count),
            rng=rng,
            laplacian_eigenvalues=laplacian_eigenvalues,
            eigenvectors=eigenvectors,
            detector_xy=detector_xy,
        )
        rows.append(samples)
    combined = np.stack(rows, axis=1)
    return torch.as_tensor(combined, dtype=torch.float32).flatten(1, 2)


def _make_whitening(
    *,
    method: str,
    family: str,
    fits: dict[str, list[CovarianceFit]],
    eigenvectors_by_view: list[np.ndarray],
    scale_by_view: torch.Tensor,
    config: dict[str, Any],
    eigenpairs: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> DetectorCovarianceWhitening:
    predictive = method not in {
        "oracle_covariance",
        "wrong_graph_assumption",
    }
    selected_fits = fits.get(method)
    if method == "wrong_graph_assumption":
        selected_fits = [
            _oracle_fit(
                "graph_heat",
                laplacian_eigenvalues=eigenvalues,
                rays_per_view=int(config["geometry"]["rays_per_view"]),
            )
            for eigenvalues, _ in eigenpairs
        ]
    if selected_fits is None:
        raise ValueError(f"unknown whitening method: {method}")
    whitening = DetectorCovarianceWhitening(
        selected_fits,
        eigenvectors_by_view=eigenvectors_by_view,
        scale_by_view=scale_by_view,
        predictive_mean_correction=predictive,
        dtype=torch.float32,
    ).to(device)
    if method == "wrong_graph_assumption":
        rays = int(config["geometry"]["rays_per_view"])
        index = torch.arange(2 * rays, device=device).reshape(rays, 2)
        index = torch.flip(index, dims=(0,)).reshape(-1)
        with torch.no_grad():
            permuted = whitening.matrix[:, index][:, :, index]
            whitening.matrix.copy_(permuted)
    return whitening


def _metric_rows(
    *,
    method: str,
    noise_family: str,
    families: list[str],
    relative_noise: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    elapsed_seconds: float,
    call_report: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for index, family in enumerate(families):
        rows.append(
            {
                "sample_index": index,
                "reaction_family": family,
                "noise_family": noise_family,
                "relative_noise": float(relative_noise[index]),
                "method": method,
                "field_relative_l2": float(
                    metrics["field_relative_l2"][index]
                ),
                "gradient_relative_l2": float(
                    metrics["gradient_relative_l2"][index]
                ),
                "front_top10_f1": float(
                    metrics["front_top10_f1"][index]
                ),
                "solver_elapsed_seconds": float(elapsed_seconds),
                "forward_calls": int(call_report["forward_calls"]),
                "adjoint_calls": int(call_report["adjoint_calls"]),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        (str(row["noise_family"]), int(row["sample_index"])): row
        for row in rows
        if row["method"] == "component_iid"
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["noise_family"]), str(row["method"])),
            [],
        ).append(row)
    output = []
    for (noise_family, method), selected in sorted(groups.items()):
        field = np.asarray(
            [float(row["field_relative_l2"]) for row in selected],
            dtype=np.float64,
        )
        gradient = np.asarray(
            [float(row["gradient_relative_l2"]) for row in selected],
            dtype=np.float64,
        )
        front = np.asarray(
            [float(row["front_top10_f1"]) for row in selected],
            dtype=np.float64,
        )
        gains = []
        front_gains = []
        for row in selected:
            reference = baseline[(noise_family, int(row["sample_index"]))]
            reference_field = float(reference["field_relative_l2"])
            gains.append(
                100.0
                * (reference_field - float(row["field_relative_l2"]))
                / max(reference_field, 1e-12)
            )
            front_gains.append(
                float(row["front_top10_f1"])
                - float(reference["front_top10_f1"])
            )
        gain = np.asarray(gains, dtype=np.float64)
        output.append(
            {
                "noise_family": noise_family,
                "method": method,
                "sample_count": len(selected),
                "field_relative_l2_mean": float(np.mean(field)),
                "field_relative_l2_median": float(np.median(field)),
                "gradient_relative_l2_mean": float(np.mean(gradient)),
                "front_top10_f1_mean": float(np.mean(front)),
                "field_gain_vs_component_iid_percent_mean": float(
                    np.mean(gain)
                ),
                "field_gain_vs_component_iid_percent_median": float(
                    np.median(gain)
                ),
                "field_harm_over_one_percent_rate": float(
                    np.mean(gain < -1.0)
                ),
                "front_f1_gain_vs_component_iid_mean": float(
                    np.mean(front_gains)
                ),
            }
        )
    return output


def _mechanism_decision(
    summaries: list[dict[str, Any]],
    *,
    criteria: dict[str, Any],
) -> dict[str, Any]:
    lookup = {
        (str(row["noise_family"]), str(row["method"])): row
        for row in summaries
    }
    heat_oracle = lookup[("graph_heat", "oracle_covariance")]
    heat_gated = lookup[("graph_heat", "dg_covgate")]
    iid_gated = lookup[("component_iid", "dg_covgate")]
    heat_wrong = lookup[("graph_heat", "wrong_graph_assumption")]
    oracle_gain = float(
        heat_oracle["field_gain_vs_component_iid_percent_median"]
    )
    gated_gain = float(
        heat_gated["field_gain_vs_component_iid_percent_median"]
    )
    retention = gated_gain / max(oracle_gain, 1e-12)
    checks = {
        "oracle_mechanism_visible": oracle_gain
        >= float(
            criteria[
                "graph_heat_oracle_median_field_gain_percent_minimum"
            ]
        ),
        "fitted_graph_gain": gated_gain
        >= float(
            criteria[
                "graph_heat_dg_covgate_median_field_gain_percent_minimum"
            ]
        ),
        "fitted_retains_oracle_gain": retention
        >= float(
            criteria[
                "graph_heat_dg_covgate_oracle_gain_retention_minimum"
            ]
        ),
        "iid_median_harm": -float(
            iid_gated["field_gain_vs_component_iid_percent_median"]
        )
        <= float(
            criteria[
                "component_iid_dg_covgate_median_field_harm_percent_maximum"
            ]
        ),
        "iid_harm_rate": float(
            iid_gated["field_harm_over_one_percent_rate"]
        )
        <= float(
            criteria[
                "component_iid_dg_covgate_harm_over_one_percent_rate_maximum"
            ]
        ),
        "wrong_model_trails_oracle": oracle_gain
        - float(
            heat_wrong["field_gain_vs_component_iid_percent_median"]
        )
        >= float(
            criteria["wrong_graph_must_trail_oracle_field_gain_percent"]
        ),
    }
    return {
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "graph_heat_oracle_median_field_gain_percent": oracle_gain,
        "graph_heat_dg_covgate_median_field_gain_percent": gated_gain,
        "dg_covgate_oracle_gain_retention": float(retention),
        "verdict": (
            "MECHANISM_VISIBLE_ADVANCE_TO_PREREGISTERED_MULTI_SEED_PILOT"
            if all(checks.values())
            else "DEVELOPMENT_SMOKE_DO_NOT_CLAIM_REVISE_OR_STOP"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_smoke(
    *,
    root: Path,
    config_path: Path,
    view_root: Path,
    device_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_json(config_path)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(8)
    geometry_config = config["geometry"]
    detector_config = _load_json(
        root / str(config["source_detector_config"])
    )
    graph_config = detector_config["detector_graph"]
    view_count = int(geometry_config["view_count"])
    rays_per_view = int(geometry_config["rays_per_view"])
    grid_size = int(geometry_config["grid_size"])
    started = time.perf_counter()

    geometry, _ = load_real_support_geometry(
        view_root,
        rays_per_view=rays_per_view,
        sample_count=int(
            geometry_config["finite_aperture_sample_count"]
        ),
    )
    operator = _make_operator(
        geometry,
        grid_size=grid_size,
        dtype=torch.float32,
    ).to(device)
    operator.support.copy_(
        zero_outer_boundary_support((grid_size,) * 3).to(device)
    )
    graph = build_detector_knn_graph(
        geometry["detector_xy"],
        view_count=view_count,
        rays_per_view=rays_per_view,
        neighbor_count=int(graph_config["neighbor_count"]),
        least_squares_ridge=float(
            graph_config["least_squares_ridge"]
        ),
    )
    eigenpairs = [
        detector_graph_spectral_basis(graph, view_index=view)
        for view in range(view_count)
    ]
    eigenvectors_by_view = [vectors for _, vectors in eigenpairs]

    families = [str(value) for value in config["data"]["reaction_families"]]
    field_seeds = [
        int(config["data"]["field_seed_start"]) + index
        for index in range(len(families))
    ]
    truth = reaction_morphology_batch(
        grid_size=grid_size,
        families=families,
        seeds=field_seeds,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        clean = operator(truth).detach()
    clean_by_view = clean.reshape(
        len(truth),
        view_count,
        rays_per_view,
        2,
    )
    view_rms = torch.sqrt(
        torch.mean(clean_by_view.square(), dim=(2, 3)).clamp_min(1e-20)
    )
    floor = (
        float(config["data"]["minimum_view_signal_fraction"])
        * torch.mean(view_rms, dim=1, keepdim=True)
    )
    levels = tuple(
        float(value) for value in config["data"]["relative_noise_levels"]
    )
    relative_noise = torch.as_tensor(
        [levels[index % len(levels)] for index in range(len(truth))],
        dtype=torch.float32,
        device=device,
    )
    view_factors = torch.as_tensor(
        config["data"]["view_noise_factors"],
        dtype=torch.float32,
        device=device,
    )
    scale_by_view = (
        relative_noise[:, None]
        * torch.maximum(view_rms, floor)
        * view_factors[None]
    ).clamp_min(1e-8)

    all_rows: list[dict[str, Any]] = []
    all_gate_rows: list[dict[str, Any]] = []
    method_runtime: list[dict[str, Any]] = []
    for family_index, noise_family in enumerate(
        config["data"]["noise_families"]
    ):
        fits, gate_rows = _calibrate(
            family=str(noise_family),
            family_index=family_index,
            config=config,
            graph=graph,
            eigenpairs=eigenpairs,
        )
        all_gate_rows.extend(gate_rows)
        unit_noise = _flowon_unit_noise(
            family=str(noise_family),
            family_index=family_index,
            sample_count=len(truth),
            config=config,
            graph=graph,
            eigenpairs=eigenpairs,
        ).to(device)
        expanded_scale = scale_by_view.repeat_interleave(
            rays_per_view,
            dim=1,
        )[:, :, None]
        observation = clean + unit_noise * expanded_scale

        for method in config["methods"]:
            whitening = _make_whitening(
                method=str(method),
                family=str(noise_family),
                fits=fits,
                eigenvectors_by_view=eigenvectors_by_view,
                scale_by_view=scale_by_view,
                config=config,
                eigenpairs=eigenpairs,
                device=device,
            )
            wrapped = WhitenedMeasurementOperator(
                operator,
                whitening,
            ).to(device)
            prepared = wrapped.prepare_observation(observation)
            direction = GeneralizedSobolevDirection(
                (grid_size,) * 3,
                strength=float(config["solver"]["sobolev_strength"]),
                epsilon=float(config["solver"]["sobolev_epsilon"]),
            ).to(device)
            wrapped.reset_call_counts()
            _synchronize(device)
            method_started = time.perf_counter()
            with torch.no_grad():
                result = preconditioned_cgls_reconstruction(
                    wrapped,
                    prepared,
                    sigma_by_view=torch.ones(
                        (len(truth), view_count),
                        dtype=torch.float32,
                        device=device,
                    ),
                    view_mask=torch.ones(
                        (len(truth), view_count),
                        dtype=torch.float32,
                        device=device,
                    ),
                    rays_per_view=rays_per_view,
                    stages=int(config["solver"]["stages"]),
                    preconditioner=direction,
                )
            _synchronize(device)
            elapsed = time.perf_counter() - method_started
            metrics = _field_metrics(result.volume, truth)
            report = wrapped.call_report()
            expected_calls = int(config["solver"]["stages"])
            if report != {
                "forward_calls": expected_calls,
                "adjoint_calls": expected_calls,
            }:
                raise ValueError("wrapped solver violated the call budget")
            all_rows.extend(
                _metric_rows(
                    method=str(method),
                    noise_family=str(noise_family),
                    families=families,
                    relative_noise=relative_noise,
                    metrics=metrics,
                    elapsed_seconds=elapsed,
                    call_report=report,
                )
            )
            method_runtime.append(
                {
                    "noise_family": str(noise_family),
                    "method": str(method),
                    "seconds": float(elapsed),
                }
            )

    summaries = _summarize(all_rows)
    decision = _mechanism_decision(
        summaries,
        criteria=config["mechanism_gate"],
    )
    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_scope": config["evidence_scope"],
        "config_sha256": _sha256(config_path),
        "source_dataset_doi": config["source_dataset_doi"],
        "geometry": {
            **config["geometry"],
            "real_support_geometry_loaded": True,
            "private_local_path_exported": False,
        },
        "solver": config["solver"],
        "data": {
            "noise_families": config["data"]["noise_families"],
            "reaction_families": families,
            "field_seeds": field_seeds,
            "relative_noise_levels": levels,
            "flowon_observation_count": len(families)
            * len(config["data"]["noise_families"]),
            "experimental_observation_count": 0,
        },
        "calibration": {
            "repeat_count": int(
                config["covariance_calibration"]["repeat_count"]
            ),
            "gate_rows": all_gate_rows,
        },
        "method_runtime": method_runtime,
        "summaries": summaries,
        "mechanism_decision": decision,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "wall_seconds": float(time.perf_counter() - started),
            "maximum_rss_bytes": _max_rss_bytes(),
        },
        "claim_boundary": config["claim_boundary"],
    }
    return report, all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )
    report, rows = run_smoke(
        root=root,
        config_path=config_path,
        view_root=args.view_root.resolve(),
        device_name=str(args.device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    rows_path = args.output_dir / "metric_rows.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(rows_path, rows)
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["mechanism_decision"]["verdict"],
                "report": str(report_path),
                "metric_rows": str(rows_path),
                "wall_seconds": report["runtime"]["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
